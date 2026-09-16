#!/usr/bin/env python3
"""Modular LiDAR estimation + CBP experiment with synchronized two-robot start.

Supported estimators:
    ekf : 4-state EKF, x = [rho, alpha1, alpha2, v1]
    pf  : 4-state particle filter, x = [rho, alpha1, alpha2, v1]
    mhe : MATLAB MHE

Key startup behavior
--------------------
No nonzero command is sent until:
    1) both /cmd_vel subscribers are discovered (optional),
    2) a recent valid LiDAR target exists,
    3) OptiTrack has both rigid bodies (optional),
    4) all conditions remain true for startup_barrier_hold seconds.

During the barrier, BOTH robots receive repeated zero Twist commands.
The first nonzero commands for both robots are then published in the SAME
timer callback.

Important for MHE:
    LiDAR samples are NOT inserted into MHE before synchronized start.
    Therefore the 31-sample MHE bootstrap horizon contains actual moving data,
    rather than stationary pre-start data.
"""

import math
from datetime import datetime
from typing import Optional

import rclpy
from rclpy.node import Node

from cmd_vel_block import CmdVelBlock
from controller_cbp import ConstantBearingController
from estimator_ekf import EKFEstimator
from estimator_pf import PFEstimator
from estimator_mhe_matlab import MHEMatlabEstimator
from experiment_logger import ExperimentLogger
from lidar_block import LidarBlock
from models import ControlCommand
from optitrack_block import OptiTrackBlock
from utils import message_time

try:
    from lidar_object_detection_ros2.msg import ObjectsArray
except Exception:
    ObjectsArray = None

try:
    from mocap4r2_msgs.msg import RigidBodies
except Exception:
    RigidBodies = None


class TwoAgentModularNode(Node):
    def __init__(self) -> None:
        super().__init__("two_agent_lidar_estimation_cbp_modular")

        if ObjectsArray is None:
            raise ImportError(
                "Could not import lidar_object_detection_ros2.msg.ObjectsArray."
            )
        if RigidBodies is None:
            raise ImportError(
                "Could not import mocap4r2_msgs.msg.RigidBodies."
            )

        # ==============================================================
        # ROS topics / LiDAR
        # ==============================================================
        self.declare_parameter("leader_cmd_topic", "/burger1/cmd_vel")
        self.declare_parameter("follower_cmd_topic", "/burger2/cmd_vel")
        self.declare_parameter("lidar_objects_topic", "/burger2/lod_objects")

        # Target modes:
        #   nearest       -> nearest object every frame
        #   fixed_id      -> only target_object_id
        #   physical_lock -> acquire once, then follow position continuity
        self.declare_parameter("target_mode", "nearest")
        self.declare_parameter("target_object_id", -1)
        self.declare_parameter("physical_lock_distance", 0.25)
        self.declare_parameter("physical_lock_max_lost_frames", 30)
        self.declare_parameter("physical_lock_reset_after_lost", False)

        self.declare_parameter("bearing_sign", 1.0)
        self.declare_parameter("rho_bias", 0.025)
        self.declare_parameter("alpha2_bias", -0.02)

        # ==============================================================
        # OptiTrack: validation/logging + optional startup readiness
        # ==============================================================
        self.declare_parameter("mocap_topic", "/rigid_bodies")
        self.declare_parameter("leader_rigid_body_name", "6")
        self.declare_parameter("follower_rigid_body_name", "7")
        self.declare_parameter("max_mocap_gap", 0.03)

        # ==============================================================
        # Estimator selection
        # ==============================================================
        self.declare_parameter("estimator_type", "mhe")
        self.declare_parameter(
            "mhe_matlab_folder", "/home/ros/lidar_modular/matlab"
        )

        # MHE bootstrap motion.
        self.declare_parameter("mhe_warmup_move", True)
        self.declare_parameter("mhe_warmup_follower_u", 0.0)

        # ==============================================================
        # Physical commands / CBP
        # ==============================================================
        self.declare_parameter("leader_v", 0.08)
        self.declare_parameter("leader_u", 0.0)
        self.declare_parameter("follower_v", 0.10)
        self.declare_parameter("mu", 2.0)
        self.declare_parameter("phi", math.radians(15.0))

        # 0 => automatic convention:
        #   EKF/PF -> -1 (legacy Python convention)
        #   MHE    -> +1 (MHE already outputs standard shape alpha1)
        self.declare_parameter("controller_alpha1_sign", 0.0)

        # For the 4-state EKF/PF this should normally be true if you want the
        # controller to actually use the estimated leader speed.
        self.declare_parameter("controller_use_estimated_v1", True)

        # ==============================================================
        # Shared EKF/PF measurement settings
        # ==============================================================
        self.declare_parameter("alpha1_initial_guess", math.radians(90))
        self.declare_parameter("v1_initial_guess", 0.14)
        self.declare_parameter("sigma_rho", 0.05)
        self.declare_parameter("sigma_alpha2", math.radians(2.5))

        # ==============================================================
        # 4-state EKF configuration
        # ==============================================================
        self.declare_parameter("q_rho", 0.05)
        self.declare_parameter("q_alpha1", math.radians(2.0))
        self.declare_parameter("q_alpha2", math.radians(2.0))
        self.declare_parameter("q_v1", 1.0e-6)
        self.declare_parameter("q_nominal_dt", 0.10)

        self.declare_parameter("p_rho_initial_std", 0.05)
        self.declare_parameter("p_alpha1_initial_std", 0.80)
        self.declare_parameter("p_alpha2_initial_std", 0.10)
        self.declare_parameter("p_v1_initial_std", 0.10)

        # ==============================================================
        # 4-state PF configuration
        # ==============================================================
        self.declare_parameter("pf_num_particles", 50000)
        self.declare_parameter("pf_random_seed", 2026)
        self.declare_parameter("pf_resample_neff_ratio", 0.50)
        self.declare_parameter("pf_alpha1_initial_uniform", True)

        self.declare_parameter("pf_init_rho_std", 0.30)
        self.declare_parameter("pf_init_alpha1_std", 0.80)
        self.declare_parameter("pf_init_alpha2_std", math.radians(8.0))
        self.declare_parameter("pf_init_v1_std", 0.05)

        self.declare_parameter("pf_process_rho_std", 0.05)
        self.declare_parameter("pf_process_alpha1_std", math.radians(2.0))
        self.declare_parameter("pf_process_alpha2_std", math.radians(2.0))
        self.declare_parameter("pf_process_v1_std", 0.001)
        self.declare_parameter("pf_nominal_dt", 0.10)

        # ==============================================================
        # Safety / timing
        # ==============================================================
        self.declare_parameter("rho_min", 0.01)
        self.declare_parameter("u2_max", 2.5)
        self.declare_parameter("max_measurement_age", 0.5)
        self.declare_parameter("control_rate", 30.0)

        # ==============================================================
        # SYNCHRONIZED STARTUP BARRIER
        # ==============================================================
        self.declare_parameter("require_cmd_subscribers", True)
        self.declare_parameter("startup_require_lidar", True)
        self.declare_parameter("startup_require_mocap", True)
        self.declare_parameter("startup_barrier_hold", 0.75)
        self.declare_parameter("startup_lidar_timeout", 0.50)
        self.declare_parameter("startup_status_period", 1.0)

        # ==============================================================
        # Load common parameters
        # ==============================================================
        self.leader_cmd_topic = str(
            self.get_parameter("leader_cmd_topic").value
        )
        self.follower_cmd_topic = str(
            self.get_parameter("follower_cmd_topic").value
        )

        self.estimator_type = str(
            self.get_parameter("estimator_type").value
        ).strip().lower()
        if self.estimator_type not in {"ekf", "pf", "mhe"}:
            raise ValueError(
                "estimator_type must be 'ekf', 'pf', or 'mhe'; got "
                f"{self.estimator_type!r}"
            )

        self.leader_v = float(self.get_parameter("leader_v").value)
        self.leader_u = float(self.get_parameter("leader_u").value)
        self.follower_v = float(self.get_parameter("follower_v").value)

        self.max_measurement_age = float(
            self.get_parameter("max_measurement_age").value
        )
        self.mhe_warmup_move = bool(
            self.get_parameter("mhe_warmup_move").value
        )
        self.mhe_warmup_follower_u = float(
            self.get_parameter("mhe_warmup_follower_u").value
        )

        self.control_rate = float(
            self.get_parameter("control_rate").value
        )
        if self.control_rate <= 0.0:
            raise ValueError("control_rate must be positive")

        rho_min = float(self.get_parameter("rho_min").value)

        self.require_cmd_subscribers = bool(
            self.get_parameter("require_cmd_subscribers").value
        )
        self.startup_require_lidar = bool(
            self.get_parameter("startup_require_lidar").value
        )
        self.startup_require_mocap = bool(
            self.get_parameter("startup_require_mocap").value
        )
        self.startup_barrier_hold = float(
            self.get_parameter("startup_barrier_hold").value
        )
        self.startup_lidar_timeout = float(
            self.get_parameter("startup_lidar_timeout").value
        )
        self.startup_status_period = float(
            self.get_parameter("startup_status_period").value
        )

        # ==============================================================
        # Build LiDAR block
        # ==============================================================
        self.target_mode = str(
            self.get_parameter("target_mode").value
        ).strip().lower()

        self.lidar = LidarBlock(
            target_object_id=self.get_parameter("target_object_id").value,
            bearing_sign=self.get_parameter("bearing_sign").value,
            rho_bias=self.get_parameter("rho_bias").value,
            alpha2_bias=self.get_parameter("alpha2_bias").value,
            rho_min=rho_min,
            target_mode=self.target_mode,
            physical_lock_distance=self.get_parameter(
                "physical_lock_distance"
            ).value,
            physical_lock_max_lost_frames=self.get_parameter(
                "physical_lock_max_lost_frames"
            ).value,
            physical_lock_reset_after_lost=self.get_parameter(
                "physical_lock_reset_after_lost"
            ).value,
        )

        # ==============================================================
        # Build estimator
        # ==============================================================
        if self.estimator_type == "mhe":
            self.get_logger().info(
                "Starting MATLAB Engine and persistent MHE()..."
            )
            self.estimator = MHEMatlabEstimator(
                matlab_folder=str(
                    self.get_parameter("mhe_matlab_folder").value
                )
            )
            self.get_logger().info(
                "MATLAB MHE estimator created successfully."
            )

        elif self.estimator_type == "ekf":
            self.estimator = EKFEstimator(
                leader_v=self.leader_v,
                leader_u=self.leader_u,
                alpha1_initial_guess=self.get_parameter(
                    "alpha1_initial_guess"
                ).value,
                v1_initial_guess=self.get_parameter(
                    "v1_initial_guess"
                ).value,
                sigma_rho=self.get_parameter("sigma_rho").value,
                sigma_alpha2=self.get_parameter("sigma_alpha2").value,
                q_rho=self.get_parameter("q_rho").value,
                q_alpha1=self.get_parameter("q_alpha1").value,
                q_alpha2=self.get_parameter("q_alpha2").value,
                q_v1=self.get_parameter("q_v1").value,
                q_nominal_dt=self.get_parameter("q_nominal_dt").value,
                p_rho_initial_std=self.get_parameter(
                    "p_rho_initial_std"
                ).value,
                p_alpha1_initial_std=self.get_parameter(
                    "p_alpha1_initial_std"
                ).value,
                p_alpha2_initial_std=self.get_parameter(
                    "p_alpha2_initial_std"
                ).value,
                p_v1_initial_std=self.get_parameter(
                    "p_v1_initial_std"
                ).value,
                rho_min=rho_min,
            )

        else:  # PF
            self.estimator = PFEstimator(
                leader_v=self.leader_v,
                leader_u=self.leader_u,
                rho_min=rho_min,
                sigma_rho=self.get_parameter("sigma_rho").value,
                sigma_alpha2=self.get_parameter("sigma_alpha2").value,
                num_particles=self.get_parameter(
                    "pf_num_particles"
                ).value,
                random_seed=self.get_parameter(
                    "pf_random_seed"
                ).value,
                resample_neff_ratio=self.get_parameter(
                    "pf_resample_neff_ratio"
                ).value,
                alpha1_initial_uniform=self.get_parameter(
                    "pf_alpha1_initial_uniform"
                ).value,
                alpha1_initial_guess=self.get_parameter(
                    "alpha1_initial_guess"
                ).value,
                v1_initial_guess=self.get_parameter(
                    "v1_initial_guess"
                ).value,
                pf_init_rho_std=self.get_parameter(
                    "pf_init_rho_std"
                ).value,
                pf_init_alpha1_std=self.get_parameter(
                    "pf_init_alpha1_std"
                ).value,
                pf_init_alpha2_std=self.get_parameter(
                    "pf_init_alpha2_std"
                ).value,
                pf_init_v1_std=self.get_parameter(
                    "pf_init_v1_std"
                ).value,
                process_rho_std=self.get_parameter(
                    "pf_process_rho_std"
                ).value,
                process_alpha1_std=self.get_parameter(
                    "pf_process_alpha1_std"
                ).value,
                process_alpha2_std=self.get_parameter(
                    "pf_process_alpha2_std"
                ).value,
                process_v1_std=self.get_parameter(
                    "pf_process_v1_std"
                ).value,
                pf_nominal_dt=self.get_parameter(
                    "pf_nominal_dt"
                ).value,
            )

        # ==============================================================
        # Controller
        # ==============================================================
        alpha1_sign_param = float(
            self.get_parameter("controller_alpha1_sign").value
        )
        if abs(alpha1_sign_param) < 1e-12:
            controller_alpha1_sign = (
                1.0 if self.estimator_type == "mhe" else -1.0
            )
        else:
            controller_alpha1_sign = alpha1_sign_param

        self.controller = ConstantBearingController(
            mu=self.get_parameter("mu").value,
            phi=self.get_parameter("phi").value,
            rho_min=rho_min,
            u2_max=self.get_parameter("u2_max").value,
            known_leader_v=self.leader_v,
            use_estimated_v1=self.get_parameter(
                "controller_use_estimated_v1"
            ).value,
            alpha1_sign=controller_alpha1_sign,
        )

        # ==============================================================
        # Command / OptiTrack / logger blocks
        # ==============================================================
        self.cmd_vel = CmdVelBlock(
            node=self,
            leader_topic=self.leader_cmd_topic,
            follower_topic=self.follower_cmd_topic,
        )

        self.optitrack = OptiTrackBlock(
            leader_name=str(
                self.get_parameter("leader_rigid_body_name").value
            ),
            follower_name=str(
                self.get_parameter("follower_rigid_body_name").value
            ),
            max_mocap_gap=self.get_parameter("max_mocap_gap").value,
        )

        log_name = (
            f"robot_data_lidar_{self.estimator_type}_cbp_"
            f"{datetime.now().strftime('%H%M%S')}.csv"
        )
        self.logger_block = ExperimentLogger(filename=log_name)

        # ==============================================================
        # Run state
        # ==============================================================
        node_now = self.now_seconds()

        # Experiment time zero is assigned at synchronized release.
        self.start_time: Optional[float] = None

        self.latest_measurement_time: Optional[float] = None
        self.latest_lidar_detection_time: Optional[float] = None
        self.current_estimate = None
        self.current_command = self.zero_command()

        self.last_prestart_measurement = None
        self.last_lidar_object_id = None

        self.mhe_ready_announced = False
        self.mhe_bootstrap_announced = False

        # Startup barrier state.
        self.run_started = False
        self.start_ready_since: Optional[float] = None
        self.last_startup_status_time = node_now - 1.0e9

        # ==============================================================
        # ROS subscriptions / timer
        # ==============================================================
        self.objects_sub = self.create_subscription(
            ObjectsArray,
            str(self.get_parameter("lidar_objects_topic").value),
            self.on_objects,
            10,
        )
        self.mocap_sub = self.create_subscription(
            RigidBodies,
            str(self.get_parameter("mocap_topic").value),
            self.on_mocap,
            10,
        )
        self.timer = self.create_timer(
            1.0 / self.control_rate, self.on_timer
        )

        self.get_logger().info(
            "Started modular LiDAR + "
            f"{self.estimator_type.upper()} + CBP node in SAFE WAIT mode."
        )
        self.get_logger().info(
            "Synchronized start barrier: "
            f"cmd_subscribers={self.require_cmd_subscribers}, "
            f"LiDAR={self.startup_require_lidar}, "
            f"OptiTrack={self.startup_require_mocap}, "
            f"hold={self.startup_barrier_hold:.2f} s"
        )
        self.get_logger().info(
            "No nonzero command will be sent until the barrier releases."
        )
        self.get_logger().info(
            "LiDAR target association: "
            f"mode={self.target_mode}, "
            f"target_object_id={self.get_parameter('target_object_id').value}"
        )
        self.get_logger().info(
            f"CSV={self.logger_block.filename}"
        )

    # ==============================================================
    # Helpers
    # ==============================================================
    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def zero_command(self) -> ControlCommand:
        return ControlCommand(
            leader_v=0.0,
            leader_u=0.0,
            follower_v=0.0,
            follower_u=0.0,
        )

    def bootstrap_motion_command(self) -> ControlCommand:
        """Command used immediately after synchronized release.

        Both robots receive their first nonzero command in the SAME callback.
        The follower steering is zero until an estimator/controller output is
        available. For MHE this is also the normal bootstrap command.
        """
        follower_u = (
            self.mhe_warmup_follower_u
            if self.estimator_type == "mhe"
            else 0.0
        )
        return ControlCommand(
            leader_v=self.leader_v,
            leader_u=self.leader_u,
            follower_v=self.follower_v,
            follower_u=follower_u,
        )

    def warmup_command(self) -> ControlCommand:
        return ControlCommand(
            leader_v=self.leader_v,
            leader_u=self.leader_u,
            follower_v=self.follower_v,
            follower_u=self.mhe_warmup_follower_u,
        )

    def closed_loop_command(self, estimate) -> ControlCommand:
        follower_u = self.controller.compute(
            estimate=estimate,
            follower_v=self.follower_v,
        )
        return ControlCommand(
            leader_v=self.leader_v,
            leader_u=self.leader_u,
            follower_v=self.follower_v,
            follower_u=follower_u,
        )

    def command_links_ready(self):
        leader_count = self.count_subscribers(self.leader_cmd_topic)
        follower_count = self.count_subscribers(self.follower_cmd_topic)

        if self.require_cmd_subscribers:
            ready = leader_count > 0 and follower_count > 0
        else:
            ready = True

        return ready, leader_count, follower_count

    def lidar_ready_for_start(self, now: float) -> bool:
        if not self.startup_require_lidar:
            return True
        if self.latest_lidar_detection_time is None:
            return False
        return (
            now - self.latest_lidar_detection_time
            <= self.startup_lidar_timeout
        )

    def mocap_ready_for_start(self) -> bool:
        if not self.startup_require_mocap:
            return True

        # OptiTrackBlock stores synchronized history only after it has seen
        # valid poses. Requiring at least one history entry ensures both
        # requested rigid bodies have appeared.
        history = getattr(self.optitrack, "history", None)
        return history is not None and len(history) >= 1

    def startup_conditions(self, now: float):
        cmd_ready, leader_count, follower_count = self.command_links_ready()
        lidar_ready = self.lidar_ready_for_start(now)
        mocap_ready = self.mocap_ready_for_start()

        all_ready = cmd_ready and lidar_ready and mocap_ready
        return (
            all_ready,
            cmd_ready,
            lidar_ready,
            mocap_ready,
            leader_count,
            follower_count,
        )

    def report_startup_status(
        self,
        now: float,
        cmd_ready: bool,
        lidar_ready: bool,
        mocap_ready: bool,
        leader_count: int,
        follower_count: int,
    ) -> None:
        if (
            now - self.last_startup_status_time
            < self.startup_status_period
        ):
            return

        self.last_startup_status_time = now

        hold_elapsed = (
            0.0
            if self.start_ready_since is None
            else now - self.start_ready_since
        )

        self.get_logger().info(
            "START WAIT: "
            f"cmd={cmd_ready} "
            f"(leader_subs={leader_count}, follower_subs={follower_count}), "
            f"lidar={lidar_ready}, mocap={mocap_ready}, "
            f"hold={hold_elapsed:.2f}/{self.startup_barrier_hold:.2f} s"
        )

    def release_synchronized_start(self, now: float) -> None:
        self.run_started = True
        self.start_time = now
        self.latest_measurement_time = None
        self.current_estimate = None

        # First nonzero command to BOTH robots in this same timer callback.
        command = self.bootstrap_motion_command()
        self.cmd_vel.publish(command)
        self.current_command = command

        self.get_logger().info(
            "========== SYNCHRONIZED START RELEASED =========="
        )
        self.get_logger().info(
            "First nonzero commands were sent to BOTH robots in the same "
            "timer callback."
        )
        self.get_logger().info(
            f"leader: v={command.leader_v:.3f}, u={command.leader_u:.3f}; "
            f"follower: v={command.follower_v:.3f}, "
            f"u={command.follower_u:.3f}"
        )

    # ==============================================================
    # LiDAR callback
    # ==============================================================
    def on_objects(self, msg: ObjectsArray) -> None:
        receive_now = self.now_seconds()
        sensor_time = message_time(msg, receive_now)

        measurement = self.lidar.process(msg, sensor_time)
        if measurement is None:
            return

        self.latest_lidar_detection_time = receive_now
        self.last_prestart_measurement = measurement

        if (
            self.target_mode == "physical_lock"
            and self.last_lidar_object_id is not None
            and measurement.object_id != self.last_lidar_object_id
        ):
            self.get_logger().info(
                "Physical LiDAR target retained while detector ID changed: "
                f"{self.last_lidar_object_id} -> {measurement.object_id}"
            )
        self.last_lidar_object_id = measurement.object_id

        # CRITICAL: do not fill EKF/PF/MHE with pre-start stationary data.
        if not self.run_started:
            return

        estimator_time = (
            receive_now
            if self.estimator_type == "mhe"
            else measurement.timestamp
        )

        try:
            estimate = self.estimator.step(
                t=estimator_time,
                rho=measurement.rho,
                alpha2=measurement.alpha2,
                v_self=self.current_command.follower_v,
                u_self=self.current_command.follower_u,
            )
        except Exception as exc:
            self.get_logger().error(
                f"Estimator step() failed: {exc}"
            )
            self.cmd_vel.stop()
            self.current_command = self.zero_command()
            return

        if estimate is None:
            return

        self.current_estimate = estimate
        self.latest_measurement_time = receive_now

        if (
            self.estimator_type == "mhe"
            and not estimate.ready
            and not self.mhe_bootstrap_announced
        ):
            self.get_logger().info(
                "MHE bootstrap started AFTER synchronized release. "
                "Collecting moving LiDAR history."
            )
            self.mhe_bootstrap_announced = True

        if (
            self.estimator_type == "mhe"
            and estimate.ready
            and not self.mhe_ready_announced
        ):
            self.get_logger().info(
                "MHE READY: full horizon obtained; switching to "
                "MHE-based CBP control."
            )
            self.mhe_ready_announced = True

        if estimate.ready:
            log_command = self.closed_loop_command(estimate)
        else:
            log_command = self.current_command

        if self.start_time is not None:
            self.logger_block.queue_lidar_sample(
                measurement=measurement,
                estimate=estimate,
                command=log_command,
                start_time=self.start_time,
            )
            self.logger_block.flush(self.optitrack)

    # ==============================================================
    # OptiTrack callback
    # ==============================================================
    def on_mocap(self, msg: RigidBodies) -> None:
        now = message_time(msg, self.now_seconds())
        if self.optitrack.process(msg, now):
            if self.run_started:
                self.logger_block.flush(self.optitrack)

    # ==============================================================
    # Timer
    # ==============================================================
    def on_timer(self) -> None:
        now = self.now_seconds()

        # ----------------------------------------------------------
        # PRE-START: repeated zeros + readiness barrier
        # ----------------------------------------------------------
        if not self.run_started:
            self.cmd_vel.stop()
            self.current_command = self.zero_command()

            (
                all_ready,
                cmd_ready,
                lidar_ready,
                mocap_ready,
                leader_count,
                follower_count,
            ) = self.startup_conditions(now)

            if all_ready:
                if self.start_ready_since is None:
                    self.start_ready_since = now
                    self.get_logger().info(
                        "All startup conditions are ready; beginning "
                        f"{self.startup_barrier_hold:.2f} s zero-command hold."
                    )

                if (
                    now - self.start_ready_since
                    >= self.startup_barrier_hold
                ):
                    self.release_synchronized_start(now)
                    return
            else:
                # Any readiness loss resets the continuous hold timer.
                self.start_ready_since = None

            self.report_startup_status(
                now,
                cmd_ready,
                lidar_ready,
                mocap_ready,
                leader_count,
                follower_count,
            )
            return

        # ----------------------------------------------------------
        # Immediately after release, before first estimator output:
        # keep BOTH robots moving with the common bootstrap command.
        # A recent LiDAR target was required for release, so normally
        # this lasts only until the next ~10 Hz LiDAR callback.
        # ----------------------------------------------------------
        if self.current_estimate is None:
            if self.lidar_ready_for_start(now):
                command = self.bootstrap_motion_command()
                self.cmd_vel.publish(command)
                self.current_command = command
            else:
                self.cmd_vel.stop()
                self.current_command = self.zero_command()
            return

        # ----------------------------------------------------------
        # Normal safety check after estimator starts
        # ----------------------------------------------------------
        measurement_is_valid = (
            self.latest_measurement_time is not None
            and (now - self.latest_measurement_time)
            <= self.max_measurement_age
        )

        if not measurement_is_valid:
            self.cmd_vel.stop()
            self.current_command = self.zero_command()
            return

        # ----------------------------------------------------------
        # MHE bootstrap
        # ----------------------------------------------------------
        if (
            self.estimator_type == "mhe"
            and not self.current_estimate.ready
        ):
            if self.mhe_warmup_move:
                command = self.warmup_command()
                self.cmd_vel.publish(command)
                self.current_command = command
            else:
                self.cmd_vel.stop()
                self.current_command = self.zero_command()
            return

        # ----------------------------------------------------------
        # EKF/PF normal operation, or MHE after ready
        # ----------------------------------------------------------
        try:
            predicted = self.estimator.predict(
                t=now,
                v_self=self.current_command.follower_v,
                u_self=self.current_command.follower_u,
            )
        except Exception as exc:
            self.get_logger().error(
                f"Estimator predict() failed: {exc}"
            )
            self.cmd_vel.stop()
            self.current_command = self.zero_command()
            return

        if predicted is None:
            self.cmd_vel.stop()
            self.current_command = self.zero_command()
            return

        self.current_estimate = predicted

        command = self.closed_loop_command(predicted)
        self.cmd_vel.publish(command)
        self.current_command = command

    # ==============================================================
    # Shutdown
    # ==============================================================
    def shutdown(self) -> None:
        self.cmd_vel.shutdown()
        self.current_command = self.zero_command()

        try:
            self.estimator.close()
        finally:
            self.logger_block.close()

        self.get_logger().info(
            "Stopped both robots, closed estimator, and closed logger."
        )


def main() -> None:
    rclpy.init()
    node = TwoAgentModularNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
