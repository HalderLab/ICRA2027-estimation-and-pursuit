import csv
import math
from collections import deque
from datetime import datetime

from models import ControlCommand, Estimate, LidarMeasurement
from utils import wrap_pi


class ExperimentLogger:
    """Collect data from LiDAR, estimator, controller/cmd_vel, and OptiTrack."""

    HEADER = [
        "time",
        "object_id",
        "x21_lidar",
        "y21_lidar",
        "rho_lidar_raw",
        "alpha2_lidar_raw",
        "rho_measured",
        "alpha2_measured",
        "rho_bias",
        "alpha2_bias",
        "rho_hat",
        "alpha1_hat",
        "alpha2_hat",
        "v1_hat",
        "u1_hat",
        "v1_cmd",
        "u1_cmd",
        "v2_cmd",
        "u2_cmd",
        "mocap_valid",
        "mocap_sync_mode",
        "mocap_time_error",
        "x1_mocap",
        "y1_mocap",
        "th1_mocap",
        "x2_mocap",
        "y2_mocap",
        "th2_mocap",
        "rho_mocap",
        "alpha1_mocap",
        "alpha2_mocap",
        "rho_lidar_minus_mocap",
        "alpha2_lidar_minus_mocap",
        "rho_hat_minus_mocap",
        "alpha1_hat_minus_mocap",
        "alpha2_hat_minus_mocap",
    ]

    def __init__(self, filename: str | None = None) -> None:
        self.filename = filename or (
            f"robot_data_lidar_ekf_cbp_{datetime.now().strftime('%H%M%S')}.csv"
        )
        self.csv_file = open(self.filename, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(self.HEADER)
        self.pending_rows = deque()

    def queue_lidar_sample(
        self,
        measurement: LidarMeasurement,
        estimate: Estimate,
        command: ControlCommand,
        start_time: float,
    ) -> None:
        elapsed = measurement.timestamp - start_time

        # Optional augmented estimator states. EKF/PF estimators that do not
        # provide these fields are logged as NaN, while the MHE wrapper can
        # provide v1 and u1 directly.
        v1_hat = getattr(estimate, "v1", None)
        u1_hat = getattr(estimate, "u1", None)

        def fmt_optional(value):
            if value is None:
                return "nan"
            try:
                value = float(value)
            except (TypeError, ValueError):
                return "nan"
            return f"{value:.5f}" if math.isfinite(value) else "nan"

        base_row = [
            f"{elapsed:.4f}",
            measurement.object_id,
            f"{measurement.x21:.5f}",
            f"{measurement.y21:.5f}",
            f"{measurement.rho_raw:.5f}",
            f"{measurement.alpha2_raw:.5f}",
            f"{measurement.rho:.5f}",
            f"{measurement.alpha2:.5f}",
            f"{measurement.rho_bias:.5f}",
            f"{measurement.alpha2_bias:.5f}",
            f"{estimate.rho:.5f}",
            f"{estimate.alpha1:.5f}",
            f"{estimate.alpha2:.5f}",
            fmt_optional(v1_hat),
            fmt_optional(u1_hat),
            f"{command.leader_v:.5f}",
            f"{command.leader_u:.5f}",
            f"{command.follower_v:.5f}",
            f"{command.follower_u:.5f}",
        ]

        self.pending_rows.append(
            (
                measurement.timestamp,
                base_row,
                measurement.rho,
                measurement.alpha2,
                Estimate(**estimate.__dict__),
            )
        )

    def flush(self, optitrack) -> None:
        if len(optitrack.history) < 2:
            return

        newest_mocap_time = optitrack.newest_time

        while self.pending_rows:
            lidar_time, base_row, rho_lidar, alpha2_lidar, estimate = self.pending_rows[0]

            if lidar_time > newest_mocap_time:
                break

            self.pending_rows.popleft()
            synced = optitrack.interpolate_at(lidar_time)

            if synced is None:
                mocap_columns = [
                    0,
                    "unavailable",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            else:
                x1, y1, th1, x2, y2, th2, time_error, mode = synced
                dx = x2 - x1
                dy = y2 - y1
                rho_mocap = math.hypot(dx, dy)
                phi12 = math.atan2(dy, dx)
                phi21 = math.atan2(-dy, -dx)
                alpha1_mocap = wrap_pi(phi12 - th1)
                alpha2_mocap = wrap_pi(phi21 - th2)

                mocap_columns = [
                    1,
                    mode,
                    f"{time_error:.6f}",
                    f"{x1:.5f}",
                    f"{y1:.5f}",
                    f"{th1:.5f}",
                    f"{x2:.5f}",
                    f"{y2:.5f}",
                    f"{th2:.5f}",
                    f"{rho_mocap:.5f}",
                    f"{alpha1_mocap:.5f}",
                    f"{alpha2_mocap:.5f}",
                    f"{rho_lidar - rho_mocap:.5f}",
                    f"{wrap_pi(alpha2_lidar - alpha2_mocap):.5f}",
                    f"{estimate.rho - rho_mocap:.5f}",
                    f"{wrap_pi(estimate.alpha1 - alpha1_mocap):.5f}",
                    f"{wrap_pi(estimate.alpha2 - alpha2_mocap):.5f}",
                ]

            self.writer.writerow(base_row + mocap_columns)
            self.csv_file.flush()

    def close(self) -> None:
        self.csv_file.close()
