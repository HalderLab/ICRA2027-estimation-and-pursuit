#!/usr/bin/env python3
"""One robot, one MHE worker, one cmd_vel publisher. Waits for coordinator.
Run via mmc_burger1.py or mmc_burger3.py; see README_zh.md.
"""
import csv
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import signal
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from lidar_object_detection_ros2.msg import ObjectsArray

from mmc_math import MMCMath, wrap_pi, clamp, yaw_from_quat_zup
from mmc_protocol import LocalSession, ACTIVE, ROBOTS


def control_qos():
    return QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE)


from mmc_worker import mhe_worker


class LocalMMC(Node, MMCMath):
    def __init__(self, robot_default):
        super().__init__('mmc_local_'+robot_default)
        self.declare_parameter('robot', robot_default)
        self.robot = str(self.get_parameter('robot').value)
        if self.robot not in ROBOTS:
            raise ValueError('robot must be burger1 or burger3')
        self.peer_robot = 'burger3' if self.robot == 'burger1' else 'burger1'
        defaults = dict(v1=.15, v3=.15, mu=1.46225, desired_energy_E0=.00284423,
            kd=1000., use_dissipative_term=True, theta_diff_desired_deg=180.,
            sync_gain=.1, sync_initial_fraction=.01, sync_ramp_start=15., sync_ramp_duration=5.,
            mhe_warmup_open_loop_u=.30, mhe_warmup_timeout=3., mhe_runtime_warning=.08,
            control_hz=30., require_cmd_subscribers=True, max_measurement_age=.50,
            max_estimate_age=.50, heartbeat_timeout=.6, clock_tolerance=.25,
            rho_min=.03, rho_stop=.15, u_max=2.50, rho_bias=.05, bearing_sign=1.,
            bearing_bias=0., target_object_id=-1,
            enable_optitrack_logging=True, mocap_topic='/rigid_bodies',
            burger1_rigid_body_name='6', burger3_rigid_body_name='19',
            burger1_heading_offset=0., burger3_heading_offset=0., max_mocap_age_for_log=.1,
            log_directory='~/mmc_logs')
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.p = {key: self.get_parameter(key).value for key in defaults}
        p = self.p
        if any(not math.isfinite(v) for v in p.values() if isinstance(v, (int, float))):
            raise ValueError('Numeric parameters must be finite')
        if not (0 < p['v1'] <= .22 and 0 < p['v3'] <= .22 and 0 < p['u_max'] <= 2.84):
            raise ValueError('Commands must fit Burger speed limits')
        for key in ('control_hz', 'max_measurement_age', 'max_estimate_age', 'heartbeat_timeout',
                    'clock_tolerance', 'mhe_warmup_timeout', 'rho_min', 'rho_stop'):
            if p[key] <= 0:
                raise ValueError(key+' must be positive')
        self.v1_nom, self.v3_nom = p['v1'], p['v3']
        self.mu, self.E0, self.kd = p['mu'], p['desired_energy_E0'], p['kd']
        self.use_dissipative = p['use_dissipative_term']
        self.theta_diff_d = math.radians(p['theta_diff_desired_deg'])
        self.sync_gain_final = p['sync_gain']
        self.sync_initial_fraction = p['sync_initial_fraction']
        self.sync_ramp_start, self.sync_ramp_duration = p['sync_ramp_start'], p['sync_ramp_duration']
        self.rho_min, self.u_max = p['rho_min'], p['u_max']
        shared = {k:p[k] for k in ('v1','v3','mu','desired_energy_E0','kd','use_dissipative_term',
            'theta_diff_desired_deg','sync_gain','sync_initial_fraction','sync_ramp_start',
            'sync_ramp_duration','mhe_warmup_open_loop_u','mhe_warmup_timeout')}
        self.signature = hashlib.sha256(json.dumps(shared, sort_keys=True).encode()).hexdigest()[:16]
        self.session = LocalSession(p['heartbeat_timeout'], p['clock_tolerance'], p['mhe_warmup_timeout'])
        self.origin = time.monotonic()
        self.clock_offset = time.time()-time.monotonic()
        self.v_cmd = self.u_cmd = 0.
        self.measurement = None
        self.measurement_mono = -math.inf
        self.peer = None
        self.peer_mono = -math.inf
        self.peer_seq = -1
        self.estimate = None
        self.estimate_time = -math.inf
        self.worker_ready = False
        self.last_step_runtime = math.nan
        self.step_result_seq = 0
        self.warned_step_seq = 0
        self.last_warning = -math.inf
        self.status_seq = 0
        self.last_log_flush = self.origin
        self.last_status = self.origin-1.
        self.last_state = None
        self.truth_pose = {}
        self.truth_sub = None
        self.pub = self.create_publisher(Twist, f'/{self.robot}/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, f'/{self.robot}/mmc/status', control_qos())
        self.create_subscription(ObjectsArray, f'/{self.robot}/merged/lod_objects',
                                 self.on_objects, qos_profile_sensor_data)
        self.create_subscription(String, '/mmc/experiment/command', self.on_command, control_qos())
        self.create_subscription(String, f'/{self.peer_robot}/mmc/status', self.on_peer, control_qos())
        if p['enable_optitrack_logging']:
            try:
                from mocap4r2_msgs.msg import RigidBodies
                self.truth_sub = self.create_subscription(RigidBodies, p['mocap_topic'],
                                                          self.on_mocap, qos_profile_sensor_data)
            except ImportError:
                self.get_logger().warning('mocap4r2_msgs unavailable; continuing with local logs only')
        logdir = Path(p['log_directory']).expanduser()
        logdir.mkdir(parents=True, exist_ok=True)
        suffix = time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000000)
        self.log_path = logdir/f'{self.robot}_mmc_{suffix}.csv'
        self.logfile = self.log_path.open('w', newline='')
        fields = ['wall_time','elapsed','robot','agent','run_id','mode','reason','rho_meas',
            'bearing_meas','measurement_age','rho_hat','alpha1_hat','alpha2_hat','remote_v_hat',
            'remote_u_hat','estimate_age','mhe_cost','mhe_iterations','horizon_samples','mhe_step_s',
            'v_cmd','u_cmd','E','E0','gamma','lambda','sync_error','ke','u_nom','u_dis',
            'truth1_x','truth1_y','truth1_theta','truth3_x','truth3_y','truth3_theta']
        self.writer = csv.DictWriter(self.logfile, fieldnames=fields)
        self.writer.writeheader()
        (logdir/f'{self.robot}_mmc_{suffix}_config.json').write_text(json.dumps(p, indent=2))
        ctx = mp.get_context('spawn')
        self.inbox, self.outbox = ctx.Queue(128), ctx.Queue(128)
        self.worker = ctx.Process(target=mhe_worker, args=(self.inbox, self.outbox), daemon=True)
        self.worker.start()
        self.timer = self.create_timer(1./p['control_hz'], self.on_timer)
        self.get_logger().info(f'{self.robot}: waiting for experiment start; CSV: {self.log_path}')

    def stop(self, reason):
        if self.session.state != 'STOPPED':
            self.get_logger().error(reason)
        self.session.stop(reason)
        self.command(0., 0.)

    def command(self, v, u):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(u)
        self.pub.publish(msg)
        self.v_cmd, self.u_cmd = float(v), float(u)

    def enqueue(self, kind, mono, rho=0., bearing=0.):
        try:
            self.inbox.put_nowait((kind, mono-self.origin, self.v_cmd, self.u_cmd, rho, bearing))
        except Full:
            self.stop('MHE input queue full: board cannot keep up')

    def drain_worker(self):
        while True:
            try:
                item = self.outbox.get_nowait()
            except Empty:
                break
            if item.get('error'):
                self.stop(item['error'])
            elif item.get('worker_ready'):
                self.worker_ready = True
                self.get_logger().info(f"Numba ready; precompile {item.get('compile_seconds', 0.):.1f}s; waiting for experiment start")
            elif 'output' in item:
                self.estimate = item['output']
                self.estimate_time = item['event_time']+self.origin
                if item['kind'] == 'step':
                    self.last_step_runtime = item['runtime']
                    self.step_result_seq += 1
        if not self.worker.is_alive():
            self.stop('MHE worker exited')

    def local_ready(self, now):
        return (self.worker_ready and self.worker.is_alive() and self.measurement is not None
                and now-self.measurement_mono <= self.p['max_measurement_age']
                and (not self.p['require_cmd_subscribers'] or self.pub.get_subscription_count() > 0))

    def mhe_ready(self, now):
        o = self.estimate
        return bool(o and o['ready'] and now-self.estimate_time <= self.p['max_estimate_age']
            and now-(self.origin+o['lastMeasurementTime']) <= self.p['max_measurement_age']
            and all(math.isfinite(o[k]) for k in ('rho','alpha1','alpha2','v1','u1')))

    def mhe_health_detail(self, now):
        """Diagnostics only: does not change readiness or stopping thresholds."""
        o = self.estimate or {}
        output_age = now-self.estimate_time
        correction_age = now-(self.origin+o.get('lastMeasurementTime', math.nan))
        raw_age = now-self.measurement_mono
        causes = []
        if not o or not o.get('ready', False):
            causes.append('estimator_not_ready')
        if output_age > self.p['max_estimate_age']:
            causes.append('worker_output_stale')
        if correction_age > self.p['max_measurement_age']:
            causes.append('MHE_correction_stale')
        invalid = [k for k in ('rho','alpha1','alpha2','v1','u1')
                   if not math.isfinite(o.get(k, math.nan))]
        if invalid:
            causes.append('nonfinite='+','.join(invalid))
        return (f"causes={','.join(causes) or 'none'}; "
                f"raw_lidar_age={raw_age:.3f}s; worker_output_age={output_age:.3f}s; "
                f"MHE_correction_age={correction_age:.3f}s; "
                f"step_runtime={self.last_step_runtime:.3f}s; "
                f"horizon_samples={o.get('horizonSamples', 0)}")

    def on_objects(self, msg):
        candidates = []
        for obj in msg.objects:
            oid = int(obj.id) if hasattr(obj, 'id') else -1
            x, y = float(obj.pose.x), float(obj.pose.y)
            rho = math.hypot(x, y)
            if not math.isfinite(rho) or rho < self.rho_min:
                continue
            if self.p['target_object_id'] >= 0 and oid != self.p['target_object_id']:
                continue
            candidates.append((rho, x, y))
        if not candidates:
            return
        _, x, y = min(candidates, key=lambda v:v[0])
        y *= self.p['bearing_sign']
        rho = max(math.hypot(x, y)+self.p['rho_bias'], self.rho_min)
        bearing = wrap_pi(math.atan2(y, x)+self.p['bearing_bias'])
        now = time.monotonic()
        self.measurement, self.measurement_mono = (rho, bearing), now
        if self.session.state in ('WARMUP', 'RUN'):
            self.enqueue('step', now, rho, bearing)

    def on_command(self, msg):
        try:
            packet = json.loads(msg.data)
            if not isinstance(packet, dict):
                return
            self.session.receive(packet, time.monotonic(), time.time(), self.local_ready(time.monotonic()))
            if self.session.state == 'STOPPED':
                self.command(0., 0.)
        except (ValueError, TypeError, KeyError):
            return

    def on_peer(self, msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict) or data.get('robot') != self.peer_robot:
                return
            if not math.isfinite(float(data['stamp'])) or abs(time.time()-float(data['stamp'])) > self.p['clock_tolerance']:
                if self.session.state in ACTIVE:
                    self.stop('peer clock offset / transport delay exceeds tolerance')
                return
            seq = int(data['seq'])
            # A restart is a fault once armed, not a new readiness signal.
            if seq <= self.peer_seq:
                return
            self.peer_seq, self.peer, self.peer_mono = seq, data, time.monotonic()
        except (ValueError, TypeError, KeyError):
            return

    def on_mocap(self, msg):
        now = time.monotonic()
        for rb in msg.rigidbodies:
            name = str(rb.rigid_body_name)
            for robot in ROBOTS:
                if name == self.p[robot+'_rigid_body_name']:
                    q = rb.pose.orientation
                    yaw = wrap_pi(yaw_from_quat_zup(q.x,q.y,q.z,q.w)+self.p[robot+'_heading_offset'])
                    self.truth_pose[robot] = (rb.pose.position.x, rb.pose.position.y, yaw, now)

    def publish_status(self, now):
        data = self.session.status(self.robot, self.local_ready(now), self.mhe_ready(now))
        data.update(stamp=time.time(), seq=self.status_seq, config_signature=self.signature)
        self.status_seq += 1
        msg = String(); msg.data = json.dumps(data, allow_nan=False)
        self.status_pub.publish(msg)

    def on_timer(self):
        try:
            self.tick()
        except Exception as exc:
            self.stop(f'controller exception: {type(exc).__name__}: {exc}')
            self.publish_status(time.monotonic())

    def tick(self):
        now = time.monotonic()
        self.drain_worker()
        if self.session.state == 'WAIT':
            self.clock_offset = time.time()-now
        if abs((time.time()-now)-self.clock_offset) > self.p['clock_tolerance'] and self.session.state in ACTIVE:
            self.stop('system clock jumped during experiment')
        self.session.tick(now, self.local_ready(now), self.mhe_ready(now), self.peer, now-self.peer_mono)
        if (self.session.state == 'STOPPED' and self.session.reason == 'local MHE is stale or invalid'):
            self.session.reason += '; '+self.mhe_health_detail(now)
        if self.session.just_started:
            self.estimate, self.estimate_time = None, -math.inf
            self.enqueue('reset', now)
        q, c = {}, {}
        if self.session.state in ('WARMUP', 'RUN'):
            elapsed = now-self.session.start_mono
            # Even before MHE first becomes ready, detect a stalled worker.
            if elapsed > self.p['max_estimate_age'] and now-self.estimate_time > self.p['max_estimate_age']:
                self.stop('MHE worker backlog / estimate age exceeded limit; '+self.mhe_health_detail(now))
            elif self.session.state == 'WARMUP':
                if self.measurement[0] < self.p['rho_stop']:
                    self.stop('measured separation below rho_stop')
                else:
                    self.command(self.v1_nom if self.robot=='burger1' else self.v3_nom,
                                 clamp(self.p['mhe_warmup_open_loop_u'], -self.u_max, self.u_max))
            else:
                o = self.estimate
                a1, a2 = ((o['alpha2'], o['alpha1']) if self.robot=='burger1' else (o['alpha1'], o['alpha2']))
                q = self.shape_to_mmc(np.array([o['rho'], a1, a2]))
                if q['rho'] < self.p['rho_stop']:
                    self.stop('estimated separation below rho_stop')
                else:
                    c = self.compute_local_command(q, elapsed, self.robot)
                    if not math.isfinite(c['u_self_cmd']):
                        self.stop('nonfinite MMC command')
                    else:
                        self.command(self.v1_nom if self.robot=='burger1' else self.v3_nom, c['u_self_cmd'])
            if self.session.state in ('WARMUP', 'RUN'):
                # Known applied control sample and predictor at every control tick.
                self.enqueue('predict', time.monotonic())
        else:
            self.command(0., 0.)
        if now-self.last_status >= .05:
            self.publish_status(now)
            self.last_status = now
        if self.session.state != self.last_state:
            self.get_logger().info(f'{self.robot}: {self.session.state} {self.session.reason}')
            self.last_state = self.session.state
        if (self.session.state in ('WARMUP', 'RUN') and self.step_result_seq > self.warned_step_seq
                and self.p['mhe_runtime_warning'] > 0 and self.last_step_runtime > self.p['mhe_runtime_warning']
                and now-self.last_warning > 2.):
            self.get_logger().warning(self.mhe_health_detail(now))
            self.warned_step_seq = self.step_result_seq
            self.last_warning = now
        self.log(now, q, c)

    def log(self, now, q, c):
        o = self.estimate or {}
        a1, a2 = o.get('alpha1', math.nan), o.get('alpha2', math.nan)
        if self.robot == 'burger1':
            a1, a2 = a2, a1
        row = dict(wall_time=time.time(), elapsed=now-self.session.start_mono if self.session.start_mono else math.nan,
            robot=self.robot, agent=1 if self.robot=='burger1' else 2, run_id=self.session.run_id,
            mode=self.session.state, reason=self.session.reason,
            rho_meas=self.measurement[0] if self.measurement else math.nan,
            bearing_meas=self.measurement[1] if self.measurement else math.nan,
            measurement_age=now-self.measurement_mono, rho_hat=o.get('rho', math.nan),
            alpha1_hat=a1, alpha2_hat=a2, remote_v_hat=o.get('v1', math.nan), remote_u_hat=o.get('u1', math.nan),
            estimate_age=now-self.estimate_time, mhe_cost=o.get('cost', math.nan),
            mhe_iterations=o.get('iterations', 0), horizon_samples=o.get('horizonSamples', 0),
            mhe_step_s=self.last_step_runtime, v_cmd=self.v_cmd, u_cmd=self.u_cmd,
            E=q.get('E', math.nan), E0=self.E0, gamma=q.get('gamma', math.nan),
            **{'lambda':q.get('lambda', math.nan)})
        for k in ('sync_error','ke','u_nom','u_dis'):
            row[k] = c.get(k, math.nan)
        for robot, label in [('burger1','truth1'),('burger3','truth3')]:
            pose = self.truth_pose.get(robot)
            vals = pose[:3] if pose and now-pose[3] <= self.p['max_mocap_age_for_log'] else [math.nan]*3
            for name, val in zip(('x','y','theta'), vals):
                row[label+'_'+name] = val
        self.writer.writerow(row)
        if now-self.last_log_flush >= 1.:
            self.logfile.flush(); self.last_log_flush = now

    def close(self):
        self.timer.cancel()
        self.session.stop('local controller shutdown')
        for _ in range(5):
            self.command(0.,0.)
            self.publish_status(time.monotonic())
            time.sleep(.03)
        try:
            self.inbox.put_nowait(None)
        except Full:
            pass
        self.worker.join(timeout=.5)
        if self.worker.is_alive():
            self.worker.terminate(); self.worker.join(timeout=1.)
        self.logfile.close()
        self.inbox.cancel_join_thread(); self.outbox.cancel_join_thread()
        self.inbox.close(); self.outbox.close()


def main(robot_default='burger1'):
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    node = None
    try:
        node = LocalMMC(robot_default)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close(); node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
