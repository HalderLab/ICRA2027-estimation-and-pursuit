#!/usr/bin/env python3
"""Run on ONE robot, after both local controllers are waiting.

This process remains running during the experiment. Ctrl+C broadcasts STOP.
--stop broadcasts STOP without starting an experiment.
"""
import argparse
import json
import math
import signal
import time
import uuid
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from mmc_protocol import Coordinator, ROBOTS


class ExperimentNode(Node):
    def __init__(self, args):
        super().__init__('mmc_experiment_coordinator')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.publisher = self.create_publisher(String, '/mmc/experiment/command', qos)
        self.core = Coordinator(uuid.uuid4().hex, args.start_delay, args.switch_delay,
            args.ready_hold, args.heartbeat_timeout, args.wait_timeout, args.duration)
        self.statuses, self.seqs = {}, {}
        self.clock_tolerance = args.clock_tolerance
        self.clock_offset = time.time()-time.monotonic()
        self.seq = 0
        self.last_phase = None
        self.last_wait_log = -math.inf
        if args.stop:
            self.core.stop('operator --stop')
        for robot in ROBOTS:
            self.create_subscription(String, f'/{robot}/mmc/status',
                lambda msg, r=robot: self.receive(r, msg), qos)
        self.timer = self.create_timer(.05, self.tick)
        self.get_logger().info('Coordinator active. Ctrl+C stops both robots. Waiting for local controllers.')

    def receive(self, robot, msg):
        try:
            p = json.loads(msg.data)
            if not isinstance(p, dict) or p.get('robot') != robot:
                return
            if not all(k in p for k in ('stamp','seq','state','ready','mhe_ready','run_id')):
                return
            stamp = float(p['stamp'])
            if not math.isfinite(stamp) or abs(time.time()-stamp) > self.clock_tolerance:
                if self.core.phase != 'WAIT':
                    self.core.stop('clock offset / transport delay from '+robot)
                return
            seq = int(p['seq'])
            if seq <= self.seqs.get(robot, -1):
                return
            self.seqs[robot] = seq
            self.statuses[robot] = (p, time.monotonic())
        except (ValueError, TypeError, KeyError):
            pass

    def publish(self):
        packet = self.core.packet(time.time(), self.seq)
        self.seq += 1
        msg = String(); msg.data = json.dumps(packet, allow_nan=False)
        self.publisher.publish(msg)

    def tick(self):
        mono, wall = time.monotonic(), time.time()
        if self.core.phase == 'WAIT':
            self.clock_offset = wall-mono
        elif abs((wall-mono)-self.clock_offset) > self.clock_tolerance:
            self.core.stop('coordinator system clock jumped')
        self.core.update(mono, wall, self.statuses)
        if self.core.phase != 'WAIT':
            self.publish()
        elif mono-self.last_wait_log > 2.:
            summary = {r: (self.statuses[r][0]['state'], self.statuses[r][0]['ready'])
                       if r in self.statuses else 'no recent status (check ROS domain/network/clock)'
                       for r in ROBOTS}
            self.get_logger().info(str(summary))
            self.last_wait_log = mono
        if self.last_phase != self.core.phase:
            self.get_logger().info(f'{self.core.phase}: {self.core.reason}')
            self.last_phase = self.core.phase

    def close(self):
        self.timer.cancel()
        self.core.stop(self.core.reason or 'coordinator shutdown / Ctrl+C')
        # Repeated volatile STOP also covers discovery and ordinary packet loss.
        until = time.monotonic()+1.
        while time.monotonic() < until and rclpy.ok():
            self.publish()
            rclpy.spin_once(self, timeout_sec=.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--duration', type=float, default=0., help='seconds from motion start; 0 means until Ctrl+C')
    parser.add_argument('--start-delay', type=float, default=2.)
    parser.add_argument('--switch-delay', type=float, default=.6)
    parser.add_argument('--ready-hold', type=float, default=.75)
    parser.add_argument('--heartbeat-timeout', type=float, default=.6)
    parser.add_argument('--clock-tolerance', type=float, default=.25)
    parser.add_argument('--wait-timeout', type=float, default=60.)
    args, ros_args = parser.parse_known_args()
    if not all(math.isfinite(v) for v in vars(args).values() if isinstance(v, float)):
        parser.error('numeric options must be finite')
    if not .6 <= args.start_delay <= 10 or not .3 <= args.switch_delay <= 5:
        parser.error('start-delay must be .6..10 s; switch-delay .3..5 s')
    if min(args.clock_tolerance,args.heartbeat_timeout,args.wait_timeout) <= 0 or min(args.ready_hold,args.duration) < 0:
        parser.error('invalid timing options')
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    node = None
    try:
        node = ExperimentNode(args)
        if args.stop:
            node.close()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if not args.stop:
                node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
