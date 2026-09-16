"""Pure Python coordination state machine; independent of ROS and estimators.

Commands/statuses use reliable VOLATILE ROS topics and a fresh UUID per run.
Repeated commands carry increasing sequence numbers. A stopped session is latched:
restart both controller processes before the next experiment.
"""
import math

ROBOTS = ('burger1', 'burger3')
ACTIVE = ('ARMED', 'COMMITTED', 'WARMUP', 'RUN')


class LocalSession:
    def __init__(self, heartbeat_timeout=.6, clock_tolerance=.25, warmup_timeout=3.):
        self.state = 'WAIT'
        self.run_id = ''
        self.reason = ''
        self.last_seq = -1
        self.last_command = None
        self.start_at = self.start_mono = None
        self.switch_at = self.switch_mono = None
        self.switch_committed = False
        self.heartbeat_timeout = heartbeat_timeout
        self.clock_tolerance = clock_tolerance
        self.warmup_timeout = warmup_timeout
        self.just_started = False

    def stop(self, reason):
        self.state, self.reason = 'STOPPED', str(reason)

    def receive(self, packet, mono, wall, local_ready):
        if self.state == 'STOPPED':
            return
        if packet.get('action') == 'STOP':
            # Explicit stop is global, also before PREPARE; never causes motion.
            self.stop(packet.get('reason', 'operator stop'))
            return
        rid = packet.get('run_id')
        seq = packet.get('seq')
        stamp = packet.get('stamp')
        if not isinstance(rid, str) or not rid or not isinstance(seq, int):
            return
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            return
        if abs(wall-stamp) > self.clock_tolerance:
            if self.state in ACTIVE:
                self.stop('command clock offset / transport delay exceeds tolerance')
            return
        if self.run_id and rid != self.run_id:
            self.stop('another experiment coordinator is active')
            return
        if seq <= self.last_seq:
            return
        action = packet.get('action')
        if action == 'PREPARE' and self.state == 'WAIT':
            start = packet.get('start_at')
            if not local_ready or not isinstance(start, (int, float)) or not math.isfinite(start):
                return
            if not .3 <= start-wall <= 10.:
                return
            self.run_id, self.start_at = rid, start
            self.start_mono = mono+(start-wall)
            self.state = 'ARMED'
        if not self.run_id:
            return
        if action in ('PREPARE', 'START', 'PREPARE_MMC', 'MMC'):
            if packet.get('start_at') != self.start_at:
                self.stop('start schedule changed within a run')
                return
        else:
            return
        self.last_seq, self.last_command = seq, mono
        if action == 'START' and self.state == 'ARMED':
            if mono >= self.start_mono:
                self.stop('late start commit')
            else:
                self.state = 'COMMITTED'
        if action == 'PREPARE_MMC' and self.state == 'WARMUP':
            switch = packet.get('switch_at')
            if not isinstance(switch, (int, float)) or not math.isfinite(switch):
                self.stop('invalid MMC switch time')
            elif self.switch_at is None:
                if switch-wall < .1 or switch-wall > 5:
                    self.stop('late/invalid MMC preparation')
                else:
                    self.switch_at, self.switch_mono = switch, mono+(switch-wall)
            elif switch != self.switch_at:
                self.stop('MMC switch schedule changed')
        if action == 'MMC' and self.state == 'WARMUP':
            if self.switch_at is None or packet.get('switch_at') != self.switch_at:
                self.stop('MMC commit without matching preparation')
            elif not self.switch_committed and mono >= self.switch_mono:
                self.stop('late MMC commit')
            else:
                self.switch_committed = True

    def tick(self, mono, local_ready, mhe_ready, peer, peer_age):
        self.just_started = False
        if self.state not in ACTIVE:
            return
        if not local_ready:
            self.stop('local sensor, command link, or worker not ready')
        elif self.last_command is None or mono-self.last_command > self.heartbeat_timeout:
            self.stop('coordinator heartbeat expired')
        elif peer is None or peer_age > self.heartbeat_timeout:
            self.stop('peer heartbeat expired')
        elif peer.get('state') == 'STOPPED':
            self.stop('peer stopped: '+str(peer.get('reason', '')))
        elif peer.get('run_id') not in ('', self.run_id):
            self.stop('peer belongs to a different experiment')
        if self.state not in ACTIVE:
            return
        if self.state == 'ARMED' and mono >= self.start_mono:
            self.stop('start was never committed')
        elif self.state == 'COMMITTED' and mono >= self.start_mono:
            if peer.get('run_id') != self.run_id or peer.get('state') not in ('COMMITTED', 'WARMUP', 'RUN'):
                self.stop('peer did not acknowledge start')
            else:
                self.state, self.just_started = 'WARMUP', True
        if self.state == 'WARMUP':
            if mono-self.start_mono > self.warmup_timeout:
                self.stop('MHE warmup timed out before coordinated MMC switch')
            elif self.switch_mono is not None and mono >= self.switch_mono:
                if (not self.switch_committed or not mhe_ready or not peer.get('mhe_ready')
                        or not peer.get('switch_committed') or peer.get('switch_at') != self.switch_at):
                    self.stop('MMC switch not committed/ready on both agents')
                else:
                    self.state = 'RUN'
        if self.state == 'RUN' and not mhe_ready:
            self.stop('local MHE is stale or invalid')

    def status(self, robot, ready, mhe_ready):
        return dict(robot=robot, state=self.state, run_id=self.run_id, reason=self.reason,
                    ready=bool(ready), mhe_ready=bool(mhe_ready), start_at=self.start_at,
                    switch_at=self.switch_at, switch_committed=self.switch_committed)


class Coordinator:
    def __init__(self, run_id, start_delay=2., switch_delay=.6,
                 hold=.75, heartbeat_timeout=.6, wait_timeout=60., duration=0.):
        self.run_id = run_id
        self.phase = 'WAIT'
        self.reason = ''
        self.start_at = self.switch_at = None
        self.ready_since = None
        self.created = None
        self.start_delay, self.switch_delay = start_delay, switch_delay
        self.hold, self.heartbeat_timeout = hold, heartbeat_timeout
        self.wait_timeout, self.duration = wait_timeout, duration

    def stop(self, reason):
        self.phase, self.reason = 'STOP', str(reason)

    def update(self, mono, wall, statuses):
        if self.created is None:
            self.created = mono
        if self.phase == 'STOP':
            return
        # statuses: {robot: (validated packet, monotonic reception time)}
        fresh = all(r in statuses and mono-statuses[r][1] <= self.heartbeat_timeout for r in ROBOTS)
        packets = [statuses[r][0] for r in ROBOTS] if fresh else []
        if self.phase == 'WAIT':
            if mono-self.created > self.wait_timeout:
                self.stop('timed out waiting for both controllers')
            elif fresh and any(p['state'] == 'STOPPED' for p in packets):
                self.stop('a controller is stopped; restart both controllers')
            elif fresh and any(p['state'] != 'WAIT' for p in packets):
                self.stop('controllers are already owned by another run')
            elif fresh and all(p['ready'] for p in packets):
                if packets[0].get('config_signature') != packets[1].get('config_signature'):
                    self.stop('shared MMC parameters differ between robots')
                    return
                if self.ready_since is None:
                    self.ready_since = mono
                if mono-self.ready_since >= self.hold:
                    self.start_at = wall+self.start_delay
                    self.phase = 'PREPARE'
            else:
                self.ready_since = None
            return
        if not fresh:
            self.stop('a controller heartbeat expired')
            return
        if any(p['state'] == 'STOPPED' or not p['ready'] for p in packets):
            self.stop('controller fault or loss of readiness')
            return
        if any(p['run_id'] not in ('', self.run_id) for p in packets):
            self.stop('controller belongs to another experiment')
            return
        if self.phase == 'PREPARE':
            if all(p['state'] == 'ARMED' and p['run_id'] == self.run_id for p in packets):
                self.phase = 'START'
            elif wall >= self.start_at-.3:
                self.stop('missing start preparation acknowledgement')
        if self.phase == 'START':
            if wall >= self.start_at-.2 and not all(p['state'] in ('COMMITTED', 'WARMUP', 'RUN') for p in packets):
                self.stop('missing start commit acknowledgement')
            elif all(p['state'] == 'WARMUP' and p['mhe_ready'] for p in packets):
                self.switch_at = wall+self.switch_delay
                self.phase = 'PREPARE_MMC'
        if self.phase == 'PREPARE_MMC':
            if all(p.get('switch_at') == self.switch_at and p['mhe_ready'] for p in packets):
                self.phase = 'MMC'
            elif wall >= self.switch_at-.15:
                self.stop('missing MMC preparation acknowledgement')
        if self.phase == 'MMC':
            if not all(p['mhe_ready'] for p in packets):
                self.stop('MHE readiness lost')
            elif wall >= self.switch_at-.1 and not all(p.get('switch_committed') for p in packets):
                self.stop('missing MMC commit acknowledgement')
        if self.duration > 0 and self.start_at is not None and wall-self.start_at >= self.duration:
            self.stop('requested experiment duration complete')

    def packet(self, wall, seq):
        return dict(action=self.phase, run_id=self.run_id, stamp=wall, seq=seq,
                    start_at=self.start_at, switch_at=self.switch_at, reason=self.reason)
