from collections import deque
from typing import Optional

from utils import wrap_pi, yaw_from_quaternion


class OptiTrackBlock:
    """Store synchronized two-robot mocap snapshots and interpolate in time."""

    def __init__(
        self,
        leader_name: str,
        follower_name: str,
        max_mocap_gap: float,
        history_size: int = 300,
    ) -> None:
        self.leader_name = str(leader_name)
        self.follower_name = str(follower_name)
        self.max_mocap_gap = float(max_mocap_gap)
        self.history = deque(maxlen=history_size)

    def process(self, msg, timestamp: float) -> bool:
        poses = {}

        for rb in msg.rigidbodies:
            name = str(rb.rigid_body_name)
            poses[name] = (
                float(rb.pose.position.x),
                float(rb.pose.position.y),
                yaw_from_quaternion(
                    rb.pose.orientation.x,
                    rb.pose.orientation.y,
                    rb.pose.orientation.z,
                    rb.pose.orientation.w,
                ),
            )

        if self.leader_name not in poses or self.follower_name not in poses:
            return False

        self.history.append(
            (float(timestamp), poses[self.leader_name], poses[self.follower_name])
        )
        return True

    @property
    def newest_time(self) -> Optional[float]:
        if not self.history:
            return None
        return float(self.history[-1][0])

    @staticmethod
    def interpolate_angle(a0: float, a1: float, fraction: float) -> float:
        return wrap_pi(a0 + fraction * wrap_pi(a1 - a0))

    def interpolate_at(self, target_time: float):
        if len(self.history) < 2:
            return None

        history = list(self.history)
        for i in range(len(history) - 1):
            t0, leader0, follower0 = history[i]
            t1, leader1, follower1 = history[i + 1]

            if t0 <= target_time <= t1:
                left_gap = target_time - t0
                right_gap = t1 - target_time
                if (
                    left_gap > self.max_mocap_gap
                    or right_gap > self.max_mocap_gap
                ):
                    return None

                dt = t1 - t0
                fraction = 0.0 if dt <= 0.0 else (target_time - t0) / dt

                x1 = leader0[0] + fraction * (leader1[0] - leader0[0])
                y1 = leader0[1] + fraction * (leader1[1] - leader0[1])
                th1 = self.interpolate_angle(leader0[2], leader1[2], fraction)

                x2 = follower0[0] + fraction * (follower1[0] - follower0[0])
                y2 = follower0[1] + fraction * (follower1[1] - follower0[1])
                th2 = self.interpolate_angle(follower0[2], follower1[2], fraction)

                return (x1, y1, th1, x2, y2, th2, 0.0, "interpolated")

        return None
