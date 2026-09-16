import math


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def message_time(msg, fallback_time: float) -> float:
    """Use a ROS message stamp when available, otherwise use fallback_time."""
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        stamp = msg.header.stamp
        sec = int(getattr(stamp, "sec", 0))
        nanosec = int(getattr(stamp, "nanosec", 0))
        if sec != 0 or nanosec != 0:
            return sec + nanosec * 1e-9
    return fallback_time
