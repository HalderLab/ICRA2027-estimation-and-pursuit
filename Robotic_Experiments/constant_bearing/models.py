from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LidarMeasurement:
    timestamp: float
    object_id: int
    x21: float
    y21: float
    rho_raw: float
    alpha2_raw: float
    rho: float
    alpha2: float
    rho_bias: float
    alpha2_bias: float


@dataclass
class Estimate:
    timestamp: float
    rho: float
    alpha1: float
    alpha2: float
    v1: Optional[float] = None
    u1: Optional[float] = None
    ready: bool = True
    source: str = "unknown"


@dataclass
class ControlCommand:
    leader_v: float
    leader_u: float
    follower_v: float
    follower_u: float
