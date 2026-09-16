import math

from models import Estimate
from utils import clamp


class ConstantBearingController:
    """Constant-bearing controller isolated from the estimator and ROS I/O."""

    def __init__(
        self,
        mu: float,
        phi: float,
        rho_min: float,
        u2_max: float,
        known_leader_v: float,
        use_estimated_v1: bool = False,
        alpha1_sign: float = -1.0,
    ) -> None:
        self.mu = float(mu)
        self.phi = float(phi)
        self.rho_min = float(rho_min)
        self.u2_max = float(u2_max)
        self.known_leader_v = float(known_leader_v)
        self.use_estimated_v1 = bool(use_estimated_v1)

        # The source file currently uses alpha1_hat = -x_hat[1] inside the
        # controller. Default -1.0 preserves that exact behavior while making
        # the convention explicit and easy to change later.
        self.alpha1_sign = float(alpha1_sign)

    def compute(self, estimate: Estimate, follower_v: float) -> float:
        rho_hat = max(float(estimate.rho), self.rho_min)
        alpha1_hat = self.alpha1_sign * float(estimate.alpha1)
        alpha2_hat = float(estimate.alpha2)

        if self.use_estimated_v1 and estimate.v1 is not None:
            v1_controller = float(estimate.v1)
        else:
            v1_controller = self.known_leader_v

        u2 = (
            self.mu * math.sin(alpha2_hat - self.phi)
            + (
                v1_controller * math.sin(alpha1_hat)
                + float(follower_v) * math.sin(alpha2_hat)
            )
            / rho_hat
        )

        return clamp(u2, -self.u2_max, self.u2_max)
