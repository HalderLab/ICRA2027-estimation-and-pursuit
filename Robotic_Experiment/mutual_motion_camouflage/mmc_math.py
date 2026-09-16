"""MMC equations copied unchanged from mmc_mhe_ol(1).py."""
import math
import numpy as np

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

def smoothstep_cosine01(s: float) -> float:
    s = clamp(s, 0.0, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * s)

def yaw_from_quat_zup(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

class MMCMath:
    def shape_to_mmc(self, x_hat: np.ndarray):
        rho = max(float(x_hat[0]), self.rho_min)
        a1 = float(x_hat[1])
        a2 = float(x_hat[2])

        gamma = -self.v1_nom * math.cos(a1) - self.v3_nom * math.cos(a2)
        lamb = -self.v1_nom * math.sin(a1) - self.v3_nom * math.sin(a2)
        delta_sq = (
            self.v1_nom**2
            + self.v3_nom**2
            + 2.0 * self.v1_nom * self.v3_nom * math.cos(a1 - a2)
        )
        delta = math.sqrt(max(delta_sq, 0.0))
        E = rho * rho * lamb * lamb * math.exp(-2.0 * self.mu * rho)
        theta_diff = wrap_pi(math.pi - a1 + a2)

        return {
            "rho": rho,
            "alpha1": a1,
            "alpha2": a2,
            "gamma": gamma,
            "lambda": lamb,
            "delta": delta,
            "E": E,
            "theta_diff": theta_diff,
        }

    def sync_gain_effective(self, elapsed: float) -> float:
        if elapsed < self.sync_ramp_start:
            fraction = self.sync_initial_fraction
        elif self.sync_ramp_duration <= 1e-12:
            fraction = 1.0
        else:
            s = (elapsed - self.sync_ramp_start) / self.sync_ramp_duration
            ramp = smoothstep_cosine01(s)
            fraction = self.sync_initial_fraction + (
                1.0 - self.sync_initial_fraction
            ) * ramp
        return self.sync_gain_final * fraction

    def compute_local_command(
        self,
        q,
        elapsed: float,
        robot: str,
    ):
        # Desired orbit energy is a fixed constant.
        # The instantaneous q["E"] is still reconstructed every control step
        # from this robot's current local growMHE/bootstrap shape inside shape_to_mmc().
        Ed = self.E0

        u_nom = -self.mu * q["lambda"]
        if self.use_dissipative:
            u_dis = self.kd * q["lambda"] * q["gamma"] * (q["E"] - Ed)
        else:
            u_dis = 0.0
        u_mmc_raw = u_nom + u_dis

        sync_error = wrap_pi(q["theta_diff"] - self.theta_diff_d)
        ke = self.sync_gain_effective(elapsed)

        if robot == "burger1":
            u_self_raw = u_mmc_raw - 0.5 * ke * sync_error
        elif robot == "burger3":
            u_self_raw = u_mmc_raw + 0.5 * ke * sync_error
        else:
            raise ValueError("robot must be burger1 or burger3")

        u_self_cmd = clamp(u_self_raw, -self.u_max, self.u_max)
        saturated = int(abs(u_self_raw - u_self_cmd) > 1e-12)

        return {
            "Ed": Ed,
            "u_nom": u_nom,
            "u_dis": u_dis,
            "u_mmc_raw": u_mmc_raw,
            "sync_error": sync_error,
            "ke": ke,
            "u_self_raw": u_self_raw,
            "u_self_cmd": u_self_cmd,
            "saturated": saturated,
        }
