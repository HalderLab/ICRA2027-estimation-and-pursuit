import math
from typing import Optional

import numpy as np

from models import Estimate
from utils import wrap_pi


class EKFEstimator:
    """Four-state EKF with x=[rho, alpha1, alpha2, v1].

    Measurement: z=[rho, alpha2]
    Known inputs: follower v_self=v2, u_self=u2, and leader steering u1.
    Leader-speed process model: v1_dot=0.
    """

    def __init__(
        self,
        leader_v: float,
        leader_u: float,
        alpha1_initial_guess: float,
        sigma_rho: float,
        sigma_alpha2: float,
        q_rho: float,
        q_alpha1: float,
        q_alpha2: float,
        q_nominal_dt: float,
        rho_min: float,
        v1_initial_guess: Optional[float] = None,
        q_v1: float = 1.0e-6,
        p_rho_initial_std: float = 0.05,
        p_alpha1_initial_std: float = 0.80,
        p_alpha2_initial_std: float = 0.10,
        p_v1_initial_std: float = 0.10,
    ) -> None:
        self.leader_v_cmd = float(leader_v)
        self.u1 = float(leader_u)
        self.alpha1_initial_guess = float(alpha1_initial_guess)
        self.v1_initial_guess = (
            float(v1_initial_guess)
            if v1_initial_guess is not None
            else float(leader_v)
        )
        self.q_nominal_dt = float(q_nominal_dt)
        self.rho_min = float(rho_min)

        self.P = np.diag([
            p_rho_initial_std**2,
            p_alpha1_initial_std**2,
            p_alpha2_initial_std**2,
            p_v1_initial_std**2,
        ]).astype(float)

        self.Q_nominal = np.diag([
            q_rho,
            q_alpha1,
            q_alpha2,
            q_v1,
        ]).astype(float)

        self.R = np.diag([sigma_rho**2, sigma_alpha2**2]).astype(float)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=float)
        self.I4 = np.eye(4)

        self.x_hat: Optional[np.ndarray] = None
        self.last_filter_time: Optional[float] = None
        self.last_estimate: Optional[Estimate] = None

        self.last_innovation = np.zeros(2)
        self.last_v1_kalman_gain = np.zeros(2)
        self.last_v1_correction = np.zeros(2)

    def step(self, t, rho, alpha2, v_self, u_self) -> Optional[Estimate]:
        t = float(t)
        rho = max(float(rho), self.rho_min)
        alpha2 = wrap_pi(float(alpha2))

        if self.x_hat is None:
            self.x_hat = np.array([
                rho,
                self.alpha1_initial_guess,
                alpha2,
                self.v1_initial_guess,
            ], dtype=float)
            self.last_filter_time = t
        else:
            dt = t - float(self.last_filter_time)
            if dt <= 0.0:
                return self.last_estimate
            self._predict_and_correct(rho, alpha2, dt, float(v_self), float(u_self))
            self.last_filter_time = t

        self.last_estimate = self._make_estimate(t)
        return self.last_estimate

    def predict(self, t, v_self, u_self) -> Optional[Estimate]:
        # Same modular behavior as the previous EKF: update only on LiDAR samples.
        del t, v_self, u_self
        return self.last_estimate

    def _make_estimate(self, t: float) -> Estimate:
        return Estimate(
            timestamp=t,
            rho=float(self.x_hat[0]),
            alpha1=float(self.x_hat[1]),
            alpha2=float(self.x_hat[2]),
            v1=float(self.x_hat[3]),
            u1=self.u1,  # known input, not an estimated state
            ready=True,
            source="EKF4",
        )

    def _predict_and_correct(self, rho_measured, alpha2_measured, dt, v_self, u_self):
        x = self.x_hat.copy()
        rho = max(float(x[0]), self.rho_min)
        a1 = float(x[1])
        a2 = float(x[2])
        v1 = float(x[3])

        shared = v1 * math.sin(a1) + v_self * math.sin(a2)

        x_dot = np.array([
            -v1 * math.cos(a1) - v_self * math.cos(a2),
            -self.u1 + shared / rho,
            -u_self + shared / rho,
            0.0,
        ])

        x_pred = x + dt * x_dot
        x_pred[0] = max(float(x_pred[0]), self.rho_min)
        x_pred[1] = wrap_pi(float(x_pred[1]))
        x_pred[2] = wrap_pi(float(x_pred[2]))

        Fc = np.array([
            [0.0, v1*math.sin(a1), v_self*math.sin(a2), -math.cos(a1)],
            [-shared/rho**2, v1*math.cos(a1)/rho, v_self*math.cos(a2)/rho, math.sin(a1)/rho],
            [-shared/rho**2, v1*math.cos(a1)/rho, v_self*math.cos(a2)/rho, math.sin(a1)/rho],
            [0.0, 0.0, 0.0, 0.0],
        ])

        A = self.I4 + dt * Fc
        Q = self.Q_nominal * (dt / self.q_nominal_dt)
        P_pred = A @ self.P @ A.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        innovation = np.array([
            rho_measured - x_pred[0],
            wrap_pi(alpha2_measured - x_pred[2]),
        ])

        S = self.H @ P_pred @ self.H.T + self.R
        PHt = P_pred @ self.H.T
        try:
            K = np.linalg.solve(S.T, PHt.T).T
        except np.linalg.LinAlgError:
            self.P = P_pred
            self.x_hat = x_pred
            return

        x_new = x_pred + K @ innovation
        x_new[0] = max(float(x_new[0]), self.rho_min)
        x_new[1] = wrap_pi(float(x_new[1]))
        x_new[2] = wrap_pi(float(x_new[2]))

        IKH = self.I4 - K @ self.H
        self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.x_hat = x_new

        self.last_innovation = innovation.copy()
        self.last_v1_kalman_gain = K[3, :].copy()
        self.last_v1_correction = K[3, :] * innovation

    def close(self) -> None:
        pass
