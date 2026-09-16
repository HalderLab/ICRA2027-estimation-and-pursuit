import math
from typing import Optional

import numpy as np

from models import Estimate
from utils import wrap_pi


class EKFEstimator:
    """Current 3-state EKF isolated behind a replaceable estimator interface.

    Public interface intentionally mirrors the future MHE wrapper:
        step(t, rho, alpha2, v_self, u_self)
        predict(t, v_self, u_self)
        close()

    The current EKF has no high-rate predictor between LiDAR updates, so
    predict() returns the most recent corrected estimate unchanged. This keeps
    the behavior of the original monolithic script.
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
    ) -> None:
        self.v1 = float(leader_v)
        self.u1 = float(leader_u)
        self.alpha1_initial_guess = float(alpha1_initial_guess)
        self.q_nominal_dt = float(q_nominal_dt)
        self.rho_min = float(rho_min)

        self.P = np.diag([0.05**2, 0.8**2, 0.1**2]).astype(float)
        self.Q_matlab = np.diag([q_rho, q_alpha1, q_alpha2]).astype(float)
        self.R = np.diag([sigma_rho**2, sigma_alpha2**2]).astype(float)
        self.H = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.I3 = np.eye(3)

        self.x_hat: Optional[np.ndarray] = None
        self.last_filter_time: Optional[float] = None
        self.last_estimate: Optional[Estimate] = None

    def step(
        self,
        t: float,
        rho: float,
        alpha2: float,
        v_self: float,
        u_self: float,
    ) -> Optional[Estimate]:
        if self.x_hat is None:
            self.x_hat = np.array(
                [rho, self.alpha1_initial_guess, alpha2], dtype=float
            )
            self.last_filter_time = float(t)
        else:
            dt = float(t) - float(self.last_filter_time)
            if dt <= 0.0:
                return self.last_estimate

            self._predict_and_correct(
                rho_measured=float(rho),
                alpha2_measured=float(alpha2),
                dt=dt,
                v_self=float(v_self),
                u_self=float(u_self),
            )
            self.last_filter_time = float(t)

        self.last_estimate = self._make_estimate(float(t))
        return self.last_estimate

    def predict(
        self,
        t: float,
        v_self: float,
        u_self: float,
    ) -> Optional[Estimate]:
        # Deliberately a no-op for this EKF version. The original code only
        # updates the EKF when a LiDAR object measurement arrives.
        del t, v_self, u_self
        return self.last_estimate

    def _make_estimate(self, t: float) -> Estimate:
        return Estimate(
            timestamp=t,
            rho=float(self.x_hat[0]),
            alpha1=float(self.x_hat[1]),
            alpha2=float(self.x_hat[2]),
            v1=self.v1,
            u1=self.u1,
            ready=True,
            source="EKF",
        )

    def _predict_and_correct(
        self,
        rho_measured: float,
        alpha2_measured: float,
        dt: float,
        v_self: float,
        u_self: float,
    ) -> None:
        x = self.x_hat.copy()
        rho = max(float(x[0]), self.rho_min)
        alpha1 = float(x[1])
        alpha2 = float(x[2])

        shared_term = self.v1 * math.sin(alpha1) + v_self * math.sin(alpha2)

        x_dot = np.array(
            [
                -self.v1 * math.cos(alpha1) - v_self * math.cos(alpha2),
                -self.u1 + shared_term / rho,
                -u_self + shared_term / rho,
            ],
            dtype=float,
        )

        x_pred = x + dt * x_dot
        x_pred[0] = max(x_pred[0], self.rho_min)
        x_pred[1] = wrap_pi(x_pred[1])
        x_pred[2] = wrap_pi(x_pred[2])

        Fc = np.array(
            [
                [
                    0.0,
                    self.v1 * math.sin(alpha1),
                    v_self * math.sin(alpha2),
                ],
                [
                    -shared_term / (rho**2),
                    self.v1 * math.cos(alpha1) / rho,
                    v_self * math.cos(alpha2) / rho,
                ],
                [
                    -shared_term / (rho**2),
                    self.v1 * math.cos(alpha1) / rho,
                    v_self * math.cos(alpha2) / rho,
                ],
            ],
            dtype=float,
        )

        A = self.I3 + dt * Fc
        Q = self.Q_matlab * (dt / self.q_nominal_dt)
        P_pred = A @ self.P @ A.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        innovation = np.array(
            [
                rho_measured - x_pred[0],
                wrap_pi(alpha2_measured - x_pred[2]),
            ],
            dtype=float,
        )

        S = self.H @ P_pred @ self.H.T + self.R
        PHt = P_pred @ self.H.T
        K = np.linalg.solve(S.T, PHt.T).T

        x_new = x_pred + K @ innovation
        x_new[0] = max(x_new[0], self.rho_min)
        x_new[1] = wrap_pi(x_new[1])
        x_new[2] = wrap_pi(x_new[2])

        IKH = self.I3 - K @ self.H
        self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.x_hat = x_new

    def close(self) -> None:
        pass
