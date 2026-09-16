import math
from typing import Optional

import numpy as np

from models import Estimate
from utils import wrap_pi


def _wrap_pi_array(a):
    return np.arctan2(np.sin(a), np.cos(a))


class PFEstimator:
    """Four-state particle filter with particles [rho, alpha1, alpha2, v1]."""

    def __init__(
        self,
        leader_v: float,
        leader_u: float,
        rho_min: float,
        sigma_rho: float = 0.02,
        sigma_alpha2: float = math.radians(2.5),
        num_particles: int = 50000,
        random_seed: int = 2026,
        resample_neff_ratio: float = 0.50,
        alpha1_initial_uniform: bool = True,
        alpha1_initial_guess: float = 0.0,
        v1_initial_guess: Optional[float] = None,
        pf_init_rho_std: float = 0.30,
        pf_init_alpha1_std: float = 0.80,
        pf_init_alpha2_std: float = math.radians(8.0),
        pf_init_v1_std: float = 0.05,
        process_rho_std: float = 0.05,
        process_alpha1_std: float = math.radians(2.0),
        process_alpha2_std: float = math.radians(2.0),
        process_v1_std: float = 0.001,
        pf_nominal_dt: float = 0.10,
        reflect_negative_v1: bool = True,
    ) -> None:
        self.leader_v_cmd = float(leader_v)
        self.u1 = float(leader_u)
        self.rho_min = float(rho_min)
        self.sigma_rho = float(sigma_rho)
        self.sigma_alpha2 = float(sigma_alpha2)
        self.N = int(num_particles)
        self.resample_neff_ratio = float(resample_neff_ratio)
        self.alpha1_initial_uniform = bool(alpha1_initial_uniform)
        self.alpha1_initial_guess = float(alpha1_initial_guess)
        self.v1_initial_guess = float(leader_v if v1_initial_guess is None else v1_initial_guess)

        self.init_rho_std = float(pf_init_rho_std)
        self.init_alpha1_std = float(pf_init_alpha1_std)
        self.init_alpha2_std = float(pf_init_alpha2_std)
        self.init_v1_std = float(pf_init_v1_std)
        self.process_rho_std = float(process_rho_std)
        self.process_alpha1_std = float(process_alpha1_std)
        self.process_alpha2_std = float(process_alpha2_std)
        self.process_v1_std = float(process_v1_std)
        self.pf_nominal_dt = float(pf_nominal_dt)
        self.reflect_negative_v1 = bool(reflect_negative_v1)

        self.rng = np.random.default_rng(int(random_seed))
        self.particles = None
        self.weights = None
        self.last_filter_time = None
        self.last_estimate = None
        self.neff = math.nan
        self.resampled_last_step = False

    def step(self, t, rho, alpha2, v_self, u_self) -> Optional[Estimate]:
        t = float(t)
        rho = max(float(rho), self.rho_min)
        alpha2 = wrap_pi(float(alpha2))

        if self.particles is None:
            self._initialize(rho, alpha2)
            self.last_filter_time = t
        else:
            dt = t - float(self.last_filter_time)
            if dt <= 0.0:
                return self.last_estimate
            self._predict(dt, float(v_self), float(u_self))
            self._update_weights(rho, alpha2)
            self.neff = self._effective_size()
            self.resampled_last_step = False
            if self.neff < self.resample_neff_ratio * self.N:
                self._systematic_resample()
                self.resampled_last_step = True
            self.last_filter_time = t

        self.last_estimate = self._make_estimate(t)
        return self.last_estimate

    def predict(self, t, v_self, u_self) -> Optional[Estimate]:
        # Do not inject process noise at the faster control-loop rate.
        del t, v_self, u_self
        return self.last_estimate

    def _initialize(self, rho, alpha2):
        p = np.zeros((self.N, 4))
        p[:, 0] = np.maximum(rho + self.init_rho_std*self.rng.standard_normal(self.N), self.rho_min)

        if self.alpha1_initial_uniform:
            p[:, 1] = self.rng.uniform(-math.pi, math.pi, self.N)
        else:
            p[:, 1] = _wrap_pi_array(self.alpha1_initial_guess + self.init_alpha1_std*self.rng.standard_normal(self.N))

        p[:, 2] = _wrap_pi_array(alpha2 + self.init_alpha2_std*self.rng.standard_normal(self.N))
        p[:, 3] = self.v1_initial_guess + self.init_v1_std*self.rng.standard_normal(self.N)
        if self.reflect_negative_v1:
            p[:, 3] = np.abs(p[:, 3])
        p[:, 3] = np.maximum(p[:, 3], 1e-6)

        self.particles = p
        self.weights = np.full(self.N, 1.0/self.N)
        self.neff = float(self.N)

    def _predict(self, dt, v_self, u_self):
        p = self.particles
        rho = np.maximum(p[:, 0], self.rho_min)
        a1 = p[:, 1]
        a2 = p[:, 2]
        v1 = p[:, 3]

        shared = (v1*np.sin(a1) + v_self*np.sin(a2))/rho
        rho_dot = -v1*np.cos(a1) - v_self*np.cos(a2)
        a1_dot = -self.u1 + shared
        a2_dot = -u_self + shared

        # MATLAB source noise values are per step at 0.1 s; scale std by sqrt(dt/dt0).
        s = math.sqrt(max(dt, 0.0)/self.pf_nominal_dt)
        p[:, 0] += dt*rho_dot + self.process_rho_std*s*self.rng.standard_normal(self.N)
        p[:, 1] += dt*a1_dot + self.process_alpha1_std*s*self.rng.standard_normal(self.N)
        p[:, 2] += dt*a2_dot + self.process_alpha2_std*s*self.rng.standard_normal(self.N)
        p[:, 3] += self.process_v1_std*s*self.rng.standard_normal(self.N)

        p[:, 0] = np.maximum(p[:, 0], self.rho_min)
        p[:, 1] = _wrap_pi_array(p[:, 1])
        p[:, 2] = _wrap_pi_array(p[:, 2])
        if self.reflect_negative_v1:
            p[:, 3] = np.abs(p[:, 3])
        p[:, 3] = np.maximum(p[:, 3], 1e-6)

    def _update_weights(self, rho_meas, alpha2_meas):
        dr = rho_meas - self.particles[:, 0]
        da = _wrap_pi_array(alpha2_meas - self.particles[:, 2])
        log_like = -0.5*((dr/self.sigma_rho)**2 + (da/self.sigma_alpha2)**2)

        # Numerically stable equivalent of MATLAB's weights .* exp(-0.5*score).
        log_w = np.log(np.maximum(self.weights, np.finfo(float).tiny)) + log_like
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        total = np.sum(w)
        self.weights = w/total if total > 0 and np.isfinite(total) else np.full(self.N, 1.0/self.N)

    def _make_estimate(self, t):
        w, p = self.weights, self.particles
        return Estimate(
            timestamp=float(t),
            rho=float(np.sum(w*p[:, 0])),
            alpha1=self._angle_mean(p[:, 1], w),
            alpha2=self._angle_mean(p[:, 2], w),
            v1=float(np.sum(w*p[:, 3])),
            u1=self.u1,  # known input, not an estimated state
            ready=True,
            source="PF4",
        )

    @staticmethod
    def _angle_mean(a, w):
        return wrap_pi(math.atan2(float(np.sum(w*np.sin(a))), float(np.sum(w*np.cos(a)))))

    def _effective_size(self):
        return float(1.0/np.sum(self.weights**2))

    def _systematic_resample(self):
        cdf = np.cumsum(self.weights)
        cdf[-1] = 1.0
        start = self.rng.random()/self.N
        points = start + np.arange(self.N)/self.N
        idx = np.searchsorted(cdf, points, side="left")
        self.particles = self.particles[idx].copy()
        self.weights.fill(1.0/self.N)

    def close(self) -> None:
        pass
