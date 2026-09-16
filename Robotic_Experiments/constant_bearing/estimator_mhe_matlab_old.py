"""MATLAB-Engine wrapper for the persistent MHE.m estimator.

This block exposes the same estimator interface used by estimator_ekf.py:

    step(t, rho, alpha2, v_self, u_self)
    predict(t, v_self, u_self)
    close()

The MATLAB MHE handle object is created exactly once and kept alive for the
whole ROS experiment so that its moving window, control history, warm start,
and predictor state persist between calls.
"""

import math
import os
from typing import Any, Dict, Optional

from models import Estimate


class MHEMatlabEstimator:
    """Use MHE.m through MATLAB Engine for Python."""

    def __init__(self, matlab_folder: str) -> None:
        self.matlab_folder = os.path.abspath(os.path.expanduser(matlab_folder))
        if not os.path.isdir(self.matlab_folder):
            raise FileNotFoundError(
                f"MATLAB MHE folder does not exist: {self.matlab_folder}"
            )

        # Import lazily so the modular program can still run in EKF mode on a
        # machine where MATLAB Engine is not installed/configured.
        try:
            import matlab.engine
        except Exception as exc:
            raise ImportError(
                "Could not import matlab.engine. Make sure MATLAB Engine is "
                "available to the same Python interpreter that runs ROS2."
            ) from exc

        self._matlab_engine_module = matlab.engine
        self.eng = self._matlab_engine_module.start_matlab()
        self.eng.addpath(self.matlab_folder, nargout=0)

        mhe_path = str(self.eng.which("MHE"))
        if not mhe_path:
            self.eng.quit()
            raise FileNotFoundError(
                f"MATLAB cannot find MHE.m after addpath({self.matlab_folder!r})."
            )

        # Persistent MATLAB handle object. DO NOT recreate this on each step.
        self.obj = self.eng.MHE()

        # Use relative time inside MATLAB. ROS timestamps can be very large;
        # MHE only needs consistent time differences.
        self._time_origin: Optional[float] = None

        self.last_estimate: Optional[Estimate] = None
        self.last_raw_output: Optional[Dict[str, Any]] = None
        self.last_diagnostics: Dict[str, Any] = {}
        self.closed = False

    def _matlab_time(self, t: float) -> float:
        t = float(t)
        if self._time_origin is None:
            self._time_origin = t
        return t - self._time_origin

    @staticmethod
    def _as_dict(out: Any) -> Dict[str, Any]:
        if isinstance(out, dict):
            return dict(out)
        try:
            return dict(out)
        except Exception as exc:
            raise TypeError(
                f"Unexpected MATLAB MHE output type: {type(out)!r}"
            ) from exc

    @staticmethod
    def _float_field(raw: Dict[str, Any], key: str, default: float = math.nan) -> float:
        value = raw.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _convert_output(self, raw_out: Any, ros_timestamp: float) -> Estimate:
        raw = self._as_dict(raw_out)
        self.last_raw_output = raw

        ready = bool(raw.get("ready", False))

        estimate = Estimate(
            timestamp=float(ros_timestamp),
            rho=self._float_field(raw, "rho"),
            alpha1=self._float_field(raw, "alpha1"),
            alpha2=self._float_field(raw, "alpha2"),
            v1=self._float_field(raw, "v1"),
            u1=self._float_field(raw, "u1"),
            ready=ready,
            source="MHE",
        )

        # Keep MHE-specific information available without forcing the common
        # Estimate dataclass or controller to know about MATLAB details.
        self.last_diagnostics = {
            "ready": ready,
            "cost": self._float_field(raw, "cost"),
            "iterations": self._float_field(raw, "iterations", 0.0),
            "isPrediction": bool(raw.get("isPrediction", False)),
            "lastMeasurementTime": self._float_field(raw, "lastMeasurementTime"),
            "predictionAge": self._float_field(raw, "predictionAge", 0.0),
            "x": self._float_field(raw, "x"),
            "y": self._float_field(raw, "y"),
            "theta": self._float_field(raw, "theta"),
        }

        self.last_estimate = estimate
        return estimate

    def step(
        self,
        t: float,
        rho: float,
        alpha2: float,
        v_self: float,
        u_self: float,
    ) -> Estimate:
        """Add one new LiDAR measurement and run MHE correction when ready."""
        if self.closed:
            raise RuntimeError("MHEMatlabEstimator has already been closed.")

        tm = self._matlab_time(t)
        out = self.eng.step(
            self.obj,
            float(tm),
            float(rho),
            float(alpha2),
            float(v_self),
            float(u_self),
        )
        return self._convert_output(out, ros_timestamp=float(t))

    def predict(
        self,
        t: float,
        v_self: float,
        u_self: float,
    ) -> Optional[Estimate]:
        """Propagate the latest corrected MHE state between LiDAR updates."""
        if self.closed:
            raise RuntimeError("MHEMatlabEstimator has already been closed.")

        # main_modular.py does not call predict() before the first LiDAR sample,
        # but returning None here also keeps this wrapper safe if used elsewhere.
        if self._time_origin is None:
            return self.last_estimate

        tm = self._matlab_time(t)
        out = self.eng.predict(
            self.obj,
            float(tm),
            float(v_self),
            float(u_self),
        )
        return self._convert_output(out, ros_timestamp=float(t))

    def close(self) -> None:
        if self.closed:
            return

        self.closed = True
        self.obj = None

        try:
            self.eng.quit()
        except Exception:
            # Shutdown must not prevent the ROS node from stopping the robots.
            pass
