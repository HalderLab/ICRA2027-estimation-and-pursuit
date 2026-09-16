"""MATLAB-Engine wrapper for a persistent MHE / growMHE estimator.

The Python-facing estimator interface is unchanged:

    step(t, rho, alpha2, v_self, u_self)
    predict(t, v_self, u_self)
    close()

By default the wrapper searches the MATLAB folder for ``growMHE.m`` first and
falls back to ``MHE.m``.  The selected MATLAB handle object is created exactly
once so its buffer, moving/growing horizon, warm start, and predictor state
persist for the complete ROS experiment.
"""

import math
import os
from typing import Any, Dict, Optional

from models import Estimate


class MHEMatlabEstimator:
    """Use growMHE.m (preferred) or MHE.m through MATLAB Engine for Python."""

    def __init__(self, matlab_folder: str, matlab_class: str = "auto") -> None:
        self.matlab_folder = os.path.abspath(os.path.expanduser(matlab_folder))
        if not os.path.isdir(self.matlab_folder):
            raise FileNotFoundError(
                f"MATLAB MHE folder does not exist: {self.matlab_folder}"
            )

        # Import lazily so EKF/PF modes can still run on a machine where the
        # MATLAB Engine is unavailable.
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

        requested = str(matlab_class).strip()
        if not requested:
            requested = "auto"

        if requested.lower() == "auto":
            # growMHE is preferred for the current hardware experiment.  Keep
            # MHE as a compatibility fallback so existing MATLAB folders still
            # work without changing main_modular.py.
            candidates = ("growMHE", "MHE")
        else:
            candidates = (requested,)

        selected = None
        selected_path = ""
        for name in candidates:
            found = str(self.eng.which(name))
            if found:
                selected = name
                selected_path = found
                break

        if selected is None:
            self.eng.quit()
            wanted = ", ".join(f"{name}.m" for name in candidates)
            raise FileNotFoundError(
                "MATLAB cannot find a supported estimator class after "
                f"addpath({self.matlab_folder!r}). Expected: {wanted}. "
                "For growMHE, the file must be named exactly 'growMHE.m' "
                "because the MATLAB class is 'growMHE'."
            )

        self.matlab_class = selected
        self.matlab_path = selected_path

        # Persistent MATLAB handle object. DO NOT recreate this on each step.
        try:
            constructor = getattr(self.eng, self.matlab_class)
            self.obj = constructor(nargout=1)
        except Exception:
            # Do not silently fall back if the selected class exists but fails
            # to construct; that would hide a MATLAB/class implementation bug.
            try:
                self.eng.quit()
            finally:
                pass
            raise

        # Use relative time inside MATLAB. ROS timestamps can be very large;
        # both MHE implementations only need consistent time differences.
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
                f"Unexpected MATLAB estimator output type: {type(out)!r}"
            ) from exc

    @staticmethod
    def _float_field(
        raw: Dict[str, Any],
        key: str,
        default: float = math.nan,
    ) -> float:
        value = raw.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _bool_field(
        raw: Dict[str, Any],
        key: str,
        default: bool = False,
    ) -> bool:
        value = raw.get(key, default)
        try:
            return bool(value)
        except Exception:
            return bool(default)

    def _convert_output(self, raw_out: Any, ros_timestamp: float) -> Estimate:
        raw = self._as_dict(raw_out)
        self.last_raw_output = raw

        ready = self._bool_field(raw, "ready", False)

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

        # MHE-specific information is kept outside the common Estimate model.
        # The three growing-horizon fields are absent in ordinary MHE.m, so
        # defaults keep this wrapper compatible with both classes.
        self.last_diagnostics = {
            "matlabClass": self.matlab_class,
            "matlabPath": self.matlab_path,
            "ready": ready,
            "cost": self._float_field(raw, "cost"),
            "iterations": self._float_field(raw, "iterations", 0.0),
            "isPrediction": self._bool_field(raw, "isPrediction", False),
            "lastMeasurementTime": self._float_field(raw, "lastMeasurementTime"),
            "predictionAge": self._float_field(raw, "predictionAge", 0.0),
            "x": self._float_field(raw, "x"),
            "y": self._float_field(raw, "y"),
            "theta": self._float_field(raw, "theta"),
            "horizonSamples": self._float_field(raw, "horizonSamples"),
            "horizonDuration": self._float_field(raw, "horizonDuration"),
            "windowFull": self._bool_field(raw, "windowFull", False),
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
        """Add one LiDAR sample and run the MATLAB estimator correction."""
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
            nargout=1,
        )
        return self._convert_output(out, ros_timestamp=float(t))

    def predict(
        self,
        t: float,
        v_self: float,
        u_self: float,
    ) -> Optional[Estimate]:
        """Propagate the latest corrected MATLAB state between LiDAR updates."""
        if self.closed:
            raise RuntimeError("MHEMatlabEstimator has already been closed.")

        if self._time_origin is None:
            return self.last_estimate

        tm = self._matlab_time(t)
        out = self.eng.predict(
            self.obj,
            float(tm),
            float(v_self),
            float(u_self),
            nargout=1,
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
