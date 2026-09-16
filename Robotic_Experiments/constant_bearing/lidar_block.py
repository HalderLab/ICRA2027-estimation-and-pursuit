import math
from typing import Optional

from models import LidarMeasurement
from utils import wrap_pi


class LidarBlock:
    """Select one detected object and convert it into a calibrated measurement.

    Target modes
    ------------
    nearest:
        Select the nearest valid object on every LiDAR callback.

    fixed_id:
        Use only ``target_object_id``. If that ID disappears, return None.

    physical_lock:
        Acquire one object once, then track the same *physical* target by
        continuity of its LiDAR-frame position. The detector ID is treated as
        a hint only and is allowed to change. If no candidate is close enough
        to the previous target position, return None rather than switching to
        another object.

        If ``target_object_id >= 0``, that ID is used only for the initial
        acquisition. After acquisition, the physical lock can follow the same
        object even if the detector assigns a new ID.
    """

    VALID_TARGET_MODES = {"nearest", "fixed_id", "physical_lock"}

    def __init__(
        self,
        target_object_id: int,
        bearing_sign: float,
        rho_bias: float,
        alpha2_bias: float,
        rho_min: float,
        target_mode: str = "nearest",
        physical_lock_distance: float = 0.25,
        physical_lock_max_lost_frames: int = 30,
        physical_lock_reset_after_lost: bool = False,
    ) -> None:
        self.target_object_id = int(target_object_id)
        self.bearing_sign = float(bearing_sign)
        self.rho_bias = float(rho_bias)
        self.alpha2_bias = float(alpha2_bias)
        self.rho_min = float(rho_min)

        self.target_mode = str(target_mode).strip().lower()
        if self.target_mode not in self.VALID_TARGET_MODES:
            raise ValueError(
                "target_mode must be one of "
                f"{sorted(self.VALID_TARGET_MODES)}, got {target_mode!r}."
            )

        self.physical_lock_distance = max(
            0.0, float(physical_lock_distance)
        )
        self.physical_lock_max_lost_frames = max(
            1, int(physical_lock_max_lost_frames)
        )
        self.physical_lock_reset_after_lost = bool(
            physical_lock_reset_after_lost
        )

        # Physical-target lock state. Coordinates are the raw LiDAR-frame
        # object coordinates after bearing_sign has been applied.
        self.locked = False
        self.locked_id: Optional[int] = None
        self.locked_x: Optional[float] = None
        self.locked_y: Optional[float] = None
        self.lost_count = 0

    def process(self, msg, timestamp: float) -> Optional[LidarMeasurement]:
        selected = self.select_target(msg)
        if selected is None:
            return None

        obj_id, x21, y21, rho_raw = selected
        alpha2_raw = wrap_pi(math.atan2(y21, x21))

        # Preserve the original correction convention:
        # corrected = raw + bias.
        rho = max(rho_raw + self.rho_bias, self.rho_min)
        alpha2 = wrap_pi(alpha2_raw + self.alpha2_bias)

        return LidarMeasurement(
            timestamp=float(timestamp),
            object_id=obj_id,
            x21=x21,
            y21=y21,
            rho_raw=rho_raw,
            alpha2_raw=alpha2_raw,
            rho=rho,
            alpha2=alpha2,
            rho_bias=self.rho_bias,
            alpha2_bias=self.alpha2_bias,
        )

    def reset_lock(self) -> None:
        """Explicitly forget the currently locked physical target."""
        self.locked = False
        self.locked_id = None
        self.locked_x = None
        self.locked_y = None
        self.lost_count = 0

    def _candidates(self, msg):
        if not hasattr(msg, "objects") or len(msg.objects) == 0:
            return []

        candidates = []
        for obj in msg.objects:
            obj_id = int(obj.id) if hasattr(obj, "id") else -1
            x21 = float(obj.pose.x)
            y21 = float(obj.pose.y) * self.bearing_sign
            rho = math.hypot(x21, y21)

            if rho < self.rho_min:
                continue

            candidates.append((obj_id, x21, y21, rho))

        return candidates

    def _accept_physical_candidate(self, candidate):
        obj_id, x21, y21, _ = candidate
        self.locked = True
        self.locked_id = obj_id
        self.locked_x = x21
        self.locked_y = y21
        self.lost_count = 0
        return candidate

    def _mark_physical_target_missing(self):
        self.lost_count += 1

        # Default behavior is deliberately conservative: keep the lock and
        # return None instead of choosing another object. Automatic reset is
        # available only when explicitly enabled.
        if (
            self.physical_lock_reset_after_lost
            and self.lost_count >= self.physical_lock_max_lost_frames
        ):
            self.reset_lock()

        return None

    def _select_physical_lock(self, candidates):
        # --------------------------------------------------------------
        # Initial acquisition
        # --------------------------------------------------------------
        if not self.locked:
            if not candidates:
                return None

            # If an initial ID was provided, acquire only that ID. Once the
            # physical lock exists, future detector ID changes are allowed.
            if self.target_object_id >= 0:
                for candidate in candidates:
                    if candidate[0] == self.target_object_id:
                        return self._accept_physical_candidate(candidate)
                return None

            # Otherwise, the first nearest object becomes the physical target.
            candidate = min(candidates, key=lambda c: c[3])
            return self._accept_physical_candidate(candidate)

        # --------------------------------------------------------------
        # Existing lock: select by spatial continuity, not detector ID.
        # --------------------------------------------------------------
        if not candidates:
            return self._mark_physical_target_missing()

        assert self.locked_x is not None
        assert self.locked_y is not None

        def displacement(candidate):
            return math.hypot(
                candidate[1] - self.locked_x,
                candidate[2] - self.locked_y,
            )

        # Prefer the old detector ID when it still exists and is spatially
        # plausible. This is just a hint; it is not required for continuity.
        same_id_candidates = [
            c for c in candidates if c[0] == self.locked_id
        ]
        if same_id_candidates:
            same_id_best = min(same_id_candidates, key=displacement)
            if displacement(same_id_best) <= self.physical_lock_distance:
                return self._accept_physical_candidate(same_id_best)

        # If the detector ID changed, choose the object nearest the previous
        # physical target position, but only inside the association gate.
        best = min(candidates, key=displacement)
        if displacement(best) <= self.physical_lock_distance:
            return self._accept_physical_candidate(best)

        # Nothing is close enough. Do NOT fall back to nearest-to-robot.
        return self._mark_physical_target_missing()

    def select_target(self, msg):
        candidates = self._candidates(msg)

        if self.target_mode == "nearest":
            if not candidates:
                return None
            return min(candidates, key=lambda c: c[3])

        if self.target_mode == "fixed_id":
            if self.target_object_id < 0:
                return None
            for candidate in candidates:
                if candidate[0] == self.target_object_id:
                    return candidate
            return None

        # physical_lock
        return self._select_physical_lock(candidates)
