"""
Hardware Safety Guard (The Nervous System).
Hardcodes strict limits, bounds checking, and anomaly interceptors to protect physical rigs
against AI hallucinations and erroneous tool calls.

Commands: every actuator target is checked (finite number, inside the experiment's safe
bounds, below any response-limit ceiling, step size) before it reaches the rig.

Readings: every reading is checked and flagged, never silently corrected:
- non_finite: NaN or inf (e.g. a sensor dropout);
- timeout: the sensor did not answer;
- saturated / out_of_range: at or beyond the sensor's full scale;
- position_mismatch: the rig's position feedback disagrees with the commanded position by
  more than the spec's tolerance (e.g. a stuck actuator); only rigs with feedback report it;
- outlier: far outside every predictive band the caller passes (by default the current best
  fit's prediction band; optionally the GP's, see config.AnomalyConfig);
- response_limit: above the spec's hard limit on the reading (a diode's current). The guard
  then backs the actuator off to its minimum position and refuses any target at or above the
  position where the limit was exceeded, for the rest of the run. This assumes the reading
  rises with the position, as it does for a diode.
A flagged reading is marked invalid; the caller decides what to do (re-measure, discard, pause).
"""
import math
import logging
import numbers
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator
from hardware.base import BaseHardwareRig, MeasurementResult, RigStatus

logger = logging.getLogger(__name__)

class SafetyViolationError(ValueError):
    """Raised when an agent requests an action outside safe physical bounds."""
    pass

class SafetyLimits(BaseModel):
    """Per-experiment limits. Built from the ExperimentSpec, never from a global default."""
    model_config = ConfigDict(frozen=True)

    min_position: float
    max_position: float
    max_step_delta: float = Field(gt=0.0)
    sensor_range: Optional[Tuple[float, float]] = None
    sensor_resolution: float = Field(default=0.0, ge=0.0)
    position_tolerance: Optional[float] = Field(default=None, gt=0.0)
    response_limit: Optional[float] = None

    @model_validator(mode="after")
    def _check_order(self) -> "SafetyLimits":
        if not self.min_position < self.max_position:
            raise ValueError("min_position must be below max_position")
        return self

    @classmethod
    def from_spec(cls, spec) -> "SafetyLimits":
        lo, hi = spec.safe_bounds
        return cls(
            min_position=lo, max_position=hi, max_step_delta=spec.max_step_delta,
            sensor_range=spec.sensor_range, sensor_resolution=spec.sensor_resolution,
            position_tolerance=spec.position_tolerance, response_limit=spec.response_limit,
        )

class HardwareSafetyGuard:
    """
    Transparent safety wrapper around any BaseHardwareRig.
    Intercepts and validates every command before it reaches the physical or simulated rig.
    """

    def __init__(self, rig: BaseHardwareRig, limits: SafetyLimits):
        self.rig = rig
        self.limits = limits
        self._emergency_stop_triggered = False
        self._ceiling: Optional[float] = None
        self._execution_history: List[Dict[str, Any]] = []

    @property
    def ceiling(self) -> Optional[float]:
        """Targets at or above this position are refused (set when the response limit was exceeded)."""
        return self._ceiling

    def connect(self) -> bool:
        return self.rig.connect()

    def disconnect(self) -> None:
        self.rig.disconnect()

    def is_connected(self) -> bool:
        return self.rig.is_connected()

    def home(self) -> bool:
        if self._emergency_stop_triggered:
            raise SafetyViolationError("Cannot home rig while Emergency Stop is active. Reset required.")
        return self.rig.home()

    def validate_actuator_target(self, target_position: float) -> float:
        """
        Validates the proposed actuator position against safety boundaries.
        """
        if self._emergency_stop_triggered:
            raise SafetyViolationError("Rig is in Emergency Stop state. Intercepted movement request.")

        # 0. Type check: NaN and inf would slip through the comparisons below
        if (
            isinstance(target_position, bool)
            or not isinstance(target_position, numbers.Real)
            or not math.isfinite(target_position)
        ):
            raise SafetyViolationError(
                f"SAFETY INTERCEPT: Target position {target_position!r} is not a finite number."
            )

        # 1. Bounds Check
        if target_position < self.limits.min_position or target_position > self.limits.max_position:
            raise SafetyViolationError(
                f"SAFETY INTERCEPT: Target position {target_position} exceeds physical limits "
                f"[{self.limits.min_position}, {self.limits.max_position}]."
            )

        # 2. Response-limit ceiling (set after the reading exceeded its hard limit)
        if self._ceiling is not None and target_position >= self._ceiling:
            raise SafetyViolationError(
                f"SAFETY INTERCEPT: Target position {target_position} is at or above {self._ceiling:g}, "
                f"where the reading exceeded its limit of {self.limits.response_limit:g}."
            )

        # 3. Delta Check (Excessive rapid displacement)
        current_pos = self.rig.get_status().current_position
        displacement = abs(target_position - current_pos)
        if displacement > self.limits.max_step_delta:
            raise SafetyViolationError(
                f"SAFETY INTERCEPT: Requested step displacement {displacement:.2f} exceeds "
                f"maximum allowed delta {self.limits.max_step_delta:.2f}."
            )

        return float(target_position)

    def safe_move(self, target_position: float) -> Dict[str, Any]:
        """
        Executes a validated actuator movement.
        """
        validated_target = self.validate_actuator_target(target_position)
        success = self.rig.move_actuator(validated_target)
        self._execution_history.append({
            "action": "move",
            "target": validated_target,
            "success": success
        })
        return {
            "success": success,
            "new_position": validated_target,
            "status": "OK" if success else "FAILED"
        }

    def check_reading(self, result: MeasurementResult, expected: Sequence[Tuple[float, float, float]] = ()) -> List[str]:
        """
        Anomaly flags for a reading. `expected` holds (mean, sigma, threshold) predictive bands; the
        reading is an outlier when it is more than threshold x sigma from the mean of every band.
        """
        value = result.sensor_value
        if not math.isfinite(value):
            return ["non_finite"]
        flags: List[str] = []
        lim = self.limits
        if lim.sensor_range is not None:
            lo, hi = lim.sensor_range
            half = lim.sensor_resolution / 2.0
            if value >= hi - half:
                flags.append("saturated")
            elif value < lo - half:
                flags.append("out_of_range")
        if lim.response_limit is not None and value > lim.response_limit:
            flags.append("response_limit")
        feedback = result.metadata.get("position_feedback")
        if (lim.position_tolerance is not None and isinstance(feedback, (int, float))
                and abs(feedback - result.actuator_position) > lim.position_tolerance):
            flags.append("position_mismatch")
        bands = [(mean, sigma, k) for mean, sigma, k in expected if sigma > 0]
        if bands and all(abs(value - mean) > k * sigma for mean, sigma, k in bands):
            flags.append("outlier")
        return flags

    def safe_read(self, expected: Sequence[Tuple[float, float, float]] = ()) -> MeasurementResult:
        """
        Takes a sensor reading and flags it (see the module docstring). A sensor timeout is
        returned as a flagged reading, not raised.
        """
        if self._emergency_stop_triggered:
            raise SafetyViolationError("Rig is in Emergency Stop state.")

        try:
            result = self.rig.read_sensor()
        except TimeoutError as exc:
            status = self.rig.get_status()
            logger.warning("Sensor timeout: %s", exc)
            return MeasurementResult(
                actuator_position=status.current_position, sensor_value=math.nan, is_valid=False,
                flags=["timeout"], metadata={"cycle": status.telemetry.get("cycle"), "error": str(exc)},
            )

        result.flags = self.check_reading(result, expected)
        result.is_valid = not result.flags
        if result.flags:
            logger.warning("Reading %s at %s flagged: %s", result.sensor_value, result.actuator_position,
                           ", ".join(result.flags))
        if "response_limit" in result.flags:
            self._trip_response_limit(result.actuator_position, result.sensor_value)
        return result

    def _trip_response_limit(self, position: float, value: float) -> None:
        self._ceiling = position if self._ceiling is None else min(self._ceiling, position)
        logger.critical("Reading %s exceeded the response limit %s at %s: backing off to %s; targets >= %s refused.",
                        value, self.limits.response_limit, position, self.limits.min_position, self._ceiling)
        self.rig.move_actuator(self.limits.min_position)
        self._execution_history.append({"action": "response_limit_backoff", "target": self.limits.min_position,
                                        "tripped_at": position, "value": value})

    def pause(self, seconds: float) -> None:
        """Wait before the next command (used before re-measuring after a sensor dropout)."""
        if self._emergency_stop_triggered:
            raise SafetyViolationError("Rig is in Emergency Stop state.")
        self.rig.wait(seconds)

    def emergency_stop(self) -> None:
        """Immediately halts operations."""
        self._emergency_stop_triggered = True
        logger.critical("EMERGENCY STOP TRIGGERED.")

    def reset_estop(self) -> None:
        """Resets emergency stop flag after operator confirmation."""
        self._emergency_stop_triggered = False
        logger.info("Emergency Stop reset.")

    def get_status(self) -> RigStatus:
        status = self.rig.get_status()
        if self._emergency_stop_triggered:
            status.last_error = "EMERGENCY_STOP_ACTIVE"
        return status
