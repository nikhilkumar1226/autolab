"""
Hardware Safety Guard (The Nervous System).
Hardcodes strict limits, bounds checking, and anomaly interceptors to protect physical rigs
against AI hallucinations and erroneous tool calls.

Commands: every actuator target is checked, in this order, before it reaches the rig:
1. the E-stop is not latched;
2. the watchdog has not expired (a valid reading within `watchdog_s` of rig time; if it has,
   the E-stop latches);
3. the target is a finite real number (not bool, str, None, NaN or inf);
4. it is inside the experiment's safe bounds and below any response-limit ceiling;
5. the step from the current position is at most `max_step_delta`;
6. the rate limit: fewer than `max_moves_per_s` moves in the last second of rig time;
7. the oscillation interlock: at most `max_moves_without_reading` moves in a row without a
   sensor reading (every measurement is one move and one reading, so only a runaway caller
   moves repeatedly without reading).
The rig's clock is simulated time on the digital twin (so these checks are reproducible) and
wall time on hardware.

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

E-stop: latched by emergency_stop() (an operator, or the watchdog). While latched every
command is refused. Only reset_estop(operator=<name>) clears it.

Audit: every request, decision and reason goes to an append-only, hash-chained audit log
(hardware/audit.py), which the engine saves in the run record.
"""
import math
import logging
import numbers
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator
from hardware.audit import AuditEntry, AuditLog
from hardware.base import BaseHardwareRig, MeasurementResult, RigStatus

logger = logging.getLogger(__name__)

class SafetyViolationError(ValueError):
    """Raised when an agent requests an action outside safe physical bounds."""
    pass

class EmergencyStopActive(SafetyViolationError):
    """The E-stop is latched: nothing moves until an operator resets it."""

def _finite_real(value: Any) -> Optional[float]:
    """value as a finite float, or None: bool, str, None, containers, complex, Decimal, NaN, inf and
    integers too large for a float are all refused."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


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
    max_moves_per_s: Optional[float] = Field(default=None, gt=0.0, description="None = no rate limit")
    max_moves_without_reading: Optional[int] = Field(default=None, ge=1, description="None = no interlock")
    watchdog_s: Optional[float] = Field(default=None, gt=0.0, description="None = no watchdog")

    @model_validator(mode="after")
    def _check_order(self) -> "SafetyLimits":
        if not self.min_position < self.max_position:
            raise ValueError("min_position must be below max_position")
        return self

    @classmethod
    def from_spec(cls, spec, safety=None) -> "SafetyLimits":
        lo, hi = spec.safe_bounds
        return cls(
            min_position=lo, max_position=hi, max_step_delta=spec.max_step_delta,
            sensor_range=spec.sensor_range, sensor_resolution=spec.sensor_resolution,
            position_tolerance=spec.position_tolerance, response_limit=spec.response_limit,
            max_moves_per_s=safety.max_moves_per_s if safety is not None else None,
            max_moves_without_reading=safety.max_moves_without_reading if safety is not None else None,
            watchdog_s=safety.watchdog_s if safety is not None else None,
        )

class HardwareSafetyGuard:
    """
    Transparent safety wrapper around any BaseHardwareRig.
    Intercepts and validates every command before it reaches the physical or simulated rig.
    """

    def __init__(self, rig: BaseHardwareRig, limits: SafetyLimits):
        self.rig = rig
        self.limits = limits
        self.audit = AuditLog()
        self._lock = threading.RLock()  # emergency_stop() may come from another thread (the dashboard)
        self._estop_reason: Optional[str] = None
        self._ceiling: Optional[float] = None
        self._move_times: Deque[float] = deque()
        self._moves_since_read = 0
        self._last_valid_t: Optional[float] = None

    # --- state -----------------------------------------------------------------------
    @property
    def ceiling(self) -> Optional[float]:
        """Targets at or above this position are refused (set when the response limit was exceeded)."""
        return self._ceiling

    @property
    def estop_active(self) -> bool:
        return self._estop_reason is not None

    @property
    def estop_reason(self) -> Optional[str]:
        return self._estop_reason

    @property
    def audit_entries(self) -> Tuple[AuditEntry, ...]:
        return self.audit.entries

    def _now(self) -> float:
        return float(self.rig.clock())

    def _log(self, action: str, decision: str, reason: str = "", request: Any = None) -> None:
        with self._lock:
            self.audit.append(self._now(), action, decision, reason, request)

    # --- connection ------------------------------------------------------------------
    def connect(self) -> bool:
        ok = self.rig.connect()
        self._last_valid_t = self._now()
        self._log("connect", "allowed" if ok else "rejected", "" if ok else "the rig did not connect")
        return ok

    def disconnect(self) -> None:
        self.rig.disconnect()
        self._log("disconnect", "info")

    def is_connected(self) -> bool:
        return self.rig.is_connected()

    def home(self) -> bool:
        if self.estop_active:
            self._log("home", "rejected", f"E-stop latched ({self._estop_reason})")
            raise EmergencyStopActive("Cannot home rig while Emergency Stop is active. Reset required.")
        ok = self.rig.home()
        self._last_valid_t = self._now()
        self._log("home", "allowed")
        return ok

    # --- commands --------------------------------------------------------------------
    def _check_watchdog(self) -> None:
        limit = self.limits.watchdog_s
        if limit is None or self._last_valid_t is None:
            return
        idle = self._now() - self._last_valid_t
        if idle > limit:
            self._latch(f"watchdog: no valid reading for {idle:.1f} s (limit {limit:g} s)")
            raise EmergencyStopActive(
                f"SAFETY INTERCEPT: watchdog expired after {idle:.1f} s without a valid reading; E-stop latched."
            )

    def validate_actuator_target(self, target_position: float) -> float:
        """
        Validates the proposed actuator position against safety boundaries.
        """
        if self.estop_active:
            raise EmergencyStopActive(f"Rig is in Emergency Stop state ({self._estop_reason}). Intercepted movement request.")
        self._check_watchdog()

        # 0. Type check: NaN and inf would slip through the comparisons below. float() of a huge
        #    integer raises OverflowError, so convert carefully; only the converted value is used.
        target = _finite_real(target_position)
        if target is None:
            raise SafetyViolationError(
                f"SAFETY INTERCEPT: Target position {_short(target_position)} is not a finite number."
            )
        target_position = target

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

        # 4. Rate limit, on the rig's clock
        rate = self.limits.max_moves_per_s
        if rate is not None:
            now = self._now()
            while self._move_times and now - self._move_times[0] >= 1.0:
                self._move_times.popleft()
            if len(self._move_times) >= rate:
                raise SafetyViolationError(
                    f"SAFETY INTERCEPT: rate limit: {len(self._move_times)} moves in the last second "
                    f"(limit {rate:g} per second)."
                )

        # 5. Oscillation interlock: no runs of moves without readings
        limit = self.limits.max_moves_without_reading
        if limit is not None and self._moves_since_read >= limit:
            raise SafetyViolationError(
                f"SAFETY INTERCEPT: {self._moves_since_read} moves without a sensor reading (limit {limit}); "
                f"take a reading before moving again."
            )

        return float(target_position)

    def safe_move(self, target_position: float) -> Dict[str, Any]:
        """
        Executes a validated actuator movement.
        """
        with self._lock:
            try:
                validated_target = self.validate_actuator_target(target_position)
            except SafetyViolationError as exc:
                self._log("move", "rejected", str(exc), target_position)
                raise
            self._move_times.append(self._now())  # command time
            self._moves_since_read += 1
            success = self.rig.move_actuator(validated_target)
            self._log("move", "allowed" if success else "rejected", "" if success else "the rig reported a failed move",
                      validated_target)
        return {
            "success": success,
            "new_position": validated_target,
            "status": "OK" if success else "FAILED"
        }

    # --- readings --------------------------------------------------------------------
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
        if self.estop_active:
            self._log("read", "rejected", f"E-stop latched ({self._estop_reason})")
            raise EmergencyStopActive(f"Rig is in Emergency Stop state ({self._estop_reason}).")

        self._moves_since_read = 0
        try:
            result = self.rig.read_sensor()
        except TimeoutError as exc:
            status = self.rig.get_status()
            logger.warning("Sensor timeout: %s", exc)
            self._log("read", "flagged", "timeout")
            return MeasurementResult(
                actuator_position=status.current_position, sensor_value=math.nan, is_valid=False,
                flags=["timeout"], metadata={"cycle": status.telemetry.get("cycle"), "error": str(exc)},
            )

        result.flags = self.check_reading(result, expected)
        result.is_valid = not result.flags
        if result.flags:
            logger.warning("Reading %s at %s flagged: %s", result.sensor_value, result.actuator_position,
                           ", ".join(result.flags))
            self._log("read", "flagged", ", ".join(result.flags), result.sensor_value)
        else:
            self._last_valid_t = self._now()
            self._log("read", "allowed", "", result.sensor_value)
        if "response_limit" in result.flags:
            self._trip_response_limit(result.actuator_position, result.sensor_value)
        return result

    def _trip_response_limit(self, position: float, value: float) -> None:
        self._ceiling = position if self._ceiling is None else min(self._ceiling, position)
        logger.critical("Reading %s exceeded the response limit %s at %s: backing off to %s; targets >= %s refused.",
                        value, self.limits.response_limit, position, self.limits.min_position, self._ceiling)
        self.rig.move_actuator(self.limits.min_position)
        self._log("backoff", "tripped",
                  f"reading {value:.4g} over the response limit {self.limits.response_limit:g} at {position:g}; "
                  f"moved to {self.limits.min_position:g}; targets >= {self._ceiling:g} refused", position)

    def pause(self, seconds: float) -> None:
        """Wait before the next command (used before re-measuring after a sensor dropout)."""
        if self.estop_active:
            self._log("pause", "rejected", f"E-stop latched ({self._estop_reason})")
            raise EmergencyStopActive("Rig is in Emergency Stop state.")
        self.rig.wait(seconds)
        self._log("pause", "allowed", "", seconds)

    # --- E-stop ------------------------------------------------------------------------
    def _latch(self, reason: str) -> None:
        with self._lock:
            if self._estop_reason is None:
                self._estop_reason = reason
                logger.critical("EMERGENCY STOP TRIGGERED: %s", reason)
                self._log("estop", "tripped", reason)
                try:
                    self.rig.emergency_stop()  # the hardware's own E-stop too, where it has one
                except Exception as exc:
                    logger.error("The rig's own E-stop failed (the guard stays latched): %s", exc)

    def emergency_stop(self, reason: str = "operator") -> None:
        """Immediately halts operations. Latched: only reset_estop(operator=...) clears it."""
        self._latch(reason)

    def reset_estop(self, operator: str) -> None:
        """Clears a latched E-stop after an operator, named in the audit log, has checked the rig."""
        if not isinstance(operator, str) or not operator.strip():
            self._log("reset", "rejected", "a reset needs the operator's name")
            raise SafetyViolationError("E-stop reset refused: give the operator's name (explicit operator reset).")
        with self._lock:
            try:
                self.rig.reset_emergency_stop()
            except Exception as exc:
                self._log("reset", "rejected", f"the rig refused the reset: {exc}")
                raise SafetyViolationError(f"E-stop reset refused by the rig: {exc}") from exc
            previous = self._estop_reason
            self._estop_reason = None
            self._last_valid_t = self._now()  # the watchdog restarts from the reset
            self._log("reset", "allowed", f"reset by {operator.strip()} (was: {previous})")
        logger.info("Emergency Stop reset by %s.", operator.strip())

    def get_status(self) -> RigStatus:
        status = self.rig.get_status()
        if self.estop_active:
            status.last_error = "EMERGENCY_STOP_ACTIVE"
        return status
