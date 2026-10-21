"""
Hardware Safety Guard (The Nervous System).
Hardcodes strict limits, bounds checking, and anomaly interceptors to protect physical rigs
against AI hallucinations and erroneous tool calls.
"""
import math
import logging
import numbers
from typing import Dict, Any
from pydantic import BaseModel, ConfigDict, Field, model_validator
from hardware.base import BaseHardwareRig, MeasurementResult, RigStatus

logger = logging.getLogger(__name__)

class SafetyViolationError(ValueError):
    """Raised when an agent requests an action outside safe physical bounds."""
    pass

class SafetyLimits(BaseModel):
    """Per-experiment actuator limits. Built from the ExperimentSpec, never from a global default."""
    model_config = ConfigDict(frozen=True)

    min_position: float
    max_position: float
    max_step_delta: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_order(self) -> "SafetyLimits":
        if not self.min_position < self.max_position:
            raise ValueError("min_position must be below max_position")
        return self

class HardwareSafetyGuard:
    """
    Transparent safety wrapper around any BaseHardwareRig.
    Intercepts and validates every command before it reaches the physical or simulated rig.
    """

    def __init__(self, rig: BaseHardwareRig, limits: SafetyLimits):
        self.rig = rig
        self.limits = limits
        self._emergency_stop_triggered = False
        self._execution_history = []

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

        # 2. Delta Check (Excessive rapid displacement)
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

    def safe_read(self) -> MeasurementResult:
        """
        Executes a sensor reading and validates the data against physical sanity bounds.
        """
        if self._emergency_stop_triggered:
            raise SafetyViolationError("Rig is in Emergency Stop state.")

        result = self.rig.read_sensor()

        # Sanity check: optical sensor readings cannot be negative
        if result.sensor_value < 0.0:
            logger.warning(f"Sensor returned negative value: {result.sensor_value}. Clamping to 0.0.")
            result.sensor_value = 0.0

        # Sanity check: detect saturated or faulty disconnected readings
        if result.sensor_value > 1000000.0:
            logger.error("Sensor reading out of range (saturation or I2C fault detected).")
            result.is_valid = False

        return result

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
