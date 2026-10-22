"""
Abstract base class and data contracts for experimental hardware rigs.
"""
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone

class DataSource(str, Enum):
    """Where measurements come from. Shown on every dashboard, plot, report and run record."""
    DIGITAL_TWIN = "DIGITAL TWIN"
    FIRMWARE_IN_THE_LOOP = "FIRMWARE-IN-THE-LOOP (Wokwi)"
    PHYSICAL_RIG = "PHYSICAL RIG"

class MeasurementResult(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actuator_position: float = Field(description="Actuator setting (e.g. angle in degrees, position in mm)")
    sensor_value: float = Field(description="Recorded sensor value (e.g. lux, volts, counts)")
    units_actuator: str = Field(default="degrees", description="Units for actuator position")
    units_sensor: str = Field(default="lux", description="Units for sensor reading")
    is_valid: bool = Field(default=True, description="Safety and validity status of measurement")
    flags: List[str] = Field(
        default_factory=list,
        description="Why the safety guard considers the reading invalid (non_finite, timeout, saturated, "
                    "position_mismatch, outlier, response_limit); empty when valid",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional debug/environment info")

class RigStatus(BaseModel):
    is_connected: bool
    is_homed: bool
    current_position: float
    is_busy: bool
    last_error: Optional[str] = None
    telemetry: Dict[str, Any] = Field(default_factory=dict)

class BaseHardwareRig(ABC):
    """
    Abstract Hardware Rig defining the standard interface for both physical
    microcontroller drivers and digital twin simulators.
    """

    @property
    @abstractmethod
    def data_source(self) -> DataSource:
        """Label for where this rig's measurements come from."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the hardware or initialize simulation state."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Safely release the hardware port or simulation resources."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check connection health."""
        pass

    @abstractmethod
    def home(self) -> bool:
        """Home the actuator to zero reference."""
        pass

    @abstractmethod
    def move_actuator(self, position: float) -> bool:
        """
        Move the actuator to target position.
        :param position: Target position (e.g., angle in degrees or distance in mm).
        :return: True if movement completed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def read_sensor(self) -> MeasurementResult:
        """
        Take a sensor measurement at the current actuator position.
        :return: MeasurementResult object.
        """
        pass

    @abstractmethod
    def get_status(self) -> RigStatus:
        """
        Retrieve current hardware status and telemetry.
        """
        pass

    def wait(self, seconds: float) -> None:
        """Pause before the next command (e.g. to let a sensor recover). The digital twin overrides this."""
        if seconds > 0:
            time.sleep(seconds)

    def emergency_stop(self) -> None:
        """Hardware-side E-stop (e.g. the firmware releases the servo). The guard latches its own E-stop first."""

    def reset_emergency_stop(self) -> None:
        """Clear a hardware-side E-stop. Called by the guard only after an operator reset."""

    def clock(self) -> float:
        """Seconds on the rig's clock, for the safety guard's rate limit and watchdog. The twin keeps simulated time."""
        return time.monotonic()
