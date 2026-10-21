"""
Abstract base class and data contracts for experimental hardware rigs.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class MeasurementResult(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    actuator_position: float = Field(description="Actuator setting (e.g. angle in degrees, position in mm)")
    sensor_value: float = Field(description="Recorded sensor value (e.g. lux, volts, counts)")
    units_actuator: str = Field(default="degrees", description="Units for actuator position")
    units_sensor: str = Field(default="lux", description="Units for sensor reading")
    is_valid: bool = Field(default=True, description="Safety and validity status of measurement")
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
