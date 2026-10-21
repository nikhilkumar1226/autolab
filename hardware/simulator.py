"""
Physics Simulation Rig (Digital Twin) for Autonomous Experimental Design.
Provides high-fidelity mathematical and noise models for optical and physical phenomena.
"""
import time
import math
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from hardware.base import BaseHardwareRig, MeasurementResult, RigStatus
from config import SimulationConfig

class PhysicsSimulationRig(BaseHardwareRig):
    """
    Simulates physical experiments with realistic sensor noise, ambient offsets,
    and mechanical transit behavior.
    """

    def __init__(self, sim_config: Optional[SimulationConfig] = None):
        self.config = sim_config or SimulationConfig()
        self._connected = False
        self._is_homed = False
        self._current_position = 0.0
        self._is_busy = False
        self._sample_history = []

    def connect(self) -> bool:
        self._connected = True
        self._is_homed = False
        self._current_position = 0.0
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def home(self) -> bool:
        if not self._connected:
            raise ConnectionError("Cannot home disconnected rig.")
        self._is_busy = True
        if self.config.transit_delay_sec > 0:
            time.sleep(self.config.transit_delay_sec)
        self._current_position = 0.0
        self._is_homed = True
        self._is_busy = False
        return True

    def move_actuator(self, position: float) -> bool:
        if not self._connected:
            raise ConnectionError("Cannot move actuator on disconnected rig.")
        self._is_busy = True
        
        # Simulate mechanical transit delay proportional to step
        displacement = abs(position - self._current_position)
        if self.config.transit_delay_sec > 0 and displacement > 0:
            delay = min(0.2, self.config.transit_delay_sec * (displacement / 45.0 + 0.1))
            time.sleep(delay)
            
        self._current_position = float(position)
        self._is_busy = False
        return True

    def _calculate_physical_intensity(self, position: float) -> float:
        """
        Computes the theoretical physical response plus noise for the active experiment.
        """
        exp_type = self.config.experiment_type.lower()
        noise = np.random.normal(0, self.config.noise_std)

        if exp_type == "malus_law":
            # Malus's Law: I = I_0 * cos^2(theta - theta_0) + I_ambient + noise
            rad = math.radians(position - self.config.theta_offset)
            intensity = self.config.i0_intensity * (math.cos(rad) ** 2) + self.config.ambient_noise_level + noise
            return max(0.0, float(intensity))

        elif exp_type == "inverse_square_law":
            # Inverse Square Law: I = I_0 / (d + d_0)^2 + I_ambient + noise
            # position is distance in cm (e.g. 5 to 100 cm)
            d0 = 5.0 # offset distance in cm
            d = max(1.0, position + d0)
            intensity = (self.config.i0_intensity * 100.0) / (d ** 2) + self.config.ambient_noise_level + noise
            return max(0.0, float(intensity))

        elif exp_type == "beer_lambert_law":
            # Beer-Lambert: I = I_0 * exp(-epsilon * c) + I_ambient + noise
            # position represents concentration in mg/L (0 to 100)
            epsilon = 0.04
            intensity = self.config.i0_intensity * math.exp(-epsilon * position) + self.config.ambient_noise_level + noise
            return max(0.0, float(intensity))

        else:
            raise ValueError(f"Unknown experiment type: {exp_type}")

    def read_sensor(self) -> MeasurementResult:
        if not self._connected:
            raise ConnectionError("Cannot read sensor from disconnected rig.")

        intensity = self._calculate_physical_intensity(self._current_position)

        units_actuator = "degrees" if self.config.experiment_type == "malus_law" else "cm"
        units_sensor = "lux"

        result = MeasurementResult(
            timestamp=datetime.utcnow(),
            actuator_position=self._current_position,
            sensor_value=intensity,
            units_actuator=units_actuator,
            units_sensor=units_sensor,
            is_valid=True,
            metadata={
                "simulation": True,
                "experiment_type": self.config.experiment_type,
                "ground_truth_i0": self.config.i0_intensity,
            }
        )
        self._sample_history.append(result)
        return result

    def get_status(self) -> RigStatus:
        return RigStatus(
            is_connected=self._connected,
            is_homed=self._is_homed,
            current_position=self._current_position,
            is_busy=self._is_busy,
            telemetry={
                "samples_recorded": len(self._sample_history),
                "mode": "simulation",
                "experiment": self.config.experiment_type
            }
        )
