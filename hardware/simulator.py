"""
Physics Simulation Rig (Digital Twin) for Autonomous Experimental Design.
Evaluates the ground-truth law of an ExperimentSpec and adds seeded sensor noise.
"""
import time
import numpy as np
from typing import Optional
from datetime import datetime, timezone
from hardware.base import BaseHardwareRig, DataSource, MeasurementResult, RigStatus
from config import SimulationConfig
from experiments.registry import ExperimentSpec

class PhysicsSimulationRig(BaseHardwareRig):
    """
    Simulates an experiment from the registry with Gaussian sensor noise
    and mechanical transit behavior.

    All noise is drawn from the `rng` passed in, so a run is reproducible from its seed.
    """

    def __init__(self, spec: ExperimentSpec, rng: np.random.Generator, sim_config: Optional[SimulationConfig] = None):
        self.spec = spec
        self.config = sim_config or SimulationConfig()
        self._rng = rng
        self._noise_std = self.config.noise_std if self.config.noise_std is not None else spec.noise.std
        self._connected = False
        self._is_homed = False
        self._current_position = 0.0
        self._is_busy = False
        self._sample_history = []

    @property
    def data_source(self) -> DataSource:
        return DataSource.DIGITAL_TWIN

    @property
    def noise_std(self) -> float:
        return self._noise_std

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
        Ground-truth response plus Gaussian noise. A light sensor cannot read below zero.
        """
        noise = self._rng.normal(0.0, self._noise_std)
        return max(0.0, self.spec.true_response(position) + float(noise))

    def read_sensor(self) -> MeasurementResult:
        if not self._connected:
            raise ConnectionError("Cannot read sensor from disconnected rig.")

        intensity = self._calculate_physical_intensity(self._current_position)

        # Metadata deliberately carries no ground truth and no experiment name.
        result = MeasurementResult(
            timestamp=datetime.now(timezone.utc),
            actuator_position=self._current_position,
            sensor_value=intensity,
            units_actuator=self.spec.independent.units,
            units_sensor=self.spec.dependent.units,
            is_valid=True,
            metadata={"simulation": True}
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
            }
        )
