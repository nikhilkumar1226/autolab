"""
Physics Simulation Rig (Digital Twin) for Autonomous Experimental Design.

Evaluates the ground-truth law of an ExperimentSpec and adds seeded sensor noise. With
non-idealities enabled (config.NonIdealityConfig) it also models:
- a finite polariser extinction ratio (polariser experiments only);
- slow ambient drift: a random walk added to every reading, one step per rig cycle, sized
  relative to the sensor noise (so it is equally "slow" for a noisy light sensor and a precise
  current meter);
- sensor saturation (readings clipped at full scale) and ADC quantisation;
- actuator position error (gaussian) and backlash (an offset that depends on the direction
  of the last move). The rig reports the COMMANDED position, as an open-loop controller does,
  so these errors reach the data as errors in x;
- occasional outlier spikes.
Injected faults (config.FaultSpec) add a light spike, a sensor dropout (NaN or timeout) or a
stuck actuator at a given rig cycle. One cycle is one sensor read or one pause.

Two kinds of information leave the rig:
- measurements, with a cycle counter and a coarse position feedback (quantised to a quarter
  of the spec's position tolerance, like a potentiometer or encoder check, not a measurement
  of x). This is all that detection may use;
- `fault_log`: the ground truth of what was injected and where. Like the true law, it is for
  scoring only; the engine reads it after the run has stopped.

Every random draw comes from the `rng` passed in. With non-idealities off and no faults the
rig draws exactly one normal number per reading, as in Phase 1, so those runs are unchanged.
"""
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from hardware.base import BaseHardwareRig, DataSource, MeasurementResult, RigStatus
from config import FaultSpec, NonIdealityConfig, SimulationConfig
from experiments.registry import ExperimentSpec


class FaultEvent(BaseModel):
    """Ground truth: an injected fault or a random outlier acted on one rig cycle. Scoring only."""
    fault_index: Optional[int] = Field(description="Index in SimulationConfig.faults; None for a non-ideality outlier")
    type: str = Field(description="light_spike, sensor_dropout, stuck_actuator or outlier")
    cycle: int
    read: bool = Field(default=True, description="False if the cycle was a pause, so no reading was affected")
    detail: Dict[str, Any] = Field(default_factory=dict)


class PhysicsSimulationRig(BaseHardwareRig):
    """
    Simulates an experiment from the registry with sensor noise, optional non-idealities and
    injected faults, and mechanical transit behaviour.
    """

    def __init__(self, spec: ExperimentSpec, rng: np.random.Generator, sim_config: Optional[SimulationConfig] = None):
        self.spec = spec
        self.config = sim_config or SimulationConfig()
        self._rng = rng
        self._noise_std = self.config.noise_std if self.config.noise_std is not None else spec.noise.std
        self._ni: Optional[NonIdealityConfig] = self.config.nonideal if self.config.nonideal.enabled else None
        self._faults: List[FaultSpec] = list(self.config.faults)
        grid = np.linspace(*spec.safe_bounds, 201)
        truth = np.array([spec.true_response(x) for x in grid])
        self._span = float(np.ptp(truth)) or 1.0
        self._connected = False
        self._is_homed = False
        self._is_busy = False
        self._reset_state()

    def _reset_state(self) -> None:
        self._commanded = 0.0       # what the controller was told, and reports
        self._actual = 0.0          # where the actuator really is
        self._direction = 1.0       # sign of the last move, for backlash
        self._stuck_fault: Optional[int] = None
        self._cycle = 0             # completed cycles (reads and pauses)
        self._drift = 0.0
        self._sample_history: List[MeasurementResult] = []
        self._fault_log: List[FaultEvent] = []

    @property
    def data_source(self) -> DataSource:
        return DataSource.DIGITAL_TWIN

    @property
    def noise_std(self) -> float:
        return self._noise_std

    @property
    def fault_log(self) -> List[FaultEvent]:
        """Ground truth of injected faults and outliers. For scoring after the run, never for detection."""
        return list(self._fault_log)

    @property
    def response_span(self) -> float:
        return self._span

    def connect(self) -> bool:
        self._connected = True
        self._is_homed = False
        self._reset_state()
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
        self._commanded = self._actual = 0.0
        self._is_homed = True
        self._is_busy = False
        return True

    # ------------------------------------------------------------------
    def _active(self, kind: str, cycle: int) -> Optional[int]:
        for i, f in enumerate(self._faults):
            if f.type == kind and f.at_step <= cycle < f.at_step + f.duration:
                return i
        return None

    def move_actuator(self, position: float) -> bool:
        if not self._connected:
            raise ConnectionError("Cannot move actuator on disconnected rig.")
        self._is_busy = True

        # Simulate mechanical transit delay proportional to step
        displacement = abs(position - self._commanded)
        if self.config.transit_delay_sec > 0 and displacement > 0:
            delay = min(0.2, self.config.transit_delay_sec * (displacement / 45.0 + 0.1))
            time.sleep(delay)

        position = float(position)
        if position != self._commanded:
            self._direction = 1.0 if position > self._commanded else -1.0
        self._commanded = position
        stuck = self._active("stuck_actuator", self._cycle + 1)
        if stuck is not None:
            self._stuck_fault = stuck  # acknowledged, not executed
        else:
            self._stuck_fault = None
            self._actual = position + self._position_error()
        self._is_busy = False
        return True

    def _position_error(self) -> float:
        ni = self._ni
        if ni is None:
            return 0.0
        tol = self.spec.position_tolerance
        error = 0.5 * ni.backlash_rel * tol * self._direction
        if ni.position_error_rel > 0:
            error += float(self._rng.normal(0.0, ni.position_error_rel * tol))
        return error

    def _response(self, x: float) -> float:
        value = self.spec.true_response(x)
        ni = self._ni
        if ni is not None and self.spec.polariser and math.isfinite(ni.extinction_ratio):
            # A crossed pair still passes 1/ER of the lamp: I = C + A [(1 - 1/ER) cos^2 + 1/ER]
            A, C = self.spec.true_params["A"], self.spec.true_params["C"]
            value = C + (1.0 - 1.0 / ni.extinction_ratio) * (value - C) + A / ni.extinction_ratio
        return value

    def _advance_cycle(self) -> int:
        self._cycle += 1
        ni = self._ni
        if ni is not None and ni.drift_step_sigma > 0:
            self._drift += float(self._rng.normal(0.0, ni.drift_step_sigma * self._noise_std))
        return self._cycle

    def _log(self, index: Optional[int], kind: str, cycle: int, read: bool = True, **detail: Any) -> None:
        self._fault_log.append(FaultEvent(fault_index=index, type=kind, cycle=cycle, read=read, detail=detail))

    def read_sensor(self) -> MeasurementResult:
        if not self._connected:
            raise ConnectionError("Cannot read sensor from disconnected rig.")
        ni, spec = self._ni, self.spec
        cycle = self._advance_cycle()

        if self._stuck_fault is not None and abs(self._actual - self._commanded) > spec.position_tolerance:
            self._log(self._stuck_fault, "stuck_actuator", cycle, commanded=self._commanded, actual=self._actual)

        dropout = self._active("sensor_dropout", cycle)
        if dropout is not None:
            mode = self._faults[dropout].mode
            self._log(dropout, "sensor_dropout", cycle, mode=mode)
            if mode == "timeout":
                raise TimeoutError("sensor did not answer (dropout)")
            value = math.nan
        else:
            value = self._response(self._actual) + float(self._rng.normal(0.0, self._noise_std)) + self._drift
            if ni is not None and ni.outlier_prob > 0 and float(self._rng.random()) < ni.outlier_prob:
                size = (1.0 if self._rng.random() < 0.5 else -1.0) * ni.outlier_sigma * self._noise_std
                value += size
                self._log(None, "outlier", cycle, size=size)
            spike = self._active("light_spike", cycle)
            if spike is not None:
                size = self._faults[spike].magnitude * self._span
                value += size
                self._log(spike, "light_spike", cycle, size=size)
            lo, hi = spec.sensor_range
            value = max(lo, value)  # a sensor cannot read below its range (e.g. negative light)
            if ni is not None and ni.saturation:
                value = min(hi, value)
            if ni is not None and ni.quantisation:
                value = lo + round((value - lo) / spec.sensor_resolution) * spec.sensor_resolution

        # Metadata deliberately carries no ground truth and no experiment name.
        quantum = spec.position_tolerance / 4.0
        result = MeasurementResult(
            timestamp=datetime.now(timezone.utc),
            actuator_position=self._commanded,
            sensor_value=value,
            units_actuator=spec.independent.units,
            units_sensor=spec.dependent.units,
            is_valid=True,
            metadata={"simulation": True, "cycle": cycle, "position_feedback": round(self._actual / quantum) * quantum},
        )
        self._sample_history.append(result)
        return result

    def wait(self, seconds: float) -> None:
        """A pause is one rig cycle: drift moves on and time-limited faults run down. No real sleep."""
        cycle = self._advance_cycle()
        dropout = self._active("sensor_dropout", cycle)
        if dropout is not None:
            self._log(dropout, "sensor_dropout", cycle, read=False, mode=self._faults[dropout].mode)

    def get_status(self) -> RigStatus:
        return RigStatus(
            is_connected=self._connected,
            is_homed=self._is_homed,
            current_position=self._commanded,
            is_busy=self._is_busy,
            telemetry={
                "samples_recorded": len(self._sample_history),
                "mode": "simulation",
                "cycle": self._cycle,
            }
        )
