"""
Shared test fixtures. Tests run offline: digital twin only, no transit delay,
all outputs written under pytest's tmp_path.
"""
from typing import Sequence

import numpy as np
import pytest

from config import OutputConfig, SimulationConfig, SystemConfig
from experiments.registry import ExperimentSpec
from hardware.simulator import PhysicsSimulationRig


@pytest.fixture
def make_config(tmp_path):
    def _make(experiment: str = "malus_law", seed: int = 0, write_report: bool = False, **overrides) -> SystemConfig:
        return SystemConfig(
            experiment=experiment,
            seed=seed,
            simulation=SimulationConfig(transit_delay_sec=0.0),
            output=OutputConfig(
                runs_dir=tmp_path / "runs",
                plots_dir=tmp_path / "plots",
                reports_dir=tmp_path / "reports",
                write_report=write_report,
            ),
            **overrides,
        )
    return _make


@pytest.fixture
def simulate():
    """Noisy readings of an experiment's digital twin at the given positions."""
    def _simulate(spec: ExperimentSpec, positions: Sequence[float], seed: int) -> np.ndarray:
        rig = PhysicsSimulationRig(spec, np.random.default_rng(seed), SimulationConfig(transit_delay_sec=0.0))
        rig.connect()
        values = []
        for x in positions:
            rig.move_actuator(float(x))
            values.append(rig.read_sensor().sensor_value)
        return np.array(values)
    return _simulate
