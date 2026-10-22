"""
Unit tests for the Physics Simulation Rig (Digital Twin).
"""
import math

import numpy as np
import pytest

from config import HardwareConfig, SimulationConfig
from experiments.registry import get_experiment
from hardware.base import DataSource
from hardware.serial_driver import SerialHardwareRig
from hardware.simulator import PhysicsSimulationRig

def make_rig(name: str, noise_std=0.0) -> PhysicsSimulationRig:
    rig = PhysicsSimulationRig(
        get_experiment(name), np.random.default_rng(0),
        SimulationConfig(noise_std=noise_std, transit_delay_sec=0.0),
    )
    rig.connect()
    return rig

def read_at(rig: PhysicsSimulationRig, x: float) -> float:
    rig.move_actuator(x)
    return rig.read_sensor().sensor_value

def test_simulator_connection_and_homing():
    sim = PhysicsSimulationRig(get_experiment("malus_law"), np.random.default_rng(0))
    assert not sim.is_connected()
    assert sim.connect()
    assert sim.is_connected()
    assert sim.home()
    status = sim.get_status()
    assert status.is_homed
    assert status.current_position == 0.0

def test_malus_law_intensity_profile():
    rig = make_rig("malus_law")
    p = rig.spec.true_params
    # Maximum at the polariser offset, minimum 90 degrees later, half way at 45 degrees
    assert read_at(rig, p["x0"]) == pytest.approx(p["A"] + p["C"])
    assert read_at(rig, p["x0"] + 90.0) == pytest.approx(p["C"])
    assert read_at(rig, p["x0"] + 45.0) == pytest.approx(p["A"] / 2 + p["C"])
    assert read_at(rig, 0.0) == pytest.approx(p["A"] * math.cos(math.radians(p["x0"])) ** 2 + p["C"])

def test_inverse_square_law_profile():
    rig = make_rig("inverse_square_law")
    p = rig.spec.true_params
    near, far = read_at(rig, 5.0), read_at(rig, 20.0)
    assert near > far
    assert far == pytest.approx(p["A"] / (20.0 + p["x0"]) ** 2 + p["C"])

def test_beer_lambert_law_profile():
    rig = make_rig("beer_lambert_law")
    p = rig.spec.true_params
    assert read_at(rig, 0.0) == pytest.approx(p["A"] + p["C"])
    assert read_at(rig, 1.0 / p["k"]) == pytest.approx(p["A"] / math.e + p["C"])

def test_noise_level_uses_spec_default_unless_overridden():
    spec = get_experiment("malus_law")
    assert PhysicsSimulationRig(spec, np.random.default_rng(0)).noise_std == spec.noise.std
    assert make_rig("malus_law", noise_std=3.0).noise_std == 3.0

def test_measurements_carry_spec_units_and_no_ground_truth():
    rig = make_rig("beer_lambert_law")
    rig.move_actuator(10.0)
    m = rig.read_sensor()
    assert (m.units_actuator, m.units_sensor) == ("mg/L", "lux")
    assert set(m.metadata) == {"simulation", "cycle", "position_feedback"}
    assert m.metadata["position_feedback"] == 10.0 and m.metadata["cycle"] == 1
    assert "experiment" not in rig.get_status().telemetry

def test_data_source_labels():
    assert make_rig("malus_law").data_source == DataSource.DIGITAL_TWIN
    assert SerialHardwareRig(HardwareConfig(port="/dev/ttyUSB0")).data_source == DataSource.PHYSICAL_RIG
    assert SerialHardwareRig(HardwareConfig(port="rfc2217://localhost:4000")).data_source == DataSource.FIRMWARE_IN_THE_LOOP
