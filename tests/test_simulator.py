"""
Unit tests for the Physics Simulation Rig (Digital Twin).
"""
import pytest
from config import SimulationConfig
from hardware.simulator import PhysicsSimulationRig

def test_simulator_connection_and_homing():
    sim = PhysicsSimulationRig()
    assert not sim.is_connected()
    assert sim.connect()
    assert sim.is_connected()
    assert sim.home()
    status = sim.get_status()
    assert status.is_homed
    assert status.current_position == 0.0

def test_malus_law_intensity_profile():
    config = SimulationConfig(
        experiment_type="malus_law",
        i0_intensity=1000.0,
        theta_offset=0.0,
        ambient_noise_level=10.0,
        noise_std=0.0, # Zero noise for deterministic test
        transit_delay_sec=0.0
    )
    sim = PhysicsSimulationRig(config)
    sim.connect()

    # 0 degrees: peak transmission (I0 + ambient = 1010)
    sim.move_actuator(0.0)
    res_0 = sim.read_sensor()
    assert pytest.approx(res_0.sensor_value, abs=1e-3) == 1010.0

    # 90 degrees: minimum transmission (ambient = 10)
    sim.move_actuator(90.0)
    res_90 = sim.read_sensor()
    assert pytest.approx(res_90.sensor_value, abs=1e-3) == 10.0

    # 45 degrees: half transmission (I0 * 0.5 + ambient = 510)
    sim.move_actuator(45.0)
    res_45 = sim.read_sensor()
    assert pytest.approx(res_45.sensor_value, abs=1e-3) == 510.0

    # 180 degrees: peak transmission again (1010)
    sim.move_actuator(180.0)
    res_180 = sim.read_sensor()
    assert pytest.approx(res_180.sensor_value, abs=1e-3) == 1010.0

def test_inverse_square_law_profile():
    config = SimulationConfig(
        experiment_type="inverse_square_law",
        i0_intensity=1000.0,
        ambient_noise_level=5.0,
        noise_std=0.0,
        transit_delay_sec=0.0
    )
    sim = PhysicsSimulationRig(config)
    sim.connect()

    # Intensity at d1 < Intensity at d0
    sim.move_actuator(5.0)
    res_5 = sim.read_sensor()

    sim.move_actuator(20.0)
    res_20 = sim.read_sensor()

    assert res_5.sensor_value > res_20.sensor_value
