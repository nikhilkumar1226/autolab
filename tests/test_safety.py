"""
Unit tests for the Hardware Safety Guard layer.
"""
import pytest
from config import HardwareConfig
from hardware.simulator import PhysicsSimulationRig
from hardware.safety_guard import HardwareSafetyGuard, SafetyViolationError

def test_safety_guard_bounds_clamping():
    config = HardwareConfig(min_actuator_value=0.0, max_actuator_value=180.0, max_step_delta=180.0)
    sim = PhysicsSimulationRig()
    guard = HardwareSafetyGuard(sim, config)
    guard.connect()

    # Valid move
    res = guard.safe_move(90.0)
    assert res["success"]
    assert res["new_position"] == 90.0

    # Negative out of bounds
    with pytest.raises(SafetyViolationError, match="SAFETY INTERCEPT: Target position -15.0 exceeds physical limits"):
        guard.safe_move(-15.0)

    # Exceeding upper bound
    with pytest.raises(SafetyViolationError, match="SAFETY INTERCEPT: Target position 200.0 exceeds physical limits"):
        guard.safe_move(200.0)

def test_safety_guard_emergency_stop():
    guard = HardwareSafetyGuard(PhysicsSimulationRig())
    guard.connect()
    guard.emergency_stop()

    with pytest.raises(SafetyViolationError, match="Emergency Stop"):
        guard.safe_move(45.0)

    guard.reset_estop()
    res = guard.safe_move(45.0)
    assert res["success"]
