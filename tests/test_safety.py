"""
Unit tests for the Hardware Safety Guard layer.
"""
import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from agents.orchestrator import DiscoveryEngine
from analysis.stopping import StopReason
from config import LLMConfig, SimulationConfig
from experiments.registry import get_experiment
from hardware.simulator import PhysicsSimulationRig
from hardware.safety_guard import HardwareSafetyGuard, SafetyLimits, SafetyViolationError

def make_guard(name: str) -> HardwareSafetyGuard:
    spec = get_experiment(name)
    rig = PhysicsSimulationRig(spec, np.random.default_rng(0), SimulationConfig(transit_delay_sec=0.0))
    lo, hi = spec.safe_bounds
    guard = HardwareSafetyGuard(rig, SafetyLimits(min_position=lo, max_position=hi, max_step_delta=spec.max_step_delta))
    guard.connect()
    return guard

def test_safety_guard_bounds_clamping():
    guard = make_guard("malus_law")

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

@pytest.mark.parametrize("name, rejected, accepted", [
    ("malus_law", [-1.0, 180.5], [0.0, 150.0, 180.0]),
    ("inverse_square_law", [2.0, 150.0], [5.0, 50.0, 100.0]),
    ("beer_lambert_law", [-0.1, 150.0], [0.0, 100.0]),
])
def test_bounds_come_from_the_experiment_spec(name, rejected, accepted):
    guard = make_guard(name)
    for position in rejected:
        with pytest.raises(SafetyViolationError, match="exceeds physical limits"):
            guard.validate_actuator_target(position)
    for position in accepted:
        assert guard.validate_actuator_target(position) == position

@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "45", None, True])
def test_non_finite_or_non_numeric_targets_are_rejected(bad):
    with pytest.raises(SafetyViolationError, match="not a finite number"):
        make_guard("malus_law").safe_move(bad)

def test_safety_limits_must_be_ordered():
    with pytest.raises(ValidationError):
        SafetyLimits(min_position=10.0, max_position=5.0, max_step_delta=1.0)

def test_safety_guard_emergency_stop():
    guard = make_guard("malus_law")
    guard.emergency_stop()

    with pytest.raises(SafetyViolationError, match="Emergency Stop"):
        guard.safe_move(45.0)

    with pytest.raises(SafetyViolationError, match="operator's name"):
        guard.reset_estop(operator="  ")
    with pytest.raises(SafetyViolationError, match="Emergency Stop"):
        guard.safe_move(45.0)  # still latched

    guard.reset_estop(operator="test operator")
    res = guard.safe_move(45.0)
    assert res["success"]

def test_rejections_are_reported_and_abort_the_run(make_config, tmp_path):
    """Phase 1 semantics (provider none: no agent to give the rejection to)."""
    events = []
    strict = get_experiment("malus_law").model_copy(update={"max_step_delta": 1.0})
    cfg = make_config(seed=0, llm=LLMConfig(provider="none", cache_dir=tmp_path / "llm_cache"))
    record = DiscoveryEngine(cfg, on_event=events.append, spec=strict).run()

    rejections = [e for e in events if e.type == "safety_rejection"]
    assert [e.requested_position for e in rejections] == [45.0, 90.0, 180.0]
    assert all("maximum allowed delta" in e.reason for e in rejections)
    assert all(e.fed_back_to is None for e in rejections)
    assert [s.status for s in record.steps] == ["measured", "rejected", "rejected", "rejected"]
    assert record.measured_points()[0] == [0.0]  # nothing rejected reached the rig
    assert record.stop.reason == StopReason.SAFETY_ABORT
    assert record.llm_calls == []


def test_a_rejected_position_is_never_resent_blindly(make_config):
    """With an agent, every rejection is fed back; the GP fallback never repeats a rejected position."""
    events = []
    strict = get_experiment("malus_law").model_copy(update={"max_step_delta": 1.0})
    record = DiscoveryEngine(make_config(seed=0), on_event=events.append, spec=strict).run()

    assert record.stop.reason == StopReason.SAFETY_ABORT
    assert record.measured_points()[0] == [0.0]
    rejected = [s for s in record.steps if s.status == "rejected"]
    # Within a slot: the proposal, 2 Planner re-proposals (here delegated to the GP), then one GP fallback.
    assert [s.attempt_in_slot for s in rejected] == [1, 2, 3, 4] * 3
    gp_requests = [s.requested_position for s in rejected if s.source == "gp"]
    assert len(gp_requests) == len(set(gp_requests))
    fed_back = [e.fed_back_to for e in events if e.type == "safety_rejection"]
    assert fed_back == ["planner", "planner", "gp", None] * 3
    reproposals = [c for c in record.llm_calls if c.purpose == "repropose"]
    assert len(reproposals) == 6
    assert all("exceeds maximum allowed delta" in json.loads(c.user)["observation"]["guard_message"] for c in reproposals)
