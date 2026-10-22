"""
Phase 4: rate limit, oscillation interlock, watchdog, latched E-stop with operator reset,
the append-only audit log, careful type checks, step-limited GP proposals and strict LLM numbers.
"""
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from pydantic import ValidationError

from agents.models import PlannerOutput, StrategyDecision
from agents.orchestrator import DiscoveryEngine
from analysis.stopping import StopReason
from config import FaultSpec, SafetyConfig, SimulationConfig
from experiments.registry import get_experiment
from hardware.audit import verify_chain
from hardware.safety_guard import EmergencyStopActive, HardwareSafetyGuard, SafetyLimits, SafetyViolationError
from hardware.simulator import PhysicsSimulationRig
from records.run_record import load_run_record
from safety.spy import SpyRig, unsafe_commands


def guarded(name="malus_law", faults=(), **safety):
    spec = get_experiment(name)
    rig = SpyRig(PhysicsSimulationRig(spec, np.random.default_rng(0),
                                      SimulationConfig(noise_std=0.0, transit_delay_sec=0.0, faults=list(faults))))
    guard = HardwareSafetyGuard(rig, SafetyLimits.from_spec(spec, SafetyConfig(**safety)))
    guard.connect()
    guard.home()
    return guard, rig


def measure(guard, x):
    guard.safe_move(x)
    return guard.safe_read()


# --- rate limit and oscillation interlock ---------------------------------------------------------

def test_rate_limit_counts_moves_per_second_of_rig_time():
    guard, rig = guarded(max_moves_per_s=5, max_moves_without_reading=100)
    for i in range(5):  # tiny moves take 0.05 s of rig time each
        guard.safe_move(10.0 + 0.1 * i)
    with pytest.raises(SafetyViolationError, match="rate limit"):
        guard.safe_move(11.0)
    rig.wait(1.0)  # a second later the window is clear again
    guard.safe_move(11.0)


def test_normal_measuring_never_hits_the_rate_limit():
    guard, rig = guarded()
    for x in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:  # small steps, each with a reading
        assert measure(guard, x).is_valid
    assert unsafe_commands(rig.commands, guard.limits, guard.audit_entries) == []


def test_moves_without_readings_are_interlocked():
    guard, _ = guarded(max_moves_without_reading=3)
    guard.safe_move(0.0)
    guard.safe_move(180.0)
    guard.safe_move(0.0)
    with pytest.raises(SafetyViolationError, match="3 moves without a sensor reading"):
        guard.safe_move(180.0)
    guard.safe_read()  # a reading clears it
    guard.safe_move(180.0)


# --- watchdog and E-stop ---------------------------------------------------------------------------

def test_the_watchdog_latches_the_estop_without_valid_readings():
    guard, rig = guarded(faults=[FaultSpec(type="sensor_dropout", at_step=1, duration=50)], watchdog_s=5.0)
    guard.safe_move(90.0)
    for _ in range(6):
        assert not guard.safe_read().is_valid
        guard.pause(1.0)
    with pytest.raises(EmergencyStopActive, match="watchdog expired"):
        guard.safe_move(45.0)
    assert guard.estop_active and guard.estop_reason.startswith("watchdog")
    with pytest.raises(EmergencyStopActive):
        guard.safe_move(45.0)
    moves = len(rig.moves)
    guard.reset_estop(operator="lab technician")
    guard.safe_move(45.0)
    assert len(rig.moves) == moves + 1


def test_a_valid_reading_feeds_the_watchdog():
    guard, rig = guarded(watchdog_s=5.0)
    for i in range(10):
        measure(guard, 90.0)
        rig.wait(3.0)  # 3 s between valid readings is fine
    assert not guard.estop_active


def test_estop_is_latched_until_a_named_operator_resets_it():
    guard, rig = guarded()
    guard.emergency_stop("dashboard button")
    for call in (lambda: guard.safe_move(90.0), guard.safe_read, lambda: guard.pause(1.0), guard.home):
        with pytest.raises(EmergencyStopActive):
            call()
    for bad in ("", "   ", None):
        with pytest.raises(SafetyViolationError, match="operator's name"):
            guard.reset_estop(operator=bad)
    assert guard.estop_active and rig.moves == []
    guard.reset_estop(operator="Nikhil")
    guard.safe_move(90.0)
    resets = [e for e in guard.audit_entries if e.action == "reset"]
    assert [e.decision for e in resets] == ["rejected", "rejected", "rejected", "allowed"]
    assert "reset by Nikhil (was: dashboard button)" in resets[-1].reason


def test_the_engine_aborts_when_the_estop_is_pressed_mid_run(make_config):
    holder = {}

    def press_after_third_measurement(event):
        if event.type == "measurement" and event.n_points == 3:
            holder["engine"].emergency_stop("operator pressed E-STOP")

    engine = DiscoveryEngine(make_config(seed=0), on_event=press_after_third_measurement)
    holder["engine"] = engine
    record = engine.run()
    assert record.stop.reason == StopReason.SAFETY_ABORT
    assert "E-stop latched (operator pressed E-STOP)" in record.stop.message
    assert record.result.n_measurements == 3
    assert not engine.emergency_stop()  # no run in progress any more
    tripped = [e for e in record.audit_log if e.action == "estop"]
    assert len(tripped) == 1 and tripped[0].decision == "tripped"
    after = [e for e in record.audit_log if e.seq > tripped[0].seq and e.action == "move" and e.decision == "allowed"]
    assert after == []


# --- audit log ----------------------------------------------------------------------------------------

def test_every_request_and_decision_is_in_a_hash_chained_log():
    guard, rig = guarded()
    measure(guard, 90.0)
    with pytest.raises(SafetyViolationError):
        guard.safe_move(500.0)
    actions = [(e.action, e.decision) for e in guard.audit_entries]
    assert actions == [("connect", "allowed"), ("home", "allowed"), ("move", "allowed"), ("read", "allowed"),
                       ("move", "rejected")]
    assert "exceeds physical limits" in guard.audit_entries[-1].reason and guard.audit_entries[-1].request == "500.0"
    assert verify_chain(guard.audit_entries) == (True, "")


def test_tampering_with_the_audit_log_is_detected():
    guard, _ = guarded()
    for x in (10.0, 20.0, 30.0):
        measure(guard, x)
    entries = list(guard.audit_entries)
    edited = entries[:3] + [entries[3].model_copy(update={"decision": "rejected"})] + entries[4:]
    assert verify_chain(edited) == (False, "entry 3 was changed after it was written")
    assert not verify_chain(entries[:2] + entries[3:])[0]  # deleted
    assert not verify_chain([entries[1], entries[0], *entries[2:]])[0]  # reordered
    assert not hasattr(guard.audit, "remove") and isinstance(guard.audit.entries, tuple)


def test_the_run_record_carries_an_intact_audit_log(make_config):
    record = DiscoveryEngine(make_config(seed=1)).run()
    loaded = load_run_record(record.artifacts.record)
    assert loaded.audit_log == record.audit_log and loaded.audit_intact()
    allowed_moves = [e for e in loaded.audit_log if e.action == "move" and e.decision == "allowed"]
    assert len(allowed_moves) == len(record.steps)  # every attempt that reached the rig was approved


# --- type checks ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [10 ** 400, -(10 ** 400), Decimal("90"), Decimal("NaN"), np.float64("nan"),
                                   complex(90, 0), "90", b"90", [90.0], None, True])
def test_values_that_are_not_finite_reals_are_refused_without_crashing(value):
    guard, rig = guarded()
    with pytest.raises(SafetyViolationError, match="not a finite number"):
        guard.safe_move(value)
    assert rig.moves == []


@pytest.mark.parametrize("value, expected", [(Fraction(1, 2), 0.5), (np.float64(90.0), 90.0), (np.int64(45), 45.0)])
def test_legal_numeric_types_inside_the_bounds_are_accepted_as_floats(value, expected):
    guard, rig = guarded()
    guard.safe_move(value)
    assert rig.moves[-1].position == expected and isinstance(rig.inner.get_status().current_position, float)


# --- realistic step limits ------------------------------------------------------------------------------

@pytest.mark.parametrize("name, step", [("diode_iv", 0.4), ("hookes_law", 10.0), ("stefan_boltzmann", 1.0)])
def test_gp_proposals_respect_the_step_limit(make_config, name, step):
    assert get_experiment(name).max_step_delta == step
    record = DiscoveryEngine(make_config(experiment=name, seed=2)).run()
    assert not [s for s in record.steps if s.status == "rejected"]
    previous = 0.0
    for s in record.steps:
        assert abs(s.requested_position - previous) <= step + 1e-9
        previous = s.position if s.position is not None else previous
    assert record.result.final_analysis.best.key == get_experiment(name).ground_truth_model


# --- strict numbers in LLM answers --------------------------------------------------------------------

@pytest.mark.parametrize("position", ["90", "90 deg", True, [90], {"v": 90}, math.nan, math.inf])
def test_llm_positions_must_be_json_numbers(position):
    with pytest.raises(ValidationError):
        StrategyDecision.model_validate({"choice": "self", "position": position, "reason": "r"})


def test_llm_initial_designs_must_be_json_numbers():
    base = {"hypothesis": {"statement": "s", "candidate_relationships": ["r"], "independent_variable": "x",
                           "proposed_range": [0, 180]}, "rationale": "r"}
    assert PlannerOutput.model_validate({**base, "initial_design": [0, 90.5, 180]}).initial_design == [0.0, 90.5, 180.0]
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate({**base, "initial_design": ["0", "90", "180"]})
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate({**base, "hypothesis": {**base["hypothesis"], "proposed_range": [False, True]},
                                      "initial_design": [0, 90, 180]})
