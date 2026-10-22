"""
Phase 3: anomaly detection by the safety guard, recovery in the closed loop (Executor
re-measure, Analyst decision), fault scoring against the twin's ground truth, parsimony,
and the CLI flags.
"""
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console

import main
from agents.analyst import DataAnalystAgent, heuristic_anomaly
from agents.orchestrator import DiscoveryEngine
from agents.prompts import ANOMALY_SYSTEM
from analysis.model_library import MODEL_LIBRARY, fit_library
from analysis.parsimony import apply_parsimony
from analysis.stopping import StopReason, evaluate_stopping
from cli.console import ConsoleEventPrinter
from config import (
    DEFAULT_FAULTS, AgentConfig, AnomalyConfig, FaultSpec, LLMConfig, NonIdealityConfig, SimulationConfig, StoppingConfig,
)
from experiments.registry import get_experiment, list_experiments
from hardware.base import MeasurementResult
from hardware.safety_guard import HardwareSafetyGuard, SafetyLimits, SafetyViolationError
from hardware.simulator import PhysicsSimulationRig
from records.faults import score_faults
from records.run_record import load_run_record


def guard_for(name="malus_law", faults=(), spec=None):
    spec = spec or get_experiment(name)
    rig = PhysicsSimulationRig(spec, np.random.default_rng(0),
                               SimulationConfig(noise_std=0.0, transit_delay_sec=0.0, faults=list(faults)))
    guard = HardwareSafetyGuard(rig, SafetyLimits.from_spec(spec))
    guard.connect()
    return guard


def reading(value, position=10.0, **metadata):
    return MeasurementResult(actuator_position=position, sensor_value=value, metadata=metadata)


# --- the guard's reading checks --------------------------------------------------------------

@pytest.mark.parametrize("value, metadata, expected, flags", [
    (500.0, {}, (), []),
    (math.nan, {}, (), ["non_finite"]),
    (math.inf, {}, (), ["non_finite"]),
    (1200.0, {}, (), ["saturated"]),
    (-5.0, {}, (), ["out_of_range"]),
    (500.0, {"position_feedback": 13.0}, (), ["position_mismatch"]),
    (500.0, {"position_feedback": 11.5}, (), []),
    (500.0, {}, [(300.0, 10.0, 4.0)], ["outlier"]),
    (500.0, {}, [(300.0, 10.0, 4.0), (480.0, 10.0, 4.0)], []),  # outlier only if outside every band
    (1500.0, {"position_feedback": 30.0}, [(300.0, 10.0, 4.0)], ["saturated", "position_mismatch", "outlier"]),
])
def test_the_guard_flags_suspicious_readings(value, metadata, expected, flags):
    assert guard_for().check_reading(reading(value, **metadata), expected) == flags


def test_a_sensor_timeout_becomes_a_flagged_reading():
    guard = guard_for(faults=[FaultSpec(type="sensor_dropout", at_step=1, mode="timeout")])
    guard.safe_move(45.0)
    result = guard.safe_read()
    assert result.flags == ["timeout"] and not result.is_valid and math.isnan(result.sensor_value)
    assert result.metadata["cycle"] == 1


def test_over_the_response_limit_the_guard_backs_off_and_sets_a_ceiling():
    spec = get_experiment("diode_iv").model_copy(update={"safe_bounds": (0.0, 0.9), "max_step_delta": 0.9})
    guard = guard_for(spec=spec)
    guard.safe_move(0.88)  # Shockley: about 100 mA, over the 30 mA limit
    result = guard.safe_read()
    assert "response_limit" in result.flags and not result.is_valid
    assert guard.ceiling == 0.88 and guard.get_status().current_position == 0.0
    with pytest.raises(SafetyViolationError, match="exceeded its limit of 30"):
        guard.safe_move(0.9)
    guard.safe_move(0.7)  # below the ceiling: allowed


def test_a_run_never_keeps_a_reading_over_the_response_limit(make_config):
    """Bounds set wider than the spec's (0.9 V instead of 0.8 V): the guard trips, and nothing over the limit is kept."""
    spec = get_experiment("diode_iv").model_copy(update={"safe_bounds": (0.0, 0.9), "max_step_delta": 0.9})
    events = []
    record = DiscoveryEngine(make_config(experiment="diode_iv", seed=0), on_event=events.append, spec=spec).run()
    _, values = record.measured_points()
    assert max(values) <= spec.response_limit
    tripped = [s for s in record.steps if "response_limit" in s.flags]
    assert tripped and all(s.anomaly.action == "discard" and s.anomaly.decided_by == "safety_guard" for s in tripped)
    # Every later request is below the lowest trip position: the GP respects the guard's ceiling.
    first = tripped[0].index
    ceilings = [min(s.requested_position for s in tripped if s.index < t.index) for t in record.steps if t.index > first]
    later = [s.requested_position for s in record.steps if s.index > first]
    assert all(p < c for p, c in zip(later, ceilings))
    # Repeated trips end the run as a safety abort, which is the safe outcome for wrong bounds.
    assert record.stop.reason in (StopReason.CONVERGED, StopReason.BUDGET_EXHAUSTED, StopReason.SAFETY_ABORT)


# --- recovery in the closed loop ---------------------------------------------------------------

def faulty_config(make_config, experiment="malus_law", seed=3, faults=DEFAULT_FAULTS, nonideal=False, **kw):
    return make_config(
        experiment=experiment, seed=seed,
        simulation=SimulationConfig(transit_delay_sec=0.0, faults=list(faults), nonideal=NonIdealityConfig(enabled=nonideal)),
        **kw,
    )


def test_each_default_fault_is_detected_and_recovered(make_config):
    events = []
    record = DiscoveryEngine(faulty_config(make_config, experiment="diode_iv", seed=3), on_event=events.append).run()
    by_type = {f.type: f for f in record.faults if f.scheduled}
    assert set(by_type) == {"light_spike", "stuck_actuator", "sensor_dropout"}
    assert by_type["stuck_actuator"].status == "recovered" and "position_mismatch" in by_type["stuck_actuator"].flags
    assert by_type["sensor_dropout"].status == "recovered" and by_type["sensor_dropout"].resolution == "pause"
    assert by_type["light_spike"].status == "recovered" and "outlier" in by_type["light_spike"].flags
    # A flagged reading never enters the data.
    flagged = [s for s in record.steps if s.status == "anomaly"]
    assert {s.rig_cycle for s in flagged}.isdisjoint({s.rig_cycle for s in record.steps if s.status == "measured"})
    anomaly_events = [e for e in events if e.type == "anomaly"]
    assert [e.step for e in anomaly_events] == [s.index for s in flagged]
    assert anomaly_events[0].handling.decided_by == "executor" and anomaly_events[0].handling.action == "remeasure"
    # The dropout's second flag went to the Analyst (mock policy = the heuristic): pause.
    dropout = [s for s in flagged if s.flags == ["non_finite"]]
    assert [s.anomaly.decided_by for s in dropout] == ["executor", "analyst"]
    assert dropout[1].anomaly.action == "pause" and dropout[1].anomaly.source == "mock"
    [call] = [c for c in record.llm_calls if c.purpose == "anomaly"]
    assert call.agent == "analyst" and dropout[1].anomaly.llm_call_id == call.id


def test_the_anomaly_prompt_is_blind(make_config):
    """Variable names and units only (as for every Analyst call); never the apparatus or the experiment."""
    spec = get_experiment("diode_iv")
    record = DiscoveryEngine(faulty_config(make_config, experiment="diode_iv", seed=3)).run()
    [call] = [c for c in record.llm_calls if c.purpose == "anomaly"]
    text = call.system + call.user
    assert spec.name not in text and spec.description_for_agents not in text and "silicon" not in text
    blind = DiscoveryEngine(faulty_config(make_config, experiment="diode_iv", seed=3,
                                          agent=AgentConfig(blind=True))).run()
    [blind_call] = [c for c in blind.llm_calls if c.purpose == "anomaly"]
    assert "diode" not in blind_call.user.lower() and json.loads(blind_call.user)["variables"]["x"]["name"] == "x"
    payload = json.loads(call.user)
    assert set(payload) == {"variables", "position", "flags", "readings_here", "expected", "position_feedback",
                            "n_measurements", "decisions_left"}


def test_provider_none_uses_the_heuristic_and_matches_mock(make_config, tmp_path):
    none = DiscoveryEngine(faulty_config(make_config, experiment="diode_iv", seed=3,
                                         llm=LLMConfig(provider="none", cache_dir=tmp_path))).run()
    mock = DiscoveryEngine(faulty_config(make_config, experiment="diode_iv", seed=3)).run()
    assert none.measured_points() == mock.measured_points()
    assert [f.status for f in none.faults] == [f.status for f in mock.faults]
    assert any(s.anomaly is not None and s.anomaly.source == "heuristic" for s in none.steps)
    assert none.llm_calls == []


def test_an_llm_decision_to_discard_is_followed(make_config, litellm_config, fake_litellm):
    from tests.test_agents import AgentScript
    script = AgentScript().on(ANOMALY_SYSTEM, lambda p: {"action": "discard", "reason": "give up this point"})
    fake_litellm.responder = script
    dropout = [FaultSpec(type="sensor_dropout", at_step=6, duration=5)]
    record = DiscoveryEngine(faulty_config(make_config, faults=dropout, llm=litellm_config())).run()
    decided = [s for s in record.steps if s.anomaly is not None and s.anomaly.decided_by == "analyst"]
    assert decided and decided[0].anomaly.action == "discard" and decided[0].anomaly.source == "llm"
    position = decided[0].requested_position
    later = [s for s in record.steps if s.index > decided[0].index]
    assert all(abs(s.requested_position - position) > 1.0 for s in later)  # never proposed again
    assert record.faults[0].status == "recovered"


def test_a_genuine_reading_flagged_as_an_outlier_is_accepted_after_it_is_confirmed(make_config):
    """A band that is far too narrow flags good data; the agreeing re-measurement is accepted."""
    cfg = make_config(seed=0, agent=AgentConfig(anomaly=AnomalyConfig(outlier_p=0.9)))
    record = DiscoveryEngine(cfg).run()
    confirmed = [s for s in record.steps if s.status == "measured" and "accepted as genuine" in s.message]
    assert confirmed
    assert record.stop.reason == StopReason.CONVERGED and record.result.final_analysis.best.key == "cos2_phase"


def test_a_spike_too_small_to_flag_is_scored_as_missed(make_config):
    tiny = [FaultSpec(type="light_spike", at_step=6, magnitude=0.002)]
    record = DiscoveryEngine(faulty_config(make_config, faults=tiny)).run()
    [fault] = record.faults
    assert fault.status == "missed" and fault.missed and not fault.detected


def test_a_fault_after_the_run_ended_is_not_reached(make_config):
    late = [FaultSpec(type="light_spike", at_step=500)]
    record = DiscoveryEngine(faulty_config(make_config, faults=late)).run()
    assert record.faults[0].status == "not_reached"
    assert record.fault_summary.reached == 0


def test_score_faults_rules():
    def step(index, cycle, status, flags=(), action=None):
        anomaly = SimpleNamespace(action=action) if action else None
        return SimpleNamespace(index=index, rig_cycle=cycle, status=status, flags=list(flags), anomaly=anomaly)

    steps = [step(1, 1, "measured"), step(2, 2, "anomaly", ["outlier"], "remeasure"), step(3, 3, "measured"),
             step(4, 4, "anomaly", ["position_mismatch"], "remeasure"), step(5, 5, "measured")]
    log = [SimpleNamespace(fault_index=0, type="light_spike", cycle=2, read=True),
           SimpleNamespace(fault_index=1, type="sensor_dropout", cycle=3, read=False),
           SimpleNamespace(fault_index=None, type="outlier", cycle=5, read=True)]
    schedule = [SimpleNamespace(type="light_spike", at_step=2, duration=1),
                SimpleNamespace(type="sensor_dropout", at_step=3, duration=1),
                SimpleNamespace(type="stuck_actuator", at_step=9, duration=1)]
    outcomes, summary = score_faults(steps, log, schedule, last_cycle=5, run_ok=True)
    assert [o.status for o in outcomes] == ["recovered", "no_effect", "not_reached", "missed"]
    assert outcomes[0].affected_steps == [2] and outcomes[0].resolution == "remeasure"
    assert outcomes[3].scheduled is False and outcomes[3].type == "outlier"
    assert summary.unexplained == {"position_mismatch": 1}  # cycle 4: nothing injected, a false alarm
    assert (summary.scheduled, summary.reached, summary.recovered, summary.outliers_missed) == (3, 2, 1, 1)


def test_heuristic_anomaly_policy():
    assert heuristic_anomaly({"flags": ["timeout"]})["action"] == "pause"
    assert heuristic_anomaly({"flags": ["non_finite"]})["action"] == "pause"
    assert heuristic_anomaly({"flags": ["saturated"]})["action"] == "discard"
    assert heuristic_anomaly({"flags": ["position_mismatch"]})["action"] == "remeasure"
    assert heuristic_anomaly({"flags": ["outlier"]})["action"] == "remeasure"


# --- Done when: every experiment runs with faults on and off; the record lists every fault -----

@pytest.mark.parametrize("name", list_experiments())
def test_every_experiment_runs_with_faults_and_non_idealities(make_config, name):
    record = DiscoveryEngine(faulty_config(make_config, experiment=name, seed=1, nonideal=True)).run()
    assert record.stop.reason in (StopReason.CONVERGED, StopReason.BUDGET_EXHAUSTED)
    scheduled = [f for f in record.faults if f.scheduled]
    assert [f.type for f in scheduled] == [f.type for f in DEFAULT_FAULTS]
    loaded = load_run_record(record.artifacts.record)
    assert loaded.faults == record.faults and loaded.fault_summary == record.fault_summary
    raw = json.loads(Path(record.artifacts.record).read_text(encoding="utf-8"))
    assert raw["schema_version"] == 3 and all({"status", "detected", "recovered", "missed"} <= set(f) for f in raw["faults"])


def test_without_faults_the_record_has_no_fault_entries(make_config):
    record = DiscoveryEngine(make_config(experiment="rc_charging", seed=0)).run()
    assert record.faults == [] and record.fault_summary.scheduled == 0


# --- parsimony ---------------------------------------------------------------------------------

def test_parsimony_prefers_the_line_and_lets_a_linear_law_converge(simulate):
    spec = get_experiment("hookes_law")
    x = np.linspace(0.0, 20.0, 20)
    y = simulate(spec, x, seed=0)
    library = fit_library(x, y)
    assert library.best.key != "linear"  # by BIC alone a curved model wins here
    result = DataAnalystAgent().analyze(x, y)
    assert result.best.key == "linear"
    assert result.fits[0].key == library.best.key and result.fits[0].redundant is not None  # kept, not counted
    redundant = [f for f in result.fits if f.redundant is not None]
    assert all(f.n_params == 3 and "same curve as Linear" in f.redundant for f in redundant)
    decision = evaluate_stopping(result, 20, StoppingConfig())
    assert decision.delta_bic.passed


def test_parsimony_changes_nothing_for_curved_laws(simulate):
    for name in ("malus_law", "inverse_square_law", "beer_lambert_law"):
        spec = get_experiment(name)
        x = np.linspace(*spec.safe_bounds, 12)
        y = simulate(spec, x, seed=2)
        library = fit_library(x, y)
        kept = apply_parsimony(library, x, y)
        assert all(f.redundant is None for f in kept.fits)
        assert kept.delta_bic_runner_up == pytest.approx(library.delta_bic_runner_up)


def test_only_redundant_rivals_pass_the_delta_bic_criterion():
    x = np.linspace(0.0, 20.0, 12)
    y = 4.0 * x + 50.0  # noise-free: a quadratic and an exponential reproduce the line exactly
    models = [MODEL_LIBRARY[k] for k in ("linear", "quadratic", "exp_decay")]
    result = apply_parsimony(fit_library(x, y, models=models), x, y)
    assert result.best.key == "linear" and result.only_redundant_rivals
    decision = evaluate_stopping(result, 12, StoppingConfig())
    assert decision.delta_bic.passed and decision.delta_bic.value is None
    assert "no distinct rival" in decision.delta_bic.detail


# --- CLI, console and report -------------------------------------------------------------------

def test_cli_faults_and_nonideal_flags(tmp_path):
    assert main.parse_faults("none") == [] and main.parse_faults("default") == DEFAULT_FAULTS
    custom = main.parse_faults('[{"type": "light_spike", "at_step": 6}]')
    assert custom == [FaultSpec(type="light_spike", at_step=6)]
    path = tmp_path / "faults.json"
    path.write_text('[{"type": "sensor_dropout", "at_step": 9, "duration": 2}]')
    assert main.parse_faults(str(path))[0].duration == 2
    with pytest.raises(SystemExit):
        main.parse_faults("not json")

    args = main.parse_args(["--experiment", "rc_charging", "--nonideal", "--faults", '[{"type":"light_spike","at_step":6}]'])
    cfg = main.build_config(args)
    assert cfg.simulation.nonideal.enabled and cfg.simulation.faults == custom
    rerun = main.rerun_command(args, cfg, 7)
    assert "--nonideal" in rerun and """--faults '[{"type":"light_spike","at_step":6}]'""" in rerun


def test_console_and_report_show_flagged_readings_and_faults(make_config):
    cfg = faulty_config(make_config, experiment="diode_iv", seed=3, write_report=True)
    printer = ConsoleEventPrinter(Console(file=io.StringIO(), width=200, force_terminal=False))
    record = DiscoveryEngine(cfg, on_event=printer).run()
    text = printer.console.file.getvalue()
    assert "FLAGGED (position_mismatch)" in text and "[analyst (mock)]" in text
    assert "Digital twin: ideal sensor and actuator" in text and "stuck_actuator at cycle 8" in text
    markdown = Path(record.artifacts.report_md).read_text(encoding="utf-8")
    assert "**Flagged readings** (never used in the fits)" in markdown
    assert "| stuck_actuator | 8 | recovered | yes | yes | no | position_mismatch | remeasure |" in markdown
    assert "Digital twin: ideal sensor and actuator (non-idealities off); injected faults:" in markdown


def test_the_outlier_bands_can_be_chosen():
    """Default: the best fit's band with a Student-t threshold; 'gp' or 'both' add the GP's normal band."""
    from scipy import stats
    from agents.tools import LabTools
    from optimization.bayesian_opt import ActiveLearningEngine

    spec = get_experiment("beer_lambert_law")
    x = np.linspace(0.0, 100.0, 8)
    y = [spec.true_response(v) + d for v, d in zip(x, [3, -5, 2, 4, -1, -6, 5, 1])]
    fit = fit_library(x, y).best

    def bands(mode):
        tools = LabTools(guard_for("beer_lambert_law"), ActiveLearningEngine(0.0, 100.0, random_state=0),
                         AnomalyConfig(outlier_bands=mode, outlier_p=0.001))
        tools.history_x, tools.history_y = list(x), list(y)
        tools.reference_fit = (fit, fit.residual_std)
        return tools.expected_bands(37.0)

    [model] = bands("model")
    assert model.source == "model" and model.threshold == pytest.approx(stats.t.isf(0.0005, 8 - 3))
    assert model.sigma > fit.residual_std  # the curve's own uncertainty is added to the noise
    [gp] = bands("gp")
    assert gp.source == "gp" and gp.threshold == pytest.approx(stats.norm.isf(0.0005))
    assert [b.source for b in bands("both")] == ["gp", "model"]
