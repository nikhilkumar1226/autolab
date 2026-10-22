"""
Tests for the LLM agents in the closed loop: Planner (plan, strategy, re-proposal, stop
rationale), Analyst (review, candidate models), Writer (number check), blind mode, provider
none, failure handling, and the CLI flags.

Offline only. Runs with provider litellm use the scripted fake litellm: every agent answers
like its mock policy unless a test overrides it, so the requests go through the real client
(JSON-schema mode, validation, tokens, cost, cache).
"""
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

import agents.orchestrator as orchestrator
import main
from agents.analyst import mock_review
from agents.lead_scientist import LeadScientistAgent, clip_range, mock_plan, mock_stop, mock_strategy
from agents.models import AnalystReport, PlannerOutput, StrategyDecision
from agents.orchestrator import DiscoveryEngine
from agents.prompts import ANALYST_SYSTEM, PLANNER_SYSTEM, PROMPT_VERSION, STOP_SYSTEM, STRATEGY_SYSTEM, WRITER_SYSTEM
from agents.writer import find_unverified_numbers, mock_writer
from analysis.stopping import StopReason
from config import AgentConfig, LLMConfig, OutputConfig
from experiments.registry import BLIND_DESCRIPTION, ExperimentSpec, NoiseSettings, VariableSpec, get_experiment, list_experiments
from hardware.safety_guard import HardwareSafetyGuard
from hardware.simulator import PhysicsSimulationRig
from llm.client import LLMClient, LLMUnavailableError, _litellm
from records.run_record import RunRecord, list_run_records, load_run_record

MOCKS = {PLANNER_SYSTEM: mock_plan, STRATEGY_SYSTEM: mock_strategy, ANALYST_SYSTEM: mock_review,
         STOP_SYSTEM: mock_stop, WRITER_SYSTEM: mock_writer}


class AgentScript:
    """Answers fake litellm requests with the mock policies, except for agents a test overrides."""

    def __init__(self):
        self.overrides = {}
        self.requests = []

    def on(self, system: str, handler) -> "AgentScript":
        self.overrides[system] = handler
        return self

    def __call__(self, kwargs):
        system = kwargs["messages"][0]["content"]
        payload = json.loads(kwargs["messages"][1]["content"])
        self.requests.append((system, payload))
        return self.overrides.get(system, MOCKS[system])(payload)

    def payloads(self, system: str):
        return [p for s, p in self.requests if s == system]


@pytest.fixture
def scripted(fake_litellm):
    script = AgentScript()
    fake_litellm.responder = script
    return script


def error(name: str, message: str = "boom"):
    return getattr(_litellm(), name)(message, llm_provider="anthropic", model="m")


def plan_with(**changes):
    def plan(payload):
        out = mock_plan(payload)
        for key, value in changes.items():
            if key == "proposed_range":
                out["hypothesis"]["proposed_range"] = value
            else:
                out[key] = value
        return out
    return plan


# --- schemas -------------------------------------------------------------------

def test_llm_output_schemas_validate_and_reject():
    with pytest.raises(ValidationError, match="position is required"):
        StrategyDecision(choice="self", reason="discriminate")
    assert StrategyDecision(choice="gp", position=5.0, reason="r").position is None
    with pytest.raises(ValidationError):
        StrategyDecision(choice="self", position=math.inf, reason="r")
    with pytest.raises(ValidationError):
        StrategyDecision(choice="maybe", reason="r")

    hypothesis = {"statement": "s", "candidate_relationships": ["a"], "independent_variable": "x", "proposed_range": [0, 1]}
    for design in ([1.0, 2.0], [1.0] * 6, [1.0, math.nan, 2.0]):
        with pytest.raises(ValidationError):
            PlannerOutput(hypothesis=hypothesis, initial_design=design, rationale="r")
    # Out-of-range positions are well-formed proposals: the safety guard, not the schema, rejects them.
    assert PlannerOutput(hypothesis=hypothesis, initial_design=[0, 500, -3], rationale="r").initial_design == [0, 500, -3]
    with pytest.raises(ValidationError):
        AnalystReport(interpretation="", residuals_suggest_missing_model=False)


# --- Planner -------------------------------------------------------------------

def test_mock_planner_reproduces_the_heuristic_design(tmp_path):
    for name in list_experiments():
        view = get_experiment(name).agent_view()
        hypothesis, steps = LeadScientistAgent(LLMClient(LLMConfig(provider="mock", cache_dir=tmp_path))).plan("q", view, 4)
        heuristic = LeadScientistAgent().generate_initial_plan(view, 4)
        assert [s.target_position for s in steps] == [s.target_position for s in heuristic]
        assert hypothesis.source == "mock" and hypothesis.suggested_range == list(view.safe_bounds)
        assert not hypothesis.range_clipped

        none_hypothesis, _ = LeadScientistAgent(LLMClient(LLMConfig(provider="none", cache_dir=tmp_path))).plan("q", view, 4)
        assert none_hypothesis.source == "heuristic"


def test_clip_range():
    assert clip_range([20, 60], (0, 180)) == ([20, 60], False, "")
    assert clip_range([60, 20], (0, 180))[0] == [20, 60]
    applied, clipped, note = clip_range([-50, 500], (0, 180))
    assert applied == [0, 180] and clipped and "clipped to [0, 180]" in note
    applied, clipped, note = clip_range([200, 300], (0, 180))
    assert applied == [0, 180] and clipped and "does not overlap" in note


def test_an_out_of_bounds_range_is_clipped_and_never_reaches_the_guard(make_config, litellm_config, scripted, monkeypatch):
    guards = []

    class SpyGuard(HardwareSafetyGuard):
        def __init__(self, rig, limits):
            super().__init__(rig, limits)
            guards.append(self)

    monkeypatch.setattr(orchestrator, "HardwareSafetyGuard", SpyGuard)
    scripted.on(PLANNER_SYSTEM, plan_with(proposed_range=[-50.0, 500.0]))
    record = DiscoveryEngine(make_config(llm=litellm_config())).run()

    h = record.hypothesis
    assert h.proposed_range == [-50.0, 500.0] and h.suggested_range == [0.0, 180.0]
    assert h.range_clipped and "clipped to [0, 180]" in h.range_note
    assert h.source == "llm" and h.model == "anthropic/claude-haiku-5-5"
    [guard] = guards
    assert (guard.limits.min_position, guard.limits.max_position) == get_experiment("malus_law").safe_bounds
    assert record.experiment.agent_view.safe_bounds == (0.0, 180.0)
    assert all(0.0 <= x <= 180.0 for x in record.measured_points()[0])
    assert record.stop.reason == StopReason.CONVERGED


def test_a_safety_rejection_is_fed_back_and_the_planner_re_proposes(make_config, litellm_config, scripted):
    def strategy(payload):
        if "observation" in payload:
            return {"choice": "self", "position": 170.0, "reason": "the guard allows at most 180 deg"}
        return mock_strategy(payload)

    scripted.on(PLANNER_SYSTEM, plan_with(initial_design=[0.0, 500.0, 90.0, 180.0])).on(STRATEGY_SYSTEM, strategy)
    events = []
    record = DiscoveryEngine(make_config(llm=litellm_config()), on_event=events.append).run()

    [rejection] = [e for e in events if e.type == "safety_rejection"]
    assert rejection.requested_position == 500.0 and rejection.fed_back_to == "planner"
    assert "exceeds physical limits" in rejection.reason
    [observation] = [p["observation"] for p in scripted.payloads(STRATEGY_SYSTEM) if "observation" in p]
    assert observation == {"rejected_position": 500.0, "guard_message": rejection.reason, "attempt": 1}

    rejected, retry = record.steps[1], record.steps[2]
    assert (rejected.status, rejected.source, rejected.attempt_in_slot) == ("rejected", "initial_design", 1)
    assert (retry.status, retry.source, retry.position, retry.attempt_in_slot) == ("measured", "planner", 170.0, 2)
    assert retry.rationale == "the guard allows at most 180 deg"
    assert record.llm_calls[retry.llm_call_ids[0]].purpose == "repropose"
    assert 500.0 not in record.measured_points()[0]
    later = [p for p in scripted.payloads(STRATEGY_SYSTEM) if "observation" not in p]
    assert later[0]["recent_rejections"][0]["rejected_position"] == 500.0


def test_after_two_rejected_reproposals_the_gp_takes_over(make_config, litellm_config, scripted):
    def strategy(payload):
        if "observation" in payload:
            return {"choice": "self", "position": 999.0, "reason": "try further out"}
        return mock_strategy(payload)

    scripted.on(PLANNER_SYSTEM, plan_with(initial_design=[0.0, 500.0, 90.0, 180.0])).on(STRATEGY_SYSTEM, strategy)
    events = []
    record = DiscoveryEngine(make_config(llm=litellm_config()), on_event=events.append).run()

    slot = record.steps[1:5]
    assert [(s.source, s.requested_position, s.status, s.attempt_in_slot) for s in slot[:3]] == [
        ("initial_design", 500.0, "rejected", 1), ("planner", 999.0, "rejected", 2), ("planner", 999.0, "rejected", 3),
    ]
    fallback = slot[3]
    assert (fallback.source, fallback.status, fallback.attempt_in_slot) == ("gp", "measured", 4)
    assert fallback.rationale == "GP fallback after 2 rejected re-proposal(s)"
    assert [e.fed_back_to for e in events if e.type == "safety_rejection"] == ["planner", "planner", "gp"]
    assert all(0.0 <= x <= 180.0 for x in record.measured_points()[0])


def test_the_planner_can_choose_its_own_point_with_a_reason(make_config, litellm_config, scripted):
    def strategy(payload):
        d = payload["models_disagree_most_at"]
        if d is None:
            return mock_strategy(payload)
        return {"choice": "self", "position": d["x"], "reason": f"discriminate: the top models differ by {d['difference_sigma']} sigma"}

    scripted.on(STRATEGY_SYSTEM, strategy)
    events = []
    record = DiscoveryEngine(make_config(experiment="inverse_square_law", seed=4, llm=litellm_config()),
                             on_event=events.append).run()

    own = [s for s in record.steps if s.source == "planner"]
    assert own and all(s.status == "measured" and s.rationale.startswith("discriminate") for s in own)
    assert all(s.gp is not None for s in own)  # what the GP would have chosen is still recorded
    choices = [e for e in events if e.type == "strategy"]
    assert all(e.choice == "self" and e.reason.startswith("discriminate") for e in choices)
    assert set(scripted.payloads(STRATEGY_SYSTEM)[0]) == {
        "variables", "safe_bounds", "data", "budget_left", "fits", "stopping_criteria_unmet", "gp_suggestion",
        "models_disagree_most_at", "recent_rejections",
    }


# --- Analyst ---------------------------------------------------------------------

def candidate(expression, *params, name="proposal"):
    return {"name": name, "expression": expression, "rationale": "residual pattern",
            "parameters": [{"name": p, "initial_guess": g} for p, g in params]}


def test_a_dangerous_candidate_model_is_rejected_and_the_run_continues(make_config, litellm_config, scripted, tmp_path):
    def review(payload):
        out = mock_review(payload)
        out["candidate_model"] = candidate("__import__('os').system('touch hacked')", ("A", 1.0), name="pwn")
        return out

    scripted.on(ANALYST_SYSTEM, review)
    record = DiscoveryEngine(make_config(llm=litellm_config())).run()

    assert record.candidate_models and all(c.status == "rejected" for c in record.candidate_models)
    assert "only sin, cos, exp, log and sqrt may be called" in record.candidate_models[0].reason
    assert record.stop.reason == StopReason.CONVERGED
    assert record.result.final_analysis.best.key == "cos2_phase"
    assert not Path("hacked").exists() and not (tmp_path / "hacked").exists()
    later = scripted.payloads(ANALYST_SYSTEM)[1]
    assert later["previous_proposals"][0]["status"] == "rejected"


def test_a_duplicate_of_a_library_model_is_not_added(make_config, litellm_config, scripted):
    def review(payload):
        out = mock_review(payload)
        if not payload["previous_proposals"]:
            out["candidate_model"] = candidate("I0*cos(pi*(x - p)/180)**2 + B", ("I0", 900.0), ("p", 10.0), ("B", 0.0))
        return out

    scripted.on(ANALYST_SYSTEM, review)
    record = DiscoveryEngine(make_config(llm=litellm_config())).run()
    [proposal] = record.candidate_models
    assert proposal.status == "equivalent" and "indistinguishable from Cosine-squared with phase" in proposal.reason
    assert record.stop.reason == StopReason.CONVERGED  # the duplicate cannot block the delta-BIC criterion
    assert all(f.origin == "library" for f in record.result.final_analysis.fits)


def _gaussian(x, A, x0, w, C):
    return A * math.exp(-((x - x0) ** 2) / (2 * w * w)) + C


PEAK = ExperimentSpec(
    name="test_peak",
    description_for_agents="A test apparatus with one actuator and one sensor.",
    independent=VariableSpec(name="detuning", symbol="d", units="MHz"),
    dependent=VariableSpec(name="signal", symbol="s", units="mV"),
    safe_bounds=(0.0, 100.0),
    max_step_delta=100.0,
    noise=NoiseSettings(std=5.0),
    ground_truth=_gaussian,
    ground_truth_model="not_in_library",
    true_params={"A": 800.0, "x0": 40.0, "w": 12.0, "C": 20.0},
)


def test_a_missing_law_proposed_by_the_analyst_is_fitted_and_wins(make_config, litellm_config, scripted):
    def review(payload):
        out = mock_review(payload)
        accepted = any(p["status"] == "accepted" for p in payload["previous_proposals"])
        if len(payload["data"]["x"]) >= 6 and not accepted:
            out["candidate_model"] = candidate(
                "A*exp(-(x - x0)**2/(2*w**2)) + C", ("A", 700.0), ("x0", 45.0), ("w", 10.0), ("C", 0.0),
                name="Gaussian peak",
            )
        return out

    scripted.on(ANALYST_SYSTEM, review)
    record = DiscoveryEngine(make_config(llm=litellm_config(), seed=1, write_report=True), spec=PEAK).run()

    accepted = [c for c in record.candidate_models if c.status == "accepted"]
    assert len(accepted) == 1 and accepted[0].key == "llm_1"
    best = record.result.final_analysis.best
    assert best.key == "llm_1" and best.origin == "llm" and best.name == "Gaussian peak (LLM-proposed)"
    assert best.params["x0"] == pytest.approx(40.0, abs=2.0)
    assert "x_{0} = " in record.result.equation_latex
    markdown = Path(record.artifacts.report_md).read_text(encoding="utf-8")
    assert "Gaussian peak (LLM-proposed)" in markdown and "| accepted |" in markdown
    assert Path(record.artifacts.plot).exists()
    assert load_run_record(record.artifacts.record).result.final_analysis.best == best


# --- Writer ----------------------------------------------------------------------

def test_find_unverified_numbers():
    facts = {"r2": 0.99876, "n": 12, "message": "delta-BIC 7.8", "rel_se": 0.0216, "bic": 34.83}
    assert find_unverified_numbers("R2 0.9988 with n = 12, delta-BIC 7.8, 2.16%, BIC 34.8 and 3 runs", facts) == []
    assert find_unverified_numbers("a bias of 42.7 lux, as in 1995", facts) == ["42.7", "1995"]


def test_writer_numbers_are_checked_against_the_record(make_config, litellm_config, scripted):
    def writer(payload):
        if "numbers_not_in_facts" in payload:
            return {"discussion": "Revised: an offset of 13.37 lux remains.", "conclusion": "Cos-squared fits."}
        return {"discussion": f"R2 is {payload['best_model']['r2']} with an invented 42.7 lux bias.", "conclusion": "Done."}

    scripted.on(WRITER_SYSTEM, writer)
    record = DiscoveryEngine(make_config(llm=litellm_config(), write_report=True)).run()

    first, second = scripted.payloads(WRITER_SYSTEM)
    assert second["numbers_not_in_facts"] == ["42.7"] and "numbers_not_in_facts" not in first
    d = record.discussion
    assert d.source == "llm" and d.unverified_numbers == ["13.37"] and len(d.llm_call_ids) == 2
    markdown = Path(record.artifacts.report_md).read_text(encoding="utf-8")
    assert "**Numbers not found in the run record:** 13.37" in markdown
    assert "Written by the Writer agent (anthropic/claude-haiku-5-5)" in markdown


# --- providers end to end -----------------------------------------------------------

@pytest.mark.parametrize("name", list_experiments())
def test_provider_none_takes_the_same_measurements_as_mock(make_config, tmp_path, name):
    none = DiscoveryEngine(make_config(experiment=name, seed=2, llm=LLMConfig(provider="none", cache_dir=tmp_path))).run()
    mock = DiscoveryEngine(make_config(experiment=name, seed=2)).run()

    assert none.llm_calls == [] and none.hypothesis.source == "heuristic" and none.stop_rationale is None
    assert all(s.review is None and s.rationale == "" for s in none.steps)
    assert none.measured_points() == mock.measured_points()
    assert none.stop.model_dump() == mock.stop.model_dump()
    assert none.result.final_analysis.best.key == get_experiment(name).ground_truth_model


def test_mock_run_logs_every_agent_call(make_config):
    record = DiscoveryEngine(make_config(seed=0, write_report=True)).run()

    assert {c.purpose for c in record.llm_calls} == {"plan", "strategy", "review", "stop_rationale", "discussion"}
    assert {c.agent for c in record.llm_calls} == {"planner", "analyst", "writer"}
    for c in record.llm_calls:
        assert c.provider == "mock" and c.model == "mock" and c.ok and c.prompt_version == PROMPT_VERSION
        assert c.system and json.loads(c.user) and c.raw_response and c.parsed
    assert record.llm_usage.calls == len(record.llm_calls)
    assert record.llm_usage.calls_per_model == {"mock": len(record.llm_calls)}
    assert load_run_record(record.artifacts.record).llm_calls == record.llm_calls
    markdown = Path(record.artifacts.report_md).read_text(encoding="utf-8")
    for text in ["Agents: mock provider", "## 7. Agent reasoning", "## 9. Discussion and conclusion",
                 "Mock provider: deterministic template, not a language model."]:
        assert text in markdown


def test_litellm_run_records_tokens_cost_and_models_and_replays_from_cache(
        make_config, litellm_config, scripted, fake_litellm):
    cfg = make_config(llm=litellm_config(), seed=0)
    record = DiscoveryEngine(cfg).run()

    assert record.llm_calls and all(c.provider == "litellm" for c in record.llm_calls)
    assert all(c.model == "anthropic/claude-haiku-5-5" and not c.cached for c in record.llm_calls)
    assert all((c.input_tokens, c.output_tokens) == (100, 20) and c.cost_usd > 0 for c in record.llm_calls)
    assert record.llm_usage.cost_usd == pytest.approx(sum(c.cost_usd for c in record.llm_calls))
    mock = DiscoveryEngine(make_config(seed=0)).run()
    assert record.measured_points() == mock.measured_points()

    requests = len(fake_litellm.calls)
    replay = DiscoveryEngine(cfg).run()
    assert len(fake_litellm.calls) == requests  # every answer came from llm_cache/
    assert all(c.cached and c.cost_usd == 0.0 for c in replay.llm_calls)
    assert replay.measured_points() == record.measured_points()


def test_when_every_model_fails_the_run_stops_with_a_clear_error(make_config, litellm_config, fake_litellm):
    fake_litellm.responder = lambda kwargs: error("AuthenticationError", "invalid api key")
    cfg = make_config(llm=litellm_config())
    with pytest.raises(LLMUnavailableError, match="every model in the LLM chain failed"):
        DiscoveryEngine(cfg).run()

    [path] = list_run_records(cfg.output.runs_dir)
    record = load_run_record(path)
    assert record.stop.reason == StopReason.ERROR and record.error.startswith("LLMUnavailableError")
    for model in cfg.llm.chain:
        assert f"{model}: auth" in record.error
    [call] = record.llm_calls
    assert not call.ok and call.purpose == "plan" and all(a.model != "mock" for a in call.attempts)
    assert record.steps == []


def test_a_late_llm_failure_keeps_the_result_and_flags_it(make_config, litellm_config, scripted):
    def boom(payload):
        return error("InternalServerError", "overloaded")

    scripted.on(STOP_SYSTEM, boom).on(WRITER_SYSTEM, boom)
    events = []
    record = DiscoveryEngine(make_config(llm=litellm_config(), write_report=True), on_event=events.append).run()

    assert record.stop.reason == StopReason.CONVERGED and record.error is None
    assert [w.split(":")[0] for w in record.warnings] == ["Stop rationale not generated", "Discussion not generated"]
    assert record.stop_rationale.source == "unavailable" and record.discussion.source == "unavailable"
    assert all(c.provider == "litellm" for c in record.llm_calls)
    markdown = Path(record.artifacts.report_md).read_text(encoding="utf-8")
    assert "Discussion not generated: every model in the LLM chain failed" in markdown
    assert events[-1].type == "report_ready" and events[-1].warnings == record.warnings


def test_without_any_key_the_run_fails_before_the_rig_moves(make_config, litellm_config, fake_litellm, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    homed = []
    monkeypatch.setattr(PhysicsSimulationRig, "home", lambda self: homed.append(True) or True)

    with pytest.raises(LLMUnavailableError, match="No model in the LLM chain can be called"):
        DiscoveryEngine(make_config(llm=litellm_config())).run()
    assert fake_litellm.calls == [] and homed == []


# --- what the agents see ----------------------------------------------------------------

def test_blind_mode_hides_names_and_apparatus_from_every_agent(make_config):
    record = DiscoveryEngine(make_config(seed=0, write_report=True, agent=AgentConfig(blind=True))).run()
    spec = get_experiment("malus_law")

    assert record.question == "How does y depend on x?" and record.config.agent.blind
    for call in record.llm_calls:
        text = call.user.lower()
        for hidden in [spec.description_for_agents.lower(), "polaris", "angle", "intensity", "lamp", "filter", "malus"]:
            assert hidden not in text, (call.purpose, hidden)
    plan = json.loads(record.llm_calls[0].user)
    assert plan["apparatus"] == BLIND_DESCRIPTION
    assert plan["variables"] == {"x": {"name": "x", "units": "deg"}, "y": {"name": "y", "units": "lux"}}
    markdown = Path(record.artifacts.report_md).read_text(encoding="utf-8")
    assert "Blind mode" in markdown and "polariser angle" in markdown  # the report is for people


def test_the_analyst_never_sees_the_apparatus_question_or_hypothesis(make_config):
    question = "Investigate polarized light transmission through a rotating filter"
    record = DiscoveryEngine(make_config(seed=0)).run(question=question)
    spec = get_experiment("malus_law")

    reviews = [c for c in record.llm_calls if c.agent == "analyst"]
    assert reviews
    for call in reviews:
        payload = json.loads(call.user)
        assert set(payload) == {"variables", "data", "fits", "not_fitted", "residuals", "previous_proposals",
                                "candidate_slots_left"}
        for secret in [spec.description_for_agents, question, record.hypothesis.statement, spec.name]:
            assert secret.lower() not in call.user.lower()
        assert payload["variables"]["x"] == {"name": "polariser angle", "units": "deg"}


# --- records and CLI ----------------------------------------------------------------------

def test_a_phase_1_record_still_loads(make_config, tmp_path):
    """Remove every field added in Phase 2, where it was added, and load the result."""
    record = DiscoveryEngine(make_config(seed=0, llm=LLMConfig(provider="none", cache_dir=tmp_path))).run()
    data = json.loads(record.model_dump_json())
    for key in ["llm_calls", "llm_usage", "candidate_models", "stop_rationale", "discussion", "warnings"]:
        del data[key]
    del data["timings"]["llm_s"], data["config"]["llm"]
    for key in ["blind", "max_reproposals", "max_llm_candidates", "candidate_equivalence_sigma"]:
        del data["config"]["agent"][key]
    data["config"]["agent"] |= {"model_name": "gpt-4o", "temperature": 0.2}  # Phase 1 fields, now removed
    for key in ["candidate_relationships", "proposed_range", "range_clipped", "range_note", "design_rationale",
                "source", "model", "llm_call_id"]:
        del data["hypothesis"][key]
    analyses = [data["result"]["final_analysis"]]
    for step in data["steps"]:
        for key in ["review", "rationale", "attempt_in_slot", "llm_call_ids"]:
            del step[key]
        analyses.append(step["analysis"])
    for analysis in filter(None, analyses):
        for fit in analysis["fits"]:
            del fit["origin"], fit["expression"]
    data["schema_version"] = 1

    loaded = RunRecord.model_validate(data)
    assert loaded.measured_points() == record.measured_points()
    assert loaded.llm_calls == [] and loaded.stop == record.stop and loaded.hypothesis.source == "heuristic"


def test_cli_flags_build_the_config():
    args = main.parse_args(["--provider", "none", "--model", "gemini/gemini-3.6-flash", "--blind", "--seed", "3", "--no-cache"])
    cfg = main.build_config(args)
    assert (cfg.llm.provider, cfg.llm.model, cfg.llm.cache) == ("none", "gemini/gemini-3.6-flash", False)
    assert cfg.agent.blind and cfg.seed == 3
    rerun = main.rerun_command(args, cfg, 3)
    assert "--provider none" in rerun and "--model gemini/gemini-3.6-flash" in rerun and "--blind" in rerun

    default = main.build_config(main.parse_args([]))
    assert default.llm.provider == "mock" and default.llm.cache and not default.agent.blind
    assert default.llm.chain == ["anthropic/claude-haiku-5-5", "gemini/gemini-3.6-flash", "gemini/gemini-3.5-flash-lite"]
    with pytest.raises(SystemExit):
        main.parse_args(["--provider", "openai"])


def test_cli_reports_an_unavailable_llm_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    build = main.build_config
    output = OutputConfig(runs_dir=tmp_path / "runs", plots_dir=tmp_path / "plots", reports_dir=tmp_path / "reports")
    monkeypatch.setattr(main, "build_config", lambda args: build(args).model_copy(update={"output": output}))

    assert main.main(["--provider", "litellm", "--seed", "1", "--no-report"]) == 1
    out = capsys.readouterr().out
    assert "Run stopped: the LLM is unavailable." in out and "python -m llm.check" in out
    assert len(list_run_records(tmp_path / "runs")) == 1
