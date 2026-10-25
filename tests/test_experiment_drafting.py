"""
Experiments drafted from a name: a model (or the offline mock table) drafts the variables, units, range and law;
the draft passes the same gates as a typed law plus a check that the variable names do not hint at the law; a
failed draft is re-asked with the problem; the run is blind and records where the law came from; and the Live
Lab runs nothing until the operator has confirmed the draft. Offline: no real model is called.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from agents.orchestrator import DiscoveryEngine
from app.lib.drafts import BY_LAW, BY_NAME
from app.lib.live import make_config
from config import LLMConfig
from experiments.custom import CustomExperimentError, agreement_verdict, law_agreement
from experiments.drafting import (MOCK_TABLE, DraftingUnavailable, ExperimentDraft, check_draft, draft_experiment,
                                  hinting_names, mock_draft, with_provenance)
from experiments.registry import CUSTOM_NAME
from llm.client import LLMClient
from records.run_record import load_run_record

ROOT = Path(__file__).resolve().parent.parent
GOOD = {"feasible": True, "reason": "ok", "title": "Pendulum", "apparatus": "A bob on a string.",
        "x_name": "pendulum length", "x_units": "m", "x_min": 0.1, "x_max": 2.0, "y_name": "period", "y_units": "s",
        "law": "2*pi*sqrt(x/9.81)", "constants": "g = 9.81 m/s²", "noise_pct": 1.0}


def mock_client(tmp_path):
    return LLMClient(LLMConfig(provider="mock", cache=False, cache_dir=tmp_path))


def scripted(answers):
    """A policy that returns the given answers in turn and remembers what it was shown."""
    seen = []

    def policy(payload):
        seen.append(payload)
        return answers[min(len(seen), len(answers)) - 1]
    policy.seen = seen
    return policy


# --- the offline table and the gates ---------------------------------------------------------------------

@pytest.mark.parametrize("row", MOCK_TABLE, ids=lambda r: r["keys"][0])
def test_every_offline_draft_passes_the_gates_and_hides_the_law(row, tmp_path):
    result = draft_experiment(f"the {row['keys'][0]} experiment", mock_client(tmp_path))
    assert result.ok and result.provenance.reasks == 0 and result.provenance.provider == "mock"
    spec = with_provenance(check_draft(result.draft), result.provenance)
    assert spec.name == CUSTOM_NAME and spec.law == row["law"] and spec.draft["drafted_from"].startswith("the ")
    view = json.dumps(spec.agent_view().model_dump())
    for secret in (row["law"], row["title"], row["apparatus"], row["constants"], "drafted"):
        assert secret.lower() not in view.lower()
    assert not hinting_names(result.draft)


def test_an_unknown_name_offline_and_a_bad_name_are_refused_with_a_reason(tmp_path):
    result = draft_experiment("photosynthesis rate against light", mock_client(tmp_path))
    assert not result.ok and "offline mock only knows" in result.error and "real LLM" in result.error
    assert "at least 3" in draft_experiment("ab", mock_client(tmp_path)).error
    assert "longer than" in draft_experiment("x" * 200, mock_client(tmp_path)).error


def test_there_is_no_drafting_without_a_model(tmp_path):
    with pytest.raises(DraftingUnavailable):
        draft_experiment("simple pendulum", LLMClient(LLMConfig(provider="none", cache=False, cache_dir=tmp_path)))


@pytest.mark.parametrize("bad, problem", [
    ({"law": "__import__('os').system('ls')"}, "law rejected"),
    ({"law": "g*x"}, "law rejected"),                                     # a named constant
    ({"x_min": 2.0, "x_max": 1.0}, "minimum < maximum"),
    ({"law": "1/x", "x_min": 0.0}, "not a finite number"),
    ({"law": "5 + 0*x"}, "does not change"),
    ({"y_name": "decay by Newton's law"}, "must not hint"),
    ({"x_name": "linear position"}, "must not hint"),
    ({"feasible": False, "reason": "not an experiment"}, "not an experiment"),
])
def test_drafts_go_through_the_same_gates_as_a_typed_law(bad, problem):
    with pytest.raises(CustomExperimentError, match=problem):
        check_draft(ExperimentDraft(**{**GOOD, **bad}))


def test_quantity_names_like_power_or_time_are_not_mistaken_for_hints():
    assert not hinting_names(ExperimentDraft(**{**GOOD, "x_name": "time", "y_name": "radiated power"}))
    assert not hinting_names(ExperimentDraft(**{**GOOD, "x_name": "thermal expansion", "y_name": "single reading"}))


def test_a_draft_that_fails_the_gates_is_re_asked_with_the_problem(tmp_path):
    policy = scripted([{**GOOD, "law": "open('f')"}, GOOD])
    client = mock_client(tmp_path)
    result = draft_experiment("simple pendulum", client, mock=policy)
    assert result.ok and result.provenance.reasks == 1 and len(result.provenance.calls) == 2
    assert "previous_problem" not in policy.seen[0]
    assert "law rejected" in policy.seen[1]["previous_problem"] and policy.seen[1]["previous_draft"]["law"] == "open('f')"
    assert all(c.agent == "drafter" and c.purpose == "draft_experiment" for c in result.provenance.calls)


def test_drafting_gives_up_after_the_re_asks(tmp_path):
    policy = scripted([{**GOOD, "law": "x.real"}])
    result = draft_experiment("simple pendulum", mock_client(tmp_path), mock=policy)
    assert not result.ok and len(policy.seen) == 3 and "failed the checks 3 times" in result.error


def test_a_real_model_draft_is_parsed_validated_and_recorded(tmp_path, monkeypatch):
    import litellm
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    requests = []

    def completion(**kwargs):
        requests.append(kwargs["messages"])
        return litellm.ModelResponse(
            model="gemini/test", usage={"prompt_tokens": 300, "completion_tokens": 120, "total_tokens": 420},
            choices=[{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": "```json\n" + json.dumps(GOOD) + "\n```"}}])

    client = LLMClient(LLMConfig(provider="litellm", model="gemini/test", fallbacks=[], cache=False,
                                 cache_dir=tmp_path), completion=completion)
    result = draft_experiment("simple pendulum: period against length", client)
    assert result.ok and result.provenance.provider == "litellm" and result.provenance.model == "gemini/test"
    assert json.loads(requests[0][1]["content"]) == {"experiment_name": "simple pendulum: period against length"}
    assert "Substitute every constant" in requests[0][0]["content"]
    assert result.provenance.calls[0].input_tokens == 300


# --- a full run --------------------------------------------------------------------------------------------

def test_a_drafted_experiment_runs_blind_and_records_its_origin(tmp_path):
    result = draft_experiment("simple pendulum", mock_client(tmp_path))
    prov = result.provenance.model_copy(update={"confirmed_by_operator": True})
    spec = with_provenance(check_draft(result.draft), prov)
    cfg = make_config(CUSTOM_NAME, 3, "mock", runs_dir=tmp_path / "runs", output_dir=tmp_path / "out")
    record = DiscoveryEngine(cfg, spec=spec).run()
    assert record.error is None and record.stop.reason.value == "converged"
    gap = law_agreement(record.experiment.law, record.result.final_analysis.best, spec.safe_bounds,
                        record.experiment.noise_std)
    assert gap < 3.0 and "was recovered" in agreement_verdict(gap)
    saved = load_run_record(tmp_path / "runs" / f"{record.run_id}.json")
    d = saved.experiment.draft
    assert d["drafted_from"] == "simple pendulum" and d["confirmed_by_operator"] and d["drafted_law"] == spec.law
    prompts = json.dumps([c.model_dump(mode="json") for c in saved.llm_calls])
    assert "pendulum" not in prompts.lower().replace("pendulum length", "") and "9.81" not in prompts


# --- the page ----------------------------------------------------------------------------------------------

def _page(tmp_path, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "APP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AUTOLAB_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("AUTOLAB_OUTPUT_DIR", str(tmp_path / "out"))
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "1_Live_Lab.py"), default_timeout=180)
    at.run()
    at.sidebar.selectbox[0].set_value(CUSTOM_NAME)
    at.run()
    return at


def test_the_live_lab_runs_nothing_until_the_draft_is_confirmed(tmp_path, monkeypatch):
    at = _page(tmp_path, monkeypatch)
    assert at.sidebar.radio[0].value == BY_NAME
    assert at.button[0].disabled                                   # Run: nothing drafted yet
    next(t for t in at.sidebar.text_input if t.label == "Experiment name").set_value("simple pendulum")
    at.run()
    next(b for b in at.sidebar.button if b.label == "Draft the experiment").click()
    at.run()
    assert not at.exception and not at.sidebar.error
    law = next(t for t in at.sidebar.text_input if t.label.startswith("Law"))
    assert law.value == "2*pi*sqrt(x/9.81)"
    assert any("Confirm the draft" in i.value for i in at.sidebar.info) and at.button[0].disabled
    next(c for c in at.sidebar.checkbox if c.label.startswith("I have checked")).check()
    next(sl for sl in at.sidebar.slider if sl.label == "Measurement budget").set_value(12)
    at.run()
    assert not at.button[0].disabled
    at.button[0].click()
    at.run(timeout=300)
    assert not at.exception
    assert any("Drafted experiment" in m.value and "simple pendulum" in m.value for m in at.markdown)
    assert any("Your law" in m.value for m in at.markdown)


def test_the_live_lab_explains_an_undraftable_name(tmp_path, monkeypatch):
    at = _page(tmp_path, monkeypatch)
    next(t for t in at.sidebar.text_input if t.label == "Experiment name").set_value("photosynthesis rate")
    at.run()
    next(b for b in at.sidebar.button if b.label == "Draft the experiment").click()
    at.run()
    assert not at.exception and any("Could not draft it" in e.value for e in at.sidebar.error)
    assert at.button[0].disabled
    at.sidebar.radio[0].set_value(BY_LAW)
    at.run()
    assert any(t.label.startswith("Law") for t in at.sidebar.text_input) and not at.button[0].disabled


# --- what the audience sees while it works ------------------------------------------------------------------

def test_drafting_reports_each_step(tmp_path):
    steps = []
    result = draft_experiment("simple pendulum", mock_client(tmp_path),
                              mock=scripted([{**GOOD, "law": "open('f')"}, GOOD]), on_progress=steps.append)
    assert result.ok
    text = "\n".join(steps)
    assert steps[0].startswith("Asking the offline mock table") and "again (attempt 2)" in text
    assert "Rejected: law rejected" in text and "Sending the problem back" in text
    assert "y = 2*pi*sqrt(x/9.81)" in text and steps[-1].startswith("All checks passed")


def test_demo_pace_slows_the_display_but_not_the_result(tmp_path):
    import time
    from app.lib.live import LiveRun
    spec = with_provenance(check_draft(ExperimentDraft(**GOOD)), draft_experiment(
        "simple pendulum", mock_client(tmp_path)).provenance)
    records = []
    for pace in (0.0, 0.05):
        cfg = make_config(CUSTOM_NAME, 5, "mock", runs_dir=tmp_path / "runs", output_dir=tmp_path / "out")
        t0 = time.perf_counter()
        live = LiveRun(cfg, spec=spec, pace_s=pace).start()
        live.join()
        records.append((live.snapshot().record, time.perf_counter() - t0))
    (fast, t_fast), (paced, t_paced) = records
    assert [(s.position, s.value) for s in fast.steps] == [(s.position, s.value) for s in paced.steps]
    assert fast.result.final_analysis.best.key == paced.result.final_analysis.best.key
    assert t_paced >= t_fast + 0.05 * (len(paced.steps) - 1)
