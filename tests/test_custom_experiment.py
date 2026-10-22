"""
Custom experiments: the operator types a law y = f(x). The law passes the same whitelist gate as
the Analyst's proposals (never executed), the agents never see it, a full run works on it, and
the run is judged by how closely the best model follows the law.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from agents.orchestrator import DiscoveryEngine
from analysis.candidate_models import FormulaRejected, compile_law
from app.lib.live import LiveRun, make_config
from config import SystemConfig
from experiments.custom import (EXAMPLES, CustomExperimentError, CustomLaw, agreement_verdict, build_custom_experiment,
                                law_agreement)
from experiments.registry import CUSTOM_NAME, get_experiment
from records.run_record import load_run_record

ROOT = Path(__file__).resolve().parent.parent


def _run(spec, tmp_path, seed=0, provider="mock", budget=30):
    cfg = make_config(CUSTOM_NAME, seed, provider, budget=budget, runs_dir=tmp_path / "runs", output_dir=tmp_path / "out")
    return DiscoveryEngine(cfg, spec=spec).run()


# --- the law gate ---------------------------------------------------------------------------------

def test_a_typed_law_compiles_and_evaluates():
    text, f = compile_law("y = 0.05*x^2 - 2*x + 40")
    assert text == "0.05*x**2 - 2*x + 40"
    assert np.allclose(f(np.array([0.0, 10.0, 20.0])), [40.0, 25.0, 20.0])
    _, g = compile_law("300*sin(x/15) + 400 + pi - e")
    assert np.isclose(g(np.array([0.0]))[0], 400 + np.pi - np.e)


@pytest.mark.parametrize("law", [
    "__import__('os').system('ls')", "x.real", "open('f')", "np.load('x')", "(lambda: 1)()", "x if x else 1",
    "9**9**9", "A*x + B", "1 + 2", "", "x" + " + 1" * 80, "sin(x, 2)", "exec('1')", "[x]",
])
def test_unsafe_or_malformed_laws_are_rejected(law):
    with pytest.raises(FormulaRejected):
        compile_law(law)


# --- building the experiment ------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, reason", [
    (dict(expression="1/x", x_min=0, x_max=10), "not a finite number"),
    (dict(expression="log(x)", x_min=0, x_max=10), "not a finite number"),
    (dict(expression="x - x + 3"), "does not change"),
    (dict(expression="2*x", x_min=5, x_max=5), "minimum < maximum"),
    (dict(expression="2*x", x_min=0, x_max=1e9), "within"),
    (dict(expression="import os"), "law rejected"),
])
def test_a_law_that_cannot_make_an_experiment_is_refused_with_a_reason(kwargs, reason):
    with pytest.raises(CustomExperimentError, match=reason):
        build_custom_experiment(CustomLaw(**kwargs))


def test_the_spec_is_derived_from_the_law_and_the_agents_never_see_it():
    spec = build_custom_experiment(CustomLaw(expression="800*exp(-x/25) + 30", x_name="delay", x_units="ms",
                                             y_name="signal", y_units="mV", noise_pct=2.0))
    span = 800 * (1 - np.exp(-4))
    assert spec.name == CUSTOM_NAME and spec.law == "800*exp(-x/25) + 30"
    assert spec.noise.std == pytest.approx(0.02 * span, rel=1e-3)
    assert spec.sensor_range[0] < 30 and spec.sensor_range[1] > 830          # no saturation inside the range
    assert spec.safe_bounds == (0.0, 100.0) and spec.max_step_delta == 100.0
    assert spec.true_response(25.0) == pytest.approx(800 * np.exp(-1) + 30)
    view = json.dumps(spec.agent_view().model_dump())
    assert "exp" not in view and "800" not in view and "delay" in view and "mV" in view


def test_custom_needs_its_spec_and_the_config_accepts_the_name():
    assert SystemConfig(experiment=CUSTOM_NAME).experiment == CUSTOM_NAME
    with pytest.raises(KeyError, match="custom experiment"):
        get_experiment(CUSTOM_NAME)
    with pytest.raises(KeyError):
        DiscoveryEngine(SystemConfig(experiment=CUSTOM_NAME))


# --- runs ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("example, model", [
    ("Quadratic", "quadratic"), ("Exponential decay", "exp_decay"), ("Straight line", "linear"),
])
def test_a_law_in_the_library_is_recovered(example, model, tmp_path):
    law, lo, hi = EXAMPLES[example]
    spec = build_custom_experiment(CustomLaw(expression=law, x_min=lo, x_max=hi))
    record = _run(spec, tmp_path)
    best = record.result.final_analysis.best
    assert best.key == model and record.stop.reason.value == "converged"
    gap = law_agreement(record.experiment.law, best, spec.safe_bounds, record.experiment.noise_std)
    assert gap < 3.0 and "recovered" in agreement_verdict(gap) and "not recovered" not in agreement_verdict(gap)
    saved = load_run_record(tmp_path / "runs" / f"{record.run_id}.json")
    assert saved.experiment.law == law and saved.experiment.name == CUSTOM_NAME


def test_a_law_outside_the_library_is_reported_as_not_recovered(tmp_path):
    law, lo, hi = EXAMPLES["Sine (not in the library)"]
    spec = build_custom_experiment(CustomLaw(expression=law, x_min=lo, x_max=hi))
    record = _run(spec, tmp_path, budget=16)
    assert record.error is None and record.result.n_measurements <= 16
    gap = law_agreement(record.experiment.law, record.result.final_analysis.best, spec.safe_bounds,
                        record.experiment.noise_std)
    assert gap > 3.0 and "not recovered" in agreement_verdict(gap)


def test_a_custom_law_runs_in_the_dashboards_live_run(tmp_path):
    spec = build_custom_experiment(CustomLaw(expression="4.2*x + 15"))
    live = LiveRun(make_config(CUSTOM_NAME, 1, "mock", runs_dir=tmp_path / "runs", output_dir=tmp_path / "out"),
                   spec=spec).start()
    live.join()
    s = live.snapshot()
    assert s.error is None and s.experiment == CUSTOM_NAME and s.record.experiment.law == "4.2*x + 15"


# --- the page ------------------------------------------------------------------------------------------------

def test_the_live_lab_runs_a_custom_law(tmp_path, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "APP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AUTOLAB_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("AUTOLAB_OUTPUT_DIR", str(tmp_path / "out"))
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "1_Live_Lab.py"), default_timeout=180)
    at.run()
    at.sidebar.selectbox[0].set_value(CUSTOM_NAME)
    at.run()
    law_box = next(t for t in at.sidebar.text_input if t.label.startswith("Law"))
    law_box.set_value("open('x')")
    at.run()
    assert not at.exception and any("Cannot run this law" in e.value for e in at.sidebar.error)
    law_box = next(t for t in at.sidebar.text_input if t.label.startswith("Law"))
    law_box.set_value("4.2*x + 15")
    at.run()
    assert not at.sidebar.error
    next(sl for sl in at.sidebar.slider if sl.label == "Measurement budget").set_value(8)
    at.button[0].click()
    at.run(timeout=300)
    assert not at.exception
    assert at.subheader[0].value.startswith("Result: Linear")
    assert any("Your law" in m.value for m in at.markdown)
    assert any("was recovered" in s.value for s in at.success)
