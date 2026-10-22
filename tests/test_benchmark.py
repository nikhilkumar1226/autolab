"""
Phase 5: the benchmark's metrics, the new samplers (grid, LLM-only), and a tiny end-to-end
benchmark (plan, resume, CSVs, figures, RESULTS.md). Phase 5.1: the free-tier subset (plan,
pacing, hidden apparatus, stop and resume on unavailable models, labels).
"""
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agents.lead_scientist import gap_point
from agents.orchestrator import DiscoveryEngine
from analysis.model_library import ModelFit
from benchmark import run as bench
from benchmark.figures import footer_text
from benchmark.metrics import mean_ci, param_errors, scoring_truth, wilson
from config import AgentConfig, LLMConfig, NonIdealityConfig, SimulationConfig
from experiments.registry import HIDDEN_DESCRIPTION, get_experiment
from optimization.baselines import GridSampler, RandomSampler, van_der_corput_fractions


def fit_of(key, params):
    return ModelFit(key=key, name=key, formula="", n_points=10, n_params=len(params), params=params,
                    std_errors={p: 0.1 for p in params}, rel_std_errors={p: 0.01 for p in params},
                    r2=1.0, rmse=1.0, aic=0.0, bic=0.0)


# --- metrics -----------------------------------------------------------------------------------

def test_parameter_errors_are_normalised_by_kind():
    spec = get_experiment("ldr_response")  # power law A x^b + C, true C = 0
    truth = dict(spec.true_params)
    errors = param_errors(fit_of("power_law", {"A": 510.0, "b": -0.707, "C": 0.958}), spec, truth)
    assert errors["A"]["pct"] == pytest.approx(2.0)          # scale: 10 / 500
    assert errors["b"]["pct"] == pytest.approx(1.0)          # scale: 0.007 / 0.7
    span = np.ptp([spec.true_response(x) for x in np.linspace(*spec.safe_bounds, 201)])
    assert errors["C"]["pct"] == pytest.approx(100 * 0.958 / span)  # offset: by the response span, true value 0
    malus = get_experiment("malus_law")
    loc = param_errors(fit_of("cos2_phase", {"A": 1000.0, "x0": 21.8, "C": 15.0}), malus, dict(malus.true_params))
    assert loc["x0"]["pct"] == pytest.approx(1.0)            # location: 1.8 / 180
    assert param_errors(fit_of("linear", {"a": 1.0, "b": 0.0}), malus, dict(malus.true_params)) is None


def test_scoring_uses_the_effective_parameters_of_a_leaky_polariser(make_config):
    record = DiscoveryEngine(make_config(seed=0, simulation=SimulationConfig(
        transit_delay_sec=0.0, nonideal=NonIdealityConfig(enabled=True)))).run()
    truth = scoring_truth(record, get_experiment("malus_law"))
    assert truth["A"] == pytest.approx(990.0) and truth["C"] == pytest.approx(25.0) and truth["x0"] == 20.0


def test_intervals():
    mean, std, lo, hi = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5 and std == pytest.approx(1.29099, rel=1e-4)
    assert (lo, hi) == pytest.approx((2.5 - 3.182446 * 1.29099 / 2, 2.5 + 3.182446 * 1.29099 / 2), rel=1e-4)
    assert math.isnan(mean_ci([])[0]) and math.isnan(mean_ci([5.0])[2])
    rate, lo, hi = wilson(19, 20)
    assert rate == 0.95 and lo == pytest.approx(0.7639, abs=1e-3) and hi == pytest.approx(0.9911, abs=1e-3)
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # NaN


# --- samplers ----------------------------------------------------------------------------------

def test_grid_order_is_coarse_to_fine_and_respects_the_window():
    assert van_der_corput_fractions(3) == [0.0, 1.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875]
    grid = GridSampler(0.0, 20.0)
    assert grid.suggest([0.0, 20.0, 10.0, 5.0]) == 15.0
    assert grid.suggest([0.0, 20.0, 10.0, 5.0], window=(0.0, 12.0)) == 2.5  # 15 is out of reach
    rng = RandomSampler(0.0, 20.0, np.random.default_rng(0))
    assert all(8.0 <= rng.suggest((8.0, 12.0)) <= 12.0 for _ in range(100))


def test_gap_point_prefers_a_discriminating_point_then_the_widest_gap():
    payload = {"reachable": [0.0, 100.0], "safe_bounds": [0.0, 100.0], "data": [[0, 1], [50, 2], [100, 3]],
               "models_disagree_most_at": {"x": 20.0, "difference_sigma": 4.0}}
    assert gap_point(payload)["position"] == 20.0
    payload["models_disagree_most_at"]["difference_sigma"] = 1.0
    assert gap_point(payload)["position"] in (25.0, 75.0)
    payload["reachable"] = [60.0, 100.0]
    assert gap_point(payload)["position"] == 80.0


@pytest.mark.parametrize("provider, source", [("mock", "planner"), ("none", "heuristic")])
def test_llm_only_mode_chooses_every_point_without_the_gp(make_config, tmp_path, provider, source):
    cfg = make_config(experiment="beer_lambert_law", seed=1, agent=AgentConfig(sampler="llm"),
                      llm=LLMConfig(provider=provider, cache_dir=tmp_path))
    events = []
    record = DiscoveryEngine(cfg, on_event=events.append).run()
    later = [s for s in record.steps if s.source not in ("initial_design", "remeasure")]
    assert later and all(s.source == source for s in later)
    assert not [e for e in events if e.type == "gp_update"]  # the GP is never asked
    if provider == "mock":
        assert {c.purpose for c in record.llm_calls} >= {"choose_point"}
    assert record.result.final_analysis.best.key == "exp_decay"


# --- a tiny end-to-end benchmark ---------------------------------------------------------------

def test_a_tiny_benchmark_writes_every_output_and_resumes(tmp_path, capsys):
    out, work = tmp_path / "results", tmp_path / "work"
    argv = ["--quick", "--experiments", "malus_law", "--conditions", "clean,faults", "--methods",
            "grid,gp_only,autolab", "--seeds", "1", "--llm-seeds", "1", "--workers", "1",
            "--out", str(out), "--work", str(work)]
    assert bench.main(argv) == 0
    rows = list(csv.DictReader((out / "benchmark_runs.csv").open()))
    kinds = {(r["kind"], r["method"]) for r in rows}
    assert {("adaptive", m) for m in ("grid", "gp_only", "autolab")} <= kinds
    assert {("curve", m) for m in ("grid", "gp_only", "autolab")} <= kinds
    assert {("fixed", "grid"), ("fixed", "gp_only")} <= kinds
    fixed = [r for r in rows if r["kind"] == "fixed"]
    autolab = {(r["condition"]): int(r["n_measurements"]) for r in rows if r["kind"] == "adaptive" and r["method"] == "autolab"}
    assert all(int(r["n_measurements"]) == max(5, autolab[r["condition"]]) for r in fixed)  # equal budget
    curve = next(r for r in rows if r["kind"] == "curve")
    assert int(curve["n_measurements"]) == bench.CURVE_BUDGET and len(json.loads(curve["curve"])) >= 10
    assert all(r["data_source"] == "DIGITAL TWIN" and r["status"] == "ok" for r in rows)
    # With the mock provider the Planner delegates to the GP, so AutoLab equals GP only.
    same = {r["condition"]: r["n_measurements"] for r in rows if r["kind"] == "adaptive" and r["method"] == "gp_only"}
    assert same == {r["condition"]: r["n_measurements"] for r in rows if r["kind"] == "adaptive" and r["method"] == "autolab"}
    summary = list(csv.DictReader((out / "benchmark_summary.csv").open()))
    assert any(s["experiment"] == "ALL" for s in summary) and "correct_ci_low" in summary[0]
    md = (out / "RESULTS.md").read_text()
    assert "Data source: DIGITAL TWIN" in md and "say nothing about what a real LLM adds" in md
    for name in ("fig_param_error_vs_n", "fig_law_accuracy", "fig_equal_budget_accuracy", "fig_fault_recovery",
                 "fig_cost_per_run"):
        assert (out / "figures" / f"{name}.png").exists() and (out / "figures" / f"{name}.pdf").exists()

    capsys.readouterr()
    assert bench.main(argv) == 0  # resume: nothing left to run
    assert "to do" not in capsys.readouterr().out


def test_dry_run_estimates_llm_calls_without_running(tmp_path, capsys):
    argv = ["--full", "--provider", "litellm", "--experiments", "malus_law", "--work", str(tmp_path), "--dry-run"]
    assert bench.main(argv) == 0
    out = capsys.readouterr().out
    assert "Real LLM: 40 runs to do (0 already done). Estimated" in out and not any(tmp_path.joinpath("rows").iterdir())


# --- Phase 5.1: the free-tier subset ------------------------------------------------------------

def subset_args(**kw):
    args = dict(methods=[*bench.BASELINE_METHODS, *bench.LLM_METHODS], provider="litellm", chain="abcd1234",
                llm_seeds=bench.SUBSET_SEEDS, seeds=20, experiments=bench.SUBSET_EXPERIMENTS,
                conditions=bench.SUBSET_CONDITIONS, no_curves=True, subset=True)
    return SimpleNamespace(**{**args, **kw})


def test_the_subset_plan_runs_only_the_llm_methods_on_hidden_apparatus():
    tasks = bench.plan_stage1(subset_args())
    llm = [t for t in tasks if t.method in bench.LLM_METHODS]
    base = [t for t in tasks if t.method not in bench.LLM_METHODS]
    assert len(llm) == 2 * 4 * 3 * 3 and len(base) == 3 * 4 * 2 * 20
    assert {t.condition for t in llm} == {"clean", "faults", "hidden_apparatus"}
    assert "hidden_apparatus" not in {t.condition for t in base}  # a non-LLM method never reads the description
    assert all(t.key.endswith("-cabcd1234") and t.provider == "litellm" for t in llm)
    assert all(t.provider == "none" and t.chain == "" for t in base)
    assert {t.kind for t in tasks} == {"adaptive"}
    assert bench.plan_stage2(subset_args(), [{"kind": "adaptive"}]) == []
    # Another model chain gets other rows: answers from different models never mix.
    other = bench.plan_stage1(subset_args(chain=bench.chain_fingerprint(["groq/x"])))
    assert {t.key for t in other if t.method == "autolab"}.isdisjoint({t.key for t in llm})


def test_hidden_apparatus_is_clean_physics_without_the_description_and_paced():
    task = bench.Task(kind="adaptive", method="autolab", provider="litellm", experiment="malus_law",
                      condition="hidden_apparatus", seed=0, budget=30, chain="abcd1234")
    cfg = bench.make_config(task, Path("unused"), rpm=10)
    assert cfg.agent.describe_apparatus is False and cfg.llm.min_request_interval_s == pytest.approx(6.0)
    assert cfg.simulation.faults == [] and not cfg.simulation.nonideal.enabled
    clean = bench.make_config(task.model_copy(update={"condition": "clean", "provider": "mock"}), Path("unused"), rpm=10)
    assert clean.agent.describe_apparatus is True and clean.llm.min_request_interval_s == 0.0  # mock is never paced
    assert bench.make_config(task, Path("unused")).llm.min_request_interval_s == 0.0


def test_the_planner_gets_names_and_units_but_no_apparatus_text(make_config):
    spec = get_experiment("inverse_square_law")
    view = spec.agent_view()
    hidden = view.for_agents(blind=False, describe_apparatus=False)
    assert hidden.description == HIDDEN_DESCRIPTION and hidden.independent == view.independent
    assert hidden.dependent == view.dependent and hidden.safe_bounds == view.safe_bounds
    assert view.for_agents(blind=False, describe_apparatus=True) == view
    assert view.for_agents(blind=True, describe_apparatus=False).independent.name == "x"  # blind still wins

    record = DiscoveryEngine(make_config("inverse_square_law", agent=AgentConfig(describe_apparatus=False))).run()
    plan = next(c for c in record.llm_calls if c.purpose == "plan")
    payload = json.loads(plan.user)
    assert payload["apparatus"] == HIDDEN_DESCRIPTION
    assert spec.description_for_agents not in plan.user
    assert view.independent.name in plan.user and view.dependent.units in plan.user


def fake_row(task, unavailable=False):
    return {**task.model_dump(), "key": task.key, "llm_unavailable": unavailable, "models": {}, "llm_calls": 0,
            "n_measurements": 0 if unavailable else 9, "error": "every model is rate-limited" if unavailable else ""}


def test_an_unavailable_run_is_not_stored_and_two_in_a_row_stop_the_benchmark(tmp_path, monkeypatch):
    tasks = [bench.Task(kind="adaptive", method="autolab", provider="litellm", experiment="malus_law",
                        condition="clean", seed=s, budget=30, chain="abcd1234") for s in range(5)]
    store = bench.Store(tmp_path / "rows")
    assert store.put(fake_row(tasks[0], unavailable=True)) is False and store.get(tasks[0]) is None

    # unavailable, ok, unavailable, unavailable -> stop; the ok run is kept, the others are retried later.
    script = iter([True, False, True, True])
    calls = []

    def fake_execute(task_json, records_dir, usd_to_inr, rpm=0.0):
        task = bench.Task.model_validate_json(task_json)
        calls.append((task.seed, rpm))
        return fake_row(task, unavailable=next(script))

    monkeypatch.setattr(bench, "execute", fake_execute)
    with pytest.raises(bench.ChainUnavailable, match="2 runs in a row found no model available"):
        bench.run_tasks(tasks, store, tmp_path / "records", workers=4, usd_to_inr=88.0, rpm=10)
    assert calls == [(0, 10), (1, 10), (2, 10), (3, 10)]  # one at a time in this process, paced
    assert [t.seed for t in tasks if store.get(t)] == [1]

    script = iter([False] * 4)
    calls.clear()
    rows = bench.run_tasks(tasks, store, tmp_path / "records", workers=4, usd_to_inr=88.0, rpm=10)
    assert [s for s, _ in calls] == [0, 2, 3, 4] and len(rows) == 5  # resumed: only the missing runs


def test_the_subset_exits_with_3_when_the_models_are_out_of_quota(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench, "execute", lambda task_json, *a, **k: fake_row(
        bench.Task.model_validate_json(task_json), unavailable=True))
    argv = ["--subset", "--experiments", "malus_law", "--conditions", "clean", "--methods", "autolab",
            "--work", str(tmp_path / "work"), "--out", str(tmp_path / "out")]
    assert bench.main(argv) == 3
    out = capsys.readouterr().out
    assert "Stopped:" in out and "Run the same command again later" in out
    assert not (tmp_path / "out").exists()


def test_subset_dry_run_shows_the_chain_calls_tokens_and_time(tmp_path, capsys):
    assert bench.main(["--subset", "--work", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Benchmark (reduced subset): 72 runs with the LLM provider 'litellm'" in out
    assert "plus 480 non-LLM baseline runs" in out and "Model chain" in out
    assert "Real LLM: 72 runs to do (0 already done). Estimated" in out and "output tokens" in out
    assert "At 10 requests per minute: about" in out
    assert not any(tmp_path.joinpath("rows").iterdir())


def test_the_subset_label_names_the_models_that_answered():
    rows = [{"provider": "litellm", "models": {"gemini/flash-lite": 30, "groq/gpt-oss-120b": 4}},
            {"provider": "litellm", "models": {"gemini/flash-lite": 10}}, {"provider": "none", "models": {}}]
    label = bench.subset_label(rows, "litellm")
    assert label == "Reduced subset, real LLM: gemini/flash-lite (40 calls), groq/gpt-oss-120b (4 calls)"
    assert "not a real LLM" in bench.subset_label(rows, "mock")
    footer = footer_text(SimpleNamespace(label=label, llm_seeds=3, seeds=20, provider="litellm"))
    assert label in footer and "mock" not in footer


def test_a_subset_rehearsal_with_mock_reuses_baselines_and_labels_everything(tmp_path):
    out, work = tmp_path / "results", tmp_path / "work"
    argv = ["--subset", "--provider", "mock", "--experiments", "malus_law", "--conditions", "clean,hidden_apparatus",
            "--methods", "gp_only,autolab", "--seeds", "1", "--llm-seeds", "1", "--workers", "1",
            "--out", str(out), "--work", str(work)]
    assert bench.main(argv) == 0
    rows = list(csv.DictReader((out / "benchmark_runs.csv").open()))
    got = {(r["method"], r["condition"], r["reused_from"]) for r in rows}
    assert got == {("gp_only", "clean", ""), ("gp_only", "hidden_apparatus", "clean"),
                   ("autolab", "clean", ""), ("autolab", "hidden_apparatus", "")}
    assert len(list((work / "rows").iterdir())) == 3  # the reused row is shown again, not run again
    md = (out / "RESULTS.md").read_text()
    assert "Reduced subset, mock provider (a rehearsal: not a real LLM)" in md and "Same seeds only" in md
    assert (out / "figures" / "fig_law_accuracy.png").exists()
