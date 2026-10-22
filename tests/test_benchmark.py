"""
Phase 5: the benchmark's metrics, the new samplers (grid, LLM-only), and a tiny end-to-end
benchmark (plan, resume, CSVs, figures, RESULTS.md).
"""
import csv
import json
import math

import numpy as np
import pytest

from agents.lead_scientist import gap_point
from agents.orchestrator import DiscoveryEngine
from analysis.model_library import ModelFit
from benchmark import run as bench
from benchmark.metrics import mean_ci, param_errors, scoring_truth, wilson
from config import AgentConfig, LLMConfig, NonIdealityConfig, SimulationConfig
from experiments.registry import get_experiment
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
    assert "Real LLM: about" in out and not any(tmp_path.joinpath("rows").iterdir())
