"""
Unit tests for the blind Data Analyst Agent and the candidate model library.
"""
import inspect
import math

import numpy as np
import pytest

from agents.analyst import DataAnalystAgent
from analysis.model_library import MODEL_LIBRARY, ModelFit, fit_library, fit_model
from experiments.registry import get_experiment, list_experiments

# (x range, true parameters) for each library model.
MODEL_CASES = {
    "linear": ((0.0, 100.0), {"a": 2.0, "b": 10.0}),
    "quadratic": ((0.0, 100.0), {"a": 0.05, "b": -3.0, "c": 50.0}),
    "exp_decay": ((0.0, 100.0), {"A": 800.0, "k": 0.03, "C": 20.0}),
    "cos2_phase": ((0.0, 180.0), {"A": 500.0, "x0": -30.0, "C": 40.0}),
    "inverse_square_offset": ((5.0, 100.0), {"A": 5.0e4, "x0": 3.0, "C": 10.0}),
    "power_law": ((1.0, 50.0), {"A": 3.0, "b": 1.5, "C": 5.0}),
}


@pytest.mark.parametrize("key", list(MODEL_LIBRARY))
def test_each_model_recovers_its_own_parameters(key):
    (lo, hi), true = MODEL_CASES[key]
    model = MODEL_LIBRARY[key]
    rng = np.random.default_rng(0)
    x = np.linspace(lo, hi, 25)
    y = model.predict(x, true) + rng.normal(0.0, 1.0, x.size)

    fit = fit_model(model, x, y)
    assert isinstance(fit, ModelFit), fit
    for p, value in true.items():
        se = fit.std_errors[p]
        assert se is not None
        assert abs(fit.params[p] - value) < 5 * se, f"{key}.{p}: {fit.params[p]} vs {value} (SE {se})"


@pytest.mark.parametrize("name", list_experiments())
def test_blind_analyst_identifies_each_law(name, simulate):
    """20 evenly spaced points at the spec's default noise, 20 seeds: >= 90% correct."""
    spec = get_experiment(name)
    analyst = DataAnalystAgent()
    x = np.linspace(*spec.safe_bounds, 20)
    correct = sum(
        analyst.analyze(x, simulate(spec, x, seed)).best.key == spec.ground_truth_model
        for seed in range(20)
    )
    assert correct >= 18, f"{name}: {correct}/20 identified"


def test_analysis_ranks_by_bic_and_reports_delta_bic(simulate):
    spec = get_experiment("malus_law")
    x = np.linspace(0.0, 180.0, 12)
    result = DataAnalystAgent().analyze(x, simulate(spec, x, seed=1), iteration=3)

    bics = [f.bic for f in result.fits]
    assert bics == sorted(bics)
    assert result.fits[0].delta_bic == 0.0
    assert result.delta_bic_runner_up == pytest.approx(result.fits[1].bic - result.fits[0].bic)
    assert result.delta_bic_runner_up > 0
    assert result.iteration == 3 and result.n_points == 12


def test_power_law_is_skipped_when_x_is_not_positive():
    x = np.linspace(0.0, 100.0, 10)
    result = fit_library(x, 1000.0 * np.exp(-0.04 * x) + 15.0)
    skipped = {s.key: s.reason for s in result.skipped}
    assert "x > 0" in skipped["power_law"]
    assert "power_law" not in {f.key for f in result.fits}


def test_models_needing_more_points_are_skipped():
    result = fit_library([0.0, 50.0, 100.0], [1.0, 2.0, 3.5])
    assert [f.key for f in result.fits] == ["linear"]
    assert result.delta_bic_runner_up is None


def test_noise_free_fit_has_finite_information_criteria():
    spec = get_experiment("malus_law")
    x = np.linspace(0.0, 180.0, 10)
    y = [spec.true_response(v) for v in x]
    best = fit_library(x, y).best
    assert best.key == "cos2_phase"
    assert math.isfinite(best.bic) and math.isfinite(best.aic)


def test_analyst_is_never_told_the_experiment():
    assert set(inspect.signature(DataAnalystAgent.analyze).parameters) == {"self", "x_data", "y_data", "iteration"}
    assert set(inspect.signature(DataAnalystAgent.__init__).parameters) == {"self", "models"}
