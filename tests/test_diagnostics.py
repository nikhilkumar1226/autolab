"""
Tests for the diagnostics handed to the agents: residual structure and model disagreement.
"""
import numpy as np
import pytest

from analysis.diagnostics import _runs_test, max_disagreement, residual_summary
from analysis.model_library import MODEL_LIBRARY, fit_model
from experiments.registry import get_experiment


def test_runs_test_on_a_known_sequence():
    runs, expected, z = _runs_test([1, 1, -1, -1, 1, 1, -1, -1])
    assert runs == 4 and expected == pytest.approx(5.0)
    assert z == pytest.approx((4 - 5) / np.sqrt(2 * 16 * (32 - 8) / (64 * 7)))
    assert _runs_test([1, 1, 1]) == (1, None, None)  # one sign only: no test


def test_a_wrong_model_leaves_systematic_residuals():
    x = np.linspace(0.0, 10.0, 21)
    y = 3.0 * x ** 2 + np.random.default_rng(0).normal(0.0, 0.5, x.size)
    linear = fit_model(MODEL_LIBRARY["linear"], x, y)
    summary = residual_summary(linear, x, y)
    assert summary.signs.startswith("+") and summary.signs.endswith("+") and "-" in summary.signs
    assert summary.runs == 3 and summary.runs_z < -3
    assert summary.lag1_autocorr > 0.5


def test_the_right_model_leaves_unstructured_residuals(simulate):
    spec = get_experiment("beer_lambert_law")
    x = np.linspace(0.0, 100.0, 30)
    y = simulate(spec, x, seed=1)
    fit = fit_model(MODEL_LIBRARY["exp_decay"], x, y)
    summary = residual_summary(fit, x, y)
    assert abs(summary.runs_z) < 2 and summary.n == 30 and len(summary.signs) == 30
    assert summary.max_abs_std_residual < 3.5 and abs(summary.trend_corr) < 0.4
    assert 0.0 <= summary.at_x <= 100.0


def test_max_disagreement_finds_where_two_models_differ_most(simulate):
    spec = get_experiment("inverse_square_law")
    x = np.array([5.0, 30.0, 52.5, 100.0, 75.0])
    y = simulate(spec, x, seed=4)
    exp_fit = fit_model(MODEL_LIBRARY["exp_decay"], x, y)
    inv_fit = fit_model(MODEL_LIBRARY["inverse_square_offset"], x, y)

    d = max_disagreement(inv_fit, exp_fit, spec.safe_bounds, sigma=10.0)
    assert 5.0 <= d.x <= 100.0
    a = MODEL_LIBRARY["inverse_square_offset"].predict([d.x], inv_fit.params)[0]
    b = MODEL_LIBRARY["exp_decay"].predict([d.x], exp_fit.params)[0]
    assert d.difference_sigma == pytest.approx(abs(a - b) / 10.0)
    grid = np.linspace(5.0, 100.0, 201)
    gaps = np.abs(MODEL_LIBRARY["inverse_square_offset"].predict(grid, inv_fit.params)
                  - MODEL_LIBRARY["exp_decay"].predict(grid, exp_fit.params))
    assert d.difference_sigma == pytest.approx(gaps.max() / 10.0)

    same = max_disagreement(inv_fit, inv_fit, spec.safe_bounds, sigma=10.0)
    assert same.difference_sigma == 0.0
    assert max_disagreement(inv_fit, exp_fit, spec.safe_bounds, sigma=0.0) is None
