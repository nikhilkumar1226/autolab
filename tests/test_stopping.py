"""
Unit tests for the evidence-based stopping rule.
"""
import math

import numpy as np
import pytest

from analysis.model_library import MODEL_LIBRARY, AnalysisResult, ModelFit, fit_library, fit_model
from analysis.stopping import PointPrediction, StopDecision, StopReason, evaluate_stopping, predict_new_point
from config import StoppingConfig

# The Phase 1 rule, independent of the current defaults.
CFG = StoppingConfig(min_points=5, predictive_stability=False)
STABLE = CFG.model_copy(update={"predictive_stability": True})


def make_fit(key: str, bic: float, rel: dict) -> ModelFit:
    return ModelFit(
        key=key, name=key, formula="", n_points=6, n_params=len(rel),
        params={p: 1.0 for p in rel}, std_errors={p: 0.01 for p in rel}, rel_std_errors=rel,
        r2=0.99, rmse=1.0, aic=bic, bic=bic,
    )


def make_analysis(delta_bic: float = 10.0, rel: dict = None, n: int = 6, best_key: str = "cos2_phase") -> AnalysisResult:
    rel = rel if rel is not None else {"A": 0.01, "x0": 0.001, "C": 0.002}
    best = make_fit(best_key, 0.0, rel)
    runner_up = make_fit("quadratic", delta_bic, {"a": 0.1, "b": 0.1, "c": 0.1}).model_copy(update={"delta_bic": delta_bic})
    return AnalysisResult(iteration=n, n_points=n, fits=[best, runner_up], delta_bic_runner_up=delta_bic)


def test_converges_when_all_criteria_are_met():
    d = evaluate_stopping(make_analysis(), 6, CFG)
    assert d.should_stop and d.reason == StopReason.CONVERGED
    assert d.best_model == "cos2_phase"
    assert d.delta_bic.passed and d.max_rel_se.passed and d.min_points.passed


def test_delta_bic_below_threshold_continues():
    d = evaluate_stopping(make_analysis(delta_bic=5.9), 6, CFG)
    assert not d.should_stop and d.reason is None
    assert not d.delta_bic.passed and d.delta_bic.value == pytest.approx(5.9)


def test_delta_bic_at_threshold_is_enough():
    assert evaluate_stopping(make_analysis(delta_bic=6.0), 6, CFG).reason == StopReason.CONVERGED


def test_imprecise_parameter_continues():
    d = evaluate_stopping(make_analysis(rel={"A": 0.06, "x0": 0.001, "C": 0.002}), 6, CFG)
    assert not d.should_stop
    assert d.max_rel_se.value == pytest.approx(0.06)
    assert d.max_rel_se.detail == "worst parameter: A"


def test_undefined_standard_error_continues():
    d = evaluate_stopping(make_analysis(rel={"A": 0.01, "x0": None, "C": 0.002}), 6, CFG)
    assert not d.should_stop and not d.max_rel_se.passed
    assert "x0" in d.max_rel_se.detail


def test_too_few_points_continues():
    d = evaluate_stopping(make_analysis(n=4), 4, CFG)
    assert not d.should_stop and not d.min_points.passed


def test_single_fitted_model_has_no_delta_bic():
    analysis = AnalysisResult(iteration=5, n_points=5, fits=[make_fit("linear", 0.0, {"a": 0.01, "b": 0.01})])
    d = evaluate_stopping(analysis, 5, CFG)
    assert not d.should_stop and d.delta_bic.value is None


def test_budget_stop_records_the_unmet_criteria():
    d = evaluate_stopping(make_analysis(delta_bic=2.0), CFG.max_measurements, CFG)
    assert d.should_stop and d.reason == StopReason.BUDGET_EXHAUSTED
    assert d.delta_bic.value == pytest.approx(2.0) and not d.delta_bic.passed
    assert not d.budget.passed


def test_convergence_takes_precedence_at_the_budget():
    assert evaluate_stopping(make_analysis(), CFG.max_measurements, CFG).reason == StopReason.CONVERGED


def test_budget_applies_before_any_analysis():
    assert not evaluate_stopping(None, 2, CFG).should_stop
    assert evaluate_stopping(None, 3, CFG.model_copy(update={"max_measurements": 3})).reason == StopReason.BUDGET_EXHAUSTED


def test_relative_se_threshold_is_configurable():
    analysis = make_analysis(rel={"A": 0.04, "x0": 0.001, "C": 0.002})
    assert evaluate_stopping(analysis, 6, CFG).should_stop
    assert not evaluate_stopping(analysis, 6, CFG.model_copy(update={"max_rel_se": 0.03})).should_stop


def test_override_forces_a_stop_and_keeps_the_criteria():
    d = evaluate_stopping(make_analysis(delta_bic=1.0), 6, CFG, override=StopReason.SAFETY_ABORT)
    assert d.should_stop and d.reason == StopReason.SAFETY_ABORT
    assert d.delta_bic.value == pytest.approx(1.0)


def test_location_and_offset_parameters_are_normalised_by_span():
    """A true phase of 0 and a baseline near 0 must not make the precision criterion unreachable."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 180.0, 12)
    model = MODEL_LIBRARY["cos2_phase"]
    y = model.predict(x, {"A": 1000.0, "x0": 0.0, "C": 0.5}) + rng.normal(0.0, 10.0, x.size)

    fit = fit_model(model, x, y)
    x_span, y_span = np.ptp(x), np.ptp(y)
    assert fit.rel_std_errors["x0"] == pytest.approx(fit.std_errors["x0"] / x_span)
    assert fit.rel_std_errors["C"] == pytest.approx(fit.std_errors["C"] / y_span)
    assert fit.rel_std_errors["A"] == pytest.approx(fit.std_errors["A"] / abs(fit.params["A"]))
    # Relative to the values themselves these would be far above 5%.
    assert fit.std_errors["C"] / abs(fit.params["C"]) > 0.05
    assert max(fit.rel_std_errors.values()) < CFG.max_rel_se


def test_default_is_the_configuration_chosen_by_the_seeded_sweep():
    """docs/CHANGELOG.md (Phase 1 fix): min_points = 8 was the only config with >= 19/20 correct everywhere."""
    default = StoppingConfig()
    assert default.min_points == 8 and not default.predictive_stability
    assert default.stability_window == 2 and default.prediction_max_z == 3.0 and default.noise_floor_rel == 0.005


# --- Predictive stability -----------------------------------------------------

def make_prediction(z: float = 1.0, model: str = "cos2_phase") -> PointPrediction:
    return PointPrediction(
        model=model, n_fit=5, x=10.0, observed=100.0 + 5.0 * z, predicted=100.0,
        residual_std=5.0, noise_floor=1.0, sigma=5.0, z=z,
    )


def stable_check(previous, prediction, cfg=STABLE, n=6, **analysis_kwargs):
    return evaluate_stopping(make_analysis(n=n, **analysis_kwargs), n, cfg, previous=previous, prediction=prediction)


def test_stability_converges_with_a_repeated_best_and_a_predicted_point():
    d = stable_check([make_analysis(n=5)], make_prediction(1.0))
    assert d.should_stop and d.reason == StopReason.CONVERGED
    assert d.stable_best.passed and d.stable_best.value == 2 and d.stable_best.threshold == 2
    assert d.prediction.passed and d.prediction.value == pytest.approx(1.0) and d.prediction.threshold == 3
    assert "2 consecutive analyses" in d.message


def test_disabled_stability_records_no_extra_criteria():
    d = evaluate_stopping(make_analysis(), 6, CFG, previous=[make_analysis(best_key="exp_decay")], prediction=make_prediction(9.0))
    assert d.reason == StopReason.CONVERGED
    assert d.stable_best is None and d.prediction is None


def test_stability_needs_an_earlier_analysis_and_prediction():
    d = stable_check([], None)
    assert not d.should_stop
    assert d.stable_best.value == 1 and not d.stable_best.passed
    assert d.prediction.value is None and not d.prediction.passed
    assert "no earlier fit" in d.prediction.detail
    assert "stable best model" in d.message and "prediction of the newest point" in d.message


def test_stability_resets_when_the_best_model_changes():
    d = stable_check([make_analysis(n=4), make_analysis(n=5, best_key="exp_decay")], make_prediction(1.0))
    assert not d.should_stop and d.stable_best.value == 1


def test_stability_needs_strong_evidence_in_every_analysis_of_the_streak():
    d = stable_check([make_analysis(n=5, delta_bic=5.9)], make_prediction(1.0))
    assert not d.should_stop and d.stable_best.value == 1
    # ...and in the current one, even if the earlier analyses were strong.
    d = stable_check([make_analysis(n=5)], make_prediction(1.0), delta_bic=5.9)
    assert not d.should_stop and d.stable_best.value == 0


def test_badly_predicted_new_point_continues_and_three_sigma_is_enough():
    d = stable_check([make_analysis(n=5)], make_prediction(3.1))
    assert not d.should_stop and not d.prediction.passed
    assert d.prediction.value == pytest.approx(3.1)
    assert stable_check([make_analysis(n=5)], make_prediction(3.0)).reason == StopReason.CONVERGED


def test_prediction_must_come_from_the_current_best_model():
    d = stable_check([make_analysis(n=5)], make_prediction(0.5, model="exp_decay"))
    assert not d.should_stop and not d.prediction.passed
    assert "not the current best" in d.prediction.detail


def test_stability_window_and_z_threshold_are_configurable():
    window3 = STABLE.model_copy(update={"stability_window": 3})
    assert not stable_check([make_analysis(n=5)], make_prediction(1.0), cfg=window3).should_stop
    d = stable_check([make_analysis(n=4), make_analysis(n=5)], make_prediction(1.0), cfg=window3)
    assert d.reason == StopReason.CONVERGED and d.stable_best.value == 3

    strict = STABLE.model_copy(update={"prediction_max_z": 2.0})
    assert not stable_check([make_analysis(n=5)], make_prediction(2.5), cfg=strict).should_stop


def test_budget_still_stops_with_stability_and_convergence_keeps_precedence():
    n = STABLE.max_measurements
    d = stable_check([], None, n=n)
    assert d.reason == StopReason.BUDGET_EXHAUSTED and not d.stable_best.passed
    assert stable_check([make_analysis(n=n - 1)], make_prediction(1.0), n=n).reason == StopReason.CONVERGED


def linear_analysis(rmse: float, n: int = 5) -> AnalysisResult:
    """A hand-made previous analysis whose best model is y = 2x + 1, so predictions are exact."""
    best = ModelFit(
        key="linear", name="Linear", formula="", n_points=n, n_params=2, params={"a": 2.0, "b": 1.0},
        std_errors={"a": 0.01, "b": 0.01}, rel_std_errors={"a": 0.01, "b": 0.01},
        r2=0.99, rmse=rmse, aic=0.0, bic=0.0,
    )
    runner_up = make_fit("quadratic", 8.0, {"a": 0.1, "b": 0.1, "c": 0.1}).model_copy(
        update={"params": {"a": 0.0, "b": 0.0, "c": 30.0}, "delta_bic": 8.0}
    )
    return AnalysisResult(iteration=n, n_points=n, fits=[best, runner_up], delta_bic_runner_up=8.0)


def test_prediction_uses_the_dof_corrected_residual_std():
    y_before = [0.0, 25.0, 50.0, 75.0, 100.0]
    p = predict_new_point(linear_analysis(rmse=3.0), 10.0, 25.0, y_before, STABLE)
    residual_std = 3.0 * math.sqrt(5 / 3)  # sqrt(SSR / (n - k)) with SSR = n * rmse^2
    assert p.model == "linear" and p.n_fit == 5
    assert p.predicted == pytest.approx(21.0)
    assert p.residual_std == pytest.approx(residual_std)
    assert p.noise_floor == pytest.approx(0.5)  # 0.5% of the 100-unit y-span
    assert p.sigma == pytest.approx(residual_std)
    assert p.z == pytest.approx(4.0 / residual_std)
    assert p.runner_up == "quadratic" and p.runner_up_predicted == pytest.approx(30.0)


def test_noise_floor_applies_when_the_fit_nearly_interpolates():
    """With 1-2 residual degrees of freedom a fit can pass almost exactly through noisy points."""
    p = predict_new_point(linear_analysis(rmse=1e-6), 10.0, 22.0, [0.0, 100.0], STABLE)
    assert p.sigma == pytest.approx(0.5) and p.noise_floor == pytest.approx(0.5)
    assert p.z == pytest.approx(2.0)


def test_floor_scales_with_the_measured_span_and_noise_floor_rel():
    cfg = STABLE.model_copy(update={"noise_floor_rel": 0.02})
    p = predict_new_point(linear_analysis(rmse=1e-6), 10.0, 22.0, [10.0, 60.0], cfg)
    assert p.noise_floor == pytest.approx(1.0)


def test_prediction_on_real_fits_matches_the_library_model():
    rng = np.random.default_rng(3)
    model = MODEL_LIBRARY["exp_decay"]
    x = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    y = model.predict(x, {"A": 1000.0, "k": 0.04, "C": 15.0}) + rng.normal(0.0, 10.0, x.size)
    previous = fit_library(x, y)
    best = previous.best

    p = predict_new_point(previous, 60.0, 120.0, y, STABLE)
    expected = float(MODEL_LIBRARY[best.key].predict([60.0], best.params)[0])
    assert p.model == best.key and p.n_fit == 5
    assert p.predicted == pytest.approx(expected)
    assert p.sigma == pytest.approx(max(best.residual_std, 0.005 * np.ptp(y)))
    assert p.z == pytest.approx(abs(120.0 - expected) / p.sigma)


def test_undefined_prediction_fails_the_check():
    power = ModelFit(
        key="power_law", name="Power law", formula="", n_points=5, n_params=3, params={"A": 1.0, "b": -1.0, "C": 0.0},
        std_errors={"A": 0.01, "b": 0.01, "C": 0.01}, rel_std_errors={"A": 0.01, "b": 0.01, "C": 0.01},
        r2=0.99, rmse=1.0, aic=0.0, bic=0.0,
    )
    previous = AnalysisResult(iteration=5, n_points=5, fits=[power])
    p = predict_new_point(previous, 0.0, 5.0, [1.0, 2.0], STABLE)  # 0 ** -1 is not finite
    assert p.predicted is None and p.z is None

    analysis = AnalysisResult(
        iteration=6, n_points=6, fits=[power, make_fit("linear", 9.0, {"a": 0.1, "b": 0.1}).model_copy(update={"delta_bic": 9.0})],
        delta_bic_runner_up=9.0,
    )
    previous = previous.model_copy(update={"fits": analysis.fits, "delta_bic_runner_up": 9.0})
    d = evaluate_stopping(analysis, 6, STABLE, previous=[previous], prediction=p)
    assert d.stable_best.passed and not d.prediction.passed and not d.should_stop
    assert "undefined" in d.prediction.detail


def test_no_prediction_without_an_earlier_fit():
    assert predict_new_point(None, 10.0, 1.0, [], STABLE) is None
    assert predict_new_point(AnalysisResult(iteration=4, n_points=4, fits=[]), 10.0, 1.0, [0.0], STABLE) is None


def test_stop_decisions_from_older_records_still_load():
    """Phase 1 run records have no stability fields."""
    decision = evaluate_stopping(make_analysis(), 6, CFG)
    old = decision.model_dump(mode="json")
    del old["stable_best"], old["prediction"]
    assert StopDecision.model_validate(old) == decision
