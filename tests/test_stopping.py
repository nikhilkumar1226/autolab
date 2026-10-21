"""
Unit tests for the evidence-based stopping rule.
"""
import numpy as np
import pytest

from analysis.model_library import MODEL_LIBRARY, AnalysisResult, ModelFit, fit_model
from analysis.stopping import StopReason, evaluate_stopping
from config import StoppingConfig

CFG = StoppingConfig()


def make_fit(key: str, bic: float, rel: dict) -> ModelFit:
    return ModelFit(
        key=key, name=key, formula="", n_points=6, n_params=len(rel),
        params={p: 1.0 for p in rel}, std_errors={p: 0.01 for p in rel}, rel_std_errors=rel,
        r2=0.99, rmse=1.0, aic=bic, bic=bic,
    )


def make_analysis(delta_bic: float = 10.0, rel: dict = None, n: int = 6) -> AnalysisResult:
    rel = rel if rel is not None else {"A": 0.01, "x0": 0.001, "C": 0.002}
    best = make_fit("cos2_phase", 0.0, rel)
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
    assert evaluate_stopping(None, 3, StoppingConfig(max_measurements=3)).reason == StopReason.BUDGET_EXHAUSTED


def test_relative_se_threshold_is_configurable():
    analysis = make_analysis(rel={"A": 0.04, "x0": 0.001, "C": 0.002})
    assert evaluate_stopping(analysis, 6, CFG).should_stop
    assert not evaluate_stopping(analysis, 6, StoppingConfig(max_rel_se=0.03)).should_stop


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
