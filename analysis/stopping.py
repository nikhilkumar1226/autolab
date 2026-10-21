"""
Evidence-based stopping rule.

A run converges when ALL of these hold for the best model (by BIC):
1. it beats the runner-up by delta-BIC >= threshold (default 6, "strong" evidence
   on the Kass & Raftery 1995 scale);
2. every parameter's relative standard error is below max_rel_se (default 5%),
   using the kind-normalised relative SE from analysis.model_library;
3. at least min_points measurements exist;
4. predictive stability, if enabled:
   (a) the same model was best, by delta-BIC >= threshold, in each of the last
       `stability_window` analyses, and
   (b) the fit made before the newest measurement predicted it within
       `prediction_max_z` sigma. sigma is that fit's residual std sqrt(SSR/(n-k)),
       floored at noise_floor_rel x the measured y-span (an assumed instrument
       resolution), because with 1-2 residual degrees of freedom the residual std
       can be close to zero by chance.
Independently, the run stops when the measurement budget is used up.

Why criterion 4: with 4-5 points a wrong 3-parameter model can pass almost exactly
through the noisy data and win delta-BIC >= 6 by chance. A model that has captured
the law should keep winning and should predict a measurement it has not seen.
"""
import math
from enum import Enum
from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from analysis.model_library import MODEL_LIBRARY, AnalysisResult, ModelFit
from config import StoppingConfig


class StopReason(str, Enum):
    CONVERGED = "converged"
    BUDGET_EXHAUSTED = "budget_exhausted"
    SAFETY_ABORT = "safety_abort"
    ERROR = "error"


class CriterionCheck(BaseModel):
    value: Optional[float]
    threshold: float
    passed: bool
    detail: str = ""


class PointPrediction(BaseModel):
    """A measurement as predicted by the best model fitted before it was taken."""
    model: str = Field(description="Key of the predicting model (best of the previous analysis)")
    n_fit: int = Field(description="Number of points in the predicting fit")
    x: float
    observed: float
    predicted: Optional[float] = Field(description="None if the model is undefined at x")
    residual_std: float = Field(description="sqrt(SSR / (n - k)) of the predicting fit")
    noise_floor: float = Field(description="noise_floor_rel x y-span of the data the fit used")
    sigma: float = Field(description="max(residual_std, noise_floor)")
    z: Optional[float] = Field(description="|observed - predicted| / sigma; None if it cannot be computed")
    # Diagnostics only (not used by the rule): did the new point discriminate between the top two models?
    runner_up: Optional[str] = None
    runner_up_predicted: Optional[float] = None


class StopDecision(BaseModel):
    should_stop: bool
    reason: Optional[StopReason] = None
    n_measurements: int
    best_model: Optional[str] = None
    delta_bic: CriterionCheck
    max_rel_se: CriterionCheck
    min_points: CriterionCheck
    budget: CriterionCheck
    # None when predictive stability is disabled.
    stable_best: Optional[CriterionCheck] = None
    prediction: Optional[CriterionCheck] = None
    message: str


def _predict_at(fit: ModelFit, x: float) -> Optional[float]:
    model = MODEL_LIBRARY.get(fit.key)
    if model is None:
        return None
    with np.errstate(all="ignore"):
        value = float(model.predict([x], fit.params)[0])
    return value if math.isfinite(value) else None


def predict_new_point(
    previous: Optional[AnalysisResult],
    x: float,
    y: float,
    y_before: Sequence[float],
    cfg: StoppingConfig,
) -> Optional[PointPrediction]:
    """
    Predict the new measurement (x, y) with the best model of `previous`, the analysis
    made before it was taken; `y_before` are the values that analysis used. The fit does
    not depend on the new point, so this is the prediction it made in advance.
    Returns None when there is no earlier fit.
    """
    if previous is None or previous.best is None:
        return None
    best, runner_up = previous.best, previous.runner_up
    noise_floor = cfg.noise_floor_rel * (float(np.ptp(y_before)) if len(y_before) else 0.0)
    sigma = max(best.residual_std, noise_floor)
    predicted = _predict_at(best, x)
    z = abs(y - predicted) / sigma if predicted is not None and sigma > 0 else None
    return PointPrediction(
        model=best.key,
        n_fit=best.n_points,
        x=x,
        observed=y,
        predicted=predicted,
        residual_std=best.residual_std,
        noise_floor=noise_floor,
        sigma=sigma,
        z=z,
        runner_up=runner_up.key if runner_up is not None else None,
        runner_up_predicted=_predict_at(runner_up, x) if runner_up is not None else None,
    )


def _stable_best_check(
    best: Optional[ModelFit], analyses: Sequence[AnalysisResult], cfg: StoppingConfig
) -> CriterionCheck:
    threshold = float(cfg.stability_window)
    if best is None:
        return CriterionCheck(value=None, threshold=threshold, passed=False, detail="no model fitted")
    streak = 0
    for a in reversed(analyses):
        dbic = a.delta_bic_runner_up
        if a.best is None or a.best.key != best.key or dbic is None or dbic < cfg.delta_bic_threshold:
            break
        streak += 1
    return CriterionCheck(
        value=float(streak), threshold=threshold, passed=streak >= cfg.stability_window,
        detail=f"consecutive analyses with {best.key} best by delta-BIC >= {cfg.delta_bic_threshold:g}",
    )


def _prediction_check(
    best: Optional[ModelFit], p: Optional[PointPrediction], cfg: StoppingConfig
) -> CriterionCheck:
    threshold = cfg.prediction_max_z
    if best is None:
        return CriterionCheck(value=None, threshold=threshold, passed=False, detail="no model fitted")
    if p is None:
        return CriterionCheck(
            value=None, threshold=threshold, passed=False,
            detail="no earlier fit predicted the newest measurement",
        )
    if p.z is None:
        why = f"{p.model} is undefined at x = {p.x:.4g}" if p.predicted is None else "sigma is zero"
        return CriterionCheck(value=None, threshold=threshold, passed=False, detail=why)
    detail = (
        f"x = {p.x:.4g}: observed {p.observed:.4g}, predicted {p.predicted:.4g} by the n = {p.n_fit} "
        f"{p.model} fit; sigma {p.sigma:.3g} (residual std {p.residual_std:.3g}, floor {p.noise_floor:.3g})"
    )
    if p.model != best.key:
        return CriterionCheck(
            value=p.z, threshold=threshold, passed=False,
            detail=f"predicted by {p.model}, not the current best {best.key}; {detail}",
        )
    return CriterionCheck(value=p.z, threshold=threshold, passed=p.z <= threshold, detail=detail)


def evaluate_stopping(
    analysis: Optional[AnalysisResult],
    n_measurements: int,
    cfg: StoppingConfig,
    override: Optional[StopReason] = None,
    previous: Sequence[AnalysisResult] = (),
    prediction: Optional[PointPrediction] = None,
) -> StopDecision:
    """
    Check every criterion and decide whether to stop. `override` forces a stop with
    that reason (safety abort, error) while still recording the criteria at that moment.
    `previous` holds the analyses before `analysis` (oldest first) and `prediction` the
    newest measurement as predicted before it was taken; both are used only by the
    predictive-stability criterion.
    """
    best = analysis.best if analysis is not None else None

    dbic = analysis.delta_bic_runner_up if analysis is not None else None
    delta_bic = CriterionCheck(
        value=dbic,
        threshold=cfg.delta_bic_threshold,
        passed=dbic is not None and dbic >= cfg.delta_bic_threshold,
        detail="" if dbic is not None else "fewer than two models fitted",
    )

    if best is None:
        rel_se = CriterionCheck(value=None, threshold=cfg.max_rel_se, passed=False, detail="no model fitted")
    else:
        undefined = [p for p, v in best.rel_std_errors.items() if v is None]
        if undefined:
            rel_se = CriterionCheck(
                value=None, threshold=cfg.max_rel_se, passed=False,
                detail=f"standard error undefined for {', '.join(undefined)}",
            )
        else:
            worst = max(best.rel_std_errors, key=lambda p: best.rel_std_errors[p])
            worst_value = best.rel_std_errors[worst]
            rel_se = CriterionCheck(
                value=worst_value, threshold=cfg.max_rel_se,
                passed=worst_value < cfg.max_rel_se, detail=f"worst parameter: {worst}",
            )

    min_points = CriterionCheck(
        value=float(n_measurements), threshold=float(cfg.min_points),
        passed=n_measurements >= cfg.min_points,
    )
    budget = CriterionCheck(
        value=float(n_measurements), threshold=float(cfg.max_measurements),
        passed=n_measurements < cfg.max_measurements,
        detail="passed = budget remaining",
    )

    checks = [("delta-BIC", delta_bic), ("relative SE", rel_se), ("min points", min_points)]
    stable_best = predicted = None
    if cfg.predictive_stability:
        analyses = [*previous, analysis] if analysis is not None else list(previous)
        stable_best = _stable_best_check(best, analyses, cfg)
        predicted = _prediction_check(best, prediction, cfg)
        checks += [("stable best model", stable_best), ("prediction of the newest point", predicted)]

    converged = all(check.passed for _, check in checks)
    if override is not None:
        reason, message = override, f"stopped: {override.value}"
    elif converged:
        reason = StopReason.CONVERGED
        message = (
            f"converged: {best.name} beats the runner-up by delta-BIC {dbic:.1f} "
            f"(>= {cfg.delta_bic_threshold:g}), max relative SE {rel_se.value:.2%} "
            f"(< {cfg.max_rel_se:.0%}), n = {n_measurements}"
        )
        if cfg.predictive_stability:
            message += (
                f"; best in {stable_best.value:.0f} consecutive analyses, newest point "
                f"{predicted.value:.2f} sigma from its prediction (<= {cfg.prediction_max_z:g})"
            )
    elif not budget.passed:
        reason = StopReason.BUDGET_EXHAUSTED
        message = f"measurement budget of {cfg.max_measurements} used up before the evidence criteria were all met"
    else:
        reason, message = None, "continue: " + ", ".join(
            name for name, check in checks if not check.passed
        ) + " not yet met"

    return StopDecision(
        should_stop=reason is not None,
        reason=reason,
        n_measurements=n_measurements,
        best_model=best.key if best is not None else None,
        delta_bic=delta_bic,
        max_rel_se=rel_se,
        min_points=min_points,
        budget=budget,
        stable_best=stable_best,
        prediction=predicted,
        message=message,
    )
