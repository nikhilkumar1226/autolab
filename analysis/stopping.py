"""
Evidence-based stopping rule.

A run converges when ALL of these hold for the best model (by BIC):
1. it beats the runner-up by delta-BIC >= threshold (default 6, "strong" evidence
   on the Kass & Raftery 1995 scale);
2. every parameter's relative standard error is below max_rel_se (default 5%),
   using the kind-normalised relative SE from analysis.model_library;
3. at least min_points measurements exist.
Independently, the run stops when the measurement budget is used up.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from analysis.model_library import AnalysisResult
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


class StopDecision(BaseModel):
    should_stop: bool
    reason: Optional[StopReason] = None
    n_measurements: int
    best_model: Optional[str] = None
    delta_bic: CriterionCheck
    max_rel_se: CriterionCheck
    min_points: CriterionCheck
    budget: CriterionCheck
    message: str


def evaluate_stopping(
    analysis: Optional[AnalysisResult],
    n_measurements: int,
    cfg: StoppingConfig,
    override: Optional[StopReason] = None,
) -> StopDecision:
    """
    Check every criterion and decide whether to stop. `override` forces a stop with
    that reason (safety abort, error) while still recording the criteria at that moment.
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

    converged = delta_bic.passed and rel_se.passed and min_points.passed
    if override is not None:
        reason, message = override, f"stopped: {override.value}"
    elif converged:
        reason = StopReason.CONVERGED
        message = (
            f"converged: {best.name} beats the runner-up by delta-BIC {dbic:.1f} "
            f"(>= {cfg.delta_bic_threshold:g}), max relative SE {rel_se.value:.2%} "
            f"(< {cfg.max_rel_se:.0%}), n = {n_measurements}"
        )
    elif not budget.passed:
        reason = StopReason.BUDGET_EXHAUSTED
        message = f"measurement budget of {cfg.max_measurements} used up before the evidence criteria were all met"
    else:
        reason, message = None, "continue: " + ", ".join(
            name for name, check in
            [("delta-BIC", delta_bic), ("relative SE", rel_se), ("min points", min_points)]
            if not check.passed
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
        message=message,
    )
