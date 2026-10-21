"""
Numerical diagnostics handed to the LLM agents. The agents never compute these themselves.

- residual_summary: is there structure left in a fit's residuals? Signs ordered by x, the
  Wald-Wolfowitz runs test (far fewer sign runs than chance means a systematic misfit),
  lag-1 autocorrelation, the largest standardised residual and the residuals' trend in x.
- max_disagreement: where inside the safe bounds the two best models predict most
  differently, in units of the noise. A measurement there can tell them apart
  (model-discrimination design, Box & Hill 1967). The Planner may choose it.
"""
import math
from typing import Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from analysis.candidate_models import predict_fit
from analysis.model_library import ModelFit


class ResidualSummary(BaseModel):
    model: str
    n: int
    signs: str = Field(description="Residual signs ordered by x: + above the fit, - below, 0 exactly on it")
    runs: int = Field(description="Number of runs of equal signs")
    expected_runs: Optional[float] = Field(description="Expected runs if the signs were random")
    runs_z: Optional[float] = Field(description="Runs-test z; strongly negative = systematic pattern")
    lag1_autocorr: Optional[float] = None
    max_abs_std_residual: float = Field(description="Largest |residual| / residual std")
    at_x: float = Field(description="x of the largest standardised residual")
    trend_corr: Optional[float] = Field(default=None, description="Correlation of the residuals with x")


class Disagreement(BaseModel):
    x: float
    best_prediction: float
    runner_up_prediction: float
    difference_sigma: float = Field(description="|best - runner-up| / sigma at x")
    sigma: float


def _runs_test(signs: Sequence[int]) -> Tuple[int, Optional[float], Optional[float]]:
    nonzero = [s for s in signs if s != 0]
    runs = 1 + sum(a != b for a, b in zip(nonzero, nonzero[1:])) if nonzero else 0
    n1, n2 = sum(s > 0 for s in nonzero), sum(s < 0 for s in nonzero)
    n = n1 + n2
    if n1 == 0 or n2 == 0 or n < 2:
        return runs, None, None
    expected = 2.0 * n1 * n2 / n + 1.0
    variance = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0))
    z = (runs - expected) / math.sqrt(variance) if variance > 0 else None
    return runs, expected, z


def residual_summary(fit: ModelFit, x: Sequence[float], y: Sequence[float]) -> Optional[ResidualSummary]:
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    predicted = predict_fit(fit, xs)
    if predicted is None or not np.all(np.isfinite(predicted)):
        return None
    order = np.argsort(xs, kind="stable")
    xs, residuals = xs[order], (ys - predicted)[order]
    signs = [int(np.sign(r)) for r in residuals]
    runs, expected, z = _runs_test(signs)

    scale = fit.residual_std
    standardised = residuals / scale if scale > 0 else np.zeros_like(residuals)
    worst = int(np.argmax(np.abs(standardised)))
    energy = float(np.sum(residuals ** 2))
    lag1 = float(np.sum(residuals[:-1] * residuals[1:]) / energy) if len(residuals) >= 3 and energy > 0 else None
    trend = None
    if len(residuals) >= 3 and np.std(residuals) > 0 and np.std(xs) > 0:
        trend = float(np.corrcoef(xs, residuals)[0, 1])
    return ResidualSummary(
        model=fit.key,
        n=len(residuals),
        signs="".join("+" if s > 0 else "-" if s < 0 else "0" for s in signs),
        runs=runs,
        expected_runs=expected,
        runs_z=z,
        lag1_autocorr=lag1,
        max_abs_std_residual=float(abs(standardised[worst])),
        at_x=float(xs[worst]),
        trend_corr=trend,
    )


def max_disagreement(
    best: ModelFit, runner_up: ModelFit, bounds: Tuple[float, float], sigma: float, n_grid: int = 201
) -> Optional[Disagreement]:
    """The x in the safe bounds where the two models' predictions differ most, in sigma."""
    if sigma <= 0:
        return None
    grid = np.linspace(bounds[0], bounds[1], n_grid)
    a, b = predict_fit(best, grid), predict_fit(runner_up, grid)
    if a is None or b is None:
        return None
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return None
    gap = np.where(finite, np.abs(a - b), -np.inf)
    i = int(np.argmax(gap))
    return Disagreement(
        x=float(grid[i]),
        best_prediction=float(a[i]),
        runner_up_prediction=float(b[i]),
        difference_sigma=float(gap[i] / sigma),
        sigma=sigma,
    )
