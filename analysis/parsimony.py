"""
Parsimony: a fit with more parameters that reproduces a simpler model's curve is not a
separate explanation of the data.

Several 3-parameter library models contain a straight line as a special case or a limit:
a quadratic with a = 0, a power law with b = 1, an exponential with k -> 0, a cos^2 arc much
shorter than its period. When the data are linear, those models fit as well as the line and
sit within delta-BIC ~ ln n of it (about 2-3.4 for 8-30 points), so the stopping rule's
delta-BIC >= 6 against the runner-up could never be met, and BIC alone picks the quadratic
in about 1 run in 4 (6/20 in the Phase 3 prototype).

The rule, applied after the BIC ranking, on the measured x-range. sigma is the residual std
of the lowest-BIC fit, floored at noise_floor_rel x the y-span; two curves are "the same"
when they stay within sigma_ratio x sigma of each other everywhere on the range.
1. Take the lowest-BIC fit that is not yet marked. If a fit with FEWER parameters has the
   same curve, mark the lowest-BIC fit redundant and repeat.
2. The fit that survives step 1 is the best model. Every other fit with MORE parameters than
   it and the same curve is marked redundant too.
Fits with the same number of parameters or fewer always stay competitors, whatever their
curves. Redundant fits stay in the table with the reason; they are only left out of the best
model, the runner-up and delta-BIC (analysis.model_library.AnalysisResult.competitors).

With the library models only, the rule can change a ranking only when a 2-parameter model
(the line) has the same curve as a 3-parameter one, i.e. when the data are linear within
noise on the measured range.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from analysis.candidate_models import predict_fit
from analysis.model_library import AnalysisResult, ModelFit


def _gap(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    return float(np.max(np.abs(a - b)))


def apply_parsimony(
    result: AnalysisResult,
    x: Sequence[float],
    y: Sequence[float],
    sigma_ratio: float = 0.5,
    noise_floor_rel: float = 0.005,
) -> AnalysisResult:
    fits: List[ModelFit] = [f.model_copy(update={"redundant": None}) for f in result.fits]
    if len(fits) < 2:
        return result.model_copy(update={"fits": fits})
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    grid = np.linspace(float(np.min(xs)), float(np.max(xs)), 200)
    sigma = max(fits[0].residual_std, noise_floor_rel * float(np.ptp(ys)))
    tolerance = sigma_ratio * sigma
    curves: Dict[str, Optional[np.ndarray]] = {f.key: predict_fit(f, grid) for f in fits}

    def same_curve(a: ModelFit, b: ModelFit) -> Optional[float]:
        gap = _gap(curves[a.key], curves[b.key])
        return gap if gap is not None and gap <= tolerance else None

    def reason(simpler: ModelFit, gap: float) -> str:
        return (f"same curve as {simpler.name} ({simpler.n_params} parameters) within {gap:.3g} "
                f"<= {sigma_ratio:g} sigma = {tolerance:.3g} on the measured range, with more parameters (parsimony)")

    marked: Dict[str, str] = {}
    while True:
        best = next(f for f in fits if f.key not in marked)
        simpler = [(f, same_curve(f, best)) for f in fits
                   if f.key not in marked and f.n_params < best.n_params]
        simpler = [(f, gap) for f, gap in simpler if gap is not None]
        if not simpler:
            break
        f, gap = min(simpler, key=lambda pair: (pair[0].n_params, pair[0].bic))
        marked[best.key] = reason(f, gap)
    for f in fits:
        if f.key in marked or f.key == best.key or f.n_params <= best.n_params:
            continue
        gap = same_curve(f, best)
        if gap is not None:
            marked[f.key] = reason(best, gap)

    fits = [f.model_copy(update={"redundant": marked.get(f.key)}) for f in fits]
    ranked = result.model_copy(update={"fits": fits})
    runner = ranked.runner_up
    return ranked.model_copy(update={
        "delta_bic_runner_up": runner.bic - ranked.best.bic if runner is not None else None,
    })
