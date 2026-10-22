"""
Benchmark metrics: everything is computed from a run record and the experiment's ground truth,
after the run. Nothing here is visible to the agents.

Parameter error. The fit of the TRUE law (whether or not it ranked first) is compared with the
true parameters. Each parameter's error is normalised by its kind, as the stopping rule does:
|error| / |true value| for a scale, / x-span for an x-location, / the true response span for a
y-offset (an offset's true value can be 0, e.g. the LDR's). `param_err_max` is the largest of
these, in percent. With non-idealities on, a polariser's effective parameters are used:
A' = A (1 - 1/ER), C' = C + A/ER, because those are what the data follow.
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from analysis.model_library import MODEL_LIBRARY, AnalysisResult, ModelFit
from experiments.registry import ExperimentSpec
from records.run_record import RunRecord


def scoring_truth(record: RunRecord, spec: ExperimentSpec) -> Dict[str, float]:
    """The parameters the data actually follow (effective values under a finite extinction ratio)."""
    truth = dict(spec.true_params)
    ni = record.config.simulation.nonideal
    if record.config.hardware.simulate and ni.enabled and spec.polariser and math.isfinite(ni.extinction_ratio):
        A, C = truth["A"], truth["C"]
        truth["A"], truth["C"] = A * (1.0 - 1.0 / ni.extinction_ratio), C + A / ni.extinction_ratio
    return truth


def _spans(spec: ExperimentSpec) -> Tuple[float, float]:
    grid = np.linspace(*spec.safe_bounds, 201)
    response = np.array([spec.true_response(x) for x in grid])
    return float(spec.safe_bounds[1] - spec.safe_bounds[0]), float(np.ptp(response)) or 1.0


def param_errors(fit: Optional[ModelFit], spec: ExperimentSpec, truth: Dict[str, float]) -> Optional[Dict[str, Dict[str, float]]]:
    """Absolute and kind-normalised error of each parameter of the true law's fit; None if it was not fitted."""
    if fit is None or fit.key != spec.ground_truth_model:
        return None
    model = MODEL_LIBRARY[spec.ground_truth_model]
    x_span, y_span = _spans(spec)
    out = {}
    for name, kind in zip(model.param_names, model.param_kinds):
        err = abs(fit.params[name] - truth[name])
        ref = {"scale": abs(truth[name]), "x_location": x_span, "y_offset": y_span}[kind] or 1.0
        out[name] = {"estimate": fit.params[name], "true": truth[name], "abs": err, "pct": 100.0 * err / ref}
    return out


def true_law_fit(analysis: Optional[AnalysisResult], spec: ExperimentSpec) -> Optional[ModelFit]:
    if analysis is None:
        return None
    return next((f for f in analysis.fits if f.key == spec.ground_truth_model), None)


def error_curve(record: RunRecord, spec: ExperimentSpec) -> List[Tuple[int, float]]:
    """(n, largest parameter error in %) of the true law's fit after each analysed measurement."""
    truth = scoring_truth(record, spec)
    curve = []
    for step in record.steps:
        if step.analysis is None:
            continue
        errors = param_errors(true_law_fit(step.analysis, spec), spec, truth)
        if errors:
            curve.append((step.n_points, max(e["pct"] for e in errors.values())))
    return curve


def estimated_tokens(record: RunRecord) -> Tuple[int, int]:
    """Tokens a real model would have used, estimated from the prompt and answer text (4 characters per token)."""
    inp = sum(len(c.system) + len(c.user) for c in record.llm_calls) // 4
    out = sum(len(c.raw_response or "") for c in record.llm_calls) // 4
    return inp, out


def run_metrics(record: RunRecord, spec: ExperimentSpec, usd_to_inr: float) -> Dict[str, Any]:
    """One benchmark row (without the plan columns)."""
    analysis = record.result.final_analysis if record.result else None
    best = analysis.best if analysis else None
    truth = scoring_truth(record, spec)
    errors = param_errors(true_law_fit(analysis, spec), spec, truth)
    usage = record.llm_usage
    s = record.fault_summary
    est_in, est_out = estimated_tokens(record)
    return {
        "run_id": record.run_id,
        "data_source": record.data_source.value,
        "status": "error" if record.error else "ok",
        "error": record.error or "",
        "true_model": spec.ground_truth_model,
        "best_model": best.key if best else "",
        "correct": int(best is not None and best.key == spec.ground_truth_model),
        "stop_reason": record.stop.reason.value if record.stop and record.stop.reason else "",
        "n_measurements": record.result.n_measurements if record.result else 0,
        "attempts": len(record.steps),
        "param_err_max_pct": max(e["pct"] for e in errors.values()) if errors else None,
        "param_errors": errors,
        "faults_scheduled": s.scheduled if s else 0,
        "faults_reached": s.reached if s else 0,
        "faults_detected": s.detected if s else 0,
        "faults_recovered": s.recovered if s else 0,
        "faults_missed": s.missed if s else 0,
        "fault_outcomes": {f.type: f.status for f in record.faults if f.scheduled},
        "outliers": s.outliers if s else 0,
        "outliers_detected": s.outliers_detected if s else 0,
        "false_alarms": sum(s.unexplained.values()) if s else 0,
        "llm_calls": usage.calls,
        "llm_failed_calls": usage.failed_calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "est_input_tokens": est_in,
        "est_output_tokens": est_out,
        "cost_usd": usage.cost_usd,
        "cost_inr": usage.cost_usd * usd_to_inr,
        "cost_unknown_calls": usage.cost_unknown_calls,
        "models": usage.calls_per_model,
        "wall_s": record.timings.total_s,
        "curve": error_curve(record, spec),
    }


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

def mean_ci(values: Sequence[float], level: float = 0.95) -> Tuple[float, float, float, float]:
    """(mean, std, low, high): Student-t interval for the mean. NaN where undefined."""
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(np.mean(v))
    if v.size == 1:
        return mean, math.nan, math.nan, math.nan
    std = float(np.std(v, ddof=1))
    half = float(stats.t.ppf(0.5 + level / 2, v.size - 1)) * std / math.sqrt(v.size)
    return mean, std, mean - half, mean + half


def wilson(successes: int, n: int, level: float = 0.95) -> Tuple[float, float, float]:
    """(rate, low, high): Wilson score interval for a proportion."""
    if n == 0:
        return math.nan, math.nan, math.nan
    z = float(stats.norm.ppf(0.5 + level / 2))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)
