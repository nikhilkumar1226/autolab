"""
Summary statistics over benchmark rows: one row per run kind x method x experiment x condition,
plus a pooled row per run kind x method x condition (experiment "ALL").

Proportions (law identified, faults recovered, outliers detected) have Wilson 95% intervals;
means have Student-t 95% intervals. Parameter error is skewed, so its median is given too.
"""
import math
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from benchmark.metrics import mean_ci, wilson

GROUP = ("kind", "method", "provider", "experiment", "condition")


def price_per_token(model: str) -> Optional[Tuple[float, float]]:
    """(input, output) USD per token from litellm's bundled price map, or None if the model is not in it."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    try:
        import litellm
    except ImportError:
        return None
    for name in (model, model.split("/", 1)[-1]):
        entry = litellm.model_cost.get(name)
        if entry and entry.get("input_cost_per_token") is not None:
            return float(entry["input_cost_per_token"]), float(entry.get("output_cost_per_token") or 0.0)
    return None


def _summarise(group: Dict[str, Any], rows: List[Dict[str, Any]], price: Optional[Tuple[float, float]],
               usd_to_inr: float) -> Dict[str, Any]:
    ok = [r for r in rows if r["status"] == "ok"]
    out = dict(group)
    out["n_runs"], out["n_errors"] = len(rows), len(rows) - len(ok)
    rate, lo, hi = wilson(sum(r["correct"] for r in ok), len(ok))
    out.update(correct_rate=rate, correct_ci_low=lo, correct_ci_high=hi)
    n = [r["n_measurements"] for r in ok]
    mean, std, lo, hi = mean_ci(n)
    out.update(n_meas_mean=mean, n_meas_std=std, n_meas_ci_low=lo, n_meas_ci_high=hi,
               n_meas_median=statistics.median(n) if n else math.nan)
    errs = [r["param_err_max_pct"] for r in ok if r["param_err_max_pct"] is not None]
    mean, std, lo, hi = mean_ci(errs)
    out.update(param_err_pct_median=statistics.median(errs) if errs else math.nan, param_err_pct_mean=mean,
               param_err_pct_std=std, param_err_pct_ci_low=lo, param_err_pct_ci_high=hi, param_err_n=len(errs))
    reached = sum(r["faults_reached"] for r in ok)
    rate, lo, hi = wilson(sum(r["faults_recovered"] for r in ok), reached)
    out.update(faults_reached=reached, fault_recovery_rate=rate, fault_recovery_ci_low=lo, fault_recovery_ci_high=hi)
    outliers = sum(r["outliers"] for r in ok)
    out.update(outliers=outliers, outlier_detection_rate=wilson(sum(r["outliers_detected"] for r in ok), outliers)[0],
               false_alarms_per_run=mean_ci([r["false_alarms"] for r in ok])[0])
    for col in ("llm_calls", "input_tokens", "output_tokens", "est_input_tokens", "est_output_tokens", "cost_usd",
                "cost_inr", "wall_s"):
        out[f"{col}_mean"] = mean_ci([r[col] for r in ok])[0]
    if price is not None and ok and any(r["llm_calls"] for r in ok):
        est = [r["est_input_tokens"] * price[0] + r["est_output_tokens"] * price[1] for r in ok]
        out["est_cost_usd_mean"] = mean_ci(est)[0]
        out["est_cost_inr_mean"] = out["est_cost_usd_mean"] * usd_to_inr
    else:
        out["est_cost_usd_mean"] = out["est_cost_inr_mean"] = 0.0 if ok and not any(r["llm_calls"] for r in ok) else math.nan
    return out


def summarise(rows: Iterable[Dict[str, Any]], price_model: str, usd_to_inr: float) -> List[Dict[str, Any]]:
    rows = list(rows)
    price = price_per_token(price_model)
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[g] for g in GROUP)].append(r)
        groups[(r["kind"], r["method"], r["provider"], "ALL", r["condition"])].append(r)
    return [_summarise(dict(zip(GROUP, key)), members, price, usd_to_inr) for key, members in sorted(groups.items())]
