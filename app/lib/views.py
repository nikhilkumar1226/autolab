"""
Tables and replay data for the pages, built from engine objects and run records. No Streamlit.
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.candidate_models import FormulaRejected, compile_law, predict_fit
from analysis.model_library import AnalysisResult
from analysis.stopping import StopDecision
from config import BASE_DIR, RUNS_DIR
from records.run_record import RunRecord

RESULTS_DIR = BASE_DIR / "results"
DEMO_RUNS_DIR = RUNS_DIR / "demo"


def fits_table(analysis: Optional[AnalysisResult]) -> pd.DataFrame:
    if analysis is None:
        return pd.DataFrame()
    rows = []
    for rank, f in enumerate(analysis.fits, 1):
        rows.append({
            "Rank": rank, "Model": f.name, "Formula": f.formula, "k": f.n_params, "BIC": round(f.bic, 2),
            "ΔBIC": round(f.delta_bic, 2), "R²": round(f.r2, 4),
            "Counted": "yes" if f.redundant is None else "no (parsimony)",
        })
    for s in analysis.skipped:
        rows.append({"Rank": None, "Model": s.name, "Formula": "", "k": None, "BIC": None, "ΔBIC": None, "R²": None,
                     "Counted": f"not fitted: {s.reason}"})
    return pd.DataFrame(rows)


def criteria_table(decision: Optional[StopDecision]) -> pd.DataFrame:
    if decision is None:
        return pd.DataFrame()
    checks = [("ΔBIC to runner-up", decision.delta_bic, "{:.1f}", "≥ {:g}"),
              ("Max relative SE", decision.max_rel_se, "{:.2%}", "< {:.0%}"),
              ("Measurements", decision.min_points, "{:.0f}", "≥ {:.0f}"),
              ("Budget remaining", decision.budget, "{:.0f}", "< {:.0f}")]
    if decision.stable_best is not None:
        checks.append(("Same best model in a row", decision.stable_best, "{:.0f}", "≥ {:.0f}"))
    if decision.prediction is not None:
        checks.append(("Newest point vs prediction (σ)", decision.prediction, "{:.2f}", "≤ {:g}"))
    return pd.DataFrame([{
        "Criterion": name,
        "Value": "–" if c.value is None else fmt.format(c.value),
        "Threshold": thr.format(c.threshold),
        "Met": "yes" if c.passed else "no",
        "Note": c.detail,
    } for name, c, fmt, thr in checks])


def best_curve(analysis: Optional[AnalysisResult], lo: float, hi: float, n: int = 200) -> pd.DataFrame:
    best = analysis.best if analysis is not None else None
    if best is None:
        return pd.DataFrame(columns=["x", "y", "model"])
    grid = np.linspace(lo, hi, n)
    y = predict_fit(best, grid)
    if y is None:
        return pd.DataFrame(columns=["x", "y", "model"])
    ok = np.isfinite(y)
    return pd.DataFrame({"x": grid[ok], "y": y[ok], "model": best.name})


def law_curve(law: Optional[str], lo: float, hi: float, n: int = 200) -> pd.DataFrame:
    """A custom experiment's law over the range (empty for registry experiments or a law that no longer compiles)."""
    if not law:
        return pd.DataFrame(columns=["x", "y"])
    try:
        _, f = compile_law(law)
    except FormulaRejected:
        return pd.DataFrame(columns=["x", "y"])
    grid = np.linspace(lo, hi, n)
    y = f(grid)
    ok = np.isfinite(y)
    return pd.DataFrame({"x": grid[ok], "y": y[ok]})


# --- run records ---------------------------------------------------------------------------

def list_records(dirs: Iterable[Path] = (RUNS_DIR, DEMO_RUNS_DIR), limit: int = 200) -> List[Path]:
    """Run records in the given folders (not their subfolders), newest first."""
    paths = []
    for d in dirs:
        if Path(d).is_dir():
            paths += list(Path(d).glob("*.json"))
    return sorted(set(paths), key=lambda p: p.name, reverse=True)[:limit]


def record_summary(path: Path) -> Dict[str, Any]:
    """A one-line description of a record, from its JSON (no validation, so old records list too)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"path": str(path), "label": f"{Path(path).name} (unreadable)"}
    result = raw.get("result") or {}
    fits = ((result.get("final_analysis") or {}).get("fits")) or []
    best = next((f.get("name") for f in fits if not f.get("redundant")), "no model")
    stop = (raw.get("stop") or {}).get("reason") or "?"
    exp = (raw.get("experiment") or {}).get("name", "?")
    provider = ((raw.get("config") or {}).get("llm") or {}).get("provider", "?")
    started = str(raw.get("started_at", ""))[:19].replace("T", " ")
    return {
        "path": str(path), "run_id": raw.get("run_id", Path(path).stem), "started": started, "experiment": exp,
        "seed": raw.get("seed"), "best": best, "stop": stop, "n": result.get("n_measurements"),
        "data_source": raw.get("data_source", "?"), "provider": provider, "demo": Path(path).parent.name == "demo",
        "label": f"{started} UTC · {exp} · seed {raw.get('seed')} · {best} · {stop} · {provider}"
                 + (" · demo" if Path(path).parent.name == "demo" else ""),
    }


def steps_frame(record: RunRecord) -> pd.DataFrame:
    rows = []
    for s in record.steps:
        rows.append({
            "attempt": s.index, "source": s.source, "requested_x": s.requested_position, "status": s.status,
            "x": s.position, "y": s.value, "n_points": s.n_points, "flags": ",".join(s.flags),
            "action": s.anomaly.action if s.anomaly else "", "best_model": s.analysis.best.key if s.analysis and s.analysis.best else "",
            "delta_bic": s.analysis.delta_bic_runner_up if s.analysis else None, "reason": s.rationale, "message": s.message,
        })
    return pd.DataFrame(rows)


def measured_steps(record: RunRecord) -> List[int]:
    return [s.index for s in record.steps if s.status == "measured"]


def replay(record: RunRecord, upto: int) -> Dict[str, Any]:
    """The state of a recorded run after attempt `upto`: points, flagged requests, the latest fits, the messages."""
    steps = [s for s in record.steps if s.index <= upto]
    points = [{"step": s.index, "x": s.position, "y": s.value, "source": s.source} for s in steps if s.status == "measured"]
    flagged = [{"step": s.index, "x": s.requested_position, "y": s.value, "kind": "rejected" if s.status == "rejected" else "flagged"}
               for s in steps if s.status in ("rejected", "anomaly") and s.requested_position is not None]
    analysis = next((s.analysis for s in reversed(steps) if s.analysis is not None), None)
    messages: List[Tuple[str, str, int]] = []
    if record.hypothesis is not None:
        messages.append(("Planner", record.hypothesis.statement, 0))
    for s in steps:
        if s.rationale and s.source in ("planner", "gp", "heuristic") and s.attempt_in_slot == 1:
            messages.append(("Planner", s.rationale, s.index))
        if s.review is not None:
            messages.append(("Analyst", s.review.interpretation, s.index))
        if s.anomaly is not None:
            messages.append(({"executor": "Executor", "safety_guard": "Safety guard", "analyst": "Analyst"}[s.anomaly.decided_by],
                             f"{s.anomaly.action.capitalize()}: {s.anomaly.reason}", s.index))
        if s.status == "rejected":
            messages.append(("Safety guard", s.message, s.index))
    return {"points": points, "flagged": flagged, "analysis": analysis, "messages": messages}


def summary_metrics(record: RunRecord) -> Dict[str, Any]:
    analysis = record.result.final_analysis if record.result else None
    best = analysis.best if analysis else None
    return {
        "best": best.name if best else "no model",
        "equation": record.result.equation_latex if record.result else None,
        "n": record.result.n_measurements if record.result else 0,
        "stop": record.stop.reason.value if record.stop and record.stop.reason else "?",
        "message": record.stop.message if record.stop else "",
        "delta_bic": analysis.delta_bic_runner_up if analysis else None,
        "llm_calls": record.llm_usage.calls,
        "cost": record.llm_usage.cost_usd,
        "audit_ok": record.audit_intact() if record.audit_log else None,
    }


def finite(v: Optional[float]) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)
