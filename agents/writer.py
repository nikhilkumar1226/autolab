"""
Writer agent: the discussion and conclusion of the generated report.

The Writer receives facts extracted from the run record (build_facts) and is told to use
only those numbers. The answer is then checked: every number in the text must match a
number in the facts at the precision written (small counts 0-10 are allowed). If one does
not, the Writer is asked once more, with the offending numbers listed. Numbers that still
do not match are kept visible under the section in the report, not silently removed.
"""
import logging
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set

from agents.models import Discussion, WriterOutput
from agents.prompts import PROMPT_VERSION, WRITER_SYSTEM, round_floats, to_payload
from config import describe_twin
from llm.client import LLMClient, NotAvailable
from records.run_record import RunRecord

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(\s?%)?(?![\w])")
SMALL_INTEGERS = set(range(11))


def agent_view_used(record: RunRecord):
    """The experiment view the LLM agents were shown in this run."""
    view = record.experiment.agent_view
    return view.blinded() if record.config.agent.blind else view


def build_facts(record: RunRecord) -> Dict[str, Any]:
    """Every number the report may quote, from the record only."""
    view = agent_view_used(record)
    analysis = record.result.final_analysis if record.result else None
    best = analysis.best if analysis else None
    runner = analysis.runner_up if analysis else None
    stop = record.stop
    last_review = next((s.review for s in reversed(record.steps) if s.review is not None), None)
    criteria = []
    if stop is not None:
        for name, c in [("delta_bic", stop.delta_bic), ("max_rel_se", stop.max_rel_se), ("min_points", stop.min_points),
                        ("budget", stop.budget), ("stable_best", stop.stable_best), ("prediction", stop.prediction)]:
            if c is not None:
                criteria.append({"criterion": name, "value": c.value, "threshold": c.threshold, "met": c.passed})
    flagged = [s for s in record.steps if s.status == "anomaly"]
    summary = record.fault_summary
    return {
        "data_source": record.data_source.value,
        "digital_twin": describe_twin(record.config) or None,
        "flagged_readings": len(flagged),
        "injected_faults": None if summary is None else {
            "scheduled": summary.scheduled, "reached": summary.reached, "detected": summary.detected,
            "recovered": summary.recovered, "missed": summary.missed,
            "random_outliers": summary.outliers, "random_outliers_detected": summary.outliers_detected,
            "false_alarms": sum(summary.unexplained.values()),
        },
        "research_question": record.question,
        "apparatus": view.description,
        "variables": {
            "x": {"name": view.independent.name, "units": view.independent.units},
            "y": {"name": view.dependent.name, "units": view.dependent.units},
        },
        "safe_bounds": list(view.safe_bounds),
        "seed": record.seed,
        "n_measurements": record.result.n_measurements if record.result else 0,
        "rejected_or_failed_attempts": sum(s.status != "measured" for s in record.steps),
        "hypothesis": None if record.hypothesis is None else {
            "statement": record.hypothesis.statement,
            "candidate_relationships": record.hypothesis.candidate_relationships,
        },
        "best_model": None if best is None else {
            "name": best.name, "formula": best.formula, "origin": best.origin, "k": best.n_params,
            "params": {p: {"value": v, "std_error": best.std_errors[p], "rel_se": best.rel_std_errors[p]}
                       for p, v in best.params.items()},
            "bic": best.bic, "r2": best.r2, "rmse": best.rmse,
        },
        "runner_up": None if runner is None else {"name": runner.name, "delta_bic": runner.delta_bic},
        "models_compared": [f.name for f in analysis.fits] if analysis else [],
        "llm_candidate_models": [
            {"expression": c.spec.expression, "status": c.status, "reason": c.reason} for c in record.candidate_models
        ],
        "analyst_interpretation": last_review.interpretation if last_review else None,
        "stop": None if stop is None else {
            "reason": stop.reason.value if stop.reason else "unknown",
            "message": stop.message,
            "criteria": criteria,
        },
    }


def _fact_numbers(obj: Any, out: Set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.add(abs(float(obj)))
    elif isinstance(obj, str):
        for match in _NUMBER.finditer(obj):
            out.add(float(match.group(1)))
    elif isinstance(obj, dict):
        for v in obj.values():
            _fact_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _fact_numbers(v, out)


def _matches(token: str, value: float, facts: Iterable[float], percent: bool) -> bool:
    decimals = len(token.split(".")[1].split("e")[0].split("E")[0]) if "." in token else 0
    digits = len(re.sub(r"[^0-9]", "", token.split("e")[0].split("E")[0]).lstrip("0")) or 1
    tolerance = 0.5 * 10 ** (-decimals)
    for fact in facts:
        for candidate in (fact, fact * 100.0) if percent else (fact,):
            if abs(candidate - value) <= tolerance * (1 + 1e-9):
                return True
            if candidate != 0 and float(f"{candidate:.{digits}g}") == value:
                return True
    return False


def find_unverified_numbers(text: str, facts: Dict[str, Any]) -> List[str]:
    """Numbers in `text` that are not in `facts` (at the precision written)."""
    known: Set[float] = set()
    _fact_numbers(round_floats(facts), known)
    _fact_numbers(facts, known)
    unverified = []
    for match in _NUMBER.finditer(text):
        token, percent = match.group(1), bool(match.group(2))
        value = float(token)
        if value in SMALL_INTEGERS and "." not in token:
            continue
        if not _matches(token, value, known, percent):
            unverified.append(token + ("%" if percent else ""))
    return list(dict.fromkeys(unverified))


def mock_writer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic template that only quotes the facts."""
    x, y = payload["variables"]["x"]["name"], payload["variables"]["y"]["name"]
    n, best, runner, stop = payload["n_measurements"], payload["best_model"], payload["runner_up"], payload["stop"]
    converged = stop is not None and stop["reason"] == "converged"
    if best is None:
        discussion = f"Data source: {payload['data_source']}. No model could be fitted to the {n} measurements."
        conclusion = "No relationship could be identified in this run."
    else:
        discussion = (
            f"Data source: {payload['data_source']}. After {n} measurements, blind model comparison ranked "
            f"{best['name']} ({best['formula']}) first"
            + (f", ahead of {runner['name']} by delta-BIC {runner['delta_bic']}" if runner else "")
            + f". The run stopped with reason {stop['reason'] if stop else 'unknown'}: {stop['message'] if stop else ''}. "
            "Only the models in the compared set could be identified."
        )
        conclusion = (
            f"{'The data support' if converged else 'Provisionally, the data favour'} {best['name']} as the "
            f"dependence of {y} on {x} within the measured range."
            + ("" if converged else " The stopping criteria were not all met.")
        )
    return {"discussion": "Mock writer (deterministic template, not a language model). " + discussion,
            "conclusion": conclusion}


class WriterAgent:
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client

    def write(self, record: RunRecord) -> Optional[Discussion]:
        """The discussion section, or None without an LLM. LLMUnavailableError propagates."""
        if self.client is None:
            return None
        facts = build_facts(record)
        try:
            out = self._ask(facts, "discussion")
        except NotAvailable:
            return None
        calls = [self.client.calls[-1]]
        unverified = find_unverified_numbers(f"{out.discussion}\n{out.conclusion}", facts)
        if unverified:
            logger.warning("Writer used numbers not in the run record: %s; asking once more", ", ".join(unverified))
            out = self._ask({**facts, "numbers_not_in_facts": unverified}, "discussion_revision")
            calls.append(self.client.calls[-1])
            unverified = find_unverified_numbers(f"{out.discussion}\n{out.conclusion}", facts)
        return Discussion(
            discussion=out.discussion,
            conclusion=out.conclusion,
            source="llm" if calls[-1].provider == "litellm" else "mock",
            model=calls[-1].model,
            llm_call_ids=[c.id for c in calls],
            unverified_numbers=unverified,
        )

    def _ask(self, facts: Dict[str, Any], purpose: str) -> WriterOutput:
        return self.client.complete_json(
            WRITER_SYSTEM, to_payload(facts), WriterOutput, agent="writer", purpose=purpose,
            mock=mock_writer, prompt_version=PROMPT_VERSION,
        )
