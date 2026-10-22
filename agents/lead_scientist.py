"""
Lead Scientist / Planner agent.

With an LLM provider the Planner:
- formulates the hypothesis and an initial design from the research question and the
  agent-facing experiment view (plan);
- before each later measurement, delegates to the GP or picks the point itself, always with
  a reason, and re-proposes after a safety rejection (decide_next);
- explains afterwards why the numerical stopping rule stopped the run (explain_stop). The
  rule itself decides.

Positions are only proposals: the hard-coded safety guard checks each one. A proposed range
is clipped to the safe bounds and recorded. It never reaches the guard, the GP or the
simulator, which are built from the experiment spec before the Planner is asked.

With provider "none" the heuristic below is used (the no-LLM baseline). The mock policies
reproduce that heuristic and always delegate to the GP, so mock and none runs take the same
measurements.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

from pydantic import BaseModel

from agents.models import (
    AgentText,
    ExperimentStep,
    GPSuggestion,
    Hypothesis,
    PlannerOutput,
    StopRationale,
    StrategyDecision,
    TextSource,
)
from agents.prompts import PLANNER_SYSTEM, PROMPT_VERSION, STOP_SYSTEM, STRATEGY_SYSTEM, to_payload
from analysis.diagnostics import Disagreement
from analysis.model_library import AnalysisResult
from analysis.stopping import StopDecision
from experiments.registry import AgentExperimentView
from llm.client import LLMClient, NotAvailable
from llm.types import LLMCallRecord

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _article(name: str) -> str:
    return name if len(name) == 1 else f"the {name}"


def default_question(view: AgentExperimentView) -> str:
    """A neutral research question built from the variable names only."""
    return f"How does {_article(view.dependent.name)} depend on {_article(view.independent.name)}?"


def heuristic_anchors(lo: float, hi: float, num_samples: int) -> List[float]:
    """Boundary survey: the ends, the middle, then the quartiles."""
    anchors = [lo, hi]
    if num_samples >= 3:
        anchors.append(round((lo + hi) / 2.0, 1))
    if num_samples >= 4:
        anchors.append(round(lo + (hi - lo) * 0.25, 1))
    if num_samples >= 5:
        anchors.append(round(lo + (hi - lo) * 0.75, 1))
    return sorted(set(anchors))


def clip_range(proposed: Sequence[float], bounds: Tuple[float, float]) -> Tuple[List[float], bool, str]:
    """Clip a proposed exploration range to the safe bounds. Returns (applied, clipped, note)."""
    lo, hi = bounds
    a, b = sorted(float(v) for v in proposed)
    applied = [max(a, lo), min(b, hi)]
    if applied[0] >= applied[1]:
        return [lo, hi], True, (
            f"proposed range [{a:g}, {b:g}] does not overlap the safe bounds [{lo:g}, {hi:g}]; the full safe range is used"
        )
    if applied != [a, b]:
        return applied, True, (
            f"proposed range [{a:g}, {b:g}] clipped to [{applied[0]:g}, {applied[1]:g}] (safe bounds [{lo:g}, {hi:g}])"
        )
    return applied, False, ""


def _variables(view: AgentExperimentView) -> Dict[str, Any]:
    return {
        "x": {"name": view.independent.name, "units": view.independent.units},
        "y": {"name": view.dependent.name, "units": view.dependent.units},
    }


def _source(call: LLMCallRecord) -> TextSource:
    return "llm" if call.provider == "litellm" else "mock"


@dataclass
class PlanningState:
    """What the Planner is shown before choosing the next measurement."""
    view: AgentExperimentView
    x: Sequence[float]
    y: Sequence[float]
    analysis: Optional[AnalysisResult]
    stop_check: Optional[StopDecision]
    gp: GPSuggestion
    disagreement: Optional[Disagreement]
    budget_left: int
    recent_rejections: List[Dict[str, Any]] = field(default_factory=list)
    recent_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    observation: Optional[Dict[str, Any]] = None


def _unmet(decision: Optional[StopDecision]) -> List[Dict[str, Any]]:
    if decision is None:
        return []
    checks = [("delta_bic", decision.delta_bic), ("max_rel_se", decision.max_rel_se), ("min_points", decision.min_points),
              ("stable_best", decision.stable_best), ("prediction", decision.prediction)]
    return [{"criterion": name, "value": c.value, "threshold": c.threshold}
            for name, c in checks if c is not None and not c.passed]


def strategy_payload(s: PlanningState) -> Dict[str, Any]:
    order = sorted(range(len(s.x)), key=lambda i: s.x[i])
    fits = [] if s.analysis is None else [
        {"model": f.name, "formula": f.formula, "k": f.n_params, "bic": f.bic, "delta_bic": f.delta_bic, "r2": f.r2}
        for f in s.analysis.fits[:4]
    ]
    d = s.disagreement
    payload = {
        "variables": _variables(s.view),
        "safe_bounds": list(s.view.safe_bounds),
        "data": [[s.x[i], s.y[i]] for i in order],
        "budget_left": s.budget_left,
        "fits": fits,
        "stopping_criteria_unmet": _unmet(s.stop_check),
        "gp_suggestion": {"x": s.gp.position, "max_sigma": s.gp.max_sigma},
        "models_disagree_most_at": None if d is None else {
            "x": d.x, "best_predicts": d.best_prediction, "runner_up_predicts": d.runner_up_prediction,
            "difference_sigma": d.difference_sigma,
        },
        "recent_rejections": s.recent_rejections[-3:],
    }
    if s.recent_anomalies:
        payload["recent_flagged_readings"] = s.recent_anomalies[-3:]
    if s.observation is not None:
        payload["observation"] = s.observation
    return payload


# ---------------------------------------------------------------------------
# Mock policies: deterministic, and only see the JSON payload a real model would see
# ---------------------------------------------------------------------------

def mock_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    lo, hi = payload["safe_bounds"]
    x, y = payload["variables"]["x"]["name"], payload["variables"]["y"]["name"]
    n = min(max(int(payload["n_points_default"]), 3), 5)
    return {
        "hypothesis": {
            "statement": f"{_article(y).capitalize()} varies systematically with {_article(x)} within the safe range; "
                         f"the form of the dependence is to be identified from the data.",
            "candidate_relationships": ["monotonic increase or decrease", "periodic variation", "power-law or exponential change"],
            "independent_variable": x,
            "proposed_range": [lo, hi],
        },
        "initial_design": heuristic_anchors(lo, hi, n),
        "rationale": "Mock planner: anchors spread evenly across the safe range (the same design as the no-LLM heuristic).",
    }


def mock_strategy(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "observation" in payload:
        return {"choice": "gp", "reason": "Mock policy: the safety guard rejected the previous proposal, so the GP chooses."}
    return {"choice": "gp", "reason": "Mock policy: delegate to GP uncertainty sampling (the no-LLM baseline)."}


def mock_stop(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"rationale": f"The numerical stopping rule ended the run ({payload['stop_reason']}): {payload['rule_message']}."}


# ---------------------------------------------------------------------------

class LeadScientistAgent:
    """
    Acts as the Principal Investigator / Planner.
    Formulates scientific hypotheses and directs the experimental methodology.
    """

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client

    def _ask(self, system: str, payload: Dict[str, Any], schema: Type[T], purpose: str,
             mock: Callable[[Dict[str, Any]], Any], step: Optional[int] = None) -> Optional[Tuple[T, LLMCallRecord]]:
        """One LLM call; None when there is no LLM (provider none). LLMUnavailableError propagates."""
        if self.client is None:
            return None
        try:
            answer = self.client.complete_json(
                system, to_payload(payload), schema, agent="planner", purpose=purpose, step=step,
                mock=mock, prompt_version=PROMPT_VERSION,
            )
        except NotAvailable:
            return None
        return answer, self.client.calls[-1]

    # --- hypothesis and initial design ---------------------------------------
    def plan(self, question: str, view: AgentExperimentView, num_samples: int) -> Tuple[Hypothesis, List[ExperimentStep]]:
        payload = {
            "research_question": question,
            "apparatus": view.description,
            "variables": _variables(view),
            "safe_bounds": list(view.safe_bounds),
            "n_points_default": num_samples,
        }
        asked = self._ask(PLANNER_SYSTEM, payload, PlannerOutput, "plan", mock_plan)
        if asked is None:
            return self.formulate_hypothesis(question, view), self.generate_initial_plan(view, num_samples)
        out, call = asked

        h = out.hypothesis
        applied, clipped, note = clip_range(h.proposed_range, view.safe_bounds)
        if clipped:
            logger.warning("Planner: %s. The safety guard limits are unchanged.", note)
        indep, dep = view.independent, view.dependent
        hypothesis = Hypothesis(
            title=question.strip(),
            statement=h.statement,
            target_relationship="; ".join(h.candidate_relationships),
            independent_variable=f"{indep.name} ({indep.units})",
            dependent_variable=f"{dep.name} ({dep.units})",
            suggested_range=applied,
            candidate_relationships=list(h.candidate_relationships),
            proposed_range=list(h.proposed_range),
            range_clipped=clipped,
            range_note=note,
            design_rationale=out.rationale,
            source=_source(call),
            model=call.model,
            llm_call_id=call.id,
        )
        # Positions are queued exactly as proposed: the safety guard, not this code, rejects bad ones.
        steps = [
            ExperimentStep(iteration=i, target_position=float(p), rationale=out.rationale,
                           acquisition_strategy="planner_initial_design")
            for i, p in enumerate(out.initial_design, 1)
        ]
        return hypothesis, steps

    # --- next point --------------------------------------------------------------
    def decide_next(self, state: PlanningState, step: int,
                    purpose: str = "strategy") -> Optional[Tuple[StrategyDecision, LLMCallRecord]]:
        """GP or own point, with a reason. None means no LLM: the caller uses the GP (Phase 1 behaviour)."""
        return self._ask(STRATEGY_SYSTEM, strategy_payload(state), StrategyDecision, purpose, mock_strategy, step=step)

    # --- stop rationale ----------------------------------------------------------
    def explain_stop(self, decision: StopDecision, best_model: Optional[str]) -> Optional[AgentText]:
        checks = [("delta_bic", decision.delta_bic), ("max_rel_se", decision.max_rel_se), ("min_points", decision.min_points),
                  ("budget", decision.budget), ("stable_best", decision.stable_best), ("prediction", decision.prediction)]
        payload = {
            "stop_reason": decision.reason.value if decision.reason else "unknown",
            "rule_message": decision.message,
            "n_measurements": decision.n_measurements,
            "best_model": best_model,
            "criteria": [
                {"criterion": name, "value": c.value, "threshold": c.threshold, "met": c.passed, "note": c.detail}
                for name, c in checks if c is not None
            ],
        }
        asked = self._ask(STOP_SYSTEM, payload, StopRationale, "stop_rationale", mock_stop)
        if asked is None:
            return None
        out, call = asked
        return AgentText(text=out.rationale, source=_source(call), model=call.model, llm_call_id=call.id)

    # --- heuristic (provider none) -------------------------------------------------
    def formulate_hypothesis(self, research_prompt: str, view: AgentExperimentView) -> Hypothesis:
        """
        Structures a testable hypothesis. No functional form is assumed: the law is
        identified afterwards by blind model comparison.
        """
        indep, dep = view.independent, view.dependent
        lo, hi = view.safe_bounds
        return Hypothesis(
            title=research_prompt.strip(),
            statement=(
                f"The {dep.name} ({dep.units}) varies systematically with the {indep.name} ({indep.units}) "
                f"over the safe range [{lo:g}, {hi:g}] {indep.units}. No functional form is assumed in advance."
            ),
            target_relationship="To be identified from the data by blind model comparison",
            independent_variable=f"{indep.name} ({indep.units})",
            dependent_variable=f"{dep.name} ({dep.units})",
            suggested_range=[lo, hi],
        )

    def generate_initial_plan(self, view: AgentExperimentView, num_samples: int = 4) -> List[ExperimentStep]:
        """
        Generates initial exploratory anchor coordinates spanning the safe bounds.
        """
        min_b, max_b = view.safe_bounds
        return [
            ExperimentStep(
                iteration=i,
                target_position=pos,
                rationale=f"Initial boundary survey anchor at {pos}",
                acquisition_strategy="grid_initialization",
            )
            for i, pos in enumerate(heuristic_anchors(min_b, max_b, num_samples), 1)
        ]
