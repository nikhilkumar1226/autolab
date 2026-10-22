"""
Data Analyst Agent (blind).

The numerical analyst sees only the measured (x, y) pairs. It is never told the experiment,
its internal name or the true law. It always fits the full candidate library, plus any
model the Analyst LLM proposed and that was accepted, and ranks them by BIC. Whether that
evidence is sufficient to stop is decided separately by analysis/stopping.py.

The Analyst LLM (review) reads the fit table and residual diagnostics, with the variable
names, units and data. It never sees the apparatus description, the research question or
the Planner's hypothesis. It returns an interpretation and may propose ONE new candidate
model per review. The proposal is compiled by analysis/candidate_models.py (whitelist
only, never executed), fitted with curve_fit and checked for equivalence with the existing
fits. It is then added to the ranking, or rejected with the reason logged.

After ranking, analysis/parsimony.py marks fits that reproduce a simpler model's curve with
more parameters; they stay in the table but do not count as the best model or the runner-up.
"""
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.models import AnalystReport, AnalystReview, AnomalyDecision, AnomalyHandling, CandidateEvaluation
from agents.prompts import ANALYST_SYSTEM, ANOMALY_SYSTEM, PROMPT_VERSION, to_payload
from analysis.candidate_models import CandidateModelSpec, FormulaRejected, compile_candidate, drop_equivalent
from analysis.diagnostics import residual_summary
from analysis.model_library import MODEL_LIBRARY, AnalysisResult, CandidateModel, SkippedModel, fit_library, fit_model
from analysis.parsimony import apply_parsimony
from experiments.registry import AgentExperimentView
from llm.client import LLMClient, NotAvailable

logger = logging.getLogger(__name__)

TOP_FITS_SHOWN = 6


def mock_review(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic stand-in: reads the numbers off the fit table and never proposes a model."""
    fits = payload["fits"]
    best = fits[0]
    runner = fits[1] if len(fits) > 1 else None
    text = f"{best['model']} fits best (BIC {best['bic']}, R2 {best['r2']})"
    text += f", ahead of {runner['model']} by delta-BIC {runner['delta_bic']}." if runner else "."
    residuals = payload["residuals"].get("best")
    z = residuals.get("runs_z") if residuals else None
    systematic = z is not None and z < -1.96
    if residuals is None:
        comment = "No residual summary is available."
    elif z is None:
        comment = f"Residual signs {residuals['signs']}; too few sign changes for a runs test."
    else:
        comment = (f"Residual signs {residuals['signs']} give a runs-test z of {z}: "
                   + ("a systematic pattern." if systematic else "no clear pattern."))
    return {"interpretation": f"Mock analyst: {text}", "residuals_suggest_missing_model": systematic,
            "residual_comment": comment, "candidate_model": None}


def heuristic_anomaly(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The no-LLM rule for a reading flagged twice (also the mock policy, so mock and none runs agree)."""
    flags = set(payload["flags"])
    if flags & {"non_finite", "timeout"}:
        return {"action": "pause", "reason": "No valid reading twice: likely a transient sensor dropout, so wait, then re-measure."}
    if flags & {"saturated", "out_of_range"}:
        return {"action": "discard", "reason": "The sensor is at the end of its range here, so another reading cannot give a valid value."}
    if "position_mismatch" in flags:
        return {"action": "remeasure", "reason": "The actuator did not reach the commanded position; command it again."}
    return {"action": "remeasure", "reason": "The reading disagrees with the expected band and with the earlier reading; take another."}


class DataAnalystAgent:
    """
    Fits every candidate model, ranks them by BIC and reports delta-BIC to the runner-up.
    """

    def __init__(
        self,
        models: Optional[Sequence[CandidateModel]] = None,
        client: Optional[LLMClient] = None,
        max_llm_candidates: int = 3,
        equivalence_sigma: float = 0.5,
        noise_floor_rel: float = 0.005,
    ):
        self.models = list(models) if models is not None else list(MODEL_LIBRARY.values())
        self.client = client
        self.max_llm_candidates = max_llm_candidates
        self.equivalence_sigma = equivalence_sigma
        self.noise_floor_rel = noise_floor_rel
        self.llm_models: List[CandidateModel] = []
        self.proposals: List[CandidateEvaluation] = []

    def analyze(self, x_data: Sequence[float], y_data: Sequence[float], iteration: int = 0) -> AnalysisResult:
        result = fit_library(x_data, y_data, iteration=iteration, models=[*self.models, *self.llm_models])
        if self.llm_models:
            result = drop_equivalent(result, x_data, y_data, self.equivalence_sigma, self.noise_floor_rel)
        result = apply_parsimony(result, x_data, y_data, self.equivalence_sigma, self.noise_floor_rel)
        if result.best is not None:
            logger.info(
                "Analyst [iter %d, n=%d]: best %s, delta-BIC to runner-up %s",
                iteration, result.n_points, result.best.key,
                f"{result.delta_bic_runner_up:.2f}" if result.delta_bic_runner_up is not None else "n/a",
            )
        return result

    # --- LLM review -------------------------------------------------------------
    def _payload(self, analysis: AnalysisResult, x: Sequence[float], y: Sequence[float],
                 view: AgentExperimentView) -> Dict[str, Any]:
        order = np.argsort(np.asarray(x, dtype=float), kind="stable")
        residuals = {}
        for label, fit in (("best", analysis.best), ("runner_up", analysis.runner_up)):
            summary = residual_summary(fit, x, y) if fit is not None else None
            if summary is not None:
                residuals[label] = summary.model_dump(exclude={"model"}) | {"model": fit.name}
        return {
            "variables": {
                "x": {"name": view.independent.name, "units": view.independent.units},
                "y": {"name": view.dependent.name, "units": view.dependent.units},
            },
            "data": {"x": [float(x[i]) for i in order], "y": [float(y[i]) for i in order]},
            "fits": [
                {"model": f.name, "formula": f.formula, "k": f.n_params, "params": f.params,
                 "rel_se": f.rel_std_errors, "bic": f.bic, "delta_bic": f.delta_bic, "r2": f.r2}
                for f in analysis.fits[:TOP_FITS_SHOWN]
            ],
            "not_fitted": [{"model": s.name, "reason": s.reason} for s in analysis.skipped],
            "residuals": residuals,
            "previous_proposals": [
                {"expression": p.spec.expression, "status": p.status, "reason": p.reason} for p in self.proposals
            ],
            "candidate_slots_left": max(self.max_llm_candidates - len(self.llm_models), 0),
        }

    def review(
        self, analysis: AnalysisResult, x: Sequence[float], y: Sequence[float], view: AgentExperimentView, step: int
    ) -> Tuple[AnalysisResult, Optional[AnalystReview]]:
        """
        Ask the Analyst LLM about this analysis. If it proposes a model that is accepted, the
        returned analysis includes it (re-ranked). Without an LLM the analysis is returned unchanged.
        """
        if self.client is None or analysis.best is None:
            return analysis, None
        try:
            report = self.client.complete_json(
                ANALYST_SYSTEM, to_payload(self._payload(analysis, x, y, view)), AnalystReport,
                agent="analyst", purpose="review", step=step, mock=mock_review, prompt_version=PROMPT_VERSION,
            )
        except NotAvailable:
            return analysis, None
        call = self.client.calls[-1]

        evaluation = None
        if report.candidate_model is not None:
            evaluation = self._evaluate(report.candidate_model, analysis, x, y, step, call.id)
            self.proposals.append(evaluation)
            if evaluation.status == "accepted":
                analysis = self.analyze(x, y, iteration=analysis.iteration)
        review = AnalystReview(
            interpretation=report.interpretation,
            residuals_suggest_missing_model=report.residuals_suggest_missing_model,
            residual_comment=report.residual_comment,
            candidate=evaluation,
            source="llm" if call.provider == "litellm" else "mock",
            model=call.model,
            llm_call_id=call.id,
        )
        return analysis, review

    # --- anomalies ------------------------------------------------------------------
    def decide_anomaly(self, payload: Dict[str, Any], step: int) -> AnomalyHandling:
        """Remeasure, pause or discard after a reading was flagged twice. Heuristic without an LLM."""
        if self.client is not None:
            try:
                decision = self.client.complete_json(
                    ANOMALY_SYSTEM, to_payload(payload), AnomalyDecision, agent="analyst", purpose="anomaly",
                    step=step, mock=heuristic_anomaly, prompt_version=PROMPT_VERSION,
                )
            except NotAvailable:
                pass
            else:
                call = self.client.calls[-1]
                return AnomalyHandling(
                    action=decision.action, decided_by="analyst", reason=decision.reason,
                    source="llm" if call.provider == "litellm" else "mock", model=call.model, llm_call_id=call.id,
                )
        rule = AnomalyDecision.model_validate(heuristic_anomaly(payload))
        return AnomalyHandling(action=rule.action, decided_by="analyst", reason=rule.reason, source="heuristic")

    def _evaluate(self, spec: CandidateModelSpec, analysis: AnalysisResult, x: Sequence[float], y: Sequence[float],
                  step: int, call_id: int) -> CandidateEvaluation:
        def outcome(status: str, reason: str, key: Optional[str] = None) -> CandidateEvaluation:
            log = logger.info if status == "accepted" else logger.warning
            log("Analyst candidate %r (%s): %s", spec.expression, status, reason)
            return CandidateEvaluation(step=step, spec=spec, status=status, reason=reason, key=key, llm_call_id=call_id)

        if len(self.llm_models) >= self.max_llm_candidates:
            return outcome("limit_reached", f"{self.max_llm_candidates} LLM-proposed models were already accepted")
        key = f"llm_{len(self.llm_models) + 1}"
        try:
            model = compile_candidate(spec, key)
        except FormulaRejected as exc:
            return outcome("rejected", str(exc))
        fit = fit_model(model, np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if isinstance(fit, SkippedModel):
            return outcome("fit_failed", fit.reason)
        trial = drop_equivalent(
            analysis.model_copy(update={"fits": [*analysis.fits, fit]}), x, y, self.equivalence_sigma, self.noise_floor_rel,
        )
        twin = next((s for s in trial.skipped if s.key == key), None)
        if twin is not None:
            return outcome("equivalent", twin.reason)
        self.llm_models.append(model)
        return outcome("accepted", f"fitted with BIC {fit.bic:.2f} (R2 {fit.r2:.4f}); added to the ranking as {key}", key)
