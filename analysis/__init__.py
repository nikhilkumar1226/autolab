"""
Blind numerical analysis: candidate model library, fitting, ranking, the stopping rule,
safely compiled LLM-proposed models and the diagnostics handed to the agents.
"""
from analysis.model_library import (
    AnalysisResult,
    CandidateModel,
    ModelFit,
    SkippedModel,
    MODEL_LIBRARY,
    fit_library,
    fit_model,
)
from analysis.stopping import (
    CriterionCheck,
    PointPrediction,
    StopDecision,
    StopReason,
    evaluate_stopping,
    predict_new_point,
)
from analysis.candidate_models import (
    CandidateModelSpec,
    FormulaRejected,
    ParameterGuess,
    compile_candidate,
    drop_equivalent,
    model_for_fit,
    predict_fit,
)
from analysis.diagnostics import Disagreement, ResidualSummary, max_disagreement, residual_summary
from analysis.parsimony import apply_parsimony

__all__ = [
    "AnalysisResult",
    "CandidateModel",
    "ModelFit",
    "SkippedModel",
    "MODEL_LIBRARY",
    "fit_library",
    "fit_model",
    "CriterionCheck",
    "PointPrediction",
    "StopDecision",
    "StopReason",
    "evaluate_stopping",
    "predict_new_point",
    "CandidateModelSpec",
    "FormulaRejected",
    "ParameterGuess",
    "compile_candidate",
    "drop_equivalent",
    "model_for_fit",
    "predict_fit",
    "Disagreement",
    "ResidualSummary",
    "max_disagreement",
    "residual_summary",
    "apply_parsimony",
]
