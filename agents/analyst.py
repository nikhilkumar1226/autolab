"""
Data Analyst Agent (blind).

The analyst sees only the measured (x, y) pairs. It is never told the experiment,
its internal name or the true law. It always fits the full candidate library and
ranks the models by BIC; whether that evidence is sufficient to stop is decided
separately by analysis/stopping.py.
"""
import logging
from typing import Optional, Sequence
from analysis.model_library import MODEL_LIBRARY, AnalysisResult, CandidateModel, fit_library

logger = logging.getLogger(__name__)

class DataAnalystAgent:
    """
    Fits every candidate model, ranks them by BIC and reports delta-BIC to the runner-up.
    """

    def __init__(self, models: Optional[Sequence[CandidateModel]] = None):
        self.models = list(models) if models is not None else list(MODEL_LIBRARY.values())

    def analyze(self, x_data: Sequence[float], y_data: Sequence[float], iteration: int = 0) -> AnalysisResult:
        result = fit_library(x_data, y_data, iteration=iteration, models=self.models)
        if result.best is not None:
            logger.info(
                "Analyst [iter %d, n=%d]: best %s, delta-BIC to runner-up %s",
                iteration, result.n_points, result.best.key,
                f"{result.delta_bic_runner_up:.2f}" if result.delta_bic_runner_up is not None else "n/a",
            )
        return result
