"""
Lead Scientist / Planner Agent (heuristic).
Formulates a testable, law-neutral hypothesis and an initial exploration design
from the agent-facing experiment description only.

Phase 2 adds an LLM planner; this heuristic remains as the no-LLM fallback.
"""
import logging
from typing import List
from agents.models import Hypothesis, ExperimentStep
from experiments.registry import AgentExperimentView

logger = logging.getLogger(__name__)

def default_question(view: AgentExperimentView) -> str:
    """A neutral research question built from the variable names only."""
    return f"How does the {view.dependent.name} depend on the {view.independent.name}?"

class LeadScientistAgent:
    """
    Acts as the Principal Investigator / Planner.
    Formulates scientific hypotheses and directs the experimental methodology.
    """

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

        anchors = [min_b, max_b]
        if num_samples >= 3:
            anchors.append(round((min_b + max_b) / 2.0, 1))
        if num_samples >= 4:
            anchors.append(round(min_b + (max_b - min_b) * 0.25, 1))
        if num_samples >= 5:
            anchors.append(round(min_b + (max_b - min_b) * 0.75, 1))

        # Sort for orderly baseline survey
        anchors = sorted(list(set(anchors)))

        steps = []
        for i, pos in enumerate(anchors, 1):
            steps.append(ExperimentStep(
                iteration=i,
                target_position=pos,
                rationale=f"Initial boundary survey anchor at {pos}",
                acquisition_strategy="grid_initialization"
            ))
        return steps
