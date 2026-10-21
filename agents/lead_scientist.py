"""
Lead Scientist / Planner Agent.
Formulates scientific hypotheses, designs initial exploration plans,
and reasons about experimental convergence.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
from agents.models import Hypothesis, ExperimentStep

logger = logging.getLogger(__name__)

class LeadScientistAgent:
    """
    Acts as the Principal Investigator / Planner.
    Formulates scientific hypotheses and directs the experimental methodology.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.api_key_available = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY"))

    def formulate_hypothesis(self, research_prompt: str) -> Hypothesis:
        """
        Interprets a high-level scientific query and structures a testable hypothesis.
        """
        # If an LLM API key is present, we can call it for dynamic hypothesis generation
        prompt_lower = research_prompt.lower()
        if "polar" in prompt_lower or "angle" in prompt_lower or "malus" in prompt_lower:
            return Hypothesis(
                title="Investigation of Polarized Light Transmission via Malus's Law",
                statement="The transmitted light intensity through a linear polarizer is directly proportional to the cosine-squared of the angle relative to the polarization axis of the incident beam.",
                target_relationship="I(theta) = I_0 * cos^2(theta - theta_0) + I_ambient",
                independent_variable="Polarizer Angle (theta)",
                dependent_variable="Transmitted Light Intensity (I)",
                suggested_range=[0.0, 180.0]
            )
        elif "distance" in prompt_lower or "inverse" in prompt_lower or "square" in prompt_lower:
            return Hypothesis(
                title="Investigation of Radiant Intensity Decay with Distance",
                statement="The intensity of light radiating from a point source decreases inversely proportional to the square of the distance from the source.",
                target_relationship="I(d) = I_0 / (d + d_0)^2 + I_ambient",
                independent_variable="Distance (d)",
                dependent_variable="Radiant Intensity (I)",
                suggested_range=[5.0, 100.0]
            )
        else:
            # Default to Malus's Law
            return Hypothesis(
                title="Optical Transmission Discovery Experiment",
                statement=f"Investigation into the physical relationship underlying: '{research_prompt}'",
                target_relationship="I(x) = f(x)",
                independent_variable="Actuator Parameter (x)",
                dependent_variable="Sensor Reading (y)",
                suggested_range=[0.0, 180.0]
            )

    def generate_initial_plan(self, hypothesis: Hypothesis, num_samples: int = 4) -> List[ExperimentStep]:
        """
        Generates initial exploratory anchor coordinates spanning the domain bounds.
        """
        min_b = hypothesis.suggested_range[0]
        max_b = hypothesis.suggested_range[1]
        
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
