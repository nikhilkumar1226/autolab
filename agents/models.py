"""
Pydantic data models and schemas for multi-agent scientific reasoning and coordination.

Two kinds of schema live here:
- LLM outputs (PlannerOutput, StrategyDecision, AnalystReport, StopRationale, WriterOutput):
  what a model must return. Numbers must be finite. A position is only ever a proposal:
  the safety guard decides.
- Record-side models (Hypothesis, AnalystReview, CandidateEvaluation, AgentText,
  Discussion): what the run record keeps, including who produced each piece of text.

Model-fit and analysis schemas live in analysis/model_library.py, next to the code that produces them.
"""
from typing import Annotated, Any, List, Literal, Optional
from pydantic import BaseModel, BeforeValidator, Field, FiniteFloat, model_validator

from analysis.candidate_models import CandidateModelSpec

TextSource = Literal["llm", "mock", "heuristic", "unavailable"]


def _json_number(value: Any) -> Any:
    """A position from a model must be a JSON number: "90", "90 deg" or true are refused, not coerced."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a JSON number (not a string, boolean, list or object)")
    return value


# A finite JSON number. Used for every position or range an LLM proposes.
Number = Annotated[FiniteFloat, BeforeValidator(_json_number)]

# ---------------------------------------------------------------------------
# LLM outputs
# ---------------------------------------------------------------------------

class HypothesisDraft(BaseModel):
    statement: str = Field(min_length=1, max_length=800, description="A testable hypothesis in one or two sentences")
    candidate_relationships: List[str] = Field(
        min_length=1, max_length=5, description="Plausible relationships between the variables, in words"
    )
    independent_variable: str = Field(min_length=1, max_length=120)
    proposed_range: List[Number] = Field(
        min_length=2, max_length=2, description="[min, max] to explore; must lie inside safe_bounds"
    )

class PlannerOutput(BaseModel):
    hypothesis: HypothesisDraft
    initial_design: List[Number] = Field(min_length=3, max_length=5, description="3-5 positions to measure first")
    rationale: str = Field(min_length=1, max_length=800)

class StrategyDecision(BaseModel):
    choice: Literal["gp", "self"] = Field(description="gp = take the GP suggestion; self = measure at `position`")
    position: Optional[Number] = Field(default=None, description="Required when choice is 'self'")
    reason: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def _position_only_for_self(self) -> "StrategyDecision":
        if self.choice == "self" and self.position is None:
            raise ValueError("position is required when choice is 'self'")
        if self.choice == "gp":
            self.position = None  # the GP's point is used; a stray number is ignored
        return self

class AnalystReport(BaseModel):
    interpretation: str = Field(min_length=1, max_length=1200)
    residuals_suggest_missing_model: bool
    residual_comment: str = Field(default="", max_length=600)
    candidate_model: Optional[CandidateModelSpec] = None

class StopRationale(BaseModel):
    rationale: str = Field(min_length=1, max_length=1200)

class WriterOutput(BaseModel):
    discussion: str = Field(min_length=1, max_length=4000)
    conclusion: str = Field(min_length=1, max_length=1500)

# ---------------------------------------------------------------------------
# Record side
# ---------------------------------------------------------------------------

class Hypothesis(BaseModel):
    title: str = Field(description="Title of the hypothesis or physical investigation")
    statement: str = Field(description="Formal scientific statement of the hypothesis")
    target_relationship: str = Field(description="Proposed relationship, or a note that it is to be identified from data")
    independent_variable: str = Field(default="theta", description="Controlled variable (e.g. angle, distance)")
    dependent_variable: str = Field(default="intensity", description="Observed variable (e.g. lux, voltage)")
    suggested_range: List[float] = Field(
        default=[0.0, 180.0], description="Range to explore, as applied: always inside the safe bounds"
    )
    candidate_relationships: List[str] = Field(default_factory=list)
    proposed_range: Optional[List[float]] = Field(default=None, description="The range the Planner proposed, before clipping")
    range_clipped: bool = False
    range_note: str = ""
    design_rationale: str = ""
    source: TextSource = "heuristic"
    model: Optional[str] = None
    llm_call_id: Optional[int] = None

class ExperimentStep(BaseModel):
    iteration: int
    target_position: float
    rationale: str
    acquisition_strategy: str = "uncertainty"

class ExpectedBand(BaseModel):
    """A predictive band at a position, used to flag outliers."""
    source: Literal["gp", "model"] = Field(
        description="gp = the GP surrogate's predictive band; model = the current best fit, sigma = its residual std (floored)"
    )
    mean: float
    sigma: float
    threshold: float = Field(default=4.0, description="Flag beyond this many sigma (normal for gp, Student-t for model)")


class ExecutionOutcome(BaseModel):
    """Result of one measurement attempt, as seen by the agents."""
    status: Literal["measured", "rejected", "error", "anomaly"]
    requested_position: Optional[float] = Field(description="None if the request was not a finite real number")
    position: Optional[float] = None
    value: Optional[float] = Field(default=None, description="None when the reading is not a finite number")
    message: str = ""
    flags: List[str] = Field(default_factory=list, description="Safety-guard anomaly flags (status anomaly)")
    expected: List[ExpectedBand] = Field(default_factory=list)
    cycle: Optional[int] = Field(default=None, description="Rig cycle counter of the reading, if the rig reports one")
    position_feedback: Optional[float] = None


class AnomalyHandling(BaseModel):
    """What was done after a flagged reading, and who decided."""
    action: Literal["remeasure", "pause", "discard"]
    decided_by: Literal["executor", "safety_guard", "analyst"] = Field(
        description="executor = the automatic first re-measurement or the decision limit; analyst = the Analyst's choice"
    )
    reason: str
    source: Optional[TextSource] = Field(default=None, description="For the analyst: llm, mock or heuristic")
    model: Optional[str] = None
    llm_call_id: Optional[int] = None


class AnomalyDecision(BaseModel):
    """The Analyst's choice after a reading was flagged twice at the same position."""
    action: Literal["remeasure", "pause", "discard"] = Field(
        description="remeasure = read again now; pause = wait, then read again; discard = give up this position"
    )
    reason: str = Field(min_length=1, max_length=600)

class GPSuggestion(BaseModel):
    position: float
    strategy: str
    max_sigma: float
    mean_sigma: float
    acquisition_score: Optional[float] = None

class GPCurve(BaseModel):
    """Downsampled GP posterior, for live display."""
    x: List[float]
    mean: List[float]
    std: List[float]

class CandidateEvaluation(BaseModel):
    """What happened to a model the Analyst LLM proposed."""
    step: int
    spec: CandidateModelSpec
    status: Literal["accepted", "rejected", "equivalent", "fit_failed", "limit_reached"]
    reason: str = ""
    key: Optional[str] = Field(default=None, description="Model key in the ranking, if accepted")
    llm_call_id: Optional[int] = None

class AnalystReview(BaseModel):
    """The Analyst LLM's reading of one analysis. The numbers themselves come from the fits."""
    interpretation: str
    residuals_suggest_missing_model: bool
    residual_comment: str = ""
    candidate: Optional[CandidateEvaluation] = None
    source: TextSource = "llm"
    model: Optional[str] = None
    llm_call_id: Optional[int] = None

class AgentText(BaseModel):
    """A piece of text written by an agent, labelled with who wrote it."""
    text: str
    source: TextSource
    model: Optional[str] = None
    llm_call_id: Optional[int] = None

class Discussion(BaseModel):
    """The Writer agent's discussion and conclusion for the report."""
    discussion: str = ""
    conclusion: str = ""
    source: TextSource
    model: Optional[str] = None
    llm_call_ids: List[int] = Field(default_factory=list)
    unverified_numbers: List[str] = Field(
        default_factory=list, description="Numbers in the text that do not appear in the run record"
    )
    note: str = ""
