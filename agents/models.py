"""
Pydantic data models and schemas for multi-agent scientific reasoning and coordination.

Model-fit and analysis schemas live in analysis/model_library.py, next to the code that produces them.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class Hypothesis(BaseModel):
    title: str = Field(description="Title of the hypothesis or physical investigation")
    statement: str = Field(description="Formal scientific statement of the hypothesis")
    target_relationship: str = Field(description="Proposed relationship, or a note that it is to be identified from data")
    independent_variable: str = Field(default="theta", description="Controlled variable (e.g. angle, distance)")
    dependent_variable: str = Field(default="intensity", description="Observed variable (e.g. lux, voltage)")
    suggested_range: List[float] = Field(default=[0.0, 180.0], description="Domain bounds [min, max]")

class ExperimentStep(BaseModel):
    iteration: int
    target_position: float
    rationale: str
    acquisition_strategy: str = "uncertainty"

class ExecutionOutcome(BaseModel):
    """Result of one measurement attempt, as seen by the agents."""
    status: Literal["measured", "rejected", "error"]
    requested_position: float
    position: Optional[float] = None
    value: Optional[float] = None
    message: str = ""

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
