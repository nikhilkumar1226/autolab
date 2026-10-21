"""
Pydantic data models and schemas for multi-agent scientific reasoning and coordination.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Hypothesis(BaseModel):
    title: str = Field(description="Title of the hypothesis or physical investigation")
    statement: str = Field(description="Formal scientific statement of the hypothesis")
    target_relationship: str = Field(description="Mathematical expression (e.g., 'I(theta) = I0 * cos^2(theta - theta0) + C')")
    independent_variable: str = Field(default="theta", description="Controlled variable (e.g. angle, distance)")
    dependent_variable: str = Field(default="intensity", description="Observed variable (e.g. lux, voltage)")
    suggested_range: List[float] = Field(default=[0.0, 180.0], description="Domain bounds [min, max]")

class ExperimentStep(BaseModel):
    iteration: int
    target_position: float
    rationale: str
    acquisition_strategy: str = "uncertainty"

class ModelFitMetric(BaseModel):
    model_name: str
    formula: str
    r2_score: float
    rmse: float
    aic: float
    bic: float
    optimal_parameters: Dict[str, float]
    parameter_errors: Dict[str, float]

class AnalysisResult(BaseModel):
    iteration: int
    total_samples: int
    candidate_fits: List[ModelFitMetric]
    best_model: ModelFitMetric
    is_converged: bool
    convergence_rationale: str
    suggested_next_position: Optional[float] = None
    next_position_strategy: Optional[str] = None

class DiscoverySummary(BaseModel):
    experiment_id: str
    start_time: datetime
    end_time: datetime
    total_iterations: int
    hypothesis: Hypothesis
    confirmed_law: str
    fitted_equation: str
    final_r2: float
    final_rmse: float
    parameters: Dict[str, float]
    observations_x: List[float]
    observations_y: List[float]
    paper_filepath: Optional[str] = None
    plot_filepath: Optional[str] = None
