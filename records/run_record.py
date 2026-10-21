"""
Run records: one JSON file per run in runs/<run_id>.json.

A record holds everything needed to audit or reproduce a run: config, seed,
experiment, data-source label, every measurement attempt with the analysis made
after it, the stop decision and timings. The `experiment` snapshot includes the
ground truth for benchmark scoring only; it is never passed to agents.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from agents.models import GPSuggestion, Hypothesis
from analysis.model_library import AnalysisResult
from analysis.stopping import StopDecision
from config import RUNS_DIR, SystemConfig
from experiments.registry import AgentExperimentView, ExperimentSpec
from hardware.base import DataSource

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

StepSource = Literal["initial_design", "gp", "random"]


class ExperimentSnapshot(BaseModel):
    name: str
    agent_view: AgentExperimentView
    noise_std: Optional[float] = Field(description="Noise std used by the simulator; None for real hardware")
    # Scoring only. Never shown to agents.
    ground_truth_model: str
    true_params: Dict[str, float]

    @classmethod
    def from_spec(cls, spec: ExperimentSpec, noise_std: Optional[float]) -> "ExperimentSnapshot":
        return cls(
            name=spec.name,
            agent_view=spec.agent_view(),
            noise_std=noise_std,
            ground_truth_model=spec.ground_truth_model,
            true_params=dict(spec.true_params),
        )


class StepRecord(BaseModel):
    index: int = Field(description="1-based attempt number (rejected attempts count)")
    source: StepSource
    requested_position: Optional[float] = Field(description="None if the request was not a finite number")
    status: Literal["measured", "rejected", "error"]
    position: Optional[float] = None
    value: Optional[float] = None
    message: str = ""
    n_points: int = Field(description="Number of measurements after this step")
    gp: Optional[GPSuggestion] = None
    analysis: Optional[AnalysisResult] = Field(default=None, description="Fits after this measurement")
    duration_s: float = 0.0


class RunResult(BaseModel):
    n_measurements: int
    final_analysis: Optional[AnalysisResult] = None
    equation_latex: Optional[str] = None


class Artifacts(BaseModel):
    record: Optional[str] = None
    plot: Optional[str] = None
    report_md: Optional[str] = None
    report_tex: Optional[str] = None


class Timings(BaseModel):
    total_s: float = 0.0
    hardware_s: float = 0.0
    analysis_s: float = 0.0
    gp_s: float = 0.0
    report_s: float = 0.0


class RunRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    seed: int
    question: str
    data_source: DataSource
    experiment: ExperimentSnapshot
    config: SystemConfig
    hypothesis: Optional[Hypothesis] = None
    steps: List[StepRecord] = Field(default_factory=list)
    stop: Optional[StopDecision] = None
    result: Optional[RunResult] = None
    artifacts: Artifacts = Field(default_factory=Artifacts)
    timings: Timings = Field(default_factory=Timings)
    error: Optional[str] = None

    def measured_points(self) -> Tuple[List[float], List[float]]:
        steps = [s for s in self.steps if s.status == "measured"]
        return [s.position for s in steps], [s.value for s in steps]


def record_path(run_id: str, runs_dir: Path = RUNS_DIR) -> Path:
    return Path(runs_dir) / f"{run_id}.json"


def save_run_record(record: RunRecord, runs_dir: Path = RUNS_DIR) -> Path:
    path = record_path(record.run_id, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Run record saved to %s", path)
    return path


def load_run_record(path_or_id: Union[str, Path], runs_dir: Path = RUNS_DIR) -> RunRecord:
    """Load a record from a file path or from a run_id in runs_dir."""
    path = Path(path_or_id)
    if not path.is_file():
        path = record_path(str(path_or_id), runs_dir)
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def list_run_records(runs_dir: Path = RUNS_DIR) -> List[Path]:
    """Record files in runs_dir, oldest first (run_ids start with a UTC timestamp)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    return sorted(runs_dir.glob("*.json"))
