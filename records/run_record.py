"""
Run records: one JSON file per run in runs/<run_id>.json.

A record holds everything needed to audit or reproduce a run: config, seed,
experiment, data-source label, every measurement attempt with the analysis made
after it, the stop decision and timings. Since schema 2 it also holds every LLM
call (prompt, raw answer, parsed answer, models tried, tokens, cost, latency), the
agents' reasons, the LLM-proposed candidate models and what happened to them, and
the Writer's discussion. The `experiment` snapshot includes the ground truth for
benchmark scoring only; it is never passed to agents.

Since schema 4 it also holds the safety guard's audit log (hash-chained).
Since schema 3 it also holds each attempt's anomaly flags and how they were handled, the
digital twin's settings (non-idealities, fault schedule) and, for the twin, the outcome of
every injected fault: detected, recovered or missed (records/faults.py).

Schema 1 and 2 records still load: every newer field has a default.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from agents.models import (
    AgentText, AnalystReview, AnomalyHandling, CandidateEvaluation, Discussion, ExpectedBand, GPSuggestion, Hypothesis,
)
from analysis.model_library import AnalysisResult
from analysis.stopping import PointPrediction, StopDecision
from config import RUNS_DIR, SystemConfig
from experiments.registry import AgentExperimentView, ExperimentSpec
from hardware.base import DataSource
from hardware.audit import AuditEntry, verify_chain
from llm.types import LLMCallRecord, LLMUsage
from records.faults import FaultOutcome, FaultSummary

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 4

# planner = a position the Planner chose itself (Phase 2); remeasure = the same position again after an anomaly
# (Phase 3); grid = the coarse-to-fine grid baseline, heuristic = the no-LLM rule of LLM-only mode (Phase 5)
StepSource = Literal["initial_design", "gp", "random", "planner", "remeasure", "grid", "heuristic"]


class ExperimentSnapshot(BaseModel):
    name: str
    agent_view: AgentExperimentView
    noise_std: Optional[float] = Field(description="Noise std used by the simulator; None for real hardware")
    # Scoring only. Never shown to agents.
    ground_truth_model: str
    true_params: Dict[str, float]
    physical_params: Optional[Dict[str, float]] = None
    law: Optional[str] = Field(default=None, description="Custom experiments: the operator's law y = f(x)")

    @classmethod
    def from_spec(cls, spec: ExperimentSpec, noise_std: Optional[float]) -> "ExperimentSnapshot":
        return cls(
            name=spec.name,
            agent_view=spec.agent_view(),
            noise_std=noise_std,
            ground_truth_model=spec.ground_truth_model,
            true_params=dict(spec.true_params),
            physical_params=dict(spec.physical_params) if spec.physical_params is not None else None,
            law=spec.law,
        )


class StepRecord(BaseModel):
    index: int = Field(description="1-based attempt number (rejected attempts count)")
    source: StepSource
    requested_position: Optional[float] = Field(description="None if the request was not a finite number")
    status: Literal["measured", "rejected", "error", "anomaly"]
    position: Optional[float] = None
    value: Optional[float] = Field(default=None, description="None if the reading was not a finite number")
    message: str = ""
    flags: List[str] = Field(default_factory=list, description="Safety-guard anomaly flags (status anomaly)")
    expected: List[ExpectedBand] = Field(default_factory=list, description="The bands the reading was checked against")
    anomaly: Optional[AnomalyHandling] = Field(default=None, description="What was done after this flagged reading")
    rig_cycle: Optional[int] = Field(default=None, description="The rig's cycle counter for this reading, if reported")
    position_feedback: Optional[float] = None
    n_points: int = Field(description="Number of measurements after this step")
    gp: Optional[GPSuggestion] = None
    prediction: Optional[PointPrediction] = Field(
        default=None, description="This measurement as predicted by the best model fitted before it"
    )
    analysis: Optional[AnalysisResult] = Field(default=None, description="Fits after this measurement")
    review: Optional[AnalystReview] = Field(default=None, description="The Analyst LLM's reading of that analysis")
    rationale: str = Field(default="", description="Why this position was requested (Planner reason, fallback note)")
    attempt_in_slot: int = Field(
        default=1, description="1 = first proposal for this measurement; 2+ = re-proposals or GP fallback after rejections"
    )
    llm_call_ids: List[int] = Field(default_factory=list, description="LLM calls behind this step (ids in llm_calls)")
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
    llm_s: float = 0.0
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
    llm_calls: List[LLMCallRecord] = Field(default_factory=list)
    llm_usage: LLMUsage = Field(default_factory=LLMUsage)
    candidate_models: List[CandidateEvaluation] = Field(default_factory=list, description="Every model the Analyst LLM proposed")
    stop_rationale: Optional[AgentText] = Field(default=None, description="The Planner's explanation; the decision is in `stop`")
    discussion: Optional[Discussion] = None
    warnings: List[str] = Field(default_factory=list, description="Problems that did not stop the run, e.g. a failed Writer call")
    faults: List[FaultOutcome] = Field(
        default_factory=list, description="Digital twin: every injected fault and random outlier, and what became of it"
    )
    fault_summary: Optional[FaultSummary] = None
    audit_log: List[AuditEntry] = Field(
        default_factory=list, description="The safety guard's append-only, hash-chained log of every request and decision"
    )

    def audit_intact(self) -> bool:
        """True if the audit log's hash chain is unbroken (no entry removed, reordered or edited)."""
        return verify_chain(self.audit_log)[0]

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
