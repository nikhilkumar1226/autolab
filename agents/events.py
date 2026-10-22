"""
Typed events emitted by the DiscoveryEngine.

The engine calls `on_event(event)` for every event, in order. Consumers (the CLI
console, the Phase 7 dashboard) render them; the engine never prints. The
`experiment` name in run_started is for the operator display only: events are
never fed to agents.

Phase 2 adds `strategy` (the Planner's choice before a measurement, with its reason)
and `llm_call` (one per LLM call: agent, purpose, model that answered, tokens, cost).
The full prompts and answers are in the run record, not in the events.
"""
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from agents.models import AgentText, AnalystReview, GPCurve, GPSuggestion, Hypothesis
from analysis.model_library import AnalysisResult
from analysis.stopping import StopDecision
from hardware.base import DataSource
from llm.types import LLMUsage
from records.run_record import Artifacts, RunResult, StepSource

logger = logging.getLogger(__name__)


class _EventBase(BaseModel):
    run_id: str
    seq: int
    timestamp: datetime


class RunStarted(_EventBase):
    type: Literal["run_started"] = "run_started"
    question: str
    experiment: str
    seed: int
    data_source: DataSource
    budget: int
    sampler: str
    provider: str = "none"
    llm: str = Field(default="", description="Provider and model chain, for display")
    blind: bool = False


class HypothesisProposed(_EventBase):
    type: Literal["hypothesis"] = "hypothesis"
    hypothesis: Hypothesis
    initial_design: List[float]


class MeasurementTaken(_EventBase):
    type: Literal["measurement"] = "measurement"
    step: int
    source: StepSource
    position: float
    value: float
    units_x: str
    units_y: str
    n_points: int


class SafetyRejected(_EventBase):
    type: Literal["safety_rejection"] = "safety_rejection"
    step: int
    source: StepSource
    requested_position: Optional[float]
    reason: str
    attempt_in_slot: int = 1
    fed_back_to: Optional[Literal["planner", "gp"]] = Field(
        default=None, description="Who proposes next: the Planner (with this message as an observation) or the GP fallback"
    )


class StrategyDecided(_EventBase):
    type: Literal["strategy"] = "strategy"
    step: int
    choice: Literal["gp", "self"]
    position: float = Field(description="The position that will be requested")
    reason: str
    gp_position: float
    observation: Optional[Dict[str, Any]] = Field(default=None, description="The safety rejection being answered, if any")
    model: Optional[str] = None


class LLMCalled(_EventBase):
    type: Literal["llm_call"] = "llm_call"
    call_id: int
    agent: str
    purpose: str
    step: Optional[int] = None
    provider: str
    model: Optional[str] = Field(description="The model that answered; None if every model failed")
    ok: bool
    cached: bool
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float]
    latency_s: float
    error: Optional[str] = None


class AnalysisCompleted(_EventBase):
    type: Literal["analysis"] = "analysis"
    step: int
    analysis: AnalysisResult
    stop_check: StopDecision
    review: Optional[AnalystReview] = None


class GPUpdated(_EventBase):
    type: Literal["gp_update"] = "gp_update"
    step: int
    suggestion: GPSuggestion
    curve: Optional[GPCurve] = None


class RunStopped(_EventBase):
    type: Literal["stopped"] = "stopped"
    decision: StopDecision
    result: Optional[RunResult] = None
    rationale: Optional[AgentText] = Field(default=None, description="The Planner's explanation; the decision is numerical")


class ReportReady(_EventBase):
    type: Literal["report_ready"] = "report_ready"
    artifacts: Artifacts
    llm_usage: Optional[LLMUsage] = None
    warnings: List[str] = Field(default_factory=list)


EngineEvent = Annotated[
    Union[RunStarted, HypothesisProposed, MeasurementTaken, SafetyRejected, StrategyDecided,
          LLMCalled, AnalysisCompleted, GPUpdated, RunStopped, ReportReady],
    Field(discriminator="type"),
]

EventCallback = Callable[[EngineEvent], None]


class EventEmitter:
    """Stamps events with run_id, sequence number and UTC time, and delivers them to the callback."""

    def __init__(self, run_id: str, callback: Optional[EventCallback] = None):
        self.run_id = run_id
        self._callback = callback
        self._seq = 0

    def emit(self, event_cls: type, **fields) -> _EventBase:
        self._seq += 1
        event = event_cls(run_id=self.run_id, seq=self._seq, timestamp=datetime.now(timezone.utc), **fields)
        if self._callback is not None:
            try:
                self._callback(event)
            except Exception:
                # A broken display must not abort an experiment that may be driving hardware.
                logger.exception("on_event callback failed for %s event", event.type)
        return event
