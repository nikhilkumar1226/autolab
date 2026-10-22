"""
Discovery Orchestrator.
Main controller managing the closed-loop autonomous scientific discovery lifecycle.

The engine never prints: it reports progress through typed events (agents/events.py)
and saves a JSON run record (records/run_record.py) for every run, including failed ones.

Who decides what:
- the safety guard and the simulator are built from the experiment spec, before any agent
  is asked; nothing an agent returns can change them;
- the Planner (LLM, or the heuristic with provider none) proposes the hypothesis, the initial
  design and, before each later measurement, either its own point or the GP's;
- the safety guard accepts or rejects every position. A rejection is reported, recorded and
  given back to the Planner as an observation; it may re-propose up to max_reproposals times,
  then the GP proposes once. Nothing rejected is ever resent blindly;
- the safety guard also flags suspicious readings (NaN, timeout, saturation, position mismatch,
  far outside the GP band, over the response limit). A flagged reading never enters the data.
  The Executor re-measures once; if the reading is flagged again, the Analyst (LLM, or a
  heuristic) decides to re-measure, pause or discard the position, at most max_decisions times;
- the Analyst fits the library blind, and the Analyst LLM may add one candidate model per review;
- the numerical stopping rule decides when to stop; the Planner only explains the decision;
- the Writer drafts the discussion from the record's numbers.
If every LLM model fails during the experiment, the run stops with an error (never a silent
switch to the mock). If it happens after the stop decision, the result is kept and the
missing text is flagged in the record, the report and the console.
On the digital twin, the injected faults are scored against the twin's fault log only after
the run has stopped (records/faults.py).
"""
import math
import uuid
import time
import logging
import secrets
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from config import SystemConfig, config as default_config, describe_twin
from experiments.registry import AgentExperimentView, ExperimentSpec, get_experiment
from hardware.base import BaseHardwareRig
from hardware.simulator import PhysicsSimulationRig
from hardware.serial_driver import SerialHardwareRig
from hardware.safety_guard import EmergencyStopActive, HardwareSafetyGuard, SafetyLimits, _finite_real
from optimization.bayesian_opt import ActiveLearningEngine
from optimization.baselines import GridSampler, RandomSampler
from agents.models import AgentText, AnomalyHandling, Discussion, GPSuggestion
from agents.events import (
    AnalysisCompleted,
    AnomalyDetected,
    EventCallback,
    EventEmitter,
    GPUpdated,
    HypothesisProposed,
    LLMCalled,
    MeasurementTaken,
    ReportReady,
    RunStarted,
    RunStopped,
    SafetyRejected,
    StrategyDecided,
)
from agents.lead_scientist import LeadScientistAgent, PlanningState, choose_point_payload, default_question, gap_point
from agents.executor import HardwareExecutorAgent
from agents.analyst import DataAnalystAgent
from agents.tools import LabTools
from agents.writer import WriterAgent
from analysis.candidate_models import model_for_fit
from analysis.diagnostics import max_disagreement
from analysis.model_library import AnalysisResult
from analysis.stopping import PointPrediction, StopDecision, StopReason, evaluate_stopping, predict_new_point
from llm.client import LLMClient, LLMUnavailableError
from llm.types import LLMCallRecord, LLMUsage
from records.faults import score_faults
from records.run_record import (
    ExperimentSnapshot,
    RunRecord,
    RunResult,
    StepRecord,
    StepSource,
    record_path,
    save_run_record,
)
from reporting.visualizer import ScienceVisualizer
from reporting.paper_generator import SciencePaperGenerator

logger = logging.getLogger(__name__)

# The analyst is first consulted at this many points (3-parameter models need n > 3).
MIN_POINTS_FOR_ANALYSIS = 4

def _equation(fit) -> Optional[str]:
    model = model_for_fit(fit) if fit is not None else None
    return model.equation(fit.params) if model is not None else None

@dataclass
class _Proposal:
    position: float
    source: StepSource
    suggestion: Optional[GPSuggestion] = None
    rationale: str = ""
    llm_call_ids: List[int] = field(default_factory=list)

@dataclass
class _Run:
    """Per-run state, so one engine can run several times without leaking history."""
    record: RunRecord
    events: EventEmitter
    rig: BaseHardwareRig
    guard: HardwareSafetyGuard
    tools: LabTools
    executor: HardwareExecutorAgent
    random_sampler: RandomSampler
    grid_sampler: GridSampler
    client: LLMClient
    planner: LeadScientistAgent
    analyst: DataAnalystAgent
    writer: WriterAgent
    view: AgentExperimentView  # what the LLM agents are shown (blinded with --blind)
    analysis: Optional[AnalysisResult] = None
    analyses: List[AnalysisResult] = field(default_factory=list)
    prediction: Optional[PointPrediction] = None
    last_decision: Optional[StopDecision] = None
    rejections: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_llm(self) -> bool:
        return self.client.provider != "none"

class DiscoveryEngine:
    """
    Coordinates the end-to-end closed-loop discovery:
    Hypothesize -> Actuate (through the safety guard) -> Observe -> Blind model selection
    -> Stopping rule -> Next point (Planner or GP) -> ... -> Report
    """

    def __init__(
        self,
        sys_config: Optional[SystemConfig] = None,
        on_event: Optional[EventCallback] = None,
        spec: Optional[ExperimentSpec] = None,
        rig_wrapper: Optional[Callable[[BaseHardwareRig], BaseHardwareRig]] = None,
        llm_completion: Optional[Callable[..., Any]] = None,
    ):
        self.config = sys_config or default_config
        # `spec` overrides the registry entry (tests and benchmark variants).
        self.spec = spec or get_experiment(self.config.experiment)
        self.on_event = on_event
        # Red-team hooks: a wrapper around the rig (e.g. a spy that records every command that reaches
        # it) and a stand-in for litellm.completion (a scripted adversarial model). Neither changes the guard.
        self.rig_wrapper = rig_wrapper
        self.llm_completion = llm_completion
        self._active_guard: Optional[HardwareSafetyGuard] = None

        # Reporting
        self.visualizer = ScienceVisualizer(self.config.output.plots_dir)
        self.paper_generator = SciencePaperGenerator(self.config.output.reports_dir)

    def emergency_stop(self, reason: str = "operator") -> bool:
        """Latch the E-stop of the run in progress (e.g. from the dashboard's E-STOP button). False if none runs."""
        guard = self._active_guard
        if guard is None:
            return False
        guard.emergency_stop(reason)
        return True

    def _build_rig(self, rng: np.random.Generator) -> BaseHardwareRig:
        if self.config.hardware.simulate:
            return PhysicsSimulationRig(self.spec, rng, self.config.simulation)
        return SerialHardwareRig(
            self.config.hardware,
            units_actuator=self.spec.independent.units,
            units_sensor=self.spec.dependent.units,
        )

    def run(self, question: Optional[str] = None) -> RunRecord:
        """
        Executes one seeded closed-loop experiment and returns its run record.
        """
        cfg = self.config
        view = self.spec.agent_view()
        if cfg.agent.blind:
            view = view.blinded()
        question = question or default_question(view)
        seed = cfg.seed if cfg.seed is not None else secrets.randbits(32)

        # Independent random streams, all derived from the one run seed. The first three are
        # the Phase 1 streams (spawn(n) children do not depend on n); the fourth is LLM backoff jitter.
        sim_seq, gp_seq, sampler_seq, llm_seq = np.random.SeedSequence(seed).spawn(4)
        rig = self._build_rig(np.random.default_rng(sim_seq))
        # Limits come from the spec only, and are fixed before any agent is asked anything.
        lo, hi = self.spec.safe_bounds
        guarded = self.rig_wrapper(rig) if self.rig_wrapper is not None else rig
        guard = HardwareSafetyGuard(guarded, SafetyLimits.from_spec(self.spec, cfg.safety))
        self._active_guard = guard
        optimizer = ActiveLearningEngine(min_bound=lo, max_bound=hi, random_state=int(gp_seq.generate_state(1)[0]))
        tools = LabTools(guard, optimizer, cfg.agent.anomaly)

        started = datetime.now(timezone.utc)
        run_id = f"{started:%Y%m%dT%H%M%SZ}_{self.spec.name}_s{seed}_{uuid.uuid4().hex[:4]}"
        noise_std = rig.noise_std if isinstance(rig, PhysicsSimulationRig) else None
        record = RunRecord(
            run_id=run_id,
            started_at=started,
            seed=seed,
            question=question,
            data_source=rig.data_source,
            experiment=ExperimentSnapshot.from_spec(self.spec, noise_std),
            config=cfg.model_copy(update={"seed": seed}),
        )
        events = EventEmitter(run_id, self.on_event)

        def on_call(call: LLMCallRecord) -> None:
            record.llm_calls.append(call)
            record.timings.llm_s += call.latency_s
            events.emit(
                LLMCalled, call_id=call.id, agent=call.agent, purpose=call.purpose, step=call.step,
                provider=call.provider, model=call.model, ok=call.ok, cached=call.cached,
                input_tokens=call.input_tokens, output_tokens=call.output_tokens, cost_usd=call.cost_usd,
                latency_s=call.latency_s, error=call.error,
            )

        client = LLMClient(cfg.llm, rng=np.random.default_rng(llm_seq), on_call=on_call, completion=self.llm_completion)
        run = _Run(
            record=record,
            events=events,
            rig=rig,
            guard=guard,
            tools=tools,
            executor=HardwareExecutorAgent(tools),
            random_sampler=RandomSampler(lo, hi, np.random.default_rng(sampler_seq)),
            grid_sampler=GridSampler(lo, hi),
            client=client,
            planner=LeadScientistAgent(client),
            analyst=DataAnalystAgent(
                client=client,
                max_llm_candidates=cfg.agent.max_llm_candidates,
                equivalence_sigma=cfg.agent.candidate_equivalence_sigma,
                noise_floor_rel=cfg.stopping.noise_floor_rel,
            ),
            writer=WriterAgent(client),
            view=view,
        )

        t0 = time.perf_counter()
        try:
            self._experiment_loop(run)
            self._finish(run, t0)
        except Exception as exc:
            if isinstance(exc, LLMUnavailableError):
                logger.error("Run %s stopped: %s", run_id, exc)  # expected and fully described: no traceback
            else:
                logger.exception("Run %s failed", run_id)
            record.error = f"{type(exc).__name__}: {exc}"
            record.stop = self._stop_check(run, len(tools.history_x), override=StopReason.ERROR)
            self._score_faults(run)
            record.llm_usage = LLMUsage.from_calls(record.llm_calls)
            record.finished_at = datetime.now(timezone.utc)
            record.timings.total_s = time.perf_counter() - t0
            self._save(run)
            raise
        finally:
            guard.disconnect()
            self._active_guard = None
        return record

    def _experiment_loop(self, run: _Run) -> None:
        cfg, record, tools = self.config, run.record, run.tools

        run.events.emit(
            RunStarted,
            question=record.question,
            experiment=self.spec.name,
            seed=record.seed,
            data_source=record.data_source,
            budget=cfg.stopping.max_measurements,
            sampler=cfg.agent.sampler,
            twin=describe_twin(cfg),
            provider=run.client.provider,
            llm=run.client.describe(),
            blind=cfg.agent.blind,
        )
        # Fail before anything moves if no model in the chain can be called.
        run.client.assert_ready()

        # 1. Connect and home hardware
        run.guard.connect()
        run.guard.home()

        # 2. Hypothesis and initial design from the agent-facing view only
        hypothesis, plan = run.planner.plan(record.question, run.view, cfg.agent.initial_sample_count)
        record.hypothesis = hypothesis
        run.events.emit(HypothesisProposed, hypothesis=hypothesis, initial_design=[s.target_position for s in plan])

        # 3. Measure -> analyse -> stopping rule -> next point, until a stop criterion fires
        queue: Deque[float] = deque(s.target_position for s in plan)
        failed_slots = 0
        while True:
            n = len(tools.history_x)
            if n >= cfg.stopping.max_measurements:
                decision = self._stop_check(run, n)
                break
            try:
                measured, decision = self._slot(run, queue)
            except EmergencyStopActive:
                measured, decision = False, None
            if run.guard.estop_active:
                decision = self._stop_check(run, len(tools.history_x), override=StopReason.SAFETY_ABORT).model_copy(
                    update={"message": f"aborted: E-stop latched ({run.guard.estop_reason})"})
                break
            if measured:
                failed_slots = 0
                if decision is not None and decision.should_stop:
                    break
                continue
            failed_slots += 1
            if failed_slots >= cfg.stopping.max_consecutive_failures:
                decision = self._stop_check(run, n, override=StopReason.SAFETY_ABORT).model_copy(update={
                    "message": f"aborted: {failed_slots} consecutive measurement slots ended without a measurement "
                               f"(rejected or failed attempts)"
                })
                break

        record.stop = decision

    # ------------------------------------------------------------------
    # One measurement slot: proposal -> guard -> (re-proposals -> GP fallback)
    # ------------------------------------------------------------------
    def _slot(self, run: _Run, queue: Deque[float]) -> Tuple[bool, Optional[StopDecision]]:
        """
        One intended measurement. Returns (measured, stop decision of the analysis after it).
        After a safety rejection the Planner is given the guard's message and may re-propose up
        to max_reproposals times; then the GP proposes once. Without an LLM there is no one to
        tell, so a rejection ends the slot (Phase 1 behaviour).
        """
        max_reproposals = self.config.agent.max_reproposals
        proposal = self._first_proposal(run, queue, len(run.record.steps) + 1)
        reproposals, gp_fallback_used, attempt = 0, False, 1
        while True:
            step, decision = self._attempt(run, proposal, attempt)
            if step.status == "measured":
                return True, decision
            if step.status == "anomaly":
                return self._recover(run, step)
            if step.status != "rejected":
                return False, None  # actuator or driver error: no re-proposal

            if run.guard.estop_active:
                return False, None  # nothing moves until an operator resets it; the loop aborts the run
            observation = {"rejected_position": step.requested_position, "guard_message": step.message, "attempt": attempt}
            run.rejections.append(observation)
            if run.has_llm and reproposals < max_reproposals:
                next_by = "planner"
            elif run.has_llm and not gp_fallback_used:
                next_by = "gp"  # in LLM-only mode: the no-LLM rule instead (there is no GP)
            else:
                next_by = None
            run.events.emit(
                SafetyRejected, step=step.index, source=step.source, requested_position=step.requested_position,
                reason=step.message, attempt_in_slot=attempt, fed_back_to=next_by,
            )
            if next_by is None:
                return False, None

            attempt += 1
            index = len(run.record.steps) + 1
            if next_by == "planner":
                reproposals += 1
                proposal = (self._llm_only_proposal(run, index, observation) if self.config.agent.sampler == "llm"
                            else self._planner_proposal(run, index, observation))
            elif self.config.agent.sampler == "llm":
                gp_fallback_used = True
                rule = gap_point(choose_point_payload(self._planning_state(run, None, None)))
                proposal = _Proposal(rule["position"], "heuristic",
                                     rationale=f"rule fallback after {max_reproposals} rejected re-proposal(s): {rule['reason']}")
            else:
                gp_fallback_used = True
                suggestion = self._gp_suggestion(run, index)
                proposal = _Proposal(
                    suggestion.position, "gp", suggestion,
                    rationale=f"GP fallback after {max_reproposals} rejected re-proposal(s)",
                )

    # ------------------------------------------------------------------
    # Flagged readings: re-measure, then the Analyst decides
    # ------------------------------------------------------------------
    def _recover(self, run: _Run, step: StepRecord) -> Tuple[bool, Optional[StopDecision]]:
        """
        Handle a flagged reading at one position. The Executor re-measures once; every later flag
        goes to the Analyst (remeasure / pause / discard), at most max_decisions times, then the
        position is discarded. A reading over the response limit is discarded at once (the guard
        has already backed off and refuses that position).
        """
        cfg = self.config.agent.anomaly
        position = step.requested_position
        readings: List[Optional[float]] = []
        decisions = 0
        while True:
            readings.append(step.value)
            if "response_limit" in step.flags:
                handling = AnomalyHandling(
                    action="discard", decided_by="safety_guard",
                    reason="the reading exceeded the hard response limit; the guard backed the actuator off and "
                           "refuses this position from now on",
                )
            elif len(readings) == 1:
                handling = AnomalyHandling(action="remeasure", decided_by="executor",
                                           reason="automatic re-measurement after the first flagged reading")
            elif decisions >= cfg.max_decisions:
                handling = AnomalyHandling(
                    action="discard", decided_by="executor",
                    reason=f"still flagged after {decisions} Analyst decision(s); the position is given up",
                )
            else:
                decisions += 1
                handling = run.analyst.decide_anomaly(self._anomaly_payload(run, step, readings, decisions), step=step.index)
                if handling.llm_call_id is not None:
                    step.llm_call_ids.append(handling.llm_call_id)
            step.anomaly = handling
            run.anomalies.append({"position": position, "flags": step.flags, "action": handling.action})
            run.events.emit(
                AnomalyDetected, step=step.index, source=step.source, position=position, value=step.value,
                flags=step.flags, expected=step.expected, handling=handling,
            )
            if handling.action == "discard":
                run.tools.discard(position)
                return False, None
            if handling.action == "pause":
                run.guard.pause(cfg.pause_s)
            confirm = step.value if step.flags == ["outlier"] else None
            proposal = _Proposal(position, "remeasure",
                                 rationale=f"re-measurement after {', '.join(step.flags)} ({handling.decided_by}: {handling.action})")
            step, decision = self._attempt(run, proposal, step.attempt_in_slot + 1, confirm=confirm)
            if step.status == "measured":
                return True, decision
            if step.status != "anomaly":
                return False, None

    def _anomaly_payload(self, run: _Run, step: StepRecord, readings: List[Optional[float]], decisions: int) -> Dict[str, Any]:
        view = run.view
        return {
            "variables": {"x": {"name": view.independent.name, "units": view.independent.units},
                          "y": {"name": view.dependent.name, "units": view.dependent.units}},
            "position": step.requested_position,
            "flags": step.flags,
            "readings_here": readings,
            "expected": [b.model_dump() for b in step.expected],
            "position_feedback": step.position_feedback,
            "n_measurements": len(run.tools.history_x),
            "decisions_left": self.config.agent.anomaly.max_decisions - decisions,
        }

    def _first_proposal(self, run: _Run, queue: Deque[float], index: int) -> _Proposal:
        if queue:
            h = run.record.hypothesis
            ids = [h.llm_call_id] if h is not None and h.llm_call_id is not None else []
            return _Proposal(queue.popleft(), "initial_design", llm_call_ids=ids)
        sampler, tools = self.config.agent.sampler, run.tools
        if sampler == "random":
            return _Proposal(run.random_sampler.suggest(tools.reachable_window()), "random")
        if sampler == "grid":
            position = run.grid_sampler.suggest(
                tools.history_x, tools.reachable_window(), exclude=[*tools.rejected_positions, *tools.discarded_positions],
            )
            return _Proposal(position, "grid")
        if sampler == "llm":
            return self._llm_only_proposal(run, index, observation=None)
        return self._planner_proposal(run, index, observation=None)

    def _llm_only_proposal(self, run: _Run, index: int, observation: Optional[Dict[str, Any]]) -> _Proposal:
        """LLM-only mode: the Planner chooses the point itself, with no GP. Without an LLM: the gap rule."""
        state = self._planning_state(run, None, observation)
        decided = run.planner.choose_point(state, step=index, purpose="choose_point" if observation is None else "repropose")
        if decided is None:
            rule = gap_point(choose_point_payload(state))
            return _Proposal(rule["position"], "heuristic", rationale=rule["reason"])
        decision, call = decided
        run.events.emit(
            StrategyDecided, step=index, choice="self", position=decision.position, reason=decision.reason,
            gp_position=None, observation=observation, model=call.model,
        )
        return _Proposal(decision.position, "planner", None, decision.reason, [call.id])

    def _planner_proposal(self, run: _Run, index: int, observation: Optional[Dict[str, Any]]) -> _Proposal:
        """The GP suggests; the Planner keeps it or picks its own point. Without an LLM the GP decides."""
        suggestion = self._gp_suggestion(run, index)
        decided = run.planner.decide_next(
            self._planning_state(run, suggestion, observation), step=index,
            purpose="strategy" if observation is None else "repropose",
        )
        if decided is None:
            return _Proposal(suggestion.position, "gp", suggestion)
        decision, call = decided
        own = decision.choice == "self"
        position = decision.position if own else suggestion.position
        run.events.emit(
            StrategyDecided, step=index, choice=decision.choice, position=position, reason=decision.reason,
            gp_position=suggestion.position, observation=observation, model=call.model,
        )
        return _Proposal(position, "planner" if own else "gp", suggestion, decision.reason, [call.id])

    def _planning_state(self, run: _Run, suggestion: Optional[GPSuggestion],
                        observation: Optional[Dict[str, Any]]) -> PlanningState:
        tools, analysis = run.tools, run.analysis
        disagreement = None
        if analysis is not None and analysis.best is not None and analysis.runner_up is not None:
            floor = self.config.stopping.noise_floor_rel * float(np.ptp(tools.history_y))
            disagreement = max_disagreement(
                analysis.best, analysis.runner_up, self.spec.safe_bounds, max(analysis.best.residual_std, floor),
            )
        return PlanningState(
            view=run.view,
            x=list(tools.history_x),
            y=list(tools.history_y),
            analysis=analysis,
            stop_check=run.last_decision,
            gp=suggestion,
            disagreement=disagreement,
            budget_left=self.config.stopping.max_measurements - len(tools.history_x),
            reachable=tools.reachable_window(),
            recent_rejections=[r for r in run.rejections if r is not observation][-3:],
            recent_anomalies=run.anomalies[-3:],
            observation=observation,
        )

    def _gp_suggestion(self, run: _Run, index: int) -> GPSuggestion:
        t_gp = time.perf_counter()
        suggestion = run.tools.tool_suggest_next_point(strategy="uncertainty")
        curve = run.tools.gp_curve()
        run.record.timings.gp_s += time.perf_counter() - t_gp
        run.events.emit(GPUpdated, step=index, suggestion=suggestion, curve=curve)
        return suggestion

    def _attempt(self, run: _Run, proposal: _Proposal, attempt: int,
                 confirm: Optional[float] = None) -> Tuple[StepRecord, Optional[StopDecision]]:
        """Send one position through the guard and record it; analyse after a measurement."""
        cfg, record, tools = self.config, run.record, run.tools
        index = len(record.steps) + 1
        position = proposal.position
        t_step = time.perf_counter()

        t_hw = time.perf_counter()
        outcome = run.executor.execute_step(position, rationale=proposal.source, confirm=confirm)
        record.timings.hardware_s += time.perf_counter() - t_hw

        step = StepRecord(
            index=index,
            source=proposal.source,
            requested_position=_finite_real(position),
            status=outcome.status,
            position=outcome.position,
            value=outcome.value,
            message=outcome.message,
            n_points=len(tools.history_x),
            gp=proposal.suggestion,
            rationale=proposal.rationale,
            attempt_in_slot=attempt,
            llm_call_ids=list(proposal.llm_call_ids),
            flags=outcome.flags,
            expected=outcome.expected,
            rig_cycle=outcome.cycle,
            position_feedback=outcome.position_feedback,
        )
        decision: Optional[StopDecision] = None
        if outcome.status == "measured":
            run.events.emit(
                MeasurementTaken,
                step=index, source=proposal.source, position=outcome.position, value=outcome.value,
                units_x=self.spec.independent.units, units_y=self.spec.dependent.units,
                n_points=step.n_points,
            )
            # The fit made before this measurement predicts it; only then is the point analysed.
            run.prediction = step.prediction = predict_new_point(
                run.analysis, outcome.position, outcome.value, tools.history_y[:-1], cfg.stopping
            )
            if step.n_points >= MIN_POINTS_FOR_ANALYSIS:
                t_an = time.perf_counter()
                analysis = run.analyst.analyze(tools.history_x, tools.history_y, iteration=index)
                record.timings.analysis_s += time.perf_counter() - t_an
                # The Analyst LLM reads the fits and may add a candidate before the stop check.
                analysis, review = run.analyst.review(analysis, tools.history_x, tools.history_y, run.view, step=index)
                if review is not None:
                    step.review = review
                    step.llm_call_ids.append(review.llm_call_id)
                    if review.candidate is not None:
                        record.candidate_models.append(review.candidate)
                run.analysis = analysis
                run.analyses.append(analysis)
                step.analysis = analysis
                if analysis.best is not None:
                    floor = cfg.stopping.noise_floor_rel * float(np.ptp(tools.history_y))
                    tools.reference_fit = (analysis.best, max(analysis.best.residual_std, floor))
                decision = run.last_decision = self._stop_check(run, step.n_points)
                run.events.emit(AnalysisCompleted, step=index, analysis=analysis, stop_check=decision, review=review)

        step.duration_s = time.perf_counter() - t_step
        record.steps.append(step)
        return step, decision

    def _stop_check(self, run: _Run, n: int, override: Optional[StopReason] = None) -> StopDecision:
        """Evaluate the stopping rule on the latest analysis, its predecessors and the newest prediction."""
        return evaluate_stopping(
            run.analysis, n, self.config.stopping, override=override,
            previous=run.analyses[:-1], prediction=run.prediction,
        )

    # ------------------------------------------------------------------
    # After the stop decision
    # ------------------------------------------------------------------
    def _score_faults(self, run: _Run) -> None:
        """Digital twin only: compare the twin's fault log (ground truth) with what was flagged."""
        rig = run.rig
        if not isinstance(rig, PhysicsSimulationRig):
            return
        stop = run.record.stop
        run_ok = stop is not None and stop.reason in (StopReason.CONVERGED, StopReason.BUDGET_EXHAUSTED)
        run.record.faults, run.record.fault_summary = score_faults(
            run.record.steps, rig.fault_log, self.config.simulation.faults,
            last_cycle=int(rig.get_status().telemetry.get("cycle", 0)), run_ok=run_ok,
        )

    def _finish(self, run: _Run, t0: float) -> None:
        record, tools = run.record, run.tools
        self._score_faults(run)
        best = run.analysis.best if run.analysis is not None else None
        record.result = RunResult(
            n_measurements=len(tools.history_x),
            final_analysis=run.analysis,
            equation_latex=_equation(best),
        )
        record.stop_rationale = self._late_llm(
            run, "Stop rationale",
            lambda: run.planner.explain_stop(record.stop, best.name if best is not None else None),
            lambda message: AgentText(text=message, source="unavailable"),
        )
        run.events.emit(RunStopped, decision=record.stop, result=record.result, rationale=record.stop_rationale)

        # 4. Discussion, figure and Markdown/LaTeX report
        if self.config.output.write_report and tools.history_x:
            t_rep = time.perf_counter()
            record.discussion = self._late_llm(
                run, "Discussion", lambda: run.writer.write(record),
                lambda message: Discussion(source="unavailable", note=message),
            )
            self._write_report(run)
            record.timings.report_s = time.perf_counter() - t_rep

        record.llm_usage = LLMUsage.from_calls(record.llm_calls)
        record.finished_at = datetime.now(timezone.utc)
        record.timings.total_s = time.perf_counter() - t0
        self._save(run)
        run.events.emit(ReportReady, artifacts=record.artifacts, llm_usage=record.llm_usage, warnings=record.warnings,
                        faults=record.faults, fault_summary=record.fault_summary)

    def _late_llm(self, run: _Run, what: str, ask, unavailable):
        """An LLM step after the stop decision: if every model fails, keep the result and flag the gap."""
        try:
            return ask()
        except LLMUnavailableError as exc:
            message = f"{what} not generated: {exc}"
            logger.error(message)
            run.record.warnings.append(message)
            return unavailable(message)

    def _write_report(self, run: _Run) -> None:
        record, tools = run.record, run.tools
        gp_grid = gp_mu = gp_sigma = None
        if len(tools.history_x) >= 2:
            tools.optimizer.fit(np.array(tools.history_x), np.array(tools.history_y))
            gp_grid = tools.optimizer.candidate_grid
            gp_mu, gp_sigma = tools.optimizer.predict(gp_grid)

        plot_path = self.visualizer.generate_discovery_figure(record, gp_grid=gp_grid, gp_mu=gp_mu, gp_sigma=gp_sigma)
        record.artifacts.plot = str(plot_path)
        record.artifacts.report_md = str(self.paper_generator.generate_markdown_paper(record))
        record.artifacts.report_tex = str(self.paper_generator.generate_latex_paper(record))

    def _save(self, run: _Run) -> None:
        run.record.audit_log = list(run.guard.audit_entries)
        runs_dir = self.config.output.runs_dir
        run.record.artifacts.record = str(record_path(run.record.run_id, runs_dir))
        save_run_record(run.record, runs_dir)
