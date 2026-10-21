"""
Discovery Orchestrator.
Main controller managing the closed-loop autonomous scientific discovery lifecycle.

The engine never prints: it reports progress through typed events (agents/events.py)
and saves a JSON run record (records/run_record.py) for every run, including failed ones.
"""
import math
import uuid
import time
import logging
import secrets
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, List, Optional, Tuple

import numpy as np

from config import SystemConfig, config as default_config
from experiments.registry import AgentExperimentView, ExperimentSpec, get_experiment
from hardware.base import BaseHardwareRig
from hardware.simulator import PhysicsSimulationRig
from hardware.serial_driver import SerialHardwareRig
from hardware.safety_guard import HardwareSafetyGuard, SafetyLimits
from optimization.bayesian_opt import ActiveLearningEngine
from optimization.baselines import RandomSampler
from agents.models import GPSuggestion
from agents.events import (
    AnalysisCompleted,
    EventCallback,
    EventEmitter,
    GPUpdated,
    HypothesisProposed,
    MeasurementTaken,
    ReportReady,
    RunStarted,
    RunStopped,
    SafetyRejected,
)
from agents.lead_scientist import LeadScientistAgent, default_question
from agents.executor import HardwareExecutorAgent
from agents.analyst import DataAnalystAgent
from agents.tools import LabTools
from analysis.model_library import MODEL_LIBRARY, AnalysisResult
from analysis.stopping import PointPrediction, StopDecision, StopReason, evaluate_stopping, predict_new_point
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

@dataclass
class _Run:
    """Per-run state, so one engine can run several times without leaking history."""
    record: RunRecord
    events: EventEmitter
    guard: HardwareSafetyGuard
    tools: LabTools
    executor: HardwareExecutorAgent
    random_sampler: RandomSampler
    analysis: Optional[AnalysisResult] = None
    analyses: List[AnalysisResult] = field(default_factory=list)
    prediction: Optional[PointPrediction] = None

class DiscoveryEngine:
    """
    Coordinates the end-to-end closed-loop discovery:
    Hypothesize -> Actuate (through the safety guard) -> Observe -> Blind model selection
    -> Stopping rule -> Active learning -> ... -> Report
    """

    def __init__(
        self,
        sys_config: Optional[SystemConfig] = None,
        on_event: Optional[EventCallback] = None,
        spec: Optional[ExperimentSpec] = None,
    ):
        self.config = sys_config or default_config
        # `spec` overrides the registry entry (tests and benchmark variants).
        self.spec = spec or get_experiment(self.config.experiment)
        self.on_event = on_event

        # Agent personas
        self.scientist = LeadScientistAgent()
        self.analyst = DataAnalystAgent()

        # Reporting
        self.visualizer = ScienceVisualizer(self.config.output.plots_dir)
        self.paper_generator = SciencePaperGenerator(self.config.output.reports_dir)

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
        question = question or default_question(view)
        seed = cfg.seed if cfg.seed is not None else secrets.randbits(32)

        # Independent random streams, all derived from the one run seed.
        sim_seq, gp_seq, sampler_seq = np.random.SeedSequence(seed).spawn(3)
        rig = self._build_rig(np.random.default_rng(sim_seq))
        lo, hi = self.spec.safe_bounds
        guard = HardwareSafetyGuard(
            rig, SafetyLimits(min_position=lo, max_position=hi, max_step_delta=self.spec.max_step_delta)
        )
        optimizer = ActiveLearningEngine(min_bound=lo, max_bound=hi, random_state=int(gp_seq.generate_state(1)[0]))
        tools = LabTools(guard, optimizer)

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
        run = _Run(
            record=record,
            events=EventEmitter(run_id, self.on_event),
            guard=guard,
            tools=tools,
            executor=HardwareExecutorAgent(tools),
            random_sampler=RandomSampler(lo, hi, np.random.default_rng(sampler_seq)),
        )

        t0 = time.perf_counter()
        try:
            self._experiment_loop(run, view)
            self._finish(run, t0)
        except Exception as exc:
            logger.exception("Run %s failed", run_id)
            record.error = f"{type(exc).__name__}: {exc}"
            record.stop = self._stop_check(run, len(tools.history_x), override=StopReason.ERROR)
            record.finished_at = datetime.now(timezone.utc)
            record.timings.total_s = time.perf_counter() - t0
            self._save(run)
            raise
        finally:
            guard.disconnect()
        return record

    def _experiment_loop(self, run: _Run, view: AgentExperimentView) -> None:
        cfg, record, tools = self.config, run.record, run.tools

        run.events.emit(
            RunStarted,
            question=record.question,
            experiment=self.spec.name,
            seed=record.seed,
            data_source=record.data_source,
            budget=cfg.stopping.max_measurements,
            sampler=cfg.agent.sampler,
        )

        # 1. Connect and home hardware
        run.guard.connect()
        run.guard.home()

        # 2. Formulate hypothesis and initial design from the agent-facing view only
        hypothesis = self.scientist.formulate_hypothesis(record.question, view)
        plan = self.scientist.generate_initial_plan(view, num_samples=cfg.agent.initial_sample_count)
        record.hypothesis = hypothesis
        run.events.emit(HypothesisProposed, hypothesis=hypothesis, initial_design=[s.target_position for s in plan])

        # 3. Measure -> analyse -> stopping rule -> next point, until a stop criterion fires
        queue: Deque[float] = deque(s.target_position for s in plan)
        failures = 0
        while True:
            n = len(tools.history_x)
            if n >= cfg.stopping.max_measurements:
                decision = self._stop_check(run, n)
                break

            index = len(record.steps) + 1
            t_step = time.perf_counter()
            position, source, suggestion = self._next_position(run, index, queue)

            t_hw = time.perf_counter()
            outcome = run.executor.execute_step(position, rationale=source)
            record.timings.hardware_s += time.perf_counter() - t_hw

            step = StepRecord(
                index=index,
                source=source,
                requested_position=position if math.isfinite(position) else None,
                status=outcome.status,
                position=outcome.position,
                value=outcome.value,
                message=outcome.message,
                n_points=len(tools.history_x),
                gp=suggestion,
            )
            decision: Optional[StopDecision] = None
            if outcome.status == "measured":
                failures = 0
                run.events.emit(
                    MeasurementTaken,
                    step=index, source=source, position=outcome.position, value=outcome.value,
                    units_x=self.spec.independent.units, units_y=self.spec.dependent.units,
                    n_points=step.n_points,
                )
                # The fit made before this measurement predicts it; only then is the point analysed.
                run.prediction = step.prediction = predict_new_point(
                    run.analysis, outcome.position, outcome.value, tools.history_y[:-1], cfg.stopping
                )
                if step.n_points >= MIN_POINTS_FOR_ANALYSIS:
                    t_an = time.perf_counter()
                    run.analysis = self.analyst.analyze(tools.history_x, tools.history_y, iteration=index)
                    record.timings.analysis_s += time.perf_counter() - t_an
                    run.analyses.append(run.analysis)
                    step.analysis = run.analysis
                    decision = self._stop_check(run, step.n_points)
                    run.events.emit(AnalysisCompleted, step=index, analysis=run.analysis, stop_check=decision)
            else:
                # A rejected command is reported, recorded and never resent blindly.
                failures += 1
                if outcome.status == "rejected":
                    run.events.emit(
                        SafetyRejected, step=index, source=source,
                        requested_position=step.requested_position, reason=outcome.message,
                    )
                if failures >= cfg.stopping.max_consecutive_failures:
                    decision = self._stop_check(
                        run, step.n_points, override=StopReason.SAFETY_ABORT
                    ).model_copy(update={"message": f"aborted after {failures} consecutive rejected or failed attempts"})

            step.duration_s = time.perf_counter() - t_step
            record.steps.append(step)
            if decision is not None and decision.should_stop:
                break

        record.stop = decision

    def _stop_check(self, run: _Run, n: int, override: Optional[StopReason] = None) -> StopDecision:
        """Evaluate the stopping rule on the latest analysis, its predecessors and the newest prediction."""
        return evaluate_stopping(
            run.analysis, n, self.config.stopping, override=override,
            previous=run.analyses[:-1], prediction=run.prediction,
        )

    def _next_position(
        self, run: _Run, index: int, queue: Deque[float]
    ) -> Tuple[float, StepSource, Optional[GPSuggestion]]:
        if queue:
            return queue.popleft(), "initial_design", None
        if self.config.agent.sampler == "random":
            return run.random_sampler.suggest(), "random", None

        t_gp = time.perf_counter()
        suggestion = run.tools.tool_suggest_next_point(strategy="uncertainty")
        curve = run.tools.gp_curve()
        run.record.timings.gp_s += time.perf_counter() - t_gp
        run.events.emit(GPUpdated, step=index, suggestion=suggestion, curve=curve)
        return suggestion.position, "gp", suggestion

    def _finish(self, run: _Run, t0: float) -> None:
        record, tools = run.record, run.tools
        best = run.analysis.best if run.analysis is not None else None
        record.result = RunResult(
            n_measurements=len(tools.history_x),
            final_analysis=run.analysis,
            equation_latex=MODEL_LIBRARY[best.key].equation(best.params) if best is not None else None,
        )
        run.events.emit(RunStopped, decision=record.stop, result=record.result)

        # 4. Figure and Markdown/LaTeX report
        if self.config.output.write_report and tools.history_x:
            t_rep = time.perf_counter()
            self._write_report(run)
            record.timings.report_s = time.perf_counter() - t_rep

        record.finished_at = datetime.now(timezone.utc)
        record.timings.total_s = time.perf_counter() - t0
        self._save(run)
        run.events.emit(ReportReady, artifacts=record.artifacts)

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
        runs_dir = self.config.output.runs_dir
        run.record.artifacts.record = str(record_path(run.record.run_id, runs_dir))
        save_run_record(run.record, runs_dir)
