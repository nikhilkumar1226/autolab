"""
A run in a background thread, so the page can redraw while the engine works and the E-STOP
button (which makes Streamlit rerun the page) can reach the running engine.

The engine's events are folded into a LiveState under a lock; the page reads snapshots. The
state holds only what the page shows: points, the GP band, the latest fits, the agents'
messages, safety events and the rig status.
"""
import copy
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.orchestrator import DiscoveryEngine
from analysis.model_library import AnalysisResult
from analysis.stopping import StopDecision
from config import SystemConfig
from records.run_record import RunRecord


@dataclass
class FeedItem:
    role: str                 # Planner, Analyst, Executor, Safety guard, Writer, System
    text: str
    step: Optional[int] = None
    by: str = ""              # which model or rule wrote it (LLM model, mock, heuristic)


@dataclass
class LiveState:
    run_id: str = ""
    experiment: str = ""
    question: str = ""
    data_source: str = ""
    seed: Optional[int] = None
    budget: int = 0
    llm: str = ""
    twin: str = ""
    x_units: str = ""
    y_units: str = ""
    points: List[Dict[str, Any]] = field(default_factory=list)       # measured points
    flagged: List[Dict[str, Any]] = field(default_factory=list)      # rejected or flagged requests
    gp: Optional[Dict[str, List[float]]] = None                      # x, mean, std
    gp_next: Optional[float] = None
    analysis: Optional[AnalysisResult] = None
    stop_check: Optional[StopDecision] = None
    feed: List[FeedItem] = field(default_factory=list)
    safety: List[FeedItem] = field(default_factory=list)
    position: Optional[float] = None
    estop: bool = False
    llm_calls: int = 0
    stopped: Optional[StopDecision] = None
    record: Optional[RunRecord] = None
    error: Optional[str] = None
    trace: str = ""
    finished: bool = False


def _by(source: Optional[str], model: Optional[str]) -> str:
    return {"llm": model or "LLM", "mock": "mock", "heuristic": "rule (no LLM)"}.get(source or "", source or "")


class LiveRun:
    def __init__(self, config: SystemConfig, question: Optional[str] = None):
        self.config = config
        self.question = question
        self.engine = DiscoveryEngine(config, on_event=self._on_event)
        self._state = LiveState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # --- control ----------------------------------------------------------------------
    def start(self) -> "LiveRun":
        self._thread = threading.Thread(target=self._work, name="autolab-run", daemon=True)
        self._thread.start()
        return self

    def _work(self) -> None:
        try:
            record = self.engine.run(question=self.question)
            with self._lock:
                self._state.record = record
        except Exception as exc:  # the engine saved the record; show the error
            with self._lock:
                self._state.error = f"{type(exc).__name__}: {exc}"
                self._state.feed.append(FeedItem("System", f"The run stopped with an error: {exc}"))
                self._state.trace = traceback.format_exc()
        finally:
            with self._lock:
                self._state.finished = True

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def emergency_stop(self) -> bool:
        """Latch the running engine's E-stop (the guard refuses everything from now on)."""
        done = self.engine.emergency_stop("operator pressed E-STOP on the dashboard")
        if done:
            with self._lock:
                self._state.estop = True
                self._state.safety.append(FeedItem("Safety guard", "E-STOP pressed: the guard latched; the run aborts."))
        return done

    def snapshot(self) -> LiveState:
        with self._lock:
            return copy.deepcopy(self._state)

    # --- events -------------------------------------------------------------------------
    def _on_event(self, e) -> None:
        with self._lock:
            handler = getattr(self, f"_on_{e.type}", None)
            if handler is not None:
                handler(self._state, e)

    def _on_run_started(self, s: LiveState, e) -> None:
        s.run_id, s.experiment, s.question = e.run_id, e.experiment, e.question
        s.data_source, s.seed, s.budget, s.llm, s.twin = e.data_source.value, e.seed, e.budget, e.llm, e.twin
        view = self.engine.spec
        s.x_units, s.y_units = view.independent.units, view.dependent.units

    def _on_hypothesis(self, s: LiveState, e) -> None:
        h = e.hypothesis
        text = f"{h.statement}\n\nInitial design: {', '.join(f'{p:g}' for p in e.initial_design)}."
        if h.design_rationale:
            text += f" {h.design_rationale}"
        if h.range_clipped:
            text += f"\n\nRange clipped: {h.range_note}. The safety limits are unchanged."
        s.feed.append(FeedItem("Planner", text, by=_by(h.source, h.model)))

    def _on_measurement(self, s: LiveState, e) -> None:
        s.points.append({"step": e.step, "x": e.position, "y": e.value, "source": e.source})
        s.position = e.position

    def _on_safety_rejection(self, s: LiveState, e) -> None:
        s.flagged.append({"step": e.step, "x": e.requested_position, "kind": "rejected"})
        then = {"planner": " The Planner is told and may re-propose.", "gp": " The GP chooses instead."}.get(e.fed_back_to or "", "")
        s.safety.append(FeedItem("Safety guard", f"Rejected {e.requested_position}: {e.reason}{then}", e.step))

    def _on_anomaly(self, s: LiveState, e) -> None:
        h = e.handling
        s.flagged.append({"step": e.step, "x": e.position, "y": e.value, "kind": "flagged"})
        value = "no reading" if e.value is None else f"{e.value:.4g} {s.y_units}"
        s.safety.append(FeedItem("Safety guard", f"Flagged reading at x = {e.position:g} ({value}): {', '.join(e.flags)}.", e.step))
        role = {"executor": "Executor", "safety_guard": "Safety guard", "analyst": "Analyst"}[h.decided_by]
        s.feed.append(FeedItem(role, f"{h.action.capitalize()}: {h.reason}", e.step, _by(h.source, h.model)))

    def _on_strategy(self, s: LiveState, e) -> None:
        what = "GP's point" if e.choice == "gp" else "own point"
        s.feed.append(FeedItem("Planner", f"Next: {what} x = {e.position:g}. {e.reason}", e.step, e.model or ""))

    def _on_llm_call(self, s: LiveState, e) -> None:
        s.llm_calls += 1
        if not e.ok:
            s.feed.append(FeedItem("System", f"LLM call failed ({e.agent}/{e.purpose}): {e.error}", e.step))

    def _on_analysis(self, s: LiveState, e) -> None:
        s.analysis, s.stop_check = e.analysis, e.stop_check
        r = e.review
        if r is not None:
            text = r.interpretation
            if r.candidate is not None:
                text += f"\n\nProposed model `{r.candidate.spec.expression}`: {r.candidate.status} ({r.candidate.reason})."
            s.feed.append(FeedItem("Analyst", text, e.step, _by(r.source, r.model)))

    def _on_gp_update(self, s: LiveState, e) -> None:
        s.gp_next = e.suggestion.position
        if e.curve is not None:
            s.gp = {"x": list(e.curve.x), "mean": list(e.curve.mean), "std": list(e.curve.std)}

    def _on_stopped(self, s: LiveState, e) -> None:
        s.stopped = e.decision
        s.feed.append(FeedItem("System", f"Stopped ({e.decision.reason.value if e.decision.reason else '?'}): {e.decision.message}"))
        if e.rationale is not None and e.rationale.text:
            s.feed.append(FeedItem("Planner", e.rationale.text, by=_by(e.rationale.source, e.rationale.model)))

    def _on_report_ready(self, s: LiveState, e) -> None:
        for w in e.warnings:
            s.feed.append(FeedItem("System", f"Warning: {w}"))


def make_config(experiment: str, seed: int, provider: str, model: str = "", blind: bool = False,
                nonideal: bool = False, faults: bool = False, budget: int = 30,
                runs_dir=None, output_dir=None) -> SystemConfig:
    """The SystemConfig for a dashboard run (digital twin). Folders can be redirected (tests)."""
    import os
    from pathlib import Path
    from config import (DEFAULT_FAULTS, AgentConfig, LLMConfig, NonIdealityConfig, OutputConfig, PLOTS_DIR,
                        REPORTS_DIR, RUNS_DIR, SimulationConfig, StoppingConfig)
    runs = Path(runs_dir or os.getenv("AUTOLAB_RUNS_DIR") or RUNS_DIR)
    out = Path(output_dir or os.getenv("AUTOLAB_OUTPUT_DIR")) if (output_dir or os.getenv("AUTOLAB_OUTPUT_DIR")) else None
    llm = LLMConfig(provider=provider) if provider != "litellm" else LLMConfig(provider="litellm")
    if provider == "litellm" and model.strip():
        llm = llm.model_copy(update={"model": model.strip()})
    return SystemConfig(
        experiment=experiment, seed=seed,
        simulation=SimulationConfig(
            transit_delay_sec=0.0, nonideal=NonIdealityConfig(enabled=nonideal),
            faults=list(DEFAULT_FAULTS) if faults else [],
        ),
        agent=AgentConfig(blind=blind),
        llm=llm,
        stopping=StoppingConfig(max_measurements=budget),
        output=OutputConfig(runs_dir=runs, plots_dir=(out / "plots") if out else PLOTS_DIR,
                            reports_dir=(out / "reports") if out else REPORTS_DIR, write_report=True),
    )
