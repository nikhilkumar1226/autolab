"""
Tests for the typed event stream and the console callback.
"""
import io

from pydantic import TypeAdapter
from rich.console import Console

from agents.events import EngineEvent
from agents.orchestrator import MIN_POINTS_FOR_ANALYSIS, DiscoveryEngine
from cli.console import ConsoleEventPrinter
from experiments.registry import get_experiment
from records.run_record import load_run_record


def test_event_sequence_matches_the_run(make_config):
    events = []
    record = DiscoveryEngine(make_config(experiment="beer_lambert_law", seed=0), on_event=events.append).run()
    types = [e.type for e in events]
    # The Phase 1 skeleton; llm_call and strategy events are interleaved with it.
    core = [t for t in types if t not in ("llm_call", "strategy")]

    assert core[:2] == ["run_started", "hypothesis"]
    assert core[-2:] == ["stopped", "report_ready"] and types[-1] == "report_ready"
    assert types.count("stopped") == 1
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert {e.run_id for e in events} == {record.run_id}

    measured = [s for s in record.steps if s.status == "measured"]
    assert types.count("measurement") == len(measured)
    assert types.count("analysis") == sum(s.n_points >= MIN_POINTS_FOR_ANALYSIS for s in measured)
    assert types.count("gp_update") == sum(s.gp is not None for s in record.steps)
    stopped = next(e for e in events if e.type == "stopped")
    assert stopped.decision == record.stop and stopped.rationale == record.stop_rationale
    assert events[-1].artifacts.record == record.artifacts.record
    assert events[-1].llm_usage == record.llm_usage

    # One llm_call event per recorded call, and the plan call comes before the hypothesis.
    calls = [e for e in events if e.type == "llm_call"]
    assert [e.call_id for e in calls] == [c.id for c in record.llm_calls] == list(range(len(record.llm_calls)))
    assert types.index("llm_call") < types.index("hypothesis") and calls[0].purpose == "plan"
    assert types.count("strategy") == sum(c.purpose == "strategy" for c in record.llm_calls)
    assert types.count("strategy") == sum(s.source == "gp" for s in record.steps)  # mock: every later point


def test_events_round_trip_through_json(make_config):
    events = []
    DiscoveryEngine(make_config(seed=0), on_event=events.append).run()
    adapter = TypeAdapter(EngineEvent)
    for event in events:
        assert adapter.validate_json(event.model_dump_json()) == event


def test_a_failing_callback_does_not_abort_the_run(make_config):
    def broken(event):
        raise ValueError("display crashed")
    record = DiscoveryEngine(make_config(seed=0), on_event=broken).run()
    assert record.stop.reason is not None
    assert load_run_record(record.artifacts.record) == record


def test_console_printer_renders_every_event_type(make_config):
    buffer = io.StringIO()
    printer = ConsoleEventPrinter(Console(file=buffer, width=140, force_terminal=False))
    seen = set()

    def callback(event):
        seen.add(event.type)
        printer(event)

    DiscoveryEngine(make_config(seed=0, write_report=True), on_event=callback).run()
    # A spec that forbids large moves forces safety rejections.
    strict = get_experiment("malus_law").model_copy(update={"max_step_delta": 1.0})
    DiscoveryEngine(make_config(seed=0), on_event=callback, spec=strict).run()

    assert seen == {"run_started", "hypothesis", "measurement", "safety_rejection", "strategy", "llm_call",
                    "analysis", "gp_update", "stopped", "report_ready"}
    out = buffer.getvalue()
    for text in ["DATA SOURCE: DIGITAL TWIN", "[step 01]", "SAFETY REJECTION", "STOPPED (converged)",
                 "STOPPED (safety_abort)", "Blind model comparison", "Run record:"]:
        assert text in out
