"""
End-to-end integration tests for the Autonomous Scientific Discovery loop.
"""
import io
from pathlib import Path

import pytest
from rich.console import Console

from agents.orchestrator import DiscoveryEngine
from analysis.stopping import StopReason
from cli.console import ConsoleEventPrinter
from config import StoppingConfig
from experiments.registry import get_experiment, list_experiments

@pytest.mark.parametrize("name", list_experiments())
def test_discovery_loop_identifies_each_law(make_config, name):
    record = DiscoveryEngine(make_config(experiment=name, seed=0)).run()

    assert record.result.final_analysis.best.key == get_experiment(name).ground_truth_model
    assert record.stop.reason in (StopReason.CONVERGED, StopReason.BUDGET_EXHAUSTED)
    assert record.result.n_measurements <= record.config.stopping.max_measurements
    assert Path(record.artifacts.record).exists()

def test_full_report_is_written_and_labelled(make_config):
    record = DiscoveryEngine(make_config(seed=0, write_report=True)).run(
        question="Investigate polarized light transmission as a function of filter rotation"
    )

    for path in [record.artifacts.plot, record.artifacts.report_md, record.artifacts.report_tex]:
        assert path is not None and Path(path).exists()
    markdown = Path(record.artifacts.report_md).read_text(encoding="utf-8")
    assert "Data source: DIGITAL TWIN" in markdown
    assert "Cosine-squared with phase" in markdown
    assert "Data source: DIGITAL TWIN" in Path(record.artifacts.report_tex).read_text(encoding="utf-8")

def test_predictive_stability_is_recorded_end_to_end(make_config):
    cfg = make_config(
        experiment="beer_lambert_law", seed=0, write_report=True,
        stopping=StoppingConfig(min_points=5, predictive_stability=True),
    )
    printer = ConsoleEventPrinter(Console(file=io.StringIO(), width=160, force_terminal=False))
    record = DiscoveryEngine(cfg, on_event=printer).run()

    measured = [s for s in record.steps if s.status == "measured"]
    first = next(i for i, s in enumerate(measured) if s.analysis is not None)
    assert all(s.prediction is None for s in measured[: first + 1])
    # Every later point was predicted by the best model of the analysis just before it.
    for before, step in zip(measured[first:], measured[first + 1:]):
        assert step.prediction is not None
        assert step.prediction.model == before.analysis.best.key
        assert step.prediction.n_fit == before.n_points
        assert step.prediction.x == step.position and step.prediction.observed == step.value

    stop = record.stop
    assert stop.reason == StopReason.CONVERGED
    assert stop.stable_best.value >= 2 and stop.prediction.value <= 3
    console_text = printer.console.file.getvalue()
    assert "| stable 2/2" in console_text and "| new point" in console_text
    markdown = Path(record.artifacts.report_md).read_text(encoding="utf-8")
    assert "Consecutive analyses with the same best model" in markdown
    assert "Newest point vs. earlier prediction" in markdown
