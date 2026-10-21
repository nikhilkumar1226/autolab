"""
End-to-end integration tests for the Autonomous Scientific Discovery loop.
"""
from pathlib import Path

import pytest

from agents.orchestrator import DiscoveryEngine
from analysis.stopping import StopReason
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
