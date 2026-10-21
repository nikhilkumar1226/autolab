"""
Tests for JSON run records: contents, save/load round trip, listing.
"""
from pathlib import Path

import pytest

from agents.analyst import DataAnalystAgent
from agents.orchestrator import MIN_POINTS_FOR_ANALYSIS, DiscoveryEngine
from analysis.model_library import AnalysisResult, ModelFit
from analysis.stopping import StopReason
from hardware.base import DataSource
from records.run_record import RunRecord, list_run_records, load_run_record


def test_record_round_trip(make_config):
    record = DiscoveryEngine(make_config(seed=0)).run()
    path = Path(record.artifacts.record)
    assert path.exists() and path.name == f"{record.run_id}.json"

    loaded = load_run_record(path)
    assert loaded == record


def test_load_by_run_id_and_list(make_config):
    cfg = make_config(seed=1)
    first = DiscoveryEngine(cfg).run()
    second = DiscoveryEngine(make_config(experiment="beer_lambert_law", seed=2)).run()

    assert load_run_record(first.run_id, runs_dir=cfg.output.runs_dir) == first
    assert [p.stem for p in list_run_records(cfg.output.runs_dir)] == sorted([first.run_id, second.run_id])
    assert list_run_records(cfg.output.runs_dir / "missing") == []


def test_record_contents(make_config):
    record = DiscoveryEngine(make_config(experiment="inverse_square_law", seed=4)).run()

    assert record.seed == 4 and record.config.seed == 4
    assert record.data_source == DataSource.DIGITAL_TWIN
    assert record.experiment.name == "inverse_square_law"
    assert record.experiment.noise_std == 10.0
    assert record.stop is not None and record.stop.reason is not None
    assert record.finished_at is not None and record.timings.total_s > 0

    measured = [s for s in record.steps if s.status == "measured"]
    assert len(measured) == record.result.n_measurements == len(record.measured_points()[0])
    for step in measured:
        assert (step.analysis is not None) == (step.n_points >= MIN_POINTS_FOR_ANALYSIS)
    # GP steps carry the suggestion that produced them
    assert all(s.gp is not None for s in record.steps if s.source == "gp")
    assert record.result.final_analysis == measured[-1].analysis


def test_undefined_standard_error_round_trips_as_null():
    fit = ModelFit(
        key="linear", name="Linear", formula="", n_points=3, n_params=2,
        params={"a": 1.0, "b": 0.0}, std_errors={"a": None, "b": 0.1}, rel_std_errors={"a": None, "b": 0.01},
        r2=1.0, rmse=0.0, aic=-10.0, bic=-10.0,
    )
    analysis = AnalysisResult(iteration=1, n_points=3, fits=[fit])
    text = analysis.model_dump_json()
    assert '"a":null' in text
    assert AnalysisResult.model_validate_json(text) == analysis


def test_failed_run_still_saves_a_record(make_config, monkeypatch):
    def broken(self, *args, **kwargs):
        raise RuntimeError("analyst crashed")
    monkeypatch.setattr(DataAnalystAgent, "analyze", broken)

    cfg = make_config(seed=0)
    with pytest.raises(RuntimeError, match="analyst crashed"):
        DiscoveryEngine(cfg).run()

    [path] = list_run_records(cfg.output.runs_dir)
    record: RunRecord = load_run_record(path)
    assert record.stop.reason == StopReason.ERROR
    assert "analyst crashed" in record.error
    assert record.steps  # measurements before the crash are kept
