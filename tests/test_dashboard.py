"""
Phase 7: the dashboard. The access policy, the live run streaming step by step and stopping on
E-STOP, the replay helpers, and every page rendered headless with Streamlit's AppTest.
"""
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from analysis.stopping import StopReason
from app.lib import access
from app.lib.live import LiveRun, make_config
from app.lib.views import (best_curve, criteria_table, fits_table, list_records, record_summary, replay, steps_frame,
                           summary_metrics)
from config import SimulationConfig

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY")


@pytest.fixture
def no_keys(monkeypatch):
    for k in (*KEYS, "APP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.fixture
def dash_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("AUTOLAB_OUTPUT_DIR", str(tmp_path / "out"))
    return tmp_path


# --- access policy ------------------------------------------------------------------------------

def test_the_real_llm_needs_a_key_a_password_and_the_right_entry(no_keys):
    assert not access.decide("", 0, 0).real_llm and "no API key" in access.decide("", 0, 0).reason
    no_keys.setenv("ANTHROPIC_API_KEY", "test-key")
    assert "no APP_PASSWORD" in access.decide("", 0, 0).reason
    no_keys.setenv("APP_PASSWORD", "s3cret")
    assert not access.decide("wrong", 0, 0).real_llm
    granted = access.decide("s3cret", 0, 0)
    assert granted.real_llm and granted.max_measurements == access.MAX_LLM_MEASUREMENTS
    assert not access.decide("s3cret", 0, access.MAX_LLM_RUNS_PER_SESSION).real_llm  # session cap
    assert access.provider_for("litellm", access.decide("wrong", 0, 0)) == "mock"
    assert access.provider_for("litellm", granted) == "litellm"
    assert access.provider_for("anything", granted) == "mock"
    assert access.decide("", access.MAX_RUNS_PER_SESSION, 0).max_runs == 0


def test_hosted_secrets_fill_the_environment_without_overriding_it(no_keys):
    no_keys.setenv("GEMINI_API_KEY", "from-env")
    access.load_secrets({"ANTHROPIC_API_KEY": "from-secrets", "GEMINI_API_KEY": "other", "UNRELATED": "x"})
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "from-secrets" and os.environ["GEMINI_API_KEY"] == "from-env"
    assert "UNRELATED" not in os.environ
    no_keys.delenv("ANTHROPIC_API_KEY")


# --- the live run ---------------------------------------------------------------------------------

def slow_config(tmp_path, **kw):
    cfg = make_config("malus_law", 0, "mock", budget=20, runs_dir=tmp_path / "runs", output_dir=tmp_path / "out", **kw)
    # A little real transit time, so the page has something to watch.
    return cfg.model_copy(update={"simulation": SimulationConfig(transit_delay_sec=0.02)})


def test_a_live_run_streams_step_by_step(tmp_path):
    live = LiveRun(slow_config(tmp_path)).start()
    seen = []
    while live.alive:
        seen.append(len(live.snapshot().points))
        time.sleep(0.02)
    live.join()
    final = live.snapshot()
    assert final.finished and final.error is None and final.record is not None
    assert len(set(seen)) >= 4 and seen == sorted(seen)  # the page saw the points arrive one by one
    assert final.data_source == "DIGITAL TWIN" and final.run_id == final.record.run_id
    roles = {item.role for item in final.feed}
    assert {"Planner", "Analyst", "System"} <= roles
    assert final.gp is not None and final.analysis is not None and final.stopped is not None
    assert (tmp_path / "runs" / f"{final.run_id}.json").exists()


def test_estop_from_the_dashboard_aborts_the_run(tmp_path):
    live = LiveRun(slow_config(tmp_path)).start()
    while live.alive and len(live.snapshot().points) < 3:
        time.sleep(0.01)
    assert live.emergency_stop()
    live.join(timeout=30)
    s = live.snapshot()
    assert s.estop and s.record.stop.reason == StopReason.SAFETY_ABORT
    assert "E-stop latched (operator pressed E-STOP on the dashboard)" in s.record.stop.message
    assert any(e.action == "estop" and e.decision == "tripped" for e in s.record.audit_log)
    assert not live.emergency_stop()  # nothing running any more


# --- views ---------------------------------------------------------------------------------------

def test_views_build_tables_curves_and_replays_from_a_record(tmp_path):
    live = LiveRun(make_config("beer_lambert_law", 1, "mock", runs_dir=tmp_path / "runs",
                               output_dir=tmp_path / "out")).start()
    live.join()
    record = live.snapshot().record
    analysis = record.result.final_analysis
    assert list(fits_table(analysis)["Model"])[0] == "Decaying exponential"
    assert set(criteria_table(record.stop)["Met"]) <= {"yes", "no"}
    curve = best_curve(analysis, 0.0, 100.0)
    assert len(curve) == 200 and curve["y"].iloc[0] > curve["y"].iloc[-1]
    first = replay(record, 4)
    assert len(first["points"]) == 4 and first["analysis"] is not None
    assert len(replay(record, record.steps[-1].index)["points"]) == record.result.n_measurements
    assert len(steps_frame(record)) == len(record.steps)
    [path] = list_records([tmp_path / "runs"])
    summary = record_summary(path)
    assert summary["experiment"] == "beer_lambert_law" and summary["best"] == "Decaying exponential"
    assert summary_metrics(record)["audit_ok"] is True


# --- pages (headless) -------------------------------------------------------------------------------

@pytest.mark.parametrize("page", ["app/Home.py", "app/pages/1_Live_Lab.py", "app/pages/2_Run_History.py",
                                  "app/pages/3_Benchmark.py", "app/pages/4_System.py"])
def test_every_page_renders(page, dash_dirs, no_keys):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60)
    at.run()
    assert not at.exception


def test_the_live_lab_page_runs_an_experiment(dash_dirs, no_keys):
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "1_Live_Lab.py"), default_timeout=120)
    at.run()
    assert at.sidebar.radio[0].options == ["mock (offline stand-in)", "no LLM"]  # no real LLM without a key
    at.sidebar.slider[0].set_value(12)
    at.button[0].click()
    at.run(timeout=120)
    assert not at.exception
    assert any("DATA SOURCE: DIGITAL TWIN" in m.value for m in at.markdown)
    assert at.subheader[0].value.startswith("Result: Cosine-squared with phase")
    assert {m.label for m in at.metric} >= {"Measurements", "Position", "Last reading", "LLM calls"}
    assert list((dash_dirs / "runs").glob("*.json"))


def test_the_live_lab_offers_the_real_llm_only_with_key_and_password(dash_dirs, no_keys):
    no_keys.setenv("ANTHROPIC_API_KEY", "test-key")
    no_keys.setenv("APP_PASSWORD", "s3cret")
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "1_Live_Lab.py"), default_timeout=60)
    at.run()
    assert "real LLM" not in at.sidebar.radio[0].options
    at.sidebar.text_input[0].set_value("s3cret")
    at.run()
    assert "real LLM" in at.sidebar.radio[0].options  # offered, and never chosen: no real call here


def test_run_history_replays_and_offers_downloads(dash_dirs, no_keys, tmp_path):
    live = LiveRun(make_config("rc_charging", 2, "mock", runs_dir=dash_dirs / "runs", output_dir=dash_dirs / "out")).start()
    live.join()
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "2_Run_History.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("rc_charging" in str(o) for o in at.selectbox[0].options)
    assert {m.label for m in at.metric} >= {"Law found", "Measurements", "Stop", "Audit log"}
    at.slider[0].set_value(4)
    at.run()
    assert not at.exception
    labels = [b.proto.label for b in at.get("download_button")]
    assert {"Run record (JSON)", "Attempts (CSV)", "Report (Markdown)"} <= set(labels)
