"""
Run History: open a saved run, replay it attempt by attempt, and download its record and report.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

import streamlit as st

from app.lib.charts import discovery_chart
from app.lib.style import badge, card, chip, setup_page
from app.lib.views import (DEMO_RUNS_DIR, best_curve, fits_table, law_curve, list_records, record_summary, replay,
                           steps_frame, summary_metrics)
from config import RUNS_DIR
from experiments.custom import agreement_verdict, law_agreement
from records.run_record import load_run_record

setup_page("Run History")
st.title("Run History")

runs_dir = Path(os.getenv("AUTOLAB_RUNS_DIR") or RUNS_DIR)


@st.cache_data(show_spinner=False)
def summary(path: str, mtime: float) -> dict:
    return record_summary(Path(path))


@st.cache_data(show_spinner=False, max_entries=8)
def load(path: str, mtime: float):
    return load_run_record(path)


paths = list_records([runs_dir, DEMO_RUNS_DIR])
if not paths:
    st.info("No saved runs yet. Start one on the Live Lab page, or run `python main.py` from the command line.")
    st.stop()

summaries = [summary(str(p), p.stat().st_mtime) for p in paths]
choice = st.selectbox("Run", range(len(summaries)), format_func=lambda i: summaries[i]["label"])
path = paths[choice]
try:
    record = load(str(path), path.stat().st_mtime)
except Exception as exc:
    st.error(f"This record cannot be read with the current code: {exc}")
    st.stop()

m = summary_metrics(record)
chips = [chip(f"experiment: {record.experiment.name}"), chip(f"seed {record.seed}"),
         chip(f"agents: {record.config.llm.provider}")]
if record.config.agent.blind:
    chips.append(chip("blind mode"))
if record.error:
    chips.append(chip("ended with an error", warn=True))
st.markdown(badge(record.data_source.value) + "".join(chips), unsafe_allow_html=True)

a, b, c, d, e = st.columns(5)
a.metric("Law found", m["best"])
b.metric("Measurements", m["n"])
c.metric("Stop", m["stop"])
d.metric("LLM calls", m["llm_calls"], help=f"estimated cost ${m['cost']:.4f}")
e.metric("Audit log", "intact" if m["audit_ok"] else ("not recorded" if m["audit_ok"] is None else "BROKEN"))
st.caption(m["message"])
if m["equation"]:
    st.latex(m["equation"])
if record.experiment.law:
    best_fit = record.result.final_analysis.best if record.result and record.result.final_analysis else None
    gap = law_agreement(record.experiment.law, best_fit, record.experiment.agent_view.safe_bounds,
                        record.experiment.noise_std)
    st.markdown(f"**Custom experiment.** The operator's law (never shown to the agents): `y = {record.experiment.law}`")
    (st.success if gap is not None and gap <= 3.0 else st.warning)(agreement_verdict(gap))

st.subheader("Replay")
attempts = [s.index for s in record.steps]
if not attempts:
    st.info("This run has no attempts.")
    st.stop()
upto = st.slider("Show the run up to attempt", min(attempts), max(attempts), max(attempts)) if len(attempts) > 1 else attempts[0]
state = replay(record, upto)
spec_view = record.experiment.agent_view
lo, hi = spec_view.safe_bounds
left, right = st.columns([3, 2])
with left:
    st.altair_chart(discovery_chart(
        state["points"], None, best_curve(state["analysis"], lo, hi), state["flagged"],
        f"{spec_view.independent.name} ({spec_view.independent.units})",
        f"{spec_view.dependent.name} ({spec_view.dependent.units})", [lo, hi], height=340,
        truth=law_curve(record.experiment.law, lo, hi),
    ), width="stretch")
    st.markdown("**Candidate models at this attempt**")
    st.dataframe(fits_table(state["analysis"]), hide_index=True, width="stretch")
with right:
    st.markdown("**Reasoning up to this attempt**")
    with st.container(height=470):
        for role, text, step in state["messages"]:
            st.markdown(card(role, text, step), unsafe_allow_html=True)

with st.expander("Every attempt"):
    st.dataframe(steps_frame(record), hide_index=True, width="stretch")
if record.faults:
    with st.expander("Injected faults (scored against the twin's fault log)"):
        st.dataframe([f.model_dump(include={"type", "scheduled", "at_cycle", "status", "detected", "recovered", "missed", "flags"})
                      for f in record.faults], hide_index=True, width="stretch")

st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
d1.download_button("Run record (JSON)", path.read_bytes(), file_name=path.name, mime="application/json")
d2.download_button("Attempts (CSV)", steps_frame(record).to_csv(index=False), file_name=f"{record.run_id}_steps.csv",
                   mime="text/csv")
report = record.artifacts.report_md
if report and Path(report).exists():
    d3.download_button("Report (Markdown)", Path(report).read_bytes(), file_name=Path(report).name, mime="text/markdown")
else:
    d3.caption("No report for this run (reports are written unless --no-report was used).")
