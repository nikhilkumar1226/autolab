"""
Benchmark: the figures and tables in results/, as written by the benchmark, the red team and the
twin sweep. Nothing is recomputed here.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.lib.style import badge, setup_page
from app.lib.views import RESULTS_DIR

setup_page("Benchmark")
st.title("Benchmark")
st.markdown(badge("DIGITAL TWIN"), unsafe_allow_html=True)

results_md = RESULTS_DIR / "RESULTS.md"
if not results_md.exists():
    st.info("No benchmark results yet. Run `python -m benchmark.run --quick` (about 2 minutes) or `--full`.")
    st.stop()

text = results_md.read_text(encoding="utf-8")
banner = [line[2:] for line in text.splitlines() if line.startswith("> ") and len(line) > 2]
if banner:
    st.warning("\n\n".join(banner))

FIGURES = [
    ("fig_law_accuracy", "Law identified correctly (blind), per method and condition"),
    ("fig_equal_budget_accuracy", "The same at an equal number of measurements"),
    ("fig_param_error_vs_n", "Parameter error after each measurement"),
    ("fig_fault_recovery", "What became of each injected fault"),
    ("fig_cost_per_run", "LLM calls and cost per run"),
]
tabs = st.tabs([title for _, title in FIGURES] + ["Summary table", "Red team", "Fault sweep", "RESULTS.md"])
for tab, (name, title) in zip(tabs, FIGURES):
    with tab:
        png = RESULTS_DIR / "figures" / f"{name}.png"
        if png.exists():
            st.image(str(png), width="stretch")
            pdf = png.with_suffix(".pdf")
            if pdf.exists():
                st.download_button("PDF", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf", key=f"pdf-{name}")
        else:
            st.caption("Not generated.")

with tabs[len(FIGURES)]:
    summary_csv = RESULTS_DIR / "benchmark_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        c1, c2, c3, c4 = st.columns(4)
        kind = c1.selectbox("Run kind", sorted(df["kind"].unique()), index=sorted(df["kind"].unique()).index("adaptive")
                            if "adaptive" in set(df["kind"]) else 0)
        condition = c2.selectbox("Condition", sorted(df["condition"].unique()))
        experiment = c3.selectbox("Experiment", ["ALL", *sorted(e for e in df["experiment"].unique() if e != "ALL")])
        columns = c4.multiselect("Columns", list(df.columns), default=[
            "method", "provider", "n_runs", "correct_rate", "correct_ci_low", "correct_ci_high", "n_meas_median",
            "n_meas_mean", "param_err_pct_median", "fault_recovery_rate", "llm_calls_mean", "est_cost_usd_mean"])
        view = df[(df["kind"] == kind) & (df["condition"] == condition) & (df["experiment"] == experiment)]
        st.dataframe(view[columns], hide_index=True, width="stretch")
        st.download_button("benchmark_summary.csv", summary_csv.read_bytes(), file_name=summary_csv.name, mime="text/csv")
    runs_csv = RESULTS_DIR / "benchmark_runs.csv"
    if runs_csv.exists():
        st.download_button("benchmark_runs.csv (one row per run)", runs_csv.read_bytes(), file_name=runs_csv.name,
                           mime="text/csv")

with tabs[len(FIGURES) + 1]:
    red = RESULTS_DIR / "redteam_summary.md"
    if red.exists():
        st.markdown(red.read_text(encoding="utf-8"))
        csv = RESULTS_DIR / "redteam.csv"
        if csv.exists():
            st.dataframe(pd.read_csv(csv)[["id", "category", "target", "description", "expected", "outcome",
                                           "unsafe_commands_reaching_rig", "detail"]], hide_index=True, width="stretch")
    else:
        st.caption("Run `python -m safety.redteam`.")

with tabs[len(FIGURES) + 2]:
    sweep = RESULTS_DIR / "phase3_twin_sweep.csv"
    if sweep.exists():
        df = pd.read_csv(sweep)
        table = df.groupby(["experiment", "condition"]).agg(
            runs=("correct", "size"), correct=("correct", "sum"), median_n=("n_measurements", "median"),
            false_alarms=("false_alarms", "sum")).reset_index()
        st.caption("GP sampling, no LLM, seeds 0-19: clean, faults, and faults plus non-idealities (Phase 3).")
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.caption("Run `python -m scripts.twin_sweep`.")

with tabs[len(FIGURES) + 3]:
    st.markdown(text)
