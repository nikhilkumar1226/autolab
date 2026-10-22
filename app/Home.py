"""
AutoLab dashboard: start page. Run with `streamlit run app/Home.py`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.lib.access import decide, key_configured, password_configured
from app.lib.style import badge, chip, setup_page
from app.lib.views import RESULTS_DIR

setup_page("Home")

st.title("AutoLab")
st.markdown(
    "A benchtop **self-driving laboratory**. Language-model agents plan the experiment, a Gaussian process picks the "
    "next measurement, a hard-coded safety guard checks every command before it reaches the actuator, and a blind "
    "analyst identifies the physical law from the data alone. It then writes a report."
)

st.subheader("Pages")
c1, c2, c3, c4 = st.columns(4)
c1.markdown("**Live Lab**  \nRun an experiment and watch every measurement, model fit and agent decision as it happens.")
c2.markdown("**Run History**  \nOpen any saved run, replay it step by step, and download its record and report.")
c3.markdown("**Benchmark**  \nThe seeded comparison with grid, random and GP-only sampling, the red-team results and the fault sweep.")
c4.markdown("**System**  \nArchitecture, safety layers, the standards mapping and the hardware.")

st.subheader("Where the data comes from")
st.markdown(
    badge("DIGITAL TWIN") + badge("FIRMWARE-IN-THE-LOOP (Wokwi)") + badge("PHYSICAL RIG"),
    unsafe_allow_html=True,
)
st.markdown(
    "Every run, plot and report says which of these produced its data:\n"
    "- the **digital twin**, a seeded simulator with optional non-idealities and injected faults;\n"
    "- the **real firmware** running on a simulated Arduino in Wokwi;\n"
    "- the **physical rig**.\n\n"
    "The dashboard runs the digital twin. The firmware and the rig are driven from the command line (`--hardware`)."
)

st.subheader("This session")
access = decide("", 0, 0)
chips = [chip("API key configured" if key_configured() else "no API key: mock provider only", warn=not key_configured()),
         chip("app password set" if password_configured() else "no app password: real LLM disabled",
              warn=not password_configured())]
st.markdown("".join(chips), unsafe_allow_html=True)
st.caption(f"Live runs: {access.reason}. Mock is a deterministic stand-in for the agents, not a language model.")

if (RESULTS_DIR / "RESULTS.md").exists():
    st.caption("Benchmark results are in results/ (see the Benchmark page).")
