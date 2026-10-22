"""
System: how AutoLab is built, its safety layers and standards mapping, and the hardware.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import re

import streamlit as st

from app.lib.style import setup_page

setup_page("System")
st.title("System")

st.subheader("Architecture")
st.graphviz_chart("""
digraph autolab {
  rankdir=TB; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#f3f2ee", color="#c3c2b7", fontname="Helvetica", fontsize=11];
  edge [color="#898781", fontname="Helvetica", fontsize=9];
  ui [label="Dashboard (Streamlit) · CLI · Benchmark"];
  engine [label="DiscoveryEngine\\n(orchestrator: events, run record)", fillcolor="#e8f1fc", color="#9ec5f4"];
  llm [label="LLM layer\\nlitellm · mock · none"];
  agents [label="Agents\\nPlanner · Analyst · Writer"];
  opt [label="Optimizer\\nGP uncertainty sampling"];
  analysis [label="Blind analysis\\nmodel library · BIC · parsimony · stopping rule"];
  guard [label="HardwareSafetyGuard\\nbounds · step · rate · interlock · watchdog · E-stop · audit", fillcolor="#fbe4e4", color="#e9a3a3"];
  twin [label="Digital twin\\nnon-idealities · faults"];
  serial [label="Serial driver\\nUSB or Wokwi (RFC2217)"];
  fw [label="Arduino firmware\\nown bounds · watchdog · ESTOP"];
  ui -> engine [label="config, E-STOP"];
  engine -> ui [label="events"];
  engine -> agents; agents -> llm [label="JSON only"];
  engine -> opt; engine -> analysis;
  engine -> guard [label="every position"];
  guard -> twin; guard -> serial; serial -> fw;
}
""")
st.markdown(
    "**Each measurement:** the Planner (or the GP) proposes a position as JSON → the safety guard validates it → "
    "the rig moves and reads → the guard checks the reading → the blind analyst fits every model and ranks them by BIC → "
    "the numerical stopping rule decides → the Planner chooses the next point or hands over to the GP."
)
st.markdown(
    "- The language model never talks to the hardware, never sets bounds and never writes the simulator's physics.\n"
    "- Model expressions it proposes are parsed with an AST whitelist and are never executed.\n"
    "- Every run is saved as a JSON record, including every prompt, answer, decision and the guard's hash-chained audit log."
)

doc = (ROOT / "docs" / "safety_and_standards.md")
if doc.exists():
    sections = re.split(r"(?m)^## ", doc.read_text(encoding="utf-8"))
    by_title = {s.splitlines()[0].strip(): "## " + s for s in sections[1:]}
    st.subheader("Safety")
    for title in ("Safety layers", "Safety features", "Evidence"):
        if title in by_title:
            with st.expander(title, expanded=title != "Evidence"):
                st.markdown(by_title[title].split("\n", 1)[1])
    st.subheader("Standards (mapping, not certification)")
    standards = [t for t in by_title if t.startswith(("IEC", "ISO"))]
    for tab, title in zip(st.tabs([t.split(":")[0] for t in standards]), standards):
        with tab:
            st.markdown(by_title[title])

st.subheader("Hardware")
ino = ROOT / "firmware" / "arduino_rig" / "arduino_rig.ino"
if ino.exists():
    source = ino.read_text(encoding="utf-8")
    version = re.search(r'FIRMWARE_VERSION\[\] = "([^"]+)"', source)
    st.markdown(
        f"Arduino Uno firmware **v{version.group(1) if version else '?'}**: an SG90 servo on pin 9 turns the second "
        "polariser. The sensor is chosen at build time: the emulated sensor (Malus's law computed on the chip, for "
        "Wokwi), an LDR on A0, or a TSL2591 on I2C. The firmware has its own bounds, step limit, command watchdog and "
        "ESTOP/RESET, independent of the host's safety guard."
    )
diagram = ROOT / "diagram.json"
if diagram.exists():
    wires = json.loads(diagram.read_text(encoding="utf-8")).get("connections", [])
    st.dataframe([{"from": a, "to": b} for a, b, *_ in wires], hide_index=True)
image = ROOT / "docs" / "img" / "wokwi_circuit.png"
if image.exists():
    st.image(str(image), caption="Wokwi circuit", width="stretch")
else:
    st.caption("Circuit screenshot: run the Wokwi simulation (docs/wokwi.md) and save it as docs/img/wokwi_circuit.png.")
