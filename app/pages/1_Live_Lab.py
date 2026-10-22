"""
Live Lab: set up a run, start it, and watch it step by step. The E-STOP button latches the
safety guard of the running engine.
"""
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.lib.access import decide, password_configured, provider_for
from app.lib.charts import discovery_chart
from app.lib.live import LiveRun, LiveState, make_config
from app.lib.style import badge, card, chip, setup_page
from app.lib.views import best_curve, criteria_table, fits_table
from experiments.registry import EXPERIMENTS, get_experiment

setup_page("Live Lab")
ss = st.session_state
ss.setdefault("runs", 0)
ss.setdefault("llm_runs", 0)

# --- settings (sidebar) ---------------------------------------------------------------------
with st.sidebar:
    st.header("Experiment")
    entered = st.text_input("App password (for the real LLM)", type="password") if password_configured() else ""
    access = decide(entered, ss.runs, ss.llm_runs)
    names = list(EXPERIMENTS)
    experiment = st.selectbox(
        "Experiment", names, index=0,
        format_func=lambda n: f"{get_experiment(n).independent.name} → {get_experiment(n).dependent.name} ({n})",
        help="The operator's view. The agents are never told the experiment's name or its law.",
    )
    spec = get_experiment(experiment)
    question = st.text_area("Research question (optional)", placeholder=f"How does the {spec.dependent.name} depend on "
                            f"the {spec.independent.name}?", height=80)
    providers = ["mock", "none"] + (["litellm"] if access.real_llm else [])
    provider_choice = st.radio(
        "Agents", providers, horizontal=True,
        format_func={"mock": "mock (offline stand-in)", "none": "no LLM", "litellm": "real LLM"}.get,
        help="mock: deterministic policies, not a language model. no LLM: rules only (the baseline).",
    )
    model = st.text_input("Model (litellm id)", value="", placeholder="default chain from settings") \
        if provider_choice == "litellm" else ""
    blind = st.toggle("Blind mode", value=False, help="Hide the apparatus and variable names from every agent.")
    nonideal = st.toggle("Non-idealities", value=False, help="Drift, saturation, quantisation, position error, backlash, outliers.")
    faults = st.toggle("Inject faults", value=False, help="A light spike, a stuck actuator and a sensor dropout.")
    budget = st.slider("Measurement budget", 8, access.max_measurements, min(20, access.max_measurements))
    random_seed = st.toggle("Random seed", value=True)
    seed = int(st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=0, step=1, disabled=random_seed))

# --- header -------------------------------------------------------------------------------------
st.title("Live Lab")
live = ss.get("live")
provider = provider_for(provider_choice, access)
header = st.empty()

c_run, c_stop, c_info = st.columns([1, 1, 4])
run_clicked = c_run.button("Run", type="primary", disabled=bool(live and live.alive) or access.max_runs <= 0,
                           width="stretch")
stop_clicked = c_stop.button("E-STOP", disabled=not (live and live.alive), width="stretch",
                             help="Latch the safety guard now: nothing moves until an operator resets it; the run aborts.")
c_info.caption(f"Runs left this session: {access.max_runs}. {access.reason.capitalize()}.")

if run_clicked and not (live and live.alive) and access.max_runs > 0:
    run_seed = secrets.randbits(31) if random_seed else seed
    cfg = make_config(experiment, run_seed, provider, model, blind, nonideal, faults, budget)
    live = LiveRun(cfg, question.strip() or None).start()
    ss.live = live
    ss.runs += 1
    ss.llm_runs += provider == "litellm"
if stop_clicked and live is not None and live.alive:
    live.emergency_stop()

# --- layout ----------------------------------------------------------------------------------
left, right = st.columns([3, 2])
with left:
    chart_slot = st.empty()
    st.markdown("**Candidate models** (ranked by BIC; lower is better)")
    fits_slot = st.empty()
    st.markdown("**Stopping criteria** (the numerical rule decides; the agents only explain)")
    criteria_slot = st.empty()
with right:
    status_slot = st.empty()
    st.markdown("**Agent reasoning**")
    feed_slot = st.empty()
    st.markdown("**Safety events**")
    safety_slot = st.empty()
result_slot = st.empty()


def render(s: LiveState, running: bool) -> None:
    source = s.data_source or "DIGITAL TWIN"
    chips = [chip(f"agents: {s.llm or provider}")]
    if s.twin:
        chips.append(chip(s.twin))
    if provider_choice == "litellm" and provider != "litellm":
        chips.append(chip("real LLM not allowed here: using mock", warn=True))
    if s.estop:
        chips.append(chip("E-STOP LATCHED", warn=True))
    header.markdown(badge(source) + "".join(chips), unsafe_allow_html=True)

    shown = get_experiment(s.experiment) if s.experiment else spec  # the running experiment, not the sidebar's
    lo, hi = shown.safe_bounds
    chart_slot.altair_chart(discovery_chart(
        s.points, s.gp, best_curve(s.analysis, lo, hi), s.flagged,
        f"{shown.independent.name} ({shown.independent.units})", f"{shown.dependent.name} ({shown.dependent.units})", [lo, hi],
    ), width="stretch")
    fits_slot.dataframe(fits_table(s.analysis), hide_index=True, width="stretch")
    criteria_slot.dataframe(criteria_table(s.stop_check), hide_index=True, width="stretch")

    with status_slot.container(border=True):
        a, b, c, d = st.columns(4)
        a.metric("Measurements", len(s.points), help=f"budget {s.budget}" if s.budget else None)
        b.metric("Position", "–" if s.position is None else f"{s.position:.4g} {s.x_units}")
        c.metric("Last reading", "–" if not s.points else f"{s.points[-1]['y']:.4g} {s.y_units}")
        d.metric("LLM calls", s.llm_calls)
        state = "E-stop latched" if s.estop else ("running" if running else ("finished" if s.finished else "idle"))
        st.caption(f"Rig: {state}" + (f" · GP next: {s.gp_next:.4g}" if s.gp_next is not None and running else "")
                   + (f" · run {s.run_id}" if s.run_id else ""))

    with feed_slot.container(height=430):
        if not s.feed:
            st.caption("The agents' messages appear here.")
        for item in s.feed[-60:]:
            st.markdown(card(item.role, item.text, item.step, item.by), unsafe_allow_html=True)
    with safety_slot.container(height=160):
        if not s.safety:
            st.caption("No rejected commands or flagged readings.")
        for item in s.safety[-30:]:
            st.markdown(card(item.role, item.text, item.step), unsafe_allow_html=True)

    if s.finished:
        with result_slot.container(border=True):
            if s.error:
                st.error(f"The run ended with an error: {s.error}")
            r = s.record
            if r is not None:
                best = r.result.final_analysis.best if r.result and r.result.final_analysis else None
                st.subheader(f"Result: {best.name if best else 'no model'}")
                st.caption(r.stop.message if r.stop else "")
                if r.result and r.result.equation_latex:
                    st.latex(r.result.equation_latex)
                if r.discussion is not None and r.discussion.discussion:
                    label = "Writer (mock template)" if r.discussion.source == "mock" else f"Writer ({r.discussion.model})"
                    st.markdown(card("Writer", f"{r.discussion.discussion}\n\n{r.discussion.conclusion}", by=label),
                                unsafe_allow_html=True)
                st.caption(f"Saved as {r.run_id}. Open it in Run History to replay it and download the record and report.")
    elif not running:
        result_slot.caption("Choose the settings in the sidebar and press Run.")


if live is None:
    render(LiveState(), running=False)
else:
    while live.alive:
        render(live.snapshot(), running=True)
        time.sleep(0.35)
    live.join()
    render(live.snapshot(), running=False)
