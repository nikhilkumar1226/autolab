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
from app.lib.views import law_curve
from experiments.custom import EXAMPLES, CustomLaw, agreement_verdict, build_custom_experiment, law_agreement
from experiments.registry import CUSTOM_NAME, EXPERIMENTS, get_experiment

setup_page("Live Lab")
ss = st.session_state
ss.setdefault("runs", 0)
ss.setdefault("llm_runs", 0)

# --- settings (sidebar) ---------------------------------------------------------------------
with st.sidebar:
    st.header("Experiment")
    entered = st.text_input("App password (for the real LLM)", type="password") if password_configured() else ""
    access = decide(entered, ss.runs, ss.llm_runs)
    names = [*EXPERIMENTS, CUSTOM_NAME]
    experiment = st.selectbox(
        "Experiment", names, index=0,
        format_func=lambda n: "✎ Your own law (custom experiment)" if n == CUSTOM_NAME else
        f"{get_experiment(n).independent.name} → {get_experiment(n).dependent.name} ({n})",
        help="The operator's view. The agents are never told the experiment's name or its law.",
    )
    custom_error = ""
    if experiment == CUSTOM_NAME:
        st.caption("Type any law y = f(x). The digital twin simulates it with noise; the agents see only the "
                   "variable names, units and range, never the law.")
        example = st.selectbox("Start from an example", list(EXAMPLES))
        ex_law, ex_lo, ex_hi = EXAMPLES[example]
        law_text = st.text_input(
            "Law  y = f(x)", value=ex_law, key=f"law_{example}",
            help="Numbers, x, pi, e, + - * / ** (or ^), and sin cos tan exp log sqrt abs radians. Checked by the same "
                 "whitelist as the Analyst's formulas and never run as code.")
        c1, c2 = st.columns(2)
        x_min = c1.number_input("x from", value=ex_lo, key=f"xmin_{example}")
        x_max = c2.number_input("x to", value=ex_hi, key=f"xmax_{example}")
        x_name = c1.text_input("x name", value="input")
        x_units = c2.text_input("x units", value="a.u.")
        y_name = c1.text_input("y name", value="response")
        y_units = c2.text_input("y units", value="a.u.")
        noise_pct = st.slider("Noise (% of the response span)", 0.2, 10.0, 1.0, 0.1,
                              help="Standard deviation of the simulated measurement noise.")
        try:
            spec = build_custom_experiment(CustomLaw(
                expression=law_text, x_min=x_min, x_max=x_max, x_name=x_name or "input", x_units=x_units,
                y_name=y_name or "response", y_units=y_units, noise_pct=noise_pct))
        except ValueError as exc:
            spec, custom_error = None, str(exc).splitlines()[-1] if "validation error" in str(exc) else str(exc)
            st.error(f"Cannot run this law: {custom_error}")
    else:
        spec = get_experiment(experiment)
    y_word, x_word = (spec.dependent.name, spec.independent.name) if spec is not None else ("response", "input")
    question = st.text_area("Research question (optional)", placeholder=f"How does the {y_word} depend on "
                            f"the {x_word}?", height=80)
    providers = ["mock", "none"] + (["litellm"] if access.real_llm else [])
    provider_choice = st.radio(
        "Agents", providers, horizontal=True,
        format_func={"mock": "mock (offline stand-in)", "none": "no LLM", "litellm": "real LLM"}.get,
        help="mock: deterministic policies, not a language model. no LLM: rules only (the baseline).",
    )
    if not access.real_llm:
        st.caption(f"Real LLM not offered: {access.reason}.")
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
run_clicked = c_run.button("Run", type="primary", width="stretch",
                           disabled=bool(live and live.alive) or access.max_runs <= 0 or spec is None)
stop_clicked = c_stop.button("E-STOP", disabled=not (live and live.alive), width="stretch",
                             help="Latch the safety guard now: nothing moves until an operator resets it; the run aborts.")
c_info.caption(f"Runs left this session: {access.max_runs}. {access.reason.capitalize()}.")

if run_clicked and not (live and live.alive) and access.max_runs > 0 and spec is not None:
    run_seed = secrets.randbits(31) if random_seed else seed
    cfg = make_config(experiment, run_seed, provider, model, blind, nonideal, faults, budget)
    live = LiveRun(cfg, question.strip() or None, spec=spec if experiment == CUSTOM_NAME else None).start()
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

    shown = live.engine.spec if live is not None else spec  # the running experiment, not the sidebar's
    if shown is None:
        lo, hi, x_title, y_title, law = 0.0, 1.0, "x", "y", None
    else:
        lo, hi = shown.safe_bounds
        x_title = f"{shown.independent.name} ({shown.independent.units})"
        y_title = f"{shown.dependent.name} ({shown.dependent.units})"
        law = shown.law
    chart_slot.altair_chart(discovery_chart(
        s.points, s.gp, best_curve(s.analysis, lo, hi), s.flagged, x_title, y_title, [lo, hi],
        truth=law_curve(law, lo, hi) if s.finished else None,  # the operator's law, only once the run is over
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
                if r.experiment.law:
                    gap = law_agreement(r.experiment.law, best, r.experiment.agent_view.safe_bounds, r.experiment.noise_std)
                    st.markdown(f"**Your law** (never shown to the agents): `y = {r.experiment.law}`")
                    (st.success if gap is not None and gap <= 3.0 else st.warning)(agreement_verdict(gap))
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
