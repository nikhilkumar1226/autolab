"""
Live Lab sidebar panel for an experiment drafted from its name (experiments/drafting.py).

Flow: the operator types a name and presses "Draft"; a language model (or, without access to one, the offline
mock table) drafts the variables, units, range and law; the draft fills editable fields; the operator checks
them and ticks the confirmation box. Only then does the panel return a spec, so nothing can run on a draft the
operator has not confirmed. Editing any field is allowed and is recorded.
"""
from typing import Optional, Tuple

import streamlit as st

from config import LLMConfig
from experiments.custom import CustomLaw, build_custom_experiment
from experiments.drafting import DraftingUnavailable, DraftProvenance, draft_experiment, with_provenance
from experiments.registry import ExperimentSpec
from llm.client import LLMClient

BY_NAME = "By name (the LLM drafts it)"
BY_LAW = "Type the law yourself"
MAX_DRAFTS_PER_SESSION = 10
FIELDS = ("d_law", "d_xmin", "d_xmax", "d_xname", "d_xunits", "d_yname", "d_yunits", "d_noise")


def short_error(exc: Exception) -> str:
    text = str(exc)
    return text.splitlines()[-1] if "validation error" in text else text


def _fill(draft) -> None:
    ss = st.session_state
    law = draft.to_custom_law()
    values = (law.expression, float(law.x_min), float(law.x_max), law.x_name, law.x_units, law.y_name, law.y_units,
              float(round(law.noise_pct, 1)))
    for key, value in zip(FIELDS, values):
        ss[key] = value
    ss["d_original"] = values
    ss["d_confirm"] = False


def drafted_experiment_panel(real_llm: bool) -> Tuple[Optional[ExperimentSpec], str]:
    """Render the panel; return (spec, error). The spec is None until the operator confirms a valid draft."""
    ss = st.session_state
    ss.setdefault("drafts", 0)
    st.caption("Type the name of any single-variable experiment. The LLM drafts its variables, units, range and "
               "law; you check and confirm them; then the run is blind: the agents never see the name or the law.")
    name = st.text_input("Experiment name", key="d_name", placeholder="e.g. simple pendulum: period against length")
    use_llm = real_llm and ss.drafts < MAX_DRAFTS_PER_SESSION
    if not use_llm:
        why = (f"this session has used its {MAX_DRAFTS_PER_SESSION} drafts" if real_llm
               else "the real LLM is not unlocked (API key and app password)")
        st.caption(f"Drafting with the offline mock: {why}. The mock knows only a few textbook experiments "
                   "(pendulum, resistor, cooling, free fall, capacitor discharge, gas syringe, charges, spring).")
    if st.button("Draft the experiment", disabled=len(name.strip()) < 3, width="stretch"):
        client = LLMClient(LLMConfig(provider="litellm" if use_llm else "mock"))
        status = st.status("Drafting the experiment…", expanded=True)
        log = []

        def progress(message: str) -> None:
            log.append(message)
            status.write(message)
        try:
            result = draft_experiment(name, client, on_progress=progress)
        except DraftingUnavailable as exc:
            result = None
            ss["d_error"] = str(exc)
        ok = result is not None and result.ok
        status.update(label="Draft ready: check it below" if ok else "Could not draft it",
                      state="complete" if ok else "error", expanded=True)
        ss["d_log"] = log
        if result is not None:
            ss.drafts += use_llm
            if result.ok:
                ss["d_prov"] = result.provenance.model_dump(mode="json")
                ss.pop("d_error", None)
                _fill(result.draft)
            else:
                ss["d_error"] = result.error
    elif ss.get("d_log"):
        with st.expander("How the draft was made", expanded=False):
            for message in ss["d_log"]:
                st.write(message)
    if ss.get("d_error"):
        st.error(f"Could not draft it: {ss['d_error']}")
    prov = ss.get("d_prov")
    if not prov:
        return None, "draft an experiment first"

    who = "offline mock" if prov["provider"] == "mock" else (prov.get("model") or "LLM")
    st.markdown(f"**Draft:** {prov.get('title') or prov['drafted_from']}  \n"
                f"<small>by {who} · from “{prov['drafted_from']}”</small>", unsafe_allow_html=True)
    if prov.get("apparatus"):
        st.caption(prov["apparatus"])
    if prov.get("constants"):
        st.caption(f"Constants used: {prov['constants']}")
    st.text_input("Law  y = f(x)", key="d_law", help="Check it, and edit it if it is wrong. It is never shown to "
                  "the agents and is compiled by the same whitelist as any typed law.")
    c1, c2 = st.columns(2)
    c1.number_input("x from", key="d_xmin")
    c2.number_input("x to", key="d_xmax")
    c1.text_input("x name", key="d_xname")
    c2.text_input("x units", key="d_xunits")
    c1.text_input("y name", key="d_yname")
    c2.text_input("y units", key="d_yunits")
    st.slider("Noise (% of the response span)", 0.2, 10.0, step=0.1, key="d_noise")
    confirmed = st.checkbox("I have checked the law, the range and the units", key="d_confirm")
    try:
        spec = build_custom_experiment(CustomLaw(
            expression=ss.d_law, x_min=ss.d_xmin, x_max=ss.d_xmax, x_name=ss.d_xname or "input", x_units=ss.d_xunits,
            y_name=ss.d_yname or "response", y_units=ss.d_yunits, noise_pct=ss.d_noise))
    except ValueError as exc:
        err = short_error(exc)
        st.error(f"Cannot run this draft: {err}")
        return None, err
    if not confirmed:
        st.info("Confirm the draft to run it.")
        return None, "the draft is not confirmed"
    current = tuple(ss[k] for k in FIELDS)
    edited = tuple(ss.get("d_original", ())) != current
    provenance = DraftProvenance(**{**prov, "edited_by_operator": edited, "confirmed_by_operator": True})
    return with_provenance(spec, provenance), ""
