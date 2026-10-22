"""
Page setup and the few HTML pieces the pages share: the data-source badge and message cards.
Agent text is HTML-escaped before it is shown.
"""
import html
from typing import Optional

import streamlit as st

from app.lib.access import load_secrets

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
.badge {display:inline-block; padding:5px 12px; border-radius:6px; font-weight:650; letter-spacing:.03em;
        font-size:.95rem; border:1px solid; margin:0 8px 6px 0;}
.badge-twin {background:#e8f1fc; color:#184f95; border-color:#9ec5f4;}
.badge-fil {background:#fdeee6; color:#8a3610; border-color:#f3b69a;}
.badge-rig {background:#e6f4e6; color:#006300; border-color:#9fd39f;}
.chip {display:inline-block; padding:3px 9px; border-radius:6px; font-size:.82rem; color:#52514e;
       background:#f3f2ee; border:1px solid #e1e0d9; margin:0 6px 6px 0;}
.chip-warn {color:#8a3610; background:#fdeee6; border-color:#f3b69a;}
.card {border-left:3px solid #c3c2b7; background:#f9f9f7; padding:7px 11px; margin:0 0 8px 0; border-radius:0 6px 6px 0;}
.card .head {font-size:.8rem; color:#52514e; margin-bottom:2px;}
.card .head b {color:#0b0b0b;}
.card .body {font-size:.9rem; color:#0b0b0b; white-space:pre-wrap;}
.role-planner {border-left-color:#2a78d6;} .role-analyst {border-left-color:#eb6834;}
.role-safety {border-left-color:#d03b3b;} .role-executor {border-left-color:#898781;}
.role-system {border-left-color:#c3c2b7;} .role-writer {border-left-color:#1baf7a;}
.muted {color:#898781; font-size:.85rem;}
</style>
"""

BADGE = {"DIGITAL TWIN": "badge-twin", "FIRMWARE-IN-THE-LOOP (Wokwi)": "badge-fil", "PHYSICAL RIG": "badge-rig"}
ROLE = {"Planner": "role-planner", "Analyst": "role-analyst", "Safety guard": "role-safety", "Executor": "role-executor",
        "System": "role-system", "Writer": "role-writer"}


def setup_page(title: str) -> None:
    st.set_page_config(page_title=f"AutoLab · {title}", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        load_secrets(st.secrets)
    except Exception:  # no secrets.toml locally: .env is used instead
        pass


def badge(data_source: str) -> str:
    return f'<span class="badge {BADGE.get(data_source, "badge-twin")}">DATA SOURCE: {html.escape(data_source)}</span>'


def chip(text: str, warn: bool = False) -> str:
    return f'<span class="chip{" chip-warn" if warn else ""}">{html.escape(text)}</span>'


def card(role: str, text: str, step: Optional[int] = None, by: str = "") -> str:
    head = f"<b>{html.escape(role)}</b>"
    if step:
        head += f" · attempt {step}"
    if by:
        head += f" · {html.escape(by)}"
    return (f'<div class="card {ROLE.get(role, "role-system")}"><div class="head">{head}</div>'
            f'<div class="body">{html.escape(text)}</div></div>')
