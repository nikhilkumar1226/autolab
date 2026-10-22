"""
Who may run what on the dashboard (public-demo protections).

A real LLM costs money, so it is allowed only when an API key is configured AND an APP_PASSWORD
is configured AND the visitor has entered it. Otherwise every run uses the mock provider and
the page says so. Runs per session and measurements per run are capped.

Settings come from Streamlit secrets when the app is hosted, else from the environment (.env).
Secret values are never shown, logged or put in a run record.
"""
import hmac
import os
from dataclasses import dataclass
from typing import Mapping, Optional

KEY_NAMES = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY")
SETTING_NAMES = (*KEY_NAMES, "APP_PASSWORD", "LLM_PROVIDER", "LLM_MODEL", "LLM_FALLBACKS")

MAX_RUNS_PER_SESSION = 10       # mock / no-LLM runs
MAX_LLM_RUNS_PER_SESSION = 3    # runs with a real model
MAX_MEASUREMENTS = 30           # budget cap with mock / no LLM
MAX_LLM_MEASUREMENTS = 20       # budget cap with a real model


def load_secrets(secrets: Optional[Mapping[str, str]]) -> None:
    """Copy hosted secrets into the environment (litellm reads keys from there). .env stays the fallback."""
    if not secrets:
        return
    for name in SETTING_NAMES:
        try:
            value = secrets.get(name)
        except Exception:
            value = None
        if value and not os.environ.get(name):
            os.environ[name] = str(value)


def key_configured() -> bool:
    return any(os.environ.get(k, "").strip() for k in KEY_NAMES)


def password_configured() -> bool:
    return bool(os.environ.get("APP_PASSWORD", "").strip())


def password_ok(entered: str) -> bool:
    expected = os.environ.get("APP_PASSWORD", "").strip()
    return bool(expected) and hmac.compare_digest(entered.strip().encode(), expected.encode())


@dataclass(frozen=True)
class Access:
    real_llm: bool
    reason: str
    max_measurements: int
    max_runs: int


def decide(entered_password: str, runs_so_far: int, llm_runs_so_far: int) -> Access:
    """What this visitor may do now."""
    if not key_configured():
        why = "no API key is configured, so every run uses the mock provider"
    elif not password_configured():
        why = "no APP_PASSWORD is configured, so real-LLM runs are disabled; every run uses the mock provider"
    elif not password_ok(entered_password):
        why = "enter the app password to use the real LLM; until then every run uses the mock provider"
    elif llm_runs_so_far >= MAX_LLM_RUNS_PER_SESSION:
        why = f"this session has used its {MAX_LLM_RUNS_PER_SESSION} real-LLM runs; further runs use the mock provider"
    else:
        return Access(True, "real LLM available", MAX_LLM_MEASUREMENTS, MAX_LLM_RUNS_PER_SESSION - llm_runs_so_far)
    return Access(False, why, MAX_MEASUREMENTS, max(0, MAX_RUNS_PER_SESSION - runs_so_far))


def provider_for(requested: str, access: Access) -> str:
    """The provider a run actually uses: never litellm unless access allows it."""
    if requested == "litellm" and not access.real_llm:
        return "mock"
    return requested if requested in ("litellm", "mock", "none") else "mock"
