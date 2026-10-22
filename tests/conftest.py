"""
Shared test fixtures. Tests run offline: digital twin only, no transit delay,
mock LLM provider, all outputs written under pytest's tmp_path.

No test may reach a real LLM: whatever .env says, the provider defaults to mock, and
litellm.completion raises unless a test installs the scripted `fake_litellm`.
"""
import json
import os

# Before config is imported (load_dotenv never overrides): tests never depend on the local .env.
os.environ["LLM_PROVIDER"] = "mock"
os.environ["LLM_MODEL"] = "anthropic/claude-haiku-5-5"
os.environ["LLM_FALLBACKS"] = "gemini/gemini-3.6-flash,gemini/gemini-3.5-flash-lite"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pytest

import llm.client as llm_client
from config import LLMConfig, OutputConfig, SimulationConfig, SystemConfig
from experiments.registry import ExperimentSpec
from hardware.simulator import PhysicsSimulationRig

_REAL_LITELLM = llm_client._litellm

TEST_KEYS = {"ANTHROPIC_API_KEY": "test-anthropic-key", "GEMINI_API_KEY": "test-gemini-key"}
TEST_CHAIN = ["anthropic/claude-haiku-5-5", "gemini/gemini-3.6-flash", "gemini/gemini-3.5-flash-lite"]


@pytest.fixture
def make_config(tmp_path):
    def _make(experiment: str = "malus_law", seed: int = 0, write_report: bool = False, **overrides) -> SystemConfig:
        overrides.setdefault("llm", LLMConfig(provider="mock", cache_dir=tmp_path / "llm_cache"))
        overrides.setdefault("simulation", SimulationConfig(transit_delay_sec=0.0))
        return SystemConfig(
            experiment=experiment,
            seed=seed,
            output=OutputConfig(
                runs_dir=tmp_path / "runs",
                plots_dir=tmp_path / "plots",
                reports_dir=tmp_path / "reports",
                write_report=write_report,
            ),
            **overrides,
        )
    return _make


@pytest.fixture
def litellm_config(tmp_path):
    """A litellm provider config with the documented chain and a per-test cache."""
    def _make(**overrides) -> LLMConfig:
        fields = dict(provider="litellm", model=TEST_CHAIN[0], fallbacks=TEST_CHAIN[1:], cache_dir=tmp_path / "llm_cache")
        fields.update(overrides)
        return LLMConfig(**fields)
    return _make


@pytest.fixture
def simulate():
    """Noisy readings of an experiment's digital twin at the given positions."""
    def _simulate(spec: ExperimentSpec, positions: Sequence[float], seed: int) -> np.ndarray:
        rig = PhysicsSimulationRig(spec, np.random.default_rng(seed), SimulationConfig(transit_delay_sec=0.0))
        rig.connect()
        values = []
        for x in positions:
            rig.move_actuator(float(x))
            values.append(rig.read_sensor().sensor_value)
        return np.array(values)
    return _simulate


# ---------------------------------------------------------------------------
# LLM isolation
# ---------------------------------------------------------------------------

class _OfflineLiteLLM:
    """Stands in for the litellm module: everything works except sending a request."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        if name == "completion":
            raise RuntimeError("A test tried to call a real LLM. Use the fake_litellm fixture.")
        return getattr(self._module, name)


@pytest.fixture(autouse=True)
def _offline_llm(monkeypatch):
    monkeypatch.setattr(llm_client, "_litellm", lambda: _OfflineLiteLLM(_REAL_LITELLM()))


class FakeLiteLLM:
    """
    Scripted litellm.completion. Each call takes the next queued item, or asks `responder`:
    a str is returned as the answer text, a dict/list as its JSON, an exception is raised,
    and a callable is called with the request kwargs and its result used as above.
    """

    def __init__(self, module):
        self._module = module
        self.script: List[Any] = []
        self.responder: Optional[Callable[[Dict[str, Any]], Any]] = None
        self.calls: List[Dict[str, Any]] = []
        self.usage = {"prompt_tokens": 100, "completion_tokens": 20}

    def queue(self, *items: Any) -> "FakeLiteLLM":
        self.script.extend(items)
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.script:
            item = self.script.pop(0)
        elif self.responder is not None:
            item = self.responder(kwargs)
        else:
            raise AssertionError("FakeLiteLLM: no scripted answer left")
        if callable(item) and not isinstance(item, BaseException):
            item = item(kwargs)
        if isinstance(item, BaseException):
            raise item
        content = item if isinstance(item, str) else json.dumps(item)
        return self._module.ModelResponse(
            model=kwargs["model"].split("/", 1)[-1],
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            usage={**self.usage, "total_tokens": sum(self.usage.values())},
        )

    @property
    def models(self) -> List[str]:
        return [c["model"] for c in self.calls]


@pytest.fixture
def fake_litellm(monkeypatch):
    """Installs FakeLiteLLM as litellm.completion (Piyush's patch("litellm.completion") style) with test keys set."""
    module = _REAL_LITELLM()
    fake = FakeLiteLLM(module)
    monkeypatch.setattr(module, "completion", fake)
    monkeypatch.setattr(llm_client, "_litellm", _REAL_LITELLM)
    for key, value in TEST_KEYS.items():
        monkeypatch.setenv(key, value)
    return fake
