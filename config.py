"""
Configuration settings for the Autonomous Scientific Discovery System.
Supports both Simulated Digital Twin and Real Hardware (Arduino/ESP32).

Experiment-specific settings (variables, units, safe bounds, ground truth, noise)
live in the experiment registry (experiments/registry.py), not here.
"""
import os
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from experiments.registry import EXPERIMENTS

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = BASE_DIR / "runs"
LLM_CACHE_DIR = BASE_DIR / "llm_cache"

def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default

def _env_list(name: str) -> List[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]

class HardwareConfig(BaseModel):
    simulate: bool = Field(
        default=True,
        description="If True, runs digital twin simulation. If False, connects via PySerial."
    )
    port: str = Field(
        default=os.getenv("SERIAL_PORT", "/dev/ttyUSB0"),
        description="Serial port path (e.g., /dev/ttyUSB0, COM3, /dev/tty.usbmodem1101, rfc2217://localhost:4000)"
    )
    baud_rate: int = Field(
        default=int(os.getenv("SERIAL_BAUD", "115200")),
        description="Baud rate for serial communication"
    )
    timeout_seconds: float = Field(default=2.0, description="Serial timeout in seconds")

class SimulationConfig(BaseModel):
    noise_std: Optional[float] = Field(
        default=None, ge=0.0,
        description="Override the experiment's default measurement noise std (None = use the spec's default)"
    )
    transit_delay_sec: float = Field(default=0.05, ge=0.0, description="Simulated motor transit latency")

class LLMConfig(BaseModel):
    """
    Which language model answers the agents. API keys are never part of the config
    (litellm reads them from the environment), so they never reach a run record.
    """
    provider: Literal["litellm", "mock", "none"] = Field(
        default_factory=lambda: _env("LLM_PROVIDER", "mock"),
        description="litellm = real model; mock = deterministic offline stand-in; none = no LLM (heuristic baseline)"
    )
    model: str = Field(
        default_factory=lambda: _env("LLM_MODEL", "anthropic/claude-haiku-5-5"),
        description="First model in the chain (litellm model id)"
    )
    fallbacks: List[str] = Field(
        default_factory=lambda: _env_list("LLM_FALLBACKS"),
        description="Models tried in order when the first one fails"
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    timeout_s: float = Field(default=30.0, gt=0.0, description="Per-request timeout")
    max_output_tokens: int = Field(default=1024, ge=16, description="Cap on the length (and cost) of each answer")
    max_json_retries: int = Field(
        default=2, ge=0, le=5,
        description="Re-asks of the same model, with the validation error fed back, when its JSON is invalid"
    )
    rate_limit_retries: int = Field(default=2, ge=0, le=5, description="Backoff retries on HTTP 429 before moving to the next model")
    backoff_base_s: float = Field(default=2.0, ge=0.0)
    backoff_max_s: float = Field(default=30.0, ge=0.0)
    structured_output: Literal["auto", "json_schema", "json_object"] = Field(
        default="auto",
        description="auto = JSON schema where litellm says the model supports it, else plain JSON mode"
    )
    cache: bool = Field(default=True, description="Serve identical prompts from the disk cache, so reruns cost nothing")
    cache_dir: Path = Field(default=LLM_CACHE_DIR)

    @property
    def chain(self) -> List[str]:
        """The model priority order, without duplicates."""
        return list(dict.fromkeys([self.model, *self.fallbacks]))

class AgentConfig(BaseModel):
    initial_sample_count: int = Field(default=4, ge=2, description="Number of initial exploration points before active learning")
    sampler: Literal["gp", "random"] = Field(
        default="gp",
        description="How points after the initial design are chosen: GP uncertainty sampling or uniform random (baseline)"
    )
    blind: bool = Field(
        default=False,
        description="Hide the apparatus description and variable names from every LLM agent (units and bounds stay)"
    )
    max_reproposals: int = Field(
        default=2, ge=0,
        description="After a safety rejection, how many times the Planner may re-propose before the GP takes over"
    )
    max_llm_candidates: int = Field(default=3, ge=0, description="LLM-proposed candidate models accepted per run")
    candidate_equivalence_sigma: float = Field(
        default=0.5, gt=0.0,
        description="An LLM-proposed model whose curve stays within this many sigma of a library fit adds nothing and is skipped"
    )

class StoppingConfig(BaseModel):
    # Defaults chosen by the seeded sweep in docs/CHANGELOG.md (Phase 1 fix): min_points = 8 was the
    # only configuration with >= 19/20 correct laws on every experiment.
    min_points: int = Field(default=8, ge=2, description="Never stop on evidence with fewer measurements than this")
    delta_bic_threshold: float = Field(default=6.0, description="Best model must beat the runner-up by at least this BIC margin")
    max_rel_se: float = Field(
        default=0.05, gt=0.0,
        description="Every parameter of the best model must have a relative standard error below this"
    )
    predictive_stability: bool = Field(
        default=False,
        description="Also require the same best model in consecutive analyses and a correctly predicted newest measurement"
    )
    stability_window: int = Field(
        default=2, ge=2,
        description="The same model must be best, by at least delta_bic_threshold, in this many consecutive analyses"
    )
    prediction_max_z: float = Field(
        default=3.0, gt=0.0,
        description="The newest measurement must lie within this many sigma of the earlier best fit's prediction"
    )
    noise_floor_rel: float = Field(
        default=0.005, ge=0.0,
        description="Lower bound on the prediction sigma, as a fraction of the measured y-span (assumed instrument resolution)"
    )
    max_measurements: int = Field(default=30, ge=1, description="Measurement budget")
    max_consecutive_failures: int = Field(
        default=3, ge=1,
        description="Abort after this many consecutive measurement slots without a measurement "
                    "(a slot is one proposal, the Planner's re-proposals after a safety rejection, and the GP fallback)"
    )

class OutputConfig(BaseModel):
    runs_dir: Path = Field(default=RUNS_DIR, description="Where JSON run records are written")
    plots_dir: Path = Field(default=PLOTS_DIR)
    reports_dir: Path = Field(default=REPORTS_DIR)
    write_report: bool = Field(default=True, description="Generate the figure and the Markdown/LaTeX report")

class SystemConfig(BaseModel):
    experiment: str = Field(default="malus_law", description="Registry name of the experiment to run")
    seed: Optional[int] = Field(
        default=None, ge=0,
        description="Seed for every random draw in the run. None = draw one from OS entropy and record it."
    )
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    stopping: StoppingConfig = Field(default_factory=StoppingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("experiment")
    @classmethod
    def _known_experiment(cls, name: str) -> str:
        if name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment '{name}'. Known: {', '.join(EXPERIMENTS)}")
        return name

config = SystemConfig()
