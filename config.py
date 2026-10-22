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

class NonIdealityConfig(BaseModel):
    """
    Imperfections of the digital twin. Off by default; each one is also off when its size is 0.
    Sizes are relative to the experiment's spec (noise std, position tolerance), so the same
    settings make sense for every experiment. Every random draw uses the run seed.
    """
    enabled: bool = Field(default=False, description="Master switch for every non-ideality below")
    extinction_ratio: float = Field(
        default=100.0, ge=1.0,
        description="Polariser experiments only: max/min transmission of the crossed pair (inf = ideal)"
    )
    drift_step_sigma: float = Field(
        default=0.3, ge=0.0,
        description="Slow ambient drift: random-walk step std per rig cycle, in units of the sensor noise std "
                    "(0.3 gives about 1.5 noise std after 25 cycles)"
    )
    saturation: bool = Field(default=True, description="Clip readings at the sensor's full scale")
    quantisation: bool = Field(default=True, description="Round readings to the sensor resolution (ADC LSB)")
    position_error_rel: float = Field(
        default=0.15, ge=0.0,
        description="Gaussian actuator position error std, as a fraction of the spec's position tolerance"
    )
    backlash_rel: float = Field(
        default=0.3, ge=0.0,
        description="Backlash: total direction-dependent position offset, as a fraction of the position tolerance"
    )
    outlier_prob: float = Field(default=0.03, ge=0.0, le=1.0, description="Chance that a reading is an outlier spike")
    outlier_sigma: float = Field(default=8.0, ge=0.0, description="Outlier size in units of the noise std (random sign)")

class FaultSpec(BaseModel):
    """
    One injected fault. `at_step` counts rig cycles: one cycle is one sensor read or one pause
    (1-based). Without pauses, cycle k is the k-th reading the rig is asked for, re-measurements
    included.
    - light_spike: the reading at that cycle has a transient spike added (an electrical glitch
      for non-optical sensors); `magnitude` is in units of the true response span (default 0.5).
    - sensor_dropout: readings during `duration` cycles return NaN (mode "nan") or time out ("timeout").
    - stuck_actuator: moves commanded during `duration` cycles are acknowledged but not executed;
      the position feedback keeps reporting the old position.
    """
    type: Literal["light_spike", "sensor_dropout", "stuck_actuator"]
    at_step: int = Field(ge=1)
    duration: int = Field(default=1, ge=1)
    magnitude: float = Field(default=0.5, description="light_spike only: spike size as a fraction of the response span")
    mode: Literal["nan", "timeout"] = Field(default="nan", description="sensor_dropout only")

# The ROADMAP example puts the stuck actuator at cycle 7, but cycle 7 is then always the automatic
# re-measurement after the cycle-6 spike, at the same position, so the stuck actuator could never act.
# Here each fault follows the previous one's recovery (spike 6, re-measure 7; stuck 8, re-measure 9;
# dropout 10-11, pause 12, re-measure 13), so all three act before a typical run stops at n = 8.
DEFAULT_FAULTS: List[FaultSpec] = [
    FaultSpec(type="light_spike", at_step=6),
    FaultSpec(type="stuck_actuator", at_step=8),
    FaultSpec(type="sensor_dropout", at_step=10, duration=2),
]

class SimulationConfig(BaseModel):
    noise_std: Optional[float] = Field(
        default=None, ge=0.0,
        description="Override the experiment's default measurement noise std (None = use the spec's default)"
    )
    transit_delay_sec: float = Field(default=0.05, ge=0.0, description="Simulated motor transit latency")
    # Simulated time (the twin's clock), used by the safety guard's rate limit and watchdog. It advances
    # whether or not transit_delay_sec makes the twin really sleep, so safety timing is reproducible.
    move_settle_s: float = Field(default=0.05, ge=0.0, description="Simulated time per move, plus travel time")
    full_travel_s: float = Field(default=0.5, ge=0.0, description="Simulated time to travel the whole safe range")
    read_time_s: float = Field(default=0.2, ge=0.0, description="Simulated sensor integration time per reading")
    nonideal: NonIdealityConfig = Field(default_factory=NonIdealityConfig)
    faults: List[FaultSpec] = Field(default_factory=list, description="Injected faults (digital twin only)")

class AnomalyConfig(BaseModel):
    """How suspicious readings are detected and handled (any rig, not only the twin)."""
    outlier_bands: Literal["model", "gp", "both"] = Field(
        default="model",
        description="Which predictive band flags an outlier: the current best fit's (Student-t, n - k dof), "
                    "the GP's (normal), or both (the reading must be outside both)"
    )
    outlier_p: float = Field(
        default=0.001, gt=0.0, lt=1.0,
        description="Two-sided tail probability for an outlier: a genuine reading is flagged this often if the band is right"
    )
    outlier_min_points: int = Field(default=4, ge=2, description="The outlier check needs at least this many measurements")
    confirm_z: float = Field(
        default=3.0, gt=0.0,
        description="A flagged outlier is genuine if a re-measurement agrees with it within this many sigma (x sqrt 2)"
    )
    max_decisions: int = Field(
        default=2, ge=0,
        description="After the automatic re-measurement, how many times the Analyst decides before the position is discarded"
    )
    pause_s: float = Field(default=1.0, ge=0.0, description="Length of a pause before re-measuring (one rig cycle on the twin)")

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

class SafetyConfig(BaseModel):
    """Safety-guard limits that are not experiment-specific (bounds and step size come from the spec)."""
    max_moves_per_s: float = Field(
        default=5.0, gt=0.0,
        description="Rate limit: at most this many actuator moves in any 1 s window of the rig's clock"
    )
    max_moves_without_reading: int = Field(
        default=3, ge=1,
        description="Oscillation interlock: at most this many moves in a row without a sensor reading"
    )
    watchdog_s: float = Field(
        default=120.0, gt=0.0,
        description="Latch the E-stop if the rig has gone this long (rig clock) without a valid reading. Generous, "
                    "because on hardware the rig clock is wall time and includes LLM latency between readings"
    )

class AgentConfig(BaseModel):
    initial_sample_count: int = Field(default=4, ge=2, description="Number of initial exploration points before active learning")
    sampler: Literal["gp", "random", "grid", "llm"] = Field(
        default="gp",
        description="How points after the initial design are chosen: gp = GP uncertainty sampling (the Planner may "
                    "choose its own point instead); random / grid = baselines; llm = the Planner chooses every point, "
                    "without a GP"
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
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
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
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("experiment")
    @classmethod
    def _known_experiment(cls, name: str) -> str:
        if name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment '{name}'. Known: {', '.join(EXPERIMENTS)}")
        return name

def describe_twin(cfg: SystemConfig) -> str:
    """The digital twin's settings in one line, for displays, reports and the Writer (empty for hardware)."""
    if not cfg.hardware.simulate:
        return ""
    sim = cfg.simulation
    parts = ["non-idealities on" if sim.nonideal.enabled else "ideal sensor and actuator (non-idealities off)"]
    if sim.faults:
        parts.append("injected faults: " + ", ".join(
            f"{f.type} at cycle {f.at_step}" + (f" for {f.duration} cycles" if f.duration > 1 else "") for f in sim.faults
        ))
    else:
        parts.append("no injected faults")
    return "; ".join(parts)

config = SystemConfig()
