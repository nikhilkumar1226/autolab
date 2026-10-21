"""
Configuration settings for the Autonomous Scientific Discovery System.
Supports both Simulated Digital Twin and Real Hardware (Arduino/ESP32).

Experiment-specific settings (variables, units, safe bounds, ground truth, noise)
live in the experiment registry (experiments/registry.py), not here.
"""
import os
from pathlib import Path
from typing import Literal, Optional
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

class AgentConfig(BaseModel):
    model_name: str = Field(
        default=os.getenv("LLM_MODEL", "gpt-4o"),
        description="LLM model identifier for litellm / OpenAI / Anthropic / Google"
    )
    temperature: float = Field(default=0.2, description="Sampling temperature for scientific reasoning")
    initial_sample_count: int = Field(default=4, ge=2, description="Number of initial exploration points before active learning")
    sampler: Literal["gp", "random"] = Field(
        default="gp",
        description="How points after the initial design are chosen: GP uncertainty sampling or uniform random (baseline)"
    )

class StoppingConfig(BaseModel):
    min_points: int = Field(default=5, ge=2, description="Never stop on evidence with fewer measurements than this")
    delta_bic_threshold: float = Field(default=6.0, description="Best model must beat the runner-up by at least this BIC margin")
    max_rel_se: float = Field(
        default=0.05, gt=0.0,
        description="Every parameter of the best model must have a relative standard error below this"
    )
    max_measurements: int = Field(default=30, ge=1, description="Measurement budget")
    max_consecutive_failures: int = Field(
        default=3, ge=1,
        description="Abort after this many consecutive rejected or failed measurement attempts"
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
    stopping: StoppingConfig = Field(default_factory=StoppingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("experiment")
    @classmethod
    def _known_experiment(cls, name: str) -> str:
        if name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment '{name}'. Known: {', '.join(EXPERIMENTS)}")
        return name

config = SystemConfig()
