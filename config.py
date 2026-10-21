"""
Configuration settings for the Autonomous Scientific Discovery System.
Supports both Simulated Digital Twin and Real Hardware (Arduino/ESP32).
"""
import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

class HardwareConfig(BaseModel):
    simulate: bool = Field(
        default=True,
        description="If True, runs digital twin simulation. If False, connects via PySerial."
    )
    port: str = Field(
        default=os.getenv("SERIAL_PORT", "/dev/ttyUSB0"),
        description="Serial port path (e.g., /dev/ttyUSB0, COM3, /dev/tty.usbmodem1101)"
    )
    baud_rate: int = Field(
        default=int(os.getenv("SERIAL_BAUD", "115200")),
        description="Baud rate for serial communication"
    )
    timeout_seconds: float = Field(default=2.0, description="Serial timeout in seconds")
    
    # Safety Limits
    min_actuator_value: float = Field(default=0.0, description="Minimum allowed actuator position (e.g. 0 deg)")
    max_actuator_value: float = Field(default=180.0, description="Maximum allowed actuator position (e.g. 180 deg)")
    max_step_delta: float = Field(default=180.0, description="Max allowed step displacement per move")

class SimulationConfig(BaseModel):
    experiment_type: str = Field(
        default="malus_law",
        description="Type of experiment simulation: 'malus_law' or 'inverse_square_law'"
    )
    # Malus Law ground truth: I = I_0 * cos^2(theta - theta_0) + I_ambient + noise
    i0_intensity: float = Field(default=1000.0, description="Peak light intensity (lux / raw counts)")
    theta_offset: float = Field(default=0.0, description="Initial polarization axis offset in degrees")
    ambient_noise_level: float = Field(default=15.0, description="Ambient background light baseline")
    noise_std: float = Field(default=12.0, description="Gaussian measurement noise standard deviation")
    transit_delay_sec: float = Field(default=0.05, description="Simulated motor transit latency")

class AgentConfig(BaseModel):
    model_name: str = Field(
        default=os.getenv("LLM_MODEL", "gpt-4o"),
        description="LLM model identifier for litellm / OpenAI / Anthropic / Google"
    )
    temperature: float = Field(default=0.2, description="Sampling temperature for scientific reasoning")
    max_discovery_iterations: int = Field(default=15, description="Maximum active learning loop iterations")
    convergence_r2_threshold: float = Field(default=0.985, description="R^2 threshold to declare scientific discovery confirmed")
    initial_sample_count: int = Field(default=4, description="Number of initial exploration points before Bayesian optimization")

class SystemConfig(BaseModel):
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

config = SystemConfig()
