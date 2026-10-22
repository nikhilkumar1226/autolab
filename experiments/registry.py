"""
Experiment registry.

Every experiment is described as data. A spec has two halves:
- what agents may see (`agent_view()`): apparatus description, variable names,
  units and safe bounds;
- what only the simulator and benchmark scoring may see: the ground-truth law,
  its parameters, and the library model that represents it.

Each spec also describes the instruments, as a manufacturer's datasheet would: the sensor's
range and resolution, the actuator's position tolerance and, where relevant, a hard limit on
the reading (a diode's current). The safety guard and the digital twin read these; agents
do not need them.

Ground-truth laws are written in physical form with physical parameters
(`physical_params`, e.g. R, C and the supply voltage). `true_params` gives the same law in
the form of the analysis-library model, for scoring; a test checks that the two agree.

The internal `name` (e.g. "malus_law") names the law, so it is never part of
the agent view either.
"""
import math
from typing import Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VariableSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    symbol: str
    units: str


class NoiseSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    std: float = Field(ge=0.0, description="Gaussian measurement noise std, in dependent-variable units")


BLIND_DESCRIPTION = "An actuator sets a controlled variable x; a sensor reports a response y."
HIDDEN_DESCRIPTION = ("An actuator sets the controlled variable; a sensor reports the response. "
                      "No description of the apparatus is given.")


class AgentExperimentView(BaseModel):
    """The only description of an experiment that agents are allowed to see."""
    model_config = ConfigDict(frozen=True)

    description: str
    independent: VariableSpec
    dependent: VariableSpec
    safe_bounds: Tuple[float, float]

    def without_description(self) -> "AgentExperimentView":
        """Variable names, units and bounds only: no description of the apparatus."""
        return self.model_copy(update={"description": HIDDEN_DESCRIPTION})

    def for_agents(self, blind: bool, describe_apparatus: bool) -> "AgentExperimentView":
        """The view the LLM agents get under the run's settings (blind mode also hides the names)."""
        if blind:
            return self.blinded()
        return self if describe_apparatus else self.without_description()

    def blinded(self) -> "AgentExperimentView":
        """Blind mode: no apparatus text and anonymous variable names. Units and bounds stay."""
        return AgentExperimentView(
            description=BLIND_DESCRIPTION,
            independent=VariableSpec(name="x", symbol="x", units=self.independent.units),
            dependent=VariableSpec(name="y", symbol="y", units=self.dependent.units),
            safe_bounds=self.safe_bounds,
        )


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description_for_agents: str
    independent: VariableSpec
    dependent: VariableSpec
    safe_bounds: Tuple[float, float]
    max_step_delta: float = Field(gt=0.0, description="Largest single actuator move the safety guard allows")
    noise: NoiseSettings

    # --- Instruments (datasheet values), used by the safety guard and the digital twin ---
    sensor_range: Tuple[float, float] = Field(description="Lowest and highest reading the sensor can report (full scale)")
    sensor_resolution: float = Field(gt=0.0, description="Smallest reading step (ADC LSB)")
    position_tolerance: float = Field(
        gt=0.0, description="Largest difference between commanded and fed-back actuator position that counts as on target"
    )
    response_limit: Optional[float] = Field(
        default=None, description="Hard safety limit on the reading; above it the guard backs the actuator off"
    )
    polariser: bool = Field(default=False, description="Light passes a polariser pair (finite extinction ratio in the twin)")

    # --- Simulator and benchmark scoring only. Never shown to agents. ---
    ground_truth: Callable[..., float] = Field(exclude=True, repr=False)
    physical_params: Optional[Dict[str, float]] = Field(
        default=None, description="Parameters of ground_truth in physical form; None means true_params"
    )
    ground_truth_model: str = Field(description="Key of the analysis-library model that represents the true law")
    true_params: Dict[str, float] = Field(description="True parameters, named as in the library model")
    law: Optional[str] = Field(
        default=None, description="Custom experiments only: the law y = f(x) the operator typed (never shown to agents)"
    )

    @model_validator(mode="after")
    def _check_bounds(self) -> "ExperimentSpec":
        lo, hi = self.safe_bounds
        if not lo < hi:
            raise ValueError(f"safe_bounds must satisfy min < max, got {self.safe_bounds}")
        if not self.sensor_range[0] < self.sensor_range[1]:
            raise ValueError(f"sensor_range must satisfy min < max, got {self.sensor_range}")
        return self

    def true_response(self, x: float) -> float:
        """Noise-free response of the true law at x."""
        params = self.physical_params if self.physical_params is not None else self.true_params
        return float(self.ground_truth(x, **params))

    def agent_view(self) -> AgentExperimentView:
        return AgentExperimentView(
            description=self.description_for_agents,
            independent=self.independent,
            dependent=self.dependent,
            safe_bounds=self.safe_bounds,
        )


# Ground-truth laws. These are written independently of analysis/model_library.py
# on purpose, so that a bug in one is caught by the cross-check test rather than
# shared silently by the simulator and the analyst.

def _malus(theta_deg: float, A: float, x0: float, C: float) -> float:
    # I = I0 cos^2(theta - theta0) + ambient
    return A * math.cos(math.radians(theta_deg - x0)) ** 2 + C


def _inverse_square(d_cm: float, A: float, x0: float, C: float) -> float:
    # E = P / (d + d0)^2 + ambient; d0 is the offset between the scale zero and the source
    return A / (d_cm + x0) ** 2 + C


def _beer_lambert(conc: float, A: float, k: float, C: float) -> float:
    # I = I0 exp(-epsilon * c) + ambient; k folds absorptivity and path length
    return A * math.exp(-k * conc) + C


def _rc_charging(t_ms: float, Vs: float, R_kohm: float, C_uF: float) -> float:
    # V(t) = Vs (1 - exp(-t / RC)); kOhm x uF = ms
    return Vs * (1.0 - math.exp(-t_ms / (R_kohm * C_uF)))


BOLTZMANN = 1.380649e-23          # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C
STEFAN_BOLTZMANN = 5.670374419e-8    # W m^-2 K^-4


def _shockley(v: float, Is_mA: float, n: float, T_K: float) -> float:
    # I = Is (exp(V / (n Vt)) - 1), Vt = kT/q
    vt = BOLTZMANN * T_K / ELEMENTARY_CHARGE
    return Is_mA * math.expm1(v / (n * vt))


def _ldr(lux: float, R1_kohm: float, gamma: float) -> float:
    # R = R1 E^-gamma, R1 = resistance at 1 lux
    return R1_kohm * lux ** (-gamma)


def _hooke(force_N: float, L0_mm: float, k_N_per_mm: float) -> float:
    # end position = unloaded position + F / k
    return L0_mm + force_N / k_N_per_mm


def _stefan_boltzmann(T_kK: float, emissivity: float, area_mm2: float, view_fraction: float, T_ambient_kK: float) -> float:
    # received power = f eps sigma A (T^4 - Ta^4), in mW
    t, ta = 1000.0 * T_kK, 1000.0 * T_ambient_kK
    return 1000.0 * view_fraction * emissivity * STEFAN_BOLTZMANN * area_mm2 * 1e-6 * (t ** 4 - ta ** 4)


_DIODE = {"Is_mA": 2.0e-6, "n": 1.9, "T_K": 300.0}
_DIODE_K = ELEMENTARY_CHARGE / (_DIODE["n"] * BOLTZMANN * _DIODE["T_K"])  # 1/(n Vt), per volt
_LAMP = {"emissivity": 0.35, "area_mm2": 20.0, "view_fraction": 0.01, "T_ambient_kK": 0.3}
# mW per kK^4: f eps sigma A x 1e12 (K^4 per kK^4) x 1e3 (mW per W)
_LAMP_A = _LAMP["view_fraction"] * _LAMP["emissivity"] * STEFAN_BOLTZMANN * _LAMP["area_mm2"] * 1e-6 * 1e15

LIGHT_SENSOR = dict(sensor_range=(0.0, 1200.0), sensor_resolution=1.0)


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    spec.name: spec
    for spec in [
        ExperimentSpec(
            name="malus_law",
            description_for_agents=(
                "A lamp shines through a fixed polarising filter and then through a second, "
                "rotatable polarising filter onto a light sensor. The controlled variable is the "
                "rotation angle of the second filter. The sensor reports the light that reaches it."
            ),
            independent=VariableSpec(name="polariser angle", symbol="θ", units="deg"),
            dependent=VariableSpec(name="transmitted intensity", symbol="I", units="lux"),
            safe_bounds=(0.0, 180.0),
            max_step_delta=180.0,
            noise=NoiseSettings(std=10.0),
            **LIGHT_SENSOR,
            position_tolerance=2.0,
            polariser=True,
            ground_truth=_malus,
            ground_truth_model="cos2_phase",
            true_params={"A": 1000.0, "x0": 20.0, "C": 15.0},
        ),
        ExperimentSpec(
            name="inverse_square_law",
            description_for_agents=(
                "A small lamp and a light sensor face each other on a straight rail. The controlled "
                "variable is the distance read off the rail scale. The sensor reports the illuminance "
                "it receives."
            ),
            independent=VariableSpec(name="source-sensor distance", symbol="d", units="cm"),
            dependent=VariableSpec(name="illuminance", symbol="E", units="lux"),
            safe_bounds=(5.0, 100.0),
            # The rig homes to 0, so the first move can be up to 100 cm.
            max_step_delta=100.0,
            noise=NoiseSettings(std=10.0),
            **LIGHT_SENSOR,
            position_tolerance=1.0,
            ground_truth=_inverse_square,
            ground_truth_model="inverse_square_offset",
            true_params={"A": 1.0e5, "x0": 5.0, "C": 15.0},
        ),
        ExperimentSpec(
            name="beer_lambert_law",
            description_for_agents=(
                "Light from a lamp passes through a cuvette of dye solution onto a light sensor. "
                "The controlled variable is the dye concentration in the cuvette. The sensor reports "
                "the light transmitted through the solution."
            ),
            independent=VariableSpec(name="dye concentration", symbol="c", units="mg/L"),
            dependent=VariableSpec(name="transmitted intensity", symbol="I", units="lux"),
            safe_bounds=(0.0, 100.0),
            max_step_delta=100.0,
            noise=NoiseSettings(std=10.0),
            **LIGHT_SENSOR,
            position_tolerance=1.0,
            ground_truth=_beer_lambert,
            ground_truth_model="exp_decay",
            true_params={"A": 1000.0, "k": 0.04, "C": 15.0},
        ),
        ExperimentSpec(
            name="rc_charging",
            description_for_agents=(
                "A capacitor is charged from a DC supply through a resistor, starting fully discharged when a "
                "switch closes. The controlled variable is the delay after the switch closes at which the "
                "capacitor voltage is sampled; the capacitor is discharged again before every sample. A 10-bit "
                "ADC reports the sampled voltage."
            ),
            independent=VariableSpec(name="sampling delay", symbol="t", units="ms"),
            dependent=VariableSpec(name="capacitor voltage", symbol="V", units="V"),
            safe_bounds=(0.0, 250.0),
            max_step_delta=250.0,
            noise=NoiseSettings(std=0.01),
            sensor_range=(0.0, 5.0),
            sensor_resolution=5.0 / 1023.0,
            position_tolerance=0.5,
            ground_truth=_rc_charging,
            physical_params={"Vs": 3.3, "R_kohm": 10.0, "C_uF": 4.7},
            # V = Vs - Vs exp(-t/RC): a decaying exponential with negative amplitude
            ground_truth_model="exp_decay",
            true_params={"A": -3.3, "k": 1.0 / 47.0, "C": 3.3},
        ),
        ExperimentSpec(
            name="diode_iv",
            description_for_agents=(
                "A small-signal silicon diode is connected in the forward direction to a programmable voltage "
                "source through a current meter. The controlled variable is the applied forward voltage; the "
                "meter reports the current through the diode. The current must stay below 30 mA."
            ),
            independent=VariableSpec(name="forward voltage", symbol="V", units="V"),
            dependent=VariableSpec(name="diode current", symbol="I", units="mA"),
            safe_bounds=(0.0, 0.8),
            # At most 0.4 V per step: never jump straight from 0 V to the top of the curve.
            max_step_delta=0.4,
            noise=NoiseSettings(std=0.01),
            sensor_range=(-50.0, 50.0),
            sensor_resolution=0.001,
            position_tolerance=0.005,
            response_limit=30.0,
            ground_truth=_shockley,
            physical_params=dict(_DIODE),
            # I = Is exp(V/(n Vt)) - Is
            ground_truth_model="exp_growth",
            true_params={"A": _DIODE["Is_mA"], "k": _DIODE_K, "C": -_DIODE["Is_mA"]},
        ),
        ExperimentSpec(
            name="ldr_response",
            description_for_agents=(
                "A light-dependent resistor sits under a dimmable lamp. The controlled variable is the "
                "illuminance at the resistor, set by the lamp driver and checked with a reference lux meter. "
                "An ohmmeter reports the resistance of the resistor."
            ),
            independent=VariableSpec(name="illuminance", symbol="E", units="lux"),
            dependent=VariableSpec(name="resistance", symbol="R", units="kOhm"),
            safe_bounds=(10.0, 1000.0),
            max_step_delta=1000.0,
            noise=NoiseSettings(std=0.3),
            sensor_range=(0.0, 2000.0),
            sensor_resolution=0.01,
            position_tolerance=2.0,
            ground_truth=_ldr,
            physical_params={"R1_kohm": 500.0, "gamma": 0.7},
            ground_truth_model="power_law",
            true_params={"A": 500.0, "b": -0.7, "C": 0.0},
        ),
        ExperimentSpec(
            name="hookes_law",
            description_for_agents=(
                "A coil spring hangs from a rigid support. A motorised loader pulls its free end downwards with "
                "a set force. The controlled variable is the applied force; a digital scale beside the spring "
                "reports the position of the free end, measured downwards from the support."
            ),
            independent=VariableSpec(name="applied force", symbol="F", units="N"),
            dependent=VariableSpec(name="end position", symbol="L", units="mm"),
            safe_bounds=(0.0, 20.0),
            # At most 10 N per step: no shock loading of the spring.
            max_step_delta=10.0,
            noise=NoiseSettings(std=0.2),
            sensor_range=(0.0, 300.0),
            sensor_resolution=0.1,
            position_tolerance=0.1,
            ground_truth=_hooke,
            physical_params={"L0_mm": 50.0, "k_N_per_mm": 0.25},
            ground_truth_model="linear",
            true_params={"a": 4.0, "b": 50.0},
        ),
        ExperimentSpec(
            name="stefan_boltzmann",
            description_for_agents=(
                "A lamp filament is held at a set temperature by a feedback controller. A thermopile detector in "
                "front of the lamp reports the radiant power it receives. The controlled variable is the filament "
                "temperature, in kilokelvin (1 kK = 1000 K)."
            ),
            independent=VariableSpec(name="filament temperature", symbol="T", units="kK"),
            dependent=VariableSpec(name="detected power", symbol="P", units="mW"),
            safe_bounds=(0.8, 2.8),
            # At most 1 kK per step: limits thermal shock to the filament.
            max_step_delta=1.0,
            noise=NoiseSettings(std=0.5),
            sensor_range=(0.0, 400.0),
            sensor_resolution=0.01,
            position_tolerance=0.01,
            ground_truth=_stefan_boltzmann,
            physical_params=dict(_LAMP),
            # P = A T^4 - A Ta^4
            ground_truth_model="power_law",
            true_params={"A": _LAMP_A, "b": 4.0, "C": -_LAMP_A * _LAMP["T_ambient_kK"] ** 4},
        ),
    ]
}


# Name of an operator-defined experiment (experiments/custom.py). It has no registry entry: its
# spec is built from the law the operator types and passed to the engine directly.
CUSTOM_NAME = "custom"


def list_experiments() -> List[str]:
    return list(EXPERIMENTS)


def get_experiment(name: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[name]
    except KeyError:
        if name == CUSTOM_NAME:
            raise KeyError("A custom experiment has no registry entry: build its spec with "
                           "experiments.custom.build_custom_experiment and pass it to the engine") from None
        raise KeyError(f"Unknown experiment '{name}'. Known: {', '.join(EXPERIMENTS)}") from None
