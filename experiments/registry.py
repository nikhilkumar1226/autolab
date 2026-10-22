"""
Experiment registry.

Every experiment is described as data. A spec has two halves:
- what agents may see (`agent_view()`): apparatus description, variable names,
  units and safe bounds;
- what only the simulator and benchmark scoring may see: the ground-truth law,
  its parameters, and the library model that represents it.

The internal `name` (e.g. "malus_law") names the law, so it is never part of
the agent view either.
"""
import math
from typing import Callable, Dict, List, Tuple

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


class AgentExperimentView(BaseModel):
    """The only description of an experiment that agents are allowed to see."""
    model_config = ConfigDict(frozen=True)

    description: str
    independent: VariableSpec
    dependent: VariableSpec
    safe_bounds: Tuple[float, float]

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

    # --- Simulator and benchmark scoring only. Never shown to agents. ---
    ground_truth: Callable[..., float] = Field(exclude=True, repr=False)
    ground_truth_model: str = Field(description="Key of the analysis-library model that represents the true law")
    true_params: Dict[str, float] = Field(description="True parameters, named as in the library model")

    @model_validator(mode="after")
    def _check_bounds(self) -> "ExperimentSpec":
        lo, hi = self.safe_bounds
        if not lo < hi:
            raise ValueError(f"safe_bounds must satisfy min < max, got {self.safe_bounds}")
        return self

    def true_response(self, x: float) -> float:
        """Noise-free response of the true law at x."""
        return float(self.ground_truth(x, **self.true_params))

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
            ground_truth=_beer_lambert,
            ground_truth_model="exp_decay",
            true_params={"A": 1000.0, "k": 0.04, "C": 15.0},
        ),
    ]
}


def list_experiments() -> List[str]:
    return list(EXPERIMENTS)


def get_experiment(name: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[name]
    except KeyError:
        raise KeyError(f"Unknown experiment '{name}'. Known: {', '.join(EXPERIMENTS)}") from None
