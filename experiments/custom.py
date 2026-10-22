"""
Custom experiments: the operator types a law y = f(x) and the digital twin simulates it.

This is how an examiner can try a law the project never prepared for. The law goes through the
same whitelist gate as a model proposed by the Analyst LLM (analysis.candidate_models.compile_law):
it is parsed and checked node by node, never given to eval, exec or sympy. Only the operator
writes it; no agent ever sets or sees it (rule: the LLM never sets the simulator's ground truth).

What the agents see is what they see for any experiment: the variable names and units the
operator chose, the safe range and a generic one-line description. The law itself, like a
registry experiment's ground truth, lives only in the spec and the run record.

The instruments are derived from the law so that one recipe suits any law: the noise is a
percentage of the response's span over the range, the sensor's full scale leaves half a span
of headroom on each side, and the actuator's tolerance and step limit scale with the range.

A custom law usually has no model in the analysis library, so a run cannot be scored as
"right or wrong law". law_agreement() measures instead how far the best model's curve strays
from the typed law over the range, in units of the measurement noise.
"""
import math
from typing import Callable, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from analysis.candidate_models import FormulaRejected, compile_law, predict_fit
from analysis.model_library import ModelFit
from experiments.registry import CUSTOM_NAME, ExperimentSpec, NoiseSettings, VariableSpec

MAX_ABS_X = 1e6
GRID_POINTS = 401
DEFAULT_NOISE_PCT = 1.0   # as in the registry experiments (10 lux on a 1000 lux span)
MATCH_WITHIN_NOISE = 3.0   # largest deviation, in noise std, still called a match

# name -> (law, x from, x to): starting points offered on the dashboard
EXAMPLES = {
    "Quadratic": ("0.05*x**2 - 2*x + 40", 0.0, 100.0),
    "Exponential decay": ("800*exp(-x/25) + 30", 0.0, 100.0),
    "Inverse square": ("50000/(x + 8)**2 + 20", 0.0, 100.0),
    "Square root (power law)": ("120*x**0.5 + 20", 1.0, 100.0),
    "Straight line": ("4.2*x + 15", 0.0, 100.0),
    "Sine (not in the library)": ("300*sin(x/15) + 400", 0.0, 100.0),
}


class CustomLaw(BaseModel):
    """What the operator fills in on the dashboard."""
    model_config = ConfigDict(frozen=True)

    expression: str = Field(min_length=1, max_length=300, description="The law y = f(x)")
    x_name: str = Field(default="input", min_length=1, max_length=40)
    x_units: str = Field(default="a.u.", max_length=16)
    x_min: float = 0.0
    x_max: float = 100.0
    y_name: str = Field(default="response", min_length=1, max_length=40)
    y_units: str = Field(default="a.u.", max_length=16)
    noise_pct: float = Field(default=DEFAULT_NOISE_PCT, gt=0.0, le=25.0,
                             description="Measurement noise std as a percentage of the response span")


class CustomExperimentError(ValueError):
    """The law or its settings cannot make a safe, meaningful experiment. The message says why."""


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def build_custom_experiment(law: CustomLaw) -> ExperimentSpec:
    """An ExperimentSpec for the operator's law. Raises CustomExperimentError with a reason."""
    lo, hi = float(law.x_min), float(law.x_max)
    if not (math.isfinite(lo) and math.isfinite(hi)) or max(abs(lo), abs(hi)) > MAX_ABS_X:
        raise CustomExperimentError(f"the range must be finite numbers within ±{MAX_ABS_X:g}")
    if not lo < hi:
        raise CustomExperimentError("the range must satisfy minimum < maximum")
    try:
        text, f = compile_law(law.expression)
    except FormulaRejected as exc:
        raise CustomExperimentError(f"law rejected: {exc}") from None

    grid = np.linspace(lo, hi, GRID_POINTS)
    truth = f(grid)
    bad = ~np.isfinite(truth)
    if bad.any():
        raise CustomExperimentError(
            f"the law is not a finite number at x = {grid[bad][0]:.6g} inside the range; narrow the range or change the law")
    y_lo, y_hi = float(truth.min()), float(truth.max())
    span = y_hi - y_lo
    if not span > 1e-9 * max(1.0, abs(y_lo), abs(y_hi)):
        raise CustomExperimentError("the law does not change over the range, so there is nothing to discover")
    if max(abs(y_lo), abs(y_hi)) > 1e9:
        raise CustomExperimentError("the law's values exceed ±1e9 over the range; rescale it")

    noise = span * law.noise_pct / 100.0
    x_name, y_name = _clean(law.x_name), _clean(law.y_name)
    x_units, y_units = _clean(law.x_units) or "a.u.", _clean(law.y_units) or "a.u."
    law_fn: Callable[[float], float] = lambda x: float(f(np.array([x]))[0])
    return ExperimentSpec(
        name=CUSTOM_NAME,
        description_for_agents=(
            f"An actuator sets the {x_name} between {lo:g} and {hi:g} {x_units}; a sensor reports the "
            f"{y_name} in {y_units}. Nothing else is known about the apparatus."
        ),
        independent=VariableSpec(name=x_name, symbol="x", units=x_units),
        dependent=VariableSpec(name=y_name, symbol="y", units=y_units),
        safe_bounds=(lo, hi),
        max_step_delta=hi - lo,               # the rig parks at the lowest position; one move may cross the range
        noise=NoiseSettings(std=noise),
        sensor_range=(y_lo - 0.5 * span - 6 * noise, y_hi + 0.5 * span + 6 * noise),
        sensor_resolution=span / 4000.0,
        position_tolerance=(hi - lo) / 90.0,  # 2 deg on a 180 deg polariser
        ground_truth=law_fn,
        ground_truth_model=CUSTOM_NAME,       # no library model is assumed to represent it
        true_params={},
        law=text,
    )


def law_agreement(law: str, fit: Optional[ModelFit], bounds: Tuple[float, float],
                  noise_std: Optional[float]) -> Optional[float]:
    """Largest gap between the best model's curve and the operator's law over the range, in noise std."""
    if fit is None or not noise_std or noise_std <= 0:
        return None
    try:
        _, f = compile_law(law)
    except FormulaRejected:
        return None
    grid = np.linspace(bounds[0], bounds[1], GRID_POINTS)
    pred = predict_fit(fit, grid)
    if pred is None:
        return None
    truth = f(grid)
    ok = np.isfinite(pred) & np.isfinite(truth)
    if not ok.any():
        return None
    return float(np.max(np.abs(pred[ok] - truth[ok])) / noise_std)


def agreement_verdict(gap: Optional[float]) -> str:
    """One sentence for the operator about law_agreement()."""
    if gap is None:
        return "The best model could not be compared with your law."
    if gap <= MATCH_WITHIN_NOISE:
        return (f"The best model follows your law everywhere in the range to within {gap:.2g} noise "
                f"standard deviations: the law was recovered.")
    return (f"The best model departs from your law by up to {gap:.3g} noise standard deviations, so the law was not "
            f"recovered. It is probably not in the model library (linear, quadratic, exponentials, inverse square, "
            f"power law, cos²); with a real LLM the Analyst may propose a model that fits it.")
