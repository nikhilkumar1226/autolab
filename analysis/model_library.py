"""
Candidate model library and blind model fitting.

The analyst fits every model in this library to the measured (x, y) data and
ranks them by BIC. Nothing here knows which experiment produced the data.

Initial guesses and bounds come only from the data:
- linear / quadratic: ordinary polynomial least squares;
- models of the form y = A * g(x; t) + C with one nonlinear parameter t: scan t on
  a data-scaled grid, solve (A, C) by linear least squares at each t, keep the best;
- cos^2 with phase: exact linearisation cos^2(u - u0) = 1/2 + 1/2 cos(2u - 2u0),
  so a 3-term linear least-squares fit gives the global initial guess.
Then scipy's curve_fit refines all parameters and estimates their covariance.

Relative standard errors depend on the parameter's kind:
- "scale" (amplitude, rate, exponent, slope): SE / |value|;
- "x_location" (a shift along x, e.g. a phase or distance offset): SE / x-span;
- "y_offset" (a baseline or intercept): SE / y-span.
A location's "relative" error would otherwise depend on the arbitrary choice of origin,
and is undefined when the true value is 0.
"""
import math
import logging
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import OptimizeWarning, curve_fit

logger = logging.getLogger(__name__)

ParamKind = Literal["scale", "x_location", "y_offset"]
Bounds = Tuple[List[float], List[float]]


# ---------------------------------------------------------------------------
# Result schemas
# ---------------------------------------------------------------------------

class ModelFit(BaseModel):
    key: str
    name: str
    formula: str
    n_points: int
    n_params: int
    params: Dict[str, float]
    std_errors: Dict[str, Optional[float]] = Field(description="None when the covariance is undefined")
    rel_std_errors: Dict[str, Optional[float]] = Field(description="Kind-normalised; None when undefined")
    r2: float
    rmse: float
    aic: float
    bic: float
    delta_bic: float = Field(default=0.0, description="BIC minus the best BIC in the ranking")

    @property
    def residual_std(self) -> float:
        """Degrees-of-freedom-corrected residual std, sqrt(SSR / (n - k)); fits always have n > k."""
        return self.rmse * math.sqrt(self.n_points / (self.n_points - self.n_params))

class SkippedModel(BaseModel):
    key: str
    name: str
    reason: str

class AnalysisResult(BaseModel):
    iteration: int
    n_points: int
    fits: List[ModelFit] = Field(description="Successful fits, ranked by BIC (best first)")
    skipped: List[SkippedModel] = Field(default_factory=list)
    delta_bic_runner_up: Optional[float] = Field(
        default=None, description="BIC(runner-up) - BIC(best); None with fewer than two fits"
    )

    @property
    def best(self) -> Optional[ModelFit]:
        return self.fits[0] if self.fits else None

    @property
    def runner_up(self) -> Optional[ModelFit]:
        return self.fits[1] if len(self.fits) > 1 else None


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateModel:
    key: str
    name: str
    formula: str
    param_names: Tuple[str, ...]
    param_kinds: Tuple[ParamKind, ...]
    func: Callable[..., np.ndarray]
    initial_guess: Callable[[np.ndarray, np.ndarray], Sequence[float]]
    bounds: Callable[[np.ndarray, np.ndarray], Bounds]
    latex: Callable[[Dict[str, float]], str]
    domain_error: Optional[Callable[[np.ndarray], Optional[str]]] = None

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def predict(self, x: Union[Sequence[float], np.ndarray], params: Dict[str, float]) -> np.ndarray:
        return self.func(np.asarray(x, dtype=float), *(params[p] for p in self.param_names))

    def equation(self, params: Dict[str, float]) -> str:
        return self.latex(params)


def _num(v: float) -> str:
    return f"{v:.4g}"

def _plus(v: float) -> str:
    """'+ 3' or '- 3' for appending a signed term."""
    return f"+ {_num(v)}" if v >= 0 else f"- {_num(-v)}"

def _span(v: np.ndarray) -> float:
    return float(np.max(v) - np.min(v)) if len(v) else 0.0

def _profile(basis: Callable[[np.ndarray, float], np.ndarray], grid: np.ndarray,
             x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """For y = A * basis(x, t) + C: scan t, solve (A, C) linearly, return the best (A, t, C)."""
    best = None
    ones = np.ones_like(x)
    for t in grid:
        with np.errstate(all="ignore"):
            g = basis(x, t)
        if not np.all(np.isfinite(g)):
            continue
        G = np.column_stack([g, ones])
        coef, *_ = np.linalg.lstsq(G, y, rcond=None)
        ssr = float(np.sum((G @ coef - y) ** 2))
        if best is None or ssr < best[0]:
            best = (ssr, float(coef[0]), float(t), float(coef[1]))
    if best is None:
        raise ValueError("no finite basis on the initial-guess grid")
    return best[1], best[2], best[3]


# --- linear: y = a x + b ---
def _linear(x, a, b):
    return a * x + b

# --- quadratic: y = a x^2 + b x + c ---
def _quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c

# --- exponential decay: y = A exp(-k x) + C ---
def _exp_decay(x, A, k, C):
    return A * np.exp(-k * x) + C

def _exp_decay_guess(x, y):
    xs = max(_span(x), 1e-12)
    x_min = float(np.min(x))
    # Basis shifted to x_min for conditioning; A is converted back below.
    A_shift, k, C = _profile(lambda xx, kk: np.exp(-kk * (xx - x_min)), np.geomspace(0.1 / xs, 20.0 / xs, 40), x, y)
    return [A_shift * math.exp(k * x_min), k, C]

def _exp_decay_bounds(x, y):
    xs = max(_span(x), 1e-12)
    return [-np.inf, 1e-6 / xs, -np.inf], [np.inf, 50.0 / xs, np.inf]

# --- cos^2 with phase (x in degrees): y = A cos^2(pi (x - x0) / 180) + C ---
def _cos2_phase(x, A, x0, C):
    return A * np.cos(np.radians(x - x0)) ** 2 + C

def _cos2_phase_guess(x, y):
    u = np.radians(x)
    G = np.column_stack([np.ones_like(x), np.cos(2 * u), np.sin(2 * u)])
    (a, b, g), *_ = np.linalg.lstsq(G, y, rcond=None)
    A = 2.0 * math.hypot(b, g)
    x0 = math.degrees(math.atan2(g, b) / 2.0)
    return [A, x0, a - A / 2.0]

def _cos2_phase_bounds(x, y):
    return [0.0, -180.0, -np.inf], [np.inf, 180.0, np.inf]

def _cos2_latex(p):
    shift = f"x - {_num(p['x0'])}" if p["x0"] >= 0 else f"x + {_num(-p['x0'])}"
    return f"y = {_num(p['A'])}\\cos^2\\left(\\frac{{\\pi\\,({shift})}}{{180}}\\right) {_plus(p['C'])}"

# --- inverse square with offset: y = A / (x + x0)^2 + C, requires x + x0 > 0 ---
def _inverse_square(x, A, x0, C):
    return A / (x + x0) ** 2 + C

def _inverse_square_guess(x, y):
    xs = max(_span(x), 1e-12)
    x_min = float(np.min(x))
    grid = -x_min + np.geomspace(1e-3 * xs, 10.0 * xs, 50)
    A, x0, C = _profile(lambda xx, t: 1.0 / (xx + t) ** 2, grid, x, y)
    return [A, x0, C]

def _inverse_square_bounds(x, y):
    xs = max(_span(x), 1e-12)
    x_min = float(np.min(x))
    return [-np.inf, -x_min + 1e-4 * xs, -np.inf], [np.inf, -x_min + 100.0 * xs, np.inf]

def _inverse_square_latex(p):
    shift = f"x + {_num(p['x0'])}" if p["x0"] >= 0 else f"x - {_num(-p['x0'])}"
    return f"y = \\frac{{{_num(p['A'])}}}{{({shift})^2}} {_plus(p['C'])}"

# --- power law: y = A x^b + C, requires x > 0 ---
def _power_law(x, A, b, C):
    return A * np.power(x, b) + C

def _power_law_guess(x, y):
    grid = np.array([b for b in np.linspace(-4.0, 4.0, 41) if abs(b) > 1e-9])
    A, b, C = _profile(lambda xx, bb: np.power(xx, bb), grid, x, y)
    return [A, b, C]

def _power_law_bounds(x, y):
    return [-np.inf, -6.0, -np.inf], [np.inf, 6.0, np.inf]

def _power_law_domain(x):
    return None if np.all(x > 0) else "requires every x > 0"


MODEL_LIBRARY: Dict[str, CandidateModel] = {
    m.key: m
    for m in [
        CandidateModel(
            key="linear", name="Linear", formula="y = a*x + b",
            param_names=("a", "b"), param_kinds=("scale", "y_offset"),
            func=_linear,
            initial_guess=lambda x, y: list(np.polyfit(x, y, 1)),
            bounds=lambda x, y: ([-np.inf] * 2, [np.inf] * 2),
            latex=lambda p: f"y = {_num(p['a'])}\\,x {_plus(p['b'])}",
        ),
        CandidateModel(
            key="quadratic", name="Quadratic", formula="y = a*x^2 + b*x + c",
            param_names=("a", "b", "c"), param_kinds=("scale", "scale", "y_offset"),
            func=_quadratic,
            initial_guess=lambda x, y: list(np.polyfit(x, y, 2)),
            bounds=lambda x, y: ([-np.inf] * 3, [np.inf] * 3),
            latex=lambda p: f"y = {_num(p['a'])}\\,x^2 {_plus(p['b'])}\\,x {_plus(p['c'])}",
        ),
        CandidateModel(
            key="exp_decay", name="Exponential decay", formula="y = A*exp(-k*x) + C",
            param_names=("A", "k", "C"), param_kinds=("scale", "scale", "y_offset"),
            func=_exp_decay, initial_guess=_exp_decay_guess, bounds=_exp_decay_bounds,
            latex=lambda p: f"y = {_num(p['A'])}\\,e^{{-{_num(p['k'])}\\,x}} {_plus(p['C'])}",
        ),
        CandidateModel(
            key="cos2_phase", name="Cosine-squared with phase", formula="y = A*cos^2(pi*(x - x0)/180) + C",
            param_names=("A", "x0", "C"), param_kinds=("scale", "x_location", "y_offset"),
            func=_cos2_phase, initial_guess=_cos2_phase_guess, bounds=_cos2_phase_bounds,
            latex=_cos2_latex,
        ),
        CandidateModel(
            key="inverse_square_offset", name="Inverse-square with offset", formula="y = A/(x + x0)^2 + C",
            param_names=("A", "x0", "C"), param_kinds=("scale", "x_location", "y_offset"),
            func=_inverse_square, initial_guess=_inverse_square_guess, bounds=_inverse_square_bounds,
            latex=_inverse_square_latex,
        ),
        CandidateModel(
            key="power_law", name="Power law", formula="y = A*x^b + C",
            param_names=("A", "b", "C"), param_kinds=("scale", "scale", "y_offset"),
            func=_power_law, initial_guess=_power_law_guess, bounds=_power_law_bounds,
            latex=lambda p: f"y = {_num(p['A'])}\\,x^{{{_num(p['b'])}}} {_plus(p['C'])}",
            domain_error=_power_law_domain,
        ),
    ]
}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _inside(p0: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    """Move an initial guess strictly inside finite bounds (curve_fit rejects guesses on a bound)."""
    p = np.array(p0, dtype=float)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if np.isfinite(a):
            p[i] = max(p[i], a + 1e-9 * (1.0 + abs(a)))
        if np.isfinite(b):
            p[i] = min(p[i], b - 1e-9 * (1.0 + abs(b)))
    return p

def information_criteria(ssr: float, n: int, k: int, y_scale: float) -> Tuple[float, float]:
    """Gaussian-likelihood AIC and BIC. SSR is floored so a perfect fit stays finite."""
    ssr = max(ssr, n * (1e-12 * max(1.0, y_scale)) ** 2)
    base = n * math.log(ssr / n)
    return base + 2 * k, base + k * math.log(n)

def fit_model(model: CandidateModel, x: np.ndarray, y: np.ndarray) -> Union[ModelFit, SkippedModel]:
    n, k = len(x), model.n_params
    if n <= k:
        return SkippedModel(key=model.key, name=model.name, reason=f"needs more than {k} points (have {n})")
    if model.domain_error is not None:
        reason = model.domain_error(x)
        if reason:
            return SkippedModel(key=model.key, name=model.name, reason=reason)

    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", np.exceptions.RankWarning)
            lo, hi = model.bounds(x, y)
            p0 = _inside(model.initial_guess(x, y), lo, hi)
            popt, pcov = curve_fit(model.func, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
            pred = model.func(x, *popt)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return SkippedModel(key=model.key, name=model.name, reason=f"fit failed: {exc}")
    if not np.all(np.isfinite(pred)):
        return SkippedModel(key=model.key, name=model.name, reason="non-finite prediction at a measured point")

    residuals = y - pred
    ssr = float(np.sum(residuals ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    aic, bic = information_criteria(ssr, n, k, float(np.max(np.abs(y))))

    x_span, y_span = _span(x), _span(y)
    params: Dict[str, float] = {}
    std_errors: Dict[str, Optional[float]] = {}
    rel_std_errors: Dict[str, Optional[float]] = {}
    diag = np.diag(pcov) if np.all(np.isfinite(pcov)) else np.full(k, np.nan)
    for i, (name, kind) in enumerate(zip(model.param_names, model.param_kinds)):
        value = float(popt[i])
        se = float(math.sqrt(diag[i])) if np.isfinite(diag[i]) and diag[i] >= 0 else None
        ref = {"scale": abs(value), "x_location": x_span, "y_offset": y_span}[kind]
        params[name] = value
        std_errors[name] = se
        rel_std_errors[name] = se / ref if se is not None and ref > 0 else None

    return ModelFit(
        key=model.key,
        name=model.name,
        formula=model.formula,
        n_points=n,
        n_params=k,
        params=params,
        std_errors=std_errors,
        rel_std_errors=rel_std_errors,
        r2=1.0 - ssr / sst if sst > 0 else 0.0,
        rmse=math.sqrt(ssr / n),
        aic=aic,
        bic=bic,
    )

def fit_library(
    x_data: Sequence[float],
    y_data: Sequence[float],
    iteration: int = 0,
    models: Optional[Sequence[CandidateModel]] = None,
) -> AnalysisResult:
    """Fit every candidate model and rank the successful fits by BIC."""
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    fits: List[ModelFit] = []
    skipped: List[SkippedModel] = []
    for model in models if models is not None else MODEL_LIBRARY.values():
        outcome = fit_model(model, x, y)
        if isinstance(outcome, ModelFit):
            fits.append(outcome)
        else:
            logger.debug("Skipped %s: %s", outcome.key, outcome.reason)
            skipped.append(outcome)

    fits.sort(key=lambda f: f.bic)
    if fits:
        best_bic = fits[0].bic
        fits = [f.model_copy(update={"delta_bic": f.bic - best_bic}) for f in fits]
    return AnalysisResult(
        iteration=iteration,
        n_points=len(x),
        fits=fits,
        skipped=skipped,
        delta_bic_runner_up=fits[1].delta_bic if len(fits) > 1 else None,
    )
