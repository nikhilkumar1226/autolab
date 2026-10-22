"""
Candidate models proposed by the Analyst LLM, compiled without ever executing model text.

The LLM may propose one new model per review: an expression in x with named parameters,
e.g. "A*exp(-k*x) + C". The text is never given to eval, exec, compile, sympy.parse_expr or
sympify (parse_expr and sympify call eval internally). Instead:

1. Gate. Text longer than 200 characters, or containing "__" anywhere, is rejected outright.
   Python's ast module then parses it with mode="eval" (parsing runs nothing), and every node
   is checked against a whitelist: int/float numbers (|c| <= 1e6), x, the declared parameters,
   pi, e, binary + - * / **, unary minus, and calls of sin, cos, tan, exp, log, sqrt, abs and
   radians by bare name with exactly one positional argument. Everything else is rejected:
   attributes (np.load, x.real), subscripts, strings, keyword arguments, lambdas, comparisons,
   comprehensions, undeclared names. A constant exponent must be a single number with
   |value| <= 10.
2. Evaluation. The validated tree is turned into nested closures over a fixed table of numpy
   operations and evaluated on numpy arrays. A name is looked up in the parameter vector,
   never in a Python namespace.
3. Structure checks, done numerically at fixed probe points (the same for every proposal):
   the expression must be finite somewhere and must depend on x and on every declared
   parameter. Parameter kinds (scale / x_location / y_offset), which set how the stopping rule
   normalises relative SE, come from numerical derivatives at those points: dy/dp = 1
   everywhere is a y_offset, dy/dp = +-dy/dx everywhere an x_location, anything else a scale.
   They are never taken from the LLM.

The result is an ordinary CandidateModel, so fitting, BIC and the relative standard errors
work as for the library.

An accepted LLM model must also add something: drop_equivalent() skips it while its curve
is indistinguishable from a library fit (or an earlier LLM fit) on the measured range.
Otherwise a duplicate or a nested superset of the best model would sit within
delta-BIC ~ ln n of it and block the delta-BIC criterion forever.
"""
import ast
import builtins
import functools
import keyword
import logging
import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field, FiniteFloat

from analysis.model_library import MODEL_LIBRARY, AnalysisResult, CandidateModel, ModelFit, ParamKind, SkippedModel

logger = logging.getLogger(__name__)

MAX_EXPRESSION_CHARS = 200
MAX_NODES = 64
MAX_PARAMS = 4
MAX_CONSTANT = 1e6
MAX_CONSTANT_EXPONENT = 10

FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
    "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "radians": np.radians,
}
CONSTANTS: Dict[str, float] = {"pi": math.pi, "e": math.e}
_BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
_PARAM_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")
# Python's own names are reserved too, so a reader never mistakes a parameter for a builtin.
RESERVED_NAMES = {"x", *FUNCTIONS, *CONSTANTS, *keyword.kwlist, *dir(builtins)}

# Fixed probe points for the structure checks: x over many scales and both signs, and two
# generic parameter vectors. The LLM's own initial guesses are added as a third vector; they
# can only make a kind harder to qualify for, never easier.
_PROBE_X = np.concatenate([-np.geomspace(1e3, 1e-2, 11), [0.0], np.geomspace(1e-2, 1e3, 21)])
_PROBE_PARAMS = ((1.37, 0.83, 1.91, 0.57), (0.61, 1.73, 0.89, 1.29))
_MIN_PROBES = 3


class ParameterGuess(BaseModel):
    name: str = Field(max_length=32)
    initial_guess: FiniteFloat = 1.0


class CandidateModelSpec(BaseModel):
    """A model proposed by the Analyst LLM. Sizes are only bounded here; compile_candidate is the gate."""
    name: str = Field(min_length=1, max_length=80, description="Short descriptive name")
    expression: str = Field(
        min_length=1, max_length=1000,
        description="Arithmetic in x, e.g. A*exp(-k*x) + C. Allowed: numbers, x, the parameters, pi, e, "
                    "+ - * / ** and unary minus, sin cos tan exp log sqrt abs radians (trig in radians)",
    )
    parameters: List[ParameterGuess] = Field(min_length=1, max_length=8, description="Free parameters with initial guesses")
    rationale: str = Field(default="", max_length=600)


class FormulaRejected(ValueError):
    """The proposed expression is outside the whitelist. Logged and recorded; the run continues."""


# ---------------------------------------------------------------------------
# 1. Gate: length, "__", ast parse and the node whitelist
# ---------------------------------------------------------------------------

def _single_number(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _single_number(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    return None


def _parse(text: str, params: Sequence[str]) -> ast.Expression:
    if len(text) > MAX_EXPRESSION_CHARS:
        raise FormulaRejected(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    if "__" in text:
        raise FormulaRejected("'__' is not allowed anywhere in the expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise FormulaRejected(f"not a valid expression ({exc.msg})") from None
    except (ValueError, RecursionError, MemoryError):
        raise FormulaRejected("not a valid expression") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise FormulaRejected(f"expression too complex ({len(nodes)} syntax nodes, limit {MAX_NODES})")

    allowed_names = {"x", *CONSTANTS, *params}
    call_targets = {id(n.func) for n in nodes if isinstance(n, ast.Call)}
    for node in nodes:
        if isinstance(node, (ast.Expression, ast.Load, ast.USub, *_BINARY)):
            continue
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY:
                raise FormulaRejected(f"operator {type(node.op).__name__} is not allowed")
            if isinstance(node.op, ast.Pow) and not any(isinstance(n, ast.Name) for n in ast.walk(node.right)):
                value = _single_number(node.right)
                if value is None or abs(value) > MAX_CONSTANT_EXPONENT:
                    raise FormulaRejected(
                        f"a constant exponent must be a single number with |value| <= {MAX_CONSTANT_EXPONENT}"
                    )
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ast.USub):
                raise FormulaRejected(f"operator {type(node.op).__name__} is not allowed (only unary minus)")
        elif isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FormulaRejected(f"constant {value!r} is not a number")
            if not math.isfinite(value) or abs(value) > MAX_CONSTANT:
                raise FormulaRejected(f"constant {value!r} is outside |c| <= {MAX_CONSTANT:g}")
        elif isinstance(node, ast.Name):
            if node.id in FUNCTIONS:
                if id(node) not in call_targets:
                    raise FormulaRejected(f"function '{node.id}' must be called with one argument")
            elif node.id not in allowed_names:
                raise FormulaRejected(f"unknown name '{node.id}' (allowed: x, pi, e, the declared parameters)")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise FormulaRejected(f"only {', '.join(FUNCTIONS)} may be called, by bare name")
            if node.keywords or len(node.args) != 1 or isinstance(node.args[0], ast.Starred):
                raise FormulaRejected(f"{node.func.id}() takes exactly one positional argument")
        else:
            raise FormulaRejected(f"{type(node).__name__} is not allowed")

    names = {n.id for n in nodes if isinstance(n, ast.Name)}
    if "x" not in names:
        raise FormulaRejected("the expression must depend on x")
    unused = [p for p in params if p not in names]
    if unused:
        raise FormulaRejected(f"declared parameter(s) not used: {', '.join(unused)}")
    return tree


def _check_params(params: Sequence[str]) -> None:
    if not 1 <= len(params) <= MAX_PARAMS:
        raise FormulaRejected(f"between 1 and {MAX_PARAMS} parameters are allowed (got {len(params)})")
    if len(set(params)) != len(params):
        raise FormulaRejected("parameter names must be unique")
    for p in params:
        if not _PARAM_NAME.match(p) or "__" in p or p in RESERVED_NAMES:
            raise FormulaRejected(f"invalid parameter name '{p}' (letters, digits, '_'; not a reserved name)")


# ---------------------------------------------------------------------------
# 2. Tree-walking numpy evaluator (no eval / exec / compile / sympy)
# ---------------------------------------------------------------------------

Evaluator = Callable[[np.ndarray, Sequence[float]], np.ndarray]


def _build(node: ast.AST, index: Dict[str, int]) -> Evaluator:
    """Closures over the validated tree. Only node types the gate accepted can reach here."""
    if isinstance(node, ast.Constant):
        value = float(node.value)
        return lambda x, p: value
    if isinstance(node, ast.Name):
        if node.id == "x":
            return lambda x, p: x
        if node.id in CONSTANTS:
            value = CONSTANTS[node.id]
            return lambda x, p: value
        i = index[node.id]
        return lambda x, p: p[i]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _build(node.operand, index)
        return lambda x, p: np.negative(inner(x, p))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        op, left, right = _BINARY[type(node.op)], _build(node.left, index), _build(node.right, index)
        return lambda x, p: op(left(x, p), right(x, p))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
        fn, inner = FUNCTIONS[node.func.id], _build(node.args[0], index)
        return lambda x, p: fn(inner(x, p))
    raise FormulaRejected(f"{type(node).__name__} is not allowed")  # unreachable after the gate


def _evaluate(tree: Evaluator, x: np.ndarray, p: Sequence[float]) -> np.ndarray:
    with np.errstate(all="ignore"):
        out = np.asarray(tree(x, p), dtype=float)
    return out if out.shape == x.shape else np.array(np.broadcast_to(out, x.shape))


# ---------------------------------------------------------------------------
# 3. Numerical structure checks and parameter kinds
# ---------------------------------------------------------------------------

def _structure(tree: Evaluator, params: Sequence[str], guesses: Sequence[float]) -> Tuple[ParamKind, ...]:
    k = len(params)
    vectors = [np.array(v[:k], dtype=float) for v in _PROBE_PARAMS]
    if any(not np.allclose(np.asarray(guesses, dtype=float), v) for v in vectors):
        vectors.append(np.asarray(guesses, dtype=float))

    x = _PROBE_X
    hx = 1e-6 * np.maximum(1.0, np.abs(x))
    f_all, dx_all, dp_all, x_all, p_all = [], [], [], [], []
    for v in vectors:
        f = _evaluate(tree, x, v)
        dfdp = []
        with np.errstate(all="ignore"):
            dfdx = (_evaluate(tree, x + hx, v) - _evaluate(tree, x - hx, v)) / (2 * hx)
            for i in range(k):
                h = 1e-6 * max(1.0, abs(v[i]))
                up, down = v.copy(), v.copy()
                up[i] += h
                down[i] -= h
                dfdp.append((_evaluate(tree, x, up) - _evaluate(tree, x, down)) / (2 * h))
        dfdp = np.array(dfdp)
        valid = np.isfinite(f) & np.isfinite(dfdx) & np.all(np.isfinite(dfdp), axis=0)
        f_all.append(f[valid])
        dx_all.append(dfdx[valid])
        dp_all.append(dfdp[:, valid])
        x_all.append(x[valid])
        p_all.append(np.repeat(np.abs(v)[:, None], int(valid.sum()), axis=1))

    f, dfdx, dfdp = np.concatenate(f_all), np.concatenate(dx_all), np.concatenate(dp_all, axis=1)
    xs, ps = np.concatenate(x_all), np.concatenate(p_all, axis=1)
    if f.size < _MIN_PROBES:
        raise FormulaRejected("the expression is undefined (not finite) at almost every probe point")

    # Sensitivities: the change in y for a relative change in the variable, against y's own size.
    scale = 1e-7 * (1.0 + np.abs(f))
    if not np.any(np.abs(dfdx) * np.maximum(1.0, np.abs(xs)) > scale):
        raise FormulaRejected("the expression does not depend on x")
    dead = [p for i, p in enumerate(params) if not np.any(np.abs(dfdp[i]) * np.maximum(1.0, ps[i]) > scale)]
    if dead:
        raise FormulaRejected(f"parameter(s) {', '.join(dead)} have no effect on the expression")

    tol = 1e-5 * (np.abs(dfdp) + np.abs(dfdx)) + 1e-6 * (1.0 + np.abs(f))
    kinds: List[ParamKind] = []
    for i in range(k):
        if np.all(np.abs(dfdp[i] - 1.0) <= tol[i]):
            kinds.append("y_offset")
        elif np.all(np.abs(dfdp[i] + dfdx) <= tol[i]) or np.all(np.abs(dfdp[i] - dfdx) <= tol[i]):
            kinds.append("x_location")
        else:
            kinds.append("scale")
    return tuple(kinds)


# ---------------------------------------------------------------------------
# LaTeX, printed from the validated tree
# ---------------------------------------------------------------------------

_ADD, _MUL, _NEG, _POW, _ATOM = 1, 2, 3, 4, 5
_GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta", "kappa", "mu", "nu", "rho", "sigma",
          "tau", "phi", "omega", "Gamma", "Delta", "Theta", "Phi", "Omega"}


def _num(v: float) -> str:
    return f"{v:.4g}"


def latex_name(name: str) -> str:
    """x0 -> x_{0}, I_max -> I_{max}, theta -> \\theta, rate -> \\mathrm{rate}."""
    def word(w: str) -> str:
        return f"\\{w}" if w in _GREEK else w if len(w) == 1 else f"\\mathrm{{{w}}}"
    m = re.match(r"^([A-Za-z]+)(\d+)$", name)
    if m:
        return f"{word(m.group(1))}_{{{m.group(2)}}}"
    if "_" in name:
        head, tail = name.split("_", 1)
        return f"{word(head)}_{{{tail.replace('_', ',')}}}" if head else word(tail)
    return word(name)


def _paren(text: str) -> str:
    return f"\\left({text}\\right)"


def _tex(node: ast.AST) -> Tuple[str, int]:
    if isinstance(node, ast.Constant):
        return _num(float(node.value)), _ATOM
    if isinstance(node, ast.Name):
        return {"pi": "\\pi", "e": "e", "x": "x"}.get(node.id) or latex_name(node.id), _ATOM
    if isinstance(node, ast.UnaryOp):
        text, prec = _tex(node.operand)
        return "-" + (_paren(text) if prec <= _ADD else text), _NEG
    if isinstance(node, ast.Call):
        name, (text, _) = node.func.id, _tex(node.args[0])
        if name == "exp":
            return f"e^{{{text}}}", _POW
        if name == "sqrt":
            return f"\\sqrt{{{text}}}", _ATOM
        if name == "abs":
            return f"\\left|{text}\\right|", _ATOM
        if name == "radians":
            return f"\\frac{{\\pi}}{{180}}{_paren(text)}", _MUL
        return f"\\{'ln' if name == 'log' else name}{_paren(text)}", _ATOM
    left, lp = _tex(node.left)
    right, rp = _tex(node.right)
    op = type(node.op)
    if op in (ast.Add, ast.Sub):
        if rp <= _ADD or rp == _NEG:
            right = _paren(right)
        return f"{left} {'+' if op is ast.Add else '-'} {right}", _ADD
    if op is ast.Mult:
        left = _paren(left) if lp <= _ADD else left
        right = _paren(right) if rp <= _ADD or rp == _NEG else right
        joiner = " \\cdot " if right[:1].isdigit() else "\\,"
        return f"{left}{joiner}{right}", _MUL
    if op is ast.Div:
        return f"\\frac{{{left}}}{{{right}}}", _MUL
    return f"{_paren(left) if lp < _ATOM else left}^{{{right}}}", _POW


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------

def compile_candidate(spec: CandidateModelSpec, key: str, name: Optional[str] = None) -> CandidateModel:
    """Validate and compile an LLM-proposed model. Raises FormulaRejected with a reason."""
    raw = spec.expression.strip()
    if len(raw) > MAX_EXPRESSION_CHARS:
        raise FormulaRejected(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    text = raw.replace("^", "**")
    params = [p.name for p in spec.parameters]
    _check_params(params)
    tree = _parse(text, params)
    evaluate = _build(tree.body, {p: i for i, p in enumerate(params)})
    guesses = [float(p.initial_guess) for p in spec.parameters]
    kinds = _structure(evaluate, params, guesses)

    def func(x, *p):
        return _evaluate(evaluate, np.asarray(x, dtype=float), p)

    def initial_guess(x, y):
        """The LLM's guesses or all ones, whichever fits the data better."""
        best, best_ssr = guesses, math.inf
        for start in (guesses, [1.0] * len(params)):
            ssr = float(np.sum((func(x, *start) - y) ** 2))
            if math.isfinite(ssr) and ssr < best_ssr:
                best, best_ssr = start, ssr
        return best

    symbolic = _tex(tree.body)[0]

    def latex(values: Dict[str, float]) -> str:
        listed = ",\\ ".join(f"{latex_name(p)} = {_num(values[p])}" for p in params)
        return f"y = {symbolic},\\quad {listed}"

    return CandidateModel(
        key=key,
        name=name or f"{spec.name} (LLM-proposed)",
        formula=f"y = {text}",
        param_names=tuple(params),
        param_kinds=kinds,
        func=func,
        initial_guess=initial_guess,
        bounds=lambda x, y: ([-np.inf] * len(params), [np.inf] * len(params)),
        latex=latex,
        expression=text,
    )


@functools.lru_cache(maxsize=64)
def _recompile(key: str, name: str, expression: str, params: Tuple[str, ...]) -> CandidateModel:
    spec = CandidateModelSpec(
        name=name, expression=expression, parameters=[ParameterGuess(name=p) for p in params],
    )
    return compile_candidate(spec, key, name=name)


def model_for_fit(fit: ModelFit) -> Optional[CandidateModel]:
    """The model behind a fit: a library model, or an LLM model recompiled through the same gate."""
    if fit.origin == "library":
        return MODEL_LIBRARY.get(fit.key)
    if fit.expression is None:
        return None
    try:
        return _recompile(fit.key, fit.name, fit.expression, tuple(fit.params))
    except FormulaRejected as exc:  # e.g. a hand-edited record
        logger.error("Cannot rebuild LLM model %s from the record: %s", fit.key, exc)
        return None


def predict_fit(fit: ModelFit, x: Sequence[float]) -> Optional[np.ndarray]:
    model = model_for_fit(fit)
    if model is None:
        return None
    with np.errstate(all="ignore"):
        return np.asarray(model.predict(x, fit.params), dtype=float)


def prediction_se(fit: ModelFit, x: float) -> Optional[float]:
    """Standard error of the fitted curve at x from the parameter covariance (delta method)."""
    model = model_for_fit(fit)
    if model is None or fit.covariance is None:
        return None
    names = list(model.param_names)
    theta = np.array([fit.params[p] for p in names], dtype=float)
    grad = np.empty(len(names))
    with np.errstate(all="ignore"):
        for i in range(len(names)):
            h = 1e-6 * max(1.0, abs(theta[i]))
            up, down = theta.copy(), theta.copy()
            up[i] += h
            down[i] -= h
            grad[i] = (model.func(np.array([x]), *up)[0] - model.func(np.array([x]), *down)[0]) / (2 * h)
    variance = float(grad @ np.asarray(fit.covariance, dtype=float) @ grad)
    return math.sqrt(variance) if math.isfinite(variance) and variance >= 0 else None


# ---------------------------------------------------------------------------
# Equivalence filter
# ---------------------------------------------------------------------------

def _rerank(result: AnalysisResult, fits: List[ModelFit], skipped: List[SkippedModel]) -> AnalysisResult:
    fits = sorted(fits, key=lambda f: f.bic)
    if fits:
        best_bic = fits[0].bic
        fits = [f.model_copy(update={"delta_bic": f.bic - best_bic}) for f in fits]
    return result.model_copy(update={
        "fits": fits,
        "skipped": skipped,
        "delta_bic_runner_up": fits[1].delta_bic if len(fits) > 1 else None,
    })


def drop_equivalent(
    result: AnalysisResult,
    x: Sequence[float],
    y: Sequence[float],
    sigma_ratio: float = 0.5,
    noise_floor_rel: float = 0.005,
) -> AnalysisResult:
    """
    Skip every LLM fit whose curve stays within sigma_ratio x sigma of a library fit, or of
    an LLM fit accepted earlier, everywhere on the measured x-range. sigma is the smallest
    residual std of any fit, floored at noise_floor_rel x the y-span. Library fits are never
    removed, so without LLM models the ranking is exactly fit_library's.
    """
    llm = sorted((f for f in result.fits if f.origin == "llm"), key=lambda f: f.key)
    if not llm:
        return result
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    grid = np.linspace(float(np.min(xs)), float(np.max(xs)), 200)
    sigma = max(min(f.residual_std for f in result.fits), noise_floor_rel * float(np.ptp(ys)))
    tolerance = sigma_ratio * sigma

    references = [f for f in result.fits if f.origin == "library"]
    curves = {f.key: predict_fit(f, grid) for f in result.fits}
    kept, skipped = list(references), list(result.skipped)
    for fit in llm:
        mine = curves[fit.key]
        twin = None
        for ref in references:
            other = curves[ref.key]
            if mine is None or other is None or not (np.all(np.isfinite(mine)) and np.all(np.isfinite(other))):
                continue
            gap = float(np.max(np.abs(mine - other)))
            if gap <= tolerance:
                twin = (ref, gap)
                break
        if twin is not None:
            ref, gap = twin
            skipped.append(SkippedModel(
                key=fit.key, name=fit.name,
                reason=f"indistinguishable from {ref.name} on the measured range "
                       f"(max difference {gap:.3g} <= {sigma_ratio:g} sigma = {tolerance:.3g})",
            ))
            continue
        kept.append(fit)
        references.append(fit)
    return _rerank(result, kept, skipped)
