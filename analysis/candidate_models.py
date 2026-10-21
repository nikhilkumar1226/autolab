"""
Candidate models proposed by the Analyst LLM, compiled without ever executing model text.

The LLM may propose one new model per review: an expression in x with named parameters,
e.g. "A*exp(-k*x) + C". The text passes two independent whitelists and is then compiled
by walking its expression tree:

1. String/AST gate. Python's ast module parses the text (parsing runs nothing). Allowed:
   int/float constants (|c| <= 1e6), x, the declared parameters, pi, + - * / ** and unary
   +/-, and calls of sin, cos, exp, log, sqrt by bare name with one positional argument.
   Everything else is rejected: attribute access (np.load), dunders, subscripts, strings,
   keyword arguments, comparisons, lambdas, comprehensions. A constant exponent must be a
   single number with |value| <= 10, so "9**9**9" cannot make sympy compute a huge integer.
2. sympy.parse_expr, with a namespace that holds only those symbols and functions. The
   resulting tree is walked again: only Add, Mul, Pow, numbers, pi and sin/cos/exp/log may
   appear, every parameter must still be present, and nothing may be undefined or complex.
3. A small tree-walking evaluator turns the sympy tree into a numpy function (no eval,
   exec or lambdify).

The result is an ordinary CandidateModel, so fitting, BIC and the relative standard errors
work as for the library. Parameter kinds (scale / x_location / y_offset), which set how
the stopping rule normalises relative SE, are inferred from the expression's structure,
never taken from the LLM.

An accepted LLM model must also add something: drop_equivalent() skips it while its curve
is indistinguishable from a library fit (or an earlier LLM fit) on the measured range.
Otherwise a duplicate or a nested superset of the best model would sit within
delta-BIC ~ ln n of it and block the delta-BIC criterion forever.
"""
import ast
import functools
import keyword
import logging
import math
import operator
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from pydantic import BaseModel, Field, FiniteFloat
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from analysis.model_library import MODEL_LIBRARY, AnalysisResult, CandidateModel, ModelFit, ParamKind, SkippedModel

logger = logging.getLogger(__name__)

MAX_EXPRESSION_CHARS = 200
MAX_NODES = 64
MAX_PARAMS = 4
MAX_CONSTANT = 1e6
MAX_CONSTANT_EXPONENT = 10

X = sp.Symbol("x")
FUNCTIONS = {"sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt}
CONSTANTS = {"pi": sp.pi}
_NUMPY = {sp.sin: np.sin, sp.cos: np.cos, sp.exp: np.exp, sp.log: np.log}
_TREE_FUNCS = (sp.Add, sp.Mul, sp.Pow, *_NUMPY)
_PARAM_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")
# sympy's own single-letter names are reserved too, so a reader never confuses them.
RESERVED_NAMES = {"x", *FUNCTIONS, *CONSTANTS, "E", "I", "N", "S", "O", "Q", "oo", "nan", "zoo", *keyword.kwlist}


class ParameterGuess(BaseModel):
    name: str = Field(max_length=32)
    initial_guess: FiniteFloat = 1.0


class CandidateModelSpec(BaseModel):
    """A model proposed by the Analyst LLM. Sizes are only bounded here; compile_candidate is the gate."""
    name: str = Field(min_length=1, max_length=80, description="Short descriptive name")
    expression: str = Field(
        min_length=1, max_length=1000,
        description="sympy syntax in x, e.g. A*exp(-k*x) + C. Allowed: numbers, x, the parameters, pi, "
                    "+ - * / **, sin cos exp log sqrt (trig in radians)",
    )
    parameters: List[ParameterGuess] = Field(min_length=1, max_length=8, description="Free parameters with initial guesses")
    rationale: str = Field(default="", max_length=600)


class FormulaRejected(ValueError):
    """The proposed expression is outside the whitelist. Logged and recorded; the run continues."""


# ---------------------------------------------------------------------------
# 1. String / AST gate
# ---------------------------------------------------------------------------

def _single_number(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _single_number(node.operand)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub) else inner)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    return None


def _check_ast(text: str, params: Sequence[str]) -> None:
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise FormulaRejected(f"not a valid expression ({exc.msg})") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise FormulaRejected(f"expression too complex ({len(nodes)} syntax nodes, limit {MAX_NODES})")

    allowed_names = {"x", *CONSTANTS, *params}
    call_targets = {id(n.func) for n in nodes if isinstance(n, ast.Call)}
    operators = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub, ast.Load)
    for node in nodes:
        if isinstance(node, (ast.Expression, *operators)):
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                raise FormulaRejected(f"operator {type(node.op).__name__} is not allowed")
            if isinstance(node.op, ast.Pow) and not any(isinstance(n, ast.Name) for n in ast.walk(node.right)):
                value = _single_number(node.right)
                if value is None or abs(value) > MAX_CONSTANT_EXPONENT:
                    raise FormulaRejected(
                        f"a constant exponent must be a single number with |value| <= {MAX_CONSTANT_EXPONENT}"
                    )
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise FormulaRejected(f"operator {type(node.op).__name__} is not allowed")
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
                raise FormulaRejected(f"unknown name '{node.id}' (allowed: x, pi, the declared parameters)")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise FormulaRejected("only sin, cos, exp, log and sqrt may be called, by bare name")
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


def _check_params(params: Sequence[str]) -> None:
    if not 1 <= len(params) <= MAX_PARAMS:
        raise FormulaRejected(f"between 1 and {MAX_PARAMS} parameters are allowed (got {len(params)})")
    if len(set(params)) != len(params):
        raise FormulaRejected("parameter names must be unique")
    for p in params:
        if not _PARAM_NAME.match(p) or p in RESERVED_NAMES:
            raise FormulaRejected(f"invalid parameter name '{p}' (letters, digits, '_'; not a reserved name)")


# ---------------------------------------------------------------------------
# 2. sympy parse with a whitelist-only namespace, then a tree check
# ---------------------------------------------------------------------------

def _to_sympy(text: str, params: Sequence[str]) -> sp.Expr:
    local = {"x": X, **{p: sp.Symbol(p) for p in params}, **FUNCTIONS, **CONSTANTS}
    namespace = {"Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational, "Symbol": sp.Symbol,
                 "__builtins__": {}}
    try:
        expr = parse_expr(text, local_dict=local, global_dict=namespace, transformations=standard_transformations)
    except Exception as exc:  # the text already passed the gate; any failure here is a rejection
        raise FormulaRejected(f"sympy could not parse it ({type(exc).__name__})") from None
    if not isinstance(expr, sp.Expr):
        raise FormulaRejected("not a mathematical expression")
    return expr


def _check_tree(expr: sp.Expr, params: Sequence[str]) -> None:
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise FormulaRejected("the expression is undefined (e.g. a division by zero)")
    if expr.has(sp.I):
        raise FormulaRejected("the expression is complex-valued")
    for node in sp.preorder_traversal(expr):
        if node.is_Symbol or node is sp.pi or node.is_Rational or node.is_Float:
            continue
        if node.func not in _TREE_FUNCS:
            raise FormulaRejected(f"{node.func.__name__} is not allowed")
    free = {s.name for s in expr.free_symbols}
    if "x" not in free:
        raise FormulaRejected("the expression does not depend on x after simplification")
    lost = [p for p in params if p not in free]
    if lost:
        raise FormulaRejected(f"parameter(s) {', '.join(lost)} have no effect after simplification")
    if free - {"x", *params}:
        raise FormulaRejected(f"unexpected symbols: {', '.join(sorted(free - {'x', *params}))}")


# ---------------------------------------------------------------------------
# 3. Tree-walking numpy evaluator (no eval / exec / lambdify)
# ---------------------------------------------------------------------------

Evaluator = Callable[[np.ndarray, Sequence[float]], np.ndarray]


def _evaluator(expr: sp.Expr, index: Dict[sp.Symbol, int]) -> Evaluator:
    if expr == X:
        return lambda x, p: x
    if expr.is_Symbol:
        i = index[expr]
        return lambda x, p: p[i]
    if expr.is_Rational or expr.is_Float or expr is sp.pi:
        value = float(expr)
        return lambda x, p: value
    args = [_evaluator(a, index) for a in expr.args]
    if expr.func is sp.Add:
        return lambda x, p: functools.reduce(operator.add, (f(x, p) for f in args))
    if expr.func is sp.Mul:
        return lambda x, p: functools.reduce(operator.mul, (f(x, p) for f in args))
    if expr.func is sp.Pow:
        base, exponent = args
        return lambda x, p: np.power(base(x, p), exponent(x, p))
    if expr.func in _NUMPY:
        fn, (inner,) = _NUMPY[expr.func], args
        return lambda x, p: fn(inner(x, p))
    raise FormulaRejected(f"{expr.func.__name__} is not allowed")  # unreachable after _check_tree


def _kind(expr: sp.Expr, p: sp.Symbol) -> ParamKind:
    d = sp.diff(expr, p)
    if sp.simplify(d - 1) == 0:
        return "y_offset"
    dx = sp.diff(expr, X)
    if dx != 0 and (sp.simplify(d + dx) == 0 or sp.simplify(d - dx) == 0):
        return "x_location"
    return "scale"


def _num(v: float) -> str:
    return f"{v:.4g}"


def compile_candidate(spec: CandidateModelSpec, key: str, name: Optional[str] = None) -> CandidateModel:
    """Validate and compile an LLM-proposed model. Raises FormulaRejected with a reason."""
    text = spec.expression.strip().replace("^", "**")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise FormulaRejected(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    params = [p.name for p in spec.parameters]
    _check_params(params)
    _check_ast(text, params)
    expr = _to_sympy(text, params)
    _check_tree(expr, params)

    symbols = [sp.Symbol(p) for p in params]
    tree = _evaluator(expr, {s: i for i, s in enumerate(symbols)})
    kinds = tuple(_kind(expr, s) for s in symbols)
    guesses = [float(p.initial_guess) for p in spec.parameters]

    def func(x, *p):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            out = np.asarray(tree(x, p), dtype=float)
        return out if out.shape == x.shape else np.full(x.shape, out)

    def initial_guess(x, y):
        """The LLM's guesses or all ones, whichever fits the data better."""
        best, best_ssr = guesses, math.inf
        for start in (guesses, [1.0] * len(params)):
            ssr = float(np.sum((func(x, *start) - y) ** 2))
            if math.isfinite(ssr) and ssr < best_ssr:
                best, best_ssr = start, ssr
        return best

    symbolic = sp.latex(expr)

    def latex(values: Dict[str, float]) -> str:
        listed = ",\\ ".join(f"{sp.latex(s)} = {_num(values[s.name])}" for s in symbols)
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
