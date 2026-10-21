"""
Tests for LLM-proposed candidate models: the formula whitelist, the eval-free compiler,
structural parameter kinds, fitting, and the equivalence filter.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from analysis.candidate_models import (
    CandidateModelSpec,
    FormulaRejected,
    ParameterGuess,
    compile_candidate,
    drop_equivalent,
    model_for_fit,
)
from analysis.model_library import MODEL_LIBRARY, ModelFit, fit_library, fit_model
from experiments.registry import get_experiment

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def spec(expression: str, *params, guesses=None) -> CandidateModelSpec:
    guesses = guesses or {}
    return CandidateModelSpec(
        name="test model", expression=expression,
        parameters=[ParameterGuess(name=p, initial_guess=guesses.get(p, 1.0)) for p in params],
    )


def compile_(expression: str, *params, **kw):
    return compile_candidate(spec(expression, *params, **kw), key="llm_1")


# --- the whitelist ---------------------------------------------------------------

@pytest.mark.parametrize("expression, params, reason", [
    ("__import__('os')", ["A"], "only sin, cos, exp, log and sqrt may be called"),
    ("np.load('x')", ["A"], "only sin, cos, exp, log and sqrt may be called"),
    ("().__class__", ["A"], "Attribute is not allowed"),
    ("x.__class__", ["A"], "Attribute is not allowed"),
    ("A*x.real", ["A"], "Attribute is not allowed"),
    ("exec('import os')", ["A"], "only sin, cos"),
    ("eval('x')", ["A"], "only sin, cos"),
    ("open('/etc/passwd')", ["A"], "only sin, cos"),
    ("getattr(x, 'mro')", ["A"], "only sin, cos"),
    ("lambda: A*x", ["A"], "Lambda is not allowed"),
    ("[A*x for x in A]", ["A"], "ListComp is not allowed"),
    ("A*x if A else x", ["A"], "IfExp is not allowed"),
    ("A*x; import os", ["A"], "not a valid expression"),
    ("A*x < 1", ["A"], "Compare is not allowed"),
    ("A*x[0]", ["A"], "Subscript is not allowed"),
    ("A @ x", ["A"], "MatMult is not allowed"),
    ("A*x % 2", ["A"], "Mod is not allowed"),
    ("'A*x'", ["A"], "is not a number"),
    ("A*x + True", ["A"], "is not a number"),
    ("A*x + 2j", ["A"], "is not a number"),
    ("A*x**(1j)", ["A"], "constant exponent must be a single number"),
    ("sin(x=A)", ["A"], "takes exactly one positional argument"),
    ("A*sin(x, x)", ["A"], "takes exactly one positional argument"),
    ("A*sin(*x)", ["A"], "takes exactly one positional argument"),
    ("A*sin*x", ["A"], "must be called with one argument"),
    ("A*sin(x)(x)", ["A"], "only sin, cos"),
    ("A*y", ["A"], "unknown name 'y'"),
    ("A*x + __builtins__", ["A"], "unknown name '__builtins__'"),
    ("A*x + B", ["A"], "unknown name 'B'"),
    ("A*x", ["A", "B"], "declared parameter(s) not used: B"),
    ("A + 2", ["A"], "must depend on x"),
    ("A*x", ["x"], "invalid parameter name 'x'"),
    ("A*x + pi", ["pi", "A"], "invalid parameter name 'pi'"),
    ("A*x", ["__class__"], "invalid parameter name"),
    ("A*x", ["lambda"], "invalid parameter name"),
    ("A*x + B*x + C*x + D*x + F", ["A", "B", "C", "D", "F"], "between 1 and 4 parameters"),
    ("9**9**9*A*x", ["A"], "constant exponent must be a single number"),
    ("A*x**(9**9)", ["A"], "constant exponent must be a single number"),
    ("A*x**100", ["A"], "|value| <= 10"),
    ("1e7*A*x", ["A"], "outside |c|"),
    ("A*log(-1)*x", ["A"], "complex-valued"),
    ("A*x + 1/0", ["A"], "undefined"),
    ("A*x/A", ["A"], "no effect after simplification"),
    ("A*x" + " + A*x" * 60, ["A"], "longer than 200 characters"),
])
def test_dangerous_or_invalid_formulas_are_rejected(expression, params, reason):
    with pytest.raises(FormulaRejected) as info:
        compile_(expression, *params)
    assert reason in str(info.value)


def test_the_compiler_never_uses_eval_exec_or_lambdify():
    """Static check over the whole project (tests and virtualenvs excluded)."""
    banned_calls = {"eval", "exec", "compile", "lambdify", "__import__"}
    offenders = []
    for path in PROJECT_ROOT.rglob("*.py"):
        rel = path.relative_to(PROJECT_ROOT)
        if any(part in {".venv", "venv", "tests", "reference"} for part in rel.parts):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                is_regex_compile = isinstance(fn, ast.Attribute) and name == "compile"
                if name in banned_calls and not is_regex_compile:
                    offenders.append(f"{rel}:{node.lineno}: {name}()")
            if isinstance(node, ast.ImportFrom) and any(a.name == "lambdify" for a in node.names):
                offenders.append(f"{rel}:{node.lineno}: imports lambdify")
    assert offenders == []


# --- compiling and evaluating --------------------------------------------------

@pytest.mark.parametrize("expression, params, values, x, expected", [
    ("A*exp(-k*x) + C", ["A", "k", "C"], [1000.0, 0.04, 15.0], 25.0, 1000.0 * np.exp(-1.0) + 15.0),
    ("A*cos(pi*(x - x0)/180)**2 + C", ["A", "x0", "C"], [1000.0, 20.0, 15.0], 80.0, 1000.0 * 0.25 + 15.0),
    ("A*x^b", ["A", "b"], [2.0, 3.0], 2.0, 16.0),
    ("A/(x + d)**2 + C", ["A", "d", "C"], [1e5, 5.0, 15.0], 45.0, 1e5 / 2500.0 + 15.0),
    ("A*sqrt(x) - log(x)*B + sin(x)", ["A", "B"], [3.0, 2.0], 4.0, 6.0 - 2.0 * np.log(4.0) + np.sin(4.0)),
    ("-A*x + -2.5", ["A"], [2.0], 3.0, -8.5),
])
def test_valid_formulas_compile_and_evaluate(expression, params, values, x, expected):
    model = compile_(expression, *params)
    out = model.func(np.array([x, x]), *values)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(expected, rel=1e-12)
    assert model.param_names == tuple(params) and model.key == "llm_1"
    assert model.expression == expression.replace("^", "**") and model.formula.startswith("y = ")


def test_parameter_kinds_are_inferred_from_structure():
    assert compile_("A*exp(-k*x) + C", "A", "k", "C").param_kinds == ("scale", "scale", "y_offset")
    assert compile_("A*cos(pi*(x - x0)/180)**2 + C", "A", "x0", "C").param_kinds == ("scale", "x_location", "y_offset")
    assert compile_("A/(x + d)**2", "A", "d").param_kinds == ("scale", "x_location")
    assert compile_("A*x**b - C", "A", "b", "C").param_kinds == ("scale", "scale", "scale")  # -C: dy/dC = -1


def test_latex_shows_the_formula_and_the_fitted_values():
    model = compile_("A*exp(-k*x) + C", "A", "k", "C")
    text = model.equation({"A": 1000.0, "k": 0.0401234, "C": 15.0})
    assert text.startswith("y = A e^{- k x} + C") and "k = 0.04012" in text


# --- fitting and ranking -------------------------------------------------------

def gaussian_peak(x):
    return 800.0 * np.exp(-((x - 40.0) ** 2) / (2 * 12.0 ** 2)) + 20.0


def test_a_missing_law_is_fitted_and_wins_the_ranking():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 100.0, 25)
    y = gaussian_peak(x) + rng.normal(0.0, 5.0, x.size)
    peak = compile_candidate(
        spec("A*exp(-(x - x0)**2/(2*w**2)) + C", "A", "x0", "w", "C",
             guesses={"A": 700.0, "x0": 45.0, "w": 10.0, "C": 0.0}),
        key="llm_1",
    )
    result = drop_equivalent(fit_library(x, y, models=[*MODEL_LIBRARY.values(), peak]), x, y)

    best = result.best
    assert best.key == "llm_1" and best.origin == "llm" and best.expression is not None
    assert best.params["x0"] == pytest.approx(40.0, abs=1.0) and abs(best.params["w"]) == pytest.approx(12.0, abs=1.0)
    assert result.delta_bic_runner_up > 10
    assert peak.param_kinds == ("scale", "x_location", "scale", "y_offset")


def test_a_library_duplicate_is_skipped_as_equivalent(simulate):
    spec_ = get_experiment("beer_lambert_law")
    x = np.linspace(0.0, 100.0, 12)
    y = simulate(spec_, x, seed=0)
    duplicate = compile_candidate(
        spec("I0*exp(-a*x) + B", "I0", "a", "B", guesses={"I0": 900.0, "a": 0.05, "B": 10.0}), key="llm_1",
    )
    library_only = fit_library(x, y)
    result = drop_equivalent(fit_library(x, y, models=[*MODEL_LIBRARY.values(), duplicate]), x, y)

    assert [f.key for f in result.fits] == [f.key for f in library_only.fits]
    assert result.delta_bic_runner_up == pytest.approx(library_only.delta_bic_runner_up)
    [skipped] = [s for s in result.skipped if s.key == "llm_1"]
    assert "indistinguishable from Exponential decay" in skipped.reason


def test_without_llm_models_the_filter_changes_nothing(simulate):
    x = np.linspace(5.0, 100.0, 10)
    y = simulate(get_experiment("inverse_square_law"), x, seed=3)
    result = fit_library(x, y)
    assert drop_equivalent(result, x, y) == result


def test_an_llm_fit_is_rebuilt_from_the_record():
    x = np.linspace(0.0, 100.0, 15)
    y = gaussian_peak(x)
    peak = compile_candidate(
        spec("A*exp(-(x - x0)**2/(2*w**2)) + C", "A", "x0", "w", "C",
             guesses={"A": 700.0, "x0": 45.0, "w": 10.0, "C": 0.0}),
        key="llm_2", name="Gaussian peak (LLM-proposed)",
    )
    fit = fit_model(peak, x, y)
    loaded = ModelFit.model_validate_json(fit.model_dump_json())

    rebuilt = model_for_fit(loaded)
    grid = np.linspace(0.0, 100.0, 7)
    np.testing.assert_allclose(rebuilt.predict(grid, loaded.params), peak.predict(grid, fit.params))
    assert model_for_fit(fit_library(x, y).fits[0]) is MODEL_LIBRARY[fit_library(x, y).fits[0].key]


def test_a_tampered_record_expression_is_not_rebuilt():
    fit = ModelFit(
        key="llm_1", name="evil", formula="", origin="llm", expression="__import__('os').system('true')",
        n_points=5, n_params=1, params={"A": 1.0}, std_errors={"A": None}, rel_std_errors={"A": None},
        r2=0.0, rmse=1.0, aic=0.0, bic=0.0,
    )
    assert model_for_fit(fit) is None
