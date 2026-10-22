"""
Property-based tests (hypothesis): for ANY input, nothing invalid or out of bounds reaches the rig.

The spy rig records every command that reaches the hardware; safety.spy.unsafe_commands checks
them against the envelope independently of the guard's code.
"""
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from agents.models import PlannerOutput, StrategyDecision
from agents.tools import LabTools
from analysis.candidate_models import CandidateModelSpec, FormulaRejected, ParameterGuess, compile_candidate
from config import SafetyConfig, SimulationConfig
from experiments.registry import get_experiment, list_experiments
from hardware.safety_guard import HardwareSafetyGuard, SafetyLimits, SafetyViolationError
from hardware.simulator import PhysicsSimulationRig
from optimization.bayesian_opt import ActiveLearningEngine
from safety.spy import SpyRig, unsafe_commands

SETTINGS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])

# Anything a buggy caller or a confused model could hand over as a "position".
anything = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True),
    st.floats(min_value=-500, max_value=500),
    st.integers(min_value=-10 ** 30, max_value=10 ** 30),
    st.integers().map(lambda i: i * 10 ** 400),
    st.booleans(),
    st.none(),
    st.text(max_size=20),
    st.decimals(allow_nan=True, allow_infinity=True),
    st.fractions(),
    st.complex_numbers(allow_nan=True),
    st.lists(st.floats(), max_size=3),
    st.dictionaries(st.text(max_size=3), st.floats(), max_size=2),
    st.binary(max_size=5),
    st.builds(np.float64, st.floats(allow_nan=True, allow_infinity=True)),
    st.builds(np.int64, st.integers(min_value=-(2 ** 63), max_value=2 ** 63 - 1)),
)
experiments = st.sampled_from(list_experiments())


def guarded(name):
    spec = get_experiment(name)
    spy = SpyRig(PhysicsSimulationRig(spec, np.random.default_rng(0), SimulationConfig(transit_delay_sec=0.0)))
    guard = HardwareSafetyGuard(spy, SafetyLimits.from_spec(spec, SafetyConfig()))
    guard.connect()
    guard.home()
    return guard, spy


@SETTINGS
@given(name=experiments, commands=st.lists(st.tuples(anything, st.booleans()), max_size=25))
def test_any_command_sequence_leaves_the_rig_inside_the_envelope(name, commands):
    guard, spy = guarded(name)
    for value, read in commands:
        try:
            guard.safe_move(value)
        except SafetyViolationError:
            pass
        if read:
            guard.safe_read()
    assert unsafe_commands(spy.commands, guard.limits, guard.audit_entries) == []
    lo, hi = get_experiment(name).safe_bounds
    for c in spy.moves:
        assert c.position is not None and lo <= c.position <= hi


@SETTINGS
@given(name=experiments, values=st.lists(anything, max_size=15))
def test_the_measurement_tool_never_raises_and_never_passes_bad_values(name, values):
    guard, spy = guarded(name)
    spec = get_experiment(name)
    tools = LabTools(guard, ActiveLearningEngine(*spec.safe_bounds, random_state=0))
    for value in values:
        outcome = tools.tool_take_measurement(value)
        assert outcome.status in ("measured", "rejected", "anomaly", "error")
        if outcome.requested_position is not None:
            assert math.isfinite(outcome.requested_position)
    assert unsafe_commands(spy.commands, guard.limits, guard.audit_entries) == []
    assert all(math.isfinite(x) and spec.safe_bounds[0] <= x <= spec.safe_bounds[1] for x in tools.history_x)


json_values = st.recursive(
    st.one_of(st.none(), st.booleans(), st.floats(allow_nan=True, allow_infinity=True), st.integers(), st.text(max_size=10)),
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=5), children, max_size=3),
    max_leaves=8,
)


@SETTINGS
@given(choice=st.one_of(st.sampled_from(["gp", "self"]), json_values), position=json_values, reason=json_values)
def test_a_strategy_answer_is_either_rejected_or_a_finite_number(choice, position, reason):
    try:
        decision = StrategyDecision.model_validate({"choice": choice, "position": position, "reason": reason})
    except ValidationError:
        return
    if decision.choice == "self":
        assert isinstance(decision.position, float) and math.isfinite(decision.position)
    else:
        assert decision.position is None


@SETTINGS
@given(design=json_values, proposed=json_values)
def test_a_plan_answer_is_either_rejected_or_all_finite_numbers(design, proposed):
    payload = {"hypothesis": {"statement": "s", "candidate_relationships": ["r"], "independent_variable": "x",
                              "proposed_range": proposed}, "initial_design": design, "rationale": "r"}
    try:
        plan = PlannerOutput.model_validate(payload)
    except ValidationError:
        return
    assert all(isinstance(v, float) and math.isfinite(v) for v in [*plan.initial_design, *plan.hypothesis.proposed_range])


@settings(max_examples=300, deadline=None)
@given(text=st.text(max_size=120), params=st.lists(st.text(min_size=1, max_size=6), min_size=1, max_size=4))
def test_any_formula_text_is_compiled_or_rejected_without_executing_anything(text, params):
    import builtins
    calls = []
    real_eval, real_exec = builtins.eval, builtins.exec
    builtins.eval = lambda *a, **k: calls.append("eval")
    builtins.exec = lambda *a, **k: calls.append("exec")
    try:
        spec = CandidateModelSpec(name="h", expression=text or "x", parameters=[ParameterGuess(name=p) for p in params])
        try:
            model = compile_candidate(spec, key="llm_1")
        except FormulaRejected:
            model = None
        if model is not None:
            out = model.func(np.linspace(0.5, 5.0, 4), *([1.0] * len(model.param_names)))
            assert out.shape == (4,)
    finally:
        builtins.eval, builtins.exec = real_eval, real_exec
    assert calls == []


# Expressions from the whitelist's own grammar, plus a few forbidden pieces, so that compiling
# and evaluating (not just parsing) are exercised.
leaf = st.one_of(st.sampled_from(["x", "A", "k", "pi", "e", "B"]), st.integers(-1000, 1000).map(str),
                 st.floats(-1e3, 1e3, allow_nan=False).map(lambda f: f"{f:.3g}"))
expr = st.recursive(
    leaf,
    lambda inner: st.one_of(
        st.tuples(inner, st.sampled_from(["+", "-", "*", "/", "**", "%", "//"]), inner).map(lambda t: f"({t[0]} {t[1]} {t[2]})"),
        st.tuples(st.sampled_from(["sin", "cos", "tan", "exp", "log", "sqrt", "abs", "radians", "open", "eval"]), inner)
          .map(lambda t: f"{t[0]}({t[1]})"),
        inner.map(lambda s: f"-{s}"),
        inner.map(lambda s: f"{s}.real"),
    ),
    max_leaves=12,
)


@settings(max_examples=400, deadline=None)
@given(text=expr)
def test_grammar_built_formulas_compile_and_evaluate_or_are_rejected(text):
    spec = CandidateModelSpec(name="g", expression=text,
                              parameters=[ParameterGuess(name=p, initial_guess=1.5) for p in ("A", "k") if p in text] or
                              [ParameterGuess(name="A")])
    try:
        model = compile_candidate(spec, key="llm_1")
    except FormulaRejected:
        return
    assert ".real" not in text and "open(" not in text and "eval(" not in text and "%" not in text and "//" not in text
    out = model.func(np.linspace(0.5, 50.0, 6), *([1.5] * len(model.param_names)))
    assert out.shape == (6,) and out.dtype == float
