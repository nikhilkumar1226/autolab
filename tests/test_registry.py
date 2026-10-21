"""
Tests for the experiment registry and the agent-facing view.
"""
import numpy as np
import pytest
from pydantic import ValidationError

from analysis.model_library import MODEL_LIBRARY
from config import SystemConfig
from experiments.registry import get_experiment, list_experiments

# Words that would hint at the law if an agent saw them.
LAW_HINTS = ["malus", "cos", "inverse", "square", "exponential", "exp(", "beer", "lambert", "law"]


def test_three_experiments_with_their_own_bounds_and_units():
    assert set(list_experiments()) == {"malus_law", "inverse_square_law", "beer_lambert_law"}
    expected = {
        "malus_law": ((0.0, 180.0), "deg"),
        "inverse_square_law": ((5.0, 100.0), "cm"),
        "beer_lambert_law": ((0.0, 100.0), "mg/L"),
    }
    for name, (bounds, units) in expected.items():
        spec = get_experiment(name)
        assert spec.safe_bounds == bounds
        assert spec.independent.units == units
        assert spec.ground_truth_model in MODEL_LIBRARY


def test_unknown_experiment_is_rejected():
    with pytest.raises(KeyError, match="Unknown experiment"):
        get_experiment("perpetual_motion")
    with pytest.raises(ValidationError):
        SystemConfig(experiment="perpetual_motion")


@pytest.mark.parametrize("name", list_experiments())
def test_agent_view_hides_name_and_ground_truth(name):
    spec = get_experiment(name)
    view = spec.agent_view()
    assert set(type(view).model_fields) == {"description", "independent", "dependent", "safe_bounds"}

    text = view.model_dump_json().lower()
    assert spec.name not in text
    assert spec.ground_truth_model not in text
    for hint in LAW_HINTS:
        assert hint not in text, f"agent view of {name} contains the hint '{hint}'"


@pytest.mark.parametrize("name", list_experiments())
def test_ground_truth_function_is_not_serialised(name):
    assert "ground_truth" not in get_experiment(name).model_dump()


@pytest.mark.parametrize("name", list_experiments())
def test_registry_truth_matches_library_model(name):
    """The simulator's law and the analyst's model are written independently; they must agree."""
    spec = get_experiment(name)
    xs = np.linspace(*spec.safe_bounds, 25)
    truth = np.array([spec.true_response(x) for x in xs])
    library = MODEL_LIBRARY[spec.ground_truth_model].predict(xs, spec.true_params)
    np.testing.assert_allclose(library, truth, rtol=1e-12, atol=1e-9)
