"""
Unit tests for the Data Analyst Agent.
"""
import numpy as np
from agents.analyst import DataAnalystAgent

def test_data_analyst_malus_law_identification():
    analyst = DataAnalystAgent(r2_threshold=0.98)
    
    # Generate synthetic Malus's Law data: I = 1000 * cos^2(theta) + 15
    angles = np.linspace(0, 180, 10)
    intensities = 1000.0 * (np.cos(np.radians(angles)) ** 2) + 15.0

    res = analyst.analyze_dataset(
        X_data=list(angles),
        y_data=list(intensities),
        experiment_type="malus_law",
        iteration=1
    )

    assert res.best_model.model_name == "Malus's Law (Trigonometric Cos^2)"
    assert res.best_model.r2_score > 0.99
    assert res.is_converged is True
    assert "I0" in res.best_model.optimal_parameters
    assert abs(res.best_model.optimal_parameters["I0"] - 1000.0) < 5.0
