"""
Unit tests for the Bayesian Optimization / Active Learning engine.
"""
import numpy as np
from optimization.bayesian_opt import ActiveLearningEngine, AcquisitionStrategy

def test_bayesian_optimization_uncertainty_sampling():
    engine = ActiveLearningEngine(min_bound=0.0, max_bound=180.0, resolution=100)
    
    # Observed points at 0, 90, 180
    X = np.array([0.0, 90.0, 180.0])
    y = np.array([1000.0, 10.0, 1000.0])

    engine.fit(X, y)
    mu, sigma = engine.predict()

    assert len(mu) == 100
    assert len(sigma) == 100

    # Suggest next point via uncertainty sampling
    suggestion = engine.suggest_next_sample(X, y, strategy=AcquisitionStrategy.UNCERTAINTY)
    suggested_pos = suggestion["suggested_position"]

    # The point with highest uncertainty should be in between sampled points (around 45 or 135 deg)
    assert 20.0 <= suggested_pos <= 160.0
    assert suggested_pos not in [0.0, 90.0, 180.0]
