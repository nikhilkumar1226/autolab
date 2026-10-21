"""
Bayesian Optimization and Active Learning Engine.
Selects statistically optimal measurement points using Gaussian Process surrogates
and acquisition functions to discover physical laws with minimal experimental trials.
"""
from enum import Enum
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.stats import norm

class AcquisitionStrategy(str, Enum):
    UNCERTAINTY = "uncertainty"        # Maximum variance (Pure Exploration / System Identification)
    EXPECTED_IMPROVEMENT = "ei"         # Exploration + Exploitation for extrema
    UPPER_CONFIDENCE_BOUND = "ucb"     # mu(x) + kappa * sigma(x)

class ActiveLearningEngine:
    """
    Gaussian Process Active Learning agent for physical parameter discovery.
    """

    def __init__(self, min_bound: float = 0.0, max_bound: float = 180.0, resolution: int = 300):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.resolution = resolution
        self.candidate_grid = np.linspace(min_bound, max_bound, resolution).reshape(-1, 1)
        
        # Kernel: Constant * RBF + WhiteKernel (accounting for sensor noise)
        kernel = ConstantKernel(constant_value=500.0, constant_value_bounds=(1e-2, 1e6)) * \
                 RBF(length_scale=30.0, length_scale_bounds=(5.0, 180.0)) + \
                 WhiteKernel(noise_level=10.0, noise_level_bounds=(1e-5, 1e4))
                 
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=42
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Gaussian Process surrogate model on observed experimental data.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.gp.fit(X, y)
        self.is_fitted = True

    def predict(self, X_grid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the surrogate mean and standard deviation across a grid.
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate GP model must be fitted before prediction.")
        if X_grid is None:
            X_grid = self.candidate_grid
        elif len(X_grid.shape) == 1:
            X_grid = X_grid.reshape(-1, 1)

        mu, sigma = self.gp.predict(X_grid, return_std=True)
        return mu, sigma

    def suggest_next_sample(
        self,
        X_observed: np.ndarray,
        y_observed: np.ndarray,
        strategy: AcquisitionStrategy = AcquisitionStrategy.UNCERTAINTY,
        kappa: float = 1.96,
        xi: float = 0.01
    ) -> Dict[str, Any]:
        """
        Computes acquisition function over candidate search space and returns optimal next coordinate.
        """
        if len(X_observed.shape) == 1:
            X_observed = X_observed.reshape(-1, 1)

        self.fit(X_observed, y_observed)
        mu, sigma = self.predict(self.candidate_grid)

        # Avoid proposing points already measured very closely
        min_distance_threshold = (self.max_bound - self.min_bound) / (self.resolution * 0.5)
        distances_to_observed = np.min(
            np.abs(self.candidate_grid - X_observed.reshape(1, -1)),
            axis=1
        )
        mask_novelty = distances_to_observed > min_distance_threshold

        if strategy == AcquisitionStrategy.UNCERTAINTY:
            # Maximum predictive variance / standard deviation
            acquisition_values = sigma.copy()

        elif strategy == AcquisitionStrategy.UPPER_CONFIDENCE_BOUND:
            # UCB: maximize expected value + confidence interval
            acquisition_values = mu + kappa * sigma

        elif strategy == AcquisitionStrategy.EXPECTED_IMPROVEMENT:
            # EI for finding maximum response
            y_max = np.max(y_observed)
            with np.errstate(divide='warn'):
                improvement = mu - y_max - xi
                Z = improvement / (sigma + 1e-9)
                ei = improvement * norm.cdf(Z) + (sigma + 1e-9) * norm.pdf(Z)
                ei[sigma <= 1e-9] = 0.0
            acquisition_values = ei

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        # Penalize already sampled points
        acquisition_values[~mask_novelty] = -1e9

        best_idx = int(np.argmax(acquisition_values))
        suggested_point = float(self.candidate_grid[best_idx, 0])
        max_uncertainty = float(np.max(sigma))
        mean_uncertainty = float(np.mean(sigma))

        return {
            "suggested_position": round(suggested_point, 2),
            "strategy": strategy.value,
            "max_sigma": round(max_uncertainty, 3),
            "mean_sigma": round(mean_uncertainty, 3),
            "acquisition_score": float(acquisition_values[best_idx]),
        }
