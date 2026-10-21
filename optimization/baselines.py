"""
Baseline samplers for comparison with GP active learning.
Every sampler draws from a numpy Generator passed in by the caller, so runs are reproducible.
"""
import numpy as np

class RandomSampler:
    """Uniform random positions within the safe bounds."""

    def __init__(self, min_bound: float, max_bound: float, rng: np.random.Generator):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self._rng = rng

    def suggest(self) -> float:
        return float(self._rng.uniform(self.min_bound, self.max_bound))
