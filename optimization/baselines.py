"""
Baseline samplers for comparison with GP active learning.
Every sampler draws from a numpy Generator passed in by the caller, so runs are reproducible.

Both take an optional `window` (lo, hi): the positions the safety guard would accept next
(inside the bounds, within the step limit of the current position, below any ceiling). A
baseline that proposed unreachable points would only be measuring the guard.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np

Window = Optional[Tuple[float, float]]


class RandomSampler:
    """Uniform random positions within the safe bounds (and the reachable window)."""

    def __init__(self, min_bound: float, max_bound: float, rng: np.random.Generator):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self._rng = rng

    def suggest(self, window: Window = None) -> float:
        lo, hi = self.min_bound, self.max_bound
        if window is not None:
            lo, hi = max(lo, window[0]), min(hi, window[1])
        return float(self._rng.uniform(lo, hi))


def van_der_corput_fractions(depth: int = 6) -> List[float]:
    """0, 1, then 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ...: every prefix is spread over [0, 1]."""
    # Each level adds the odd multiples of 1/2^level, in bit-reversed order of their numerators.
    ordered = [0.0, 1.0]
    for level in range(1, depth + 1):
        n = 2 ** level
        odd = list(range(1, n, 2))
        odd.sort(key=lambda i: format(i, f"0{level}b")[::-1])
        ordered += [i / n for i in odd]
    return ordered


class GridSampler:
    """
    A grid refined coarse to fine (van der Corput order: ends, middle, quarters, eighths, ...),
    so that the measured points are evenly spread whenever the stopping rule ends the run.
    Each call returns the first grid point in that order that is not yet measured and is reachable.
    """

    def __init__(self, min_bound: float, max_bound: float, depth: int = 6):
        self.min_bound, self.max_bound = min_bound, max_bound
        span = max_bound - min_bound
        self.points = [min_bound + f * span for f in van_der_corput_fractions(depth)]
        self.tolerance = span / 2 ** (depth + 2)

    def suggest(self, measured: Sequence[float], window: Window = None, exclude: Sequence[float] = ()) -> float:
        taken = np.asarray([*measured, *exclude], dtype=float)

        def free(p: float) -> bool:
            return taken.size == 0 or float(np.min(np.abs(taken - p))) > self.tolerance

        def reachable(p: float) -> bool:
            return window is None or window[0] <= p <= window[1]

        for p in self.points:
            if free(p) and reachable(p):
                return float(p)
        candidates = [p for p in self.points if free(p)] or self.points
        if window is not None:  # nothing reachable is free: the reachable point nearest the next free one
            target = candidates[0]
            return float(min(max(target, window[0]), window[1]))
        return float(candidates[0])
