"""
Tool interface for agentic execution.
Binds the hardware safety layer and the active learning engine into callable tools.
"""
import logging
from typing import List, Optional
import numpy as np
from hardware.safety_guard import HardwareSafetyGuard, SafetyViolationError
from optimization.bayesian_opt import ActiveLearningEngine, AcquisitionStrategy
from agents.models import ExecutionOutcome, GPCurve, GPSuggestion

logger = logging.getLogger(__name__)

class LabTools:
    """
    Executable toolbox accessible to Planner, Executor, and Analyst agents.
    """

    def __init__(self, safety_guard: HardwareSafetyGuard, optimizer: ActiveLearningEngine):
        self.safety_guard = safety_guard
        self.optimizer = optimizer
        self.history_x: List[float] = []
        self.history_y: List[float] = []
        # Positions the safety guard refused; the GP will not propose them again.
        self.rejected_positions: List[float] = []

    def tool_take_measurement(self, position: float) -> ExecutionOutcome:
        """
        Compound tool: moves to the target through the safety guard and records the sensor reading.
        A safety rejection is reported back as an outcome, never retried here.
        """
        try:
            move = self.safety_guard.safe_move(position)
            if not move["success"]:
                return ExecutionOutcome(status="error", requested_position=position, message="actuator reported a failed move")
            measurement = self.safety_guard.safe_read()
        except SafetyViolationError as e:
            self.rejected_positions.append(position)
            return ExecutionOutcome(status="rejected", requested_position=position, message=str(e))
        except Exception as e:
            return ExecutionOutcome(status="error", requested_position=position, message=f"{type(e).__name__}: {e}")

        if not measurement.is_valid:
            return ExecutionOutcome(
                status="error", requested_position=position, position=measurement.actuator_position,
                message="sensor reading flagged invalid by the safety guard",
            )

        self.history_x.append(measurement.actuator_position)
        self.history_y.append(measurement.sensor_value)
        return ExecutionOutcome(
            status="measured",
            requested_position=position,
            position=measurement.actuator_position,
            value=measurement.sensor_value,
        )

    def tool_suggest_next_point(self, strategy: str = "uncertainty") -> GPSuggestion:
        """
        Invokes Bayesian Optimization / Active Learning to calculate the most statistically
        informative next parameter coordinate.
        """
        if len(self.history_x) < 2:
            # Not enough points for a GP: propose the midpoint of the search space, or the
            # grid point nearest to it that is not close to a measured or rejected position.
            mid = (self.optimizer.min_bound + self.optimizer.max_bound) / 2.0
            return GPSuggestion(position=self._nearest_allowed(mid), strategy="initial_midpoint", max_sigma=1.0, mean_sigma=1.0)

        strat = AcquisitionStrategy(strategy) if strategy in [s.value for s in AcquisitionStrategy] else AcquisitionStrategy.UNCERTAINTY
        suggestion = self.optimizer.suggest_next_sample(
            np.array(self.history_x), np.array(self.history_y), strategy=strat, exclude=self.rejected_positions
        )
        return GPSuggestion(
            position=suggestion["suggested_position"],
            strategy=suggestion["strategy"],
            max_sigma=suggestion["max_sigma"],
            mean_sigma=suggestion["mean_sigma"],
            acquisition_score=suggestion["acquisition_score"],
        )

    def _nearest_allowed(self, target: float) -> float:
        """`target`, unless a rejected or measured position is within the GP's novelty distance."""
        opt = self.optimizer
        threshold = (opt.max_bound - opt.min_bound) / (opt.resolution * 0.5)
        taken = np.asarray([*self.history_x, *self.rejected_positions], dtype=float)
        taken = taken[np.isfinite(taken)]
        if taken.size == 0 or np.min(np.abs(taken - target)) > threshold:
            return round(target, 2)
        grid = opt.candidate_grid.ravel()
        allowed = grid[np.min(np.abs(grid[:, None] - taken[None, :]), axis=1) > threshold]
        if allowed.size == 0:
            return round(target, 2)
        return round(float(allowed[np.argmin(np.abs(allowed - target))]), 2)

    def gp_curve(self, n_points: int = 50) -> Optional[GPCurve]:
        """Current GP posterior on a coarse grid (None before the GP is fitted)."""
        if not self.optimizer.is_fitted:
            return None
        grid = np.linspace(self.optimizer.min_bound, self.optimizer.max_bound, n_points)
        mu, sigma = self.optimizer.predict(grid)
        return GPCurve(x=grid.tolist(), mean=mu.tolist(), std=sigma.tolist())
