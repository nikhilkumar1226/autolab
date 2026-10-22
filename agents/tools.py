"""
Tool interface for agentic execution.
Binds the hardware safety layer and the active learning engine into callable tools.
"""
import math
import logging
from typing import List, Optional, Tuple
import numpy as np
from scipy import stats
from analysis.candidate_models import predict_fit, prediction_se
from analysis.model_library import ModelFit
from config import AnomalyConfig
from hardware.safety_guard import HardwareSafetyGuard, SafetyViolationError, _finite_real
from optimization.bayesian_opt import ActiveLearningEngine, AcquisitionStrategy
from agents.models import ExecutionOutcome, ExpectedBand, GPCurve, GPSuggestion

logger = logging.getLogger(__name__)

class LabTools:
    """
    Executable toolbox accessible to Planner, Executor, and Analyst agents.
    """

    def __init__(self, safety_guard: HardwareSafetyGuard, optimizer: ActiveLearningEngine,
                 anomaly: Optional[AnomalyConfig] = None):
        self.safety_guard = safety_guard
        self.optimizer = optimizer
        self.anomaly = anomaly or AnomalyConfig()
        self.history_x: List[float] = []
        self.history_y: List[float] = []
        # Positions the safety guard refused, and positions given up after persistent anomalies;
        # the GP will not propose either again.
        self.rejected_positions: List[float] = []
        self.discarded_positions: List[float] = []
        # The current best fit and its sigma (set by the engine after each analysis): a second
        # opinion for the outlier check, because the GP's band is over-confident on steep curves.
        self.reference_fit: Optional[Tuple[ModelFit, float]] = None

    def expected_bands(self, position: float) -> List[ExpectedBand]:
        """
        Predictive bands at `position`, once enough points exist (config.AnomalyConfig.outlier_bands):
        - model: the current best fit. sigma combines the noise (residual std, floored) and the fitted
          curve's own uncertainty (delta method); the threshold is a Student-t quantile with the fit's
          n - k degrees of freedom, because sigma itself is estimated from few points;
        - gp: the GP's predictive band (noise included), with a normal quantile.
        """
        cfg = self.anomaly
        if len(self.history_x) < cfg.outlier_min_points:
            return []
        bands: List[ExpectedBand] = []
        if cfg.outlier_bands in ("gp", "both"):
            if self.optimizer.fitted_n != len(self.history_x):
                self.optimizer.fit(np.array(self.history_x), np.array(self.history_y))
            mu, sigma = self.optimizer.predict(np.array([position]))
            bands.append(ExpectedBand(source="gp", mean=float(mu[0]), sigma=float(sigma[0]),
                                      threshold=float(stats.norm.isf(cfg.outlier_p / 2))))
        if cfg.outlier_bands in ("model", "both") and self.reference_fit is not None:
            fit, fit_sigma = self.reference_fit
            predicted = predict_fit(fit, [position])
            se = prediction_se(fit, position)
            dof = fit.n_points - fit.n_params
            if predicted is not None and math.isfinite(float(predicted[0])) and se is not None and dof >= 1:
                bands.append(ExpectedBand(source="model", mean=float(predicted[0]), sigma=math.hypot(fit_sigma, se),
                                          threshold=float(stats.t.isf(cfg.outlier_p / 2, dof))))
        return bands

    def tool_take_measurement(self, position: float, confirm: Optional[float] = None) -> ExecutionOutcome:
        """
        Compound tool: moves to the target through the safety guard and records the sensor reading.
        A safety rejection is reported back as an outcome, never retried here. A reading the guard
        flags is reported as an anomaly and kept out of the data.

        `confirm` is an earlier reading at the same position that was flagged as an outlier. If this
        reading is flagged only as an outlier too, and agrees with it within the narrowest band's
        sigma, both are genuine (the expectation was wrong, not the sensor): this reading is accepted.
        """
        requested = _finite_real(position)  # what is recorded; the guard itself checks `position` as given
        try:
            move = self.safety_guard.safe_move(position)
            if not move["success"]:
                return ExecutionOutcome(status="error", requested_position=requested, message="actuator reported a failed move")
            bands = self.expected_bands(position)
            measurement = self.safety_guard.safe_read(expected=[(b.mean, b.sigma, b.threshold) for b in bands])
        except SafetyViolationError as e:
            if requested is not None:
                self.rejected_positions.append(requested)
            return ExecutionOutcome(status="rejected", requested_position=requested, message=str(e))
        except Exception as e:
            return ExecutionOutcome(status="error", requested_position=requested, message=f"{type(e).__name__}: {e}")

        value = measurement.sensor_value if math.isfinite(measurement.sensor_value) else None
        cycle = measurement.metadata.get("cycle")
        feedback = measurement.metadata.get("position_feedback")
        common = dict(
            requested_position=requested, position=measurement.actuator_position, value=value, expected=bands,
            cycle=cycle if isinstance(cycle, int) else None,
            position_feedback=float(feedback) if isinstance(feedback, (int, float)) else None,
        )
        message = ""
        if not measurement.is_valid:
            sigma = min((b.sigma for b in bands), default=0.0)
            agrees = (
                measurement.flags == ["outlier"] and confirm is not None and value is not None
                and abs(value - confirm) <= self.anomaly.confirm_z * math.sqrt(2.0) * sigma
            )
            if not agrees:
                return ExecutionOutcome(
                    status="anomaly", flags=list(measurement.flags),
                    message=f"reading flagged by the safety guard: {', '.join(measurement.flags)}", **common,
                )
            message = (f"outside the GP band, but a re-measurement agreed with the earlier reading "
                       f"({confirm:.4g}) within {self.anomaly.confirm_z:g} x sqrt(2) sigma: accepted as genuine")

        self.history_x.append(measurement.actuator_position)
        self.history_y.append(measurement.sensor_value)
        return ExecutionOutcome(status="measured", message=message, **common)

    def discard(self, position: float) -> None:
        """Give up a position after persistent anomalies; the GP will not propose it again."""
        self.discarded_positions.append(position)

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
        limits = self.safety_guard.limits
        reachable = None
        if limits.max_step_delta < limits.max_position - limits.min_position:
            here = self.safety_guard.get_status().current_position
            reachable = (here - limits.max_step_delta, here + limits.max_step_delta)
        suggestion = self.optimizer.suggest_next_sample(
            np.array(self.history_x), np.array(self.history_y), strategy=strat,
            exclude=[*self.rejected_positions, *self.discarded_positions], upper=self.safety_guard.ceiling,
            reachable=reachable,
        )
        return GPSuggestion(
            position=suggestion["suggested_position"],
            strategy=suggestion["strategy"],
            max_sigma=suggestion["max_sigma"],
            mean_sigma=suggestion["mean_sigma"],
            acquisition_score=suggestion["acquisition_score"],
        )

    def _nearest_allowed(self, target: float) -> float:
        """`target`, unless a rejected, discarded or measured position is within the GP's novelty distance."""
        opt = self.optimizer
        threshold = (opt.max_bound - opt.min_bound) / (opt.resolution * 0.5)
        taken = np.asarray([*self.history_x, *self.rejected_positions, *self.discarded_positions], dtype=float)
        taken = taken[np.isfinite(taken)]
        ceiling = self.safety_guard.ceiling
        below = ceiling is None or target < ceiling
        if below and (taken.size == 0 or np.min(np.abs(taken - target)) > threshold):
            return round(target, 2)
        grid = opt.candidate_grid.ravel()
        if ceiling is not None:
            grid = grid[grid < ceiling]
        if taken.size == 0:
            return round(float(grid[np.argmin(np.abs(grid - target))]), 2) if grid.size else round(target, 2)
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
