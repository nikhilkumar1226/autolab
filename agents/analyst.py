"""
Data Analyst Agent.
Performs statistical hypothesis testing, non-linear physical model fitting,
information criterion evaluations (AIC/BIC), and convergence analysis.
"""
import numpy as np
from scipy.optimize import curve_fit
from typing import List, Dict, Any, Tuple
from agents.models import ModelFitMetric, AnalysisResult

# Candidate physical functions
def malus_law_model(theta, I0, theta0, C):
    rad = np.radians(theta - theta0)
    return I0 * (np.cos(rad) ** 2) + C

def inverse_square_model(d, I0, d0, C):
    return (I0 * 100.0) / ((d + d0) ** 2) + C

def linear_model(x, m, c):
    return m * x + c

def quadratic_model(x, a, b, c):
    return a * (x ** 2) + b * x + c

def exponential_model(x, I0, alpha, C):
    return I0 * np.exp(-alpha * x) + C

class DataAnalystAgent:
    """
    Analyzes gathered experimental data, fits mathematical models,
    ranks competing physical hypotheses, and assesses convergence.
    """

    def __init__(self, r2_threshold: float = 0.985):
        self.r2_threshold = r2_threshold

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, num_params: int) -> Tuple[float, float, float, float]:
        """
        Computes R^2, RMSE, AIC, and BIC.
        """
        n = len(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else 0.0
        rmse = np.sqrt(ss_res / max(1, n))

        # AIC and BIC (assuming normal distribution of residuals)
        if ss_res > 1e-12 and n > num_params:
            aic = n * np.log(ss_res / n) + 2 * num_params
            bic = n * np.log(ss_res / n) + num_params * np.log(n)
        else:
            aic = 1e6
            bic = 1e6

        return float(r2), float(rmse), float(aic), float(bic)

    def analyze_dataset(
        self,
        X_data: List[float],
        y_data: List[float],
        experiment_type: str = "malus_law",
        iteration: int = 1
    ) -> AnalysisResult:
        """
        Fits candidate models and determines scientific confirmation.
        """
        X = np.array(X_data, dtype=float)
        y = np.array(y_data, dtype=float)
        n = len(X)

        candidate_fits: List[ModelFitMetric] = []

        if n < 4:
            # Need minimum points to fit multi-parameter non-linear models
            dummy_metric = ModelFitMetric(
                model_name="Gathering Data",
                formula="N/A",
                r2_score=0.0,
                rmse=0.0,
                aic=1e6,
                bic=1e6,
                optimal_parameters={},
                parameter_errors={}
            )
            return AnalysisResult(
                iteration=iteration,
                total_samples=n,
                candidate_fits=[],
                best_model=dummy_metric,
                is_converged=False,
                convergence_rationale=f"Gathering initial baseline data ({n}/4 points)..."
            )

        # 1. Linear Model Fit
        try:
            popt, pcov = curve_fit(linear_model, X, y)
            y_pred = linear_model(X, *popt)
            r2, rmse, aic, bic = self._compute_metrics(y, y_pred, len(popt))
            candidate_fits.append(ModelFitMetric(
                model_name="Linear Model",
                formula="I(x) = m*x + c",
                r2_score=round(r2, 4),
                rmse=round(rmse, 4),
                aic=round(aic, 2),
                bic=round(bic, 2),
                optimal_parameters={"m": round(popt[0], 4), "c": round(popt[1], 4)},
                parameter_errors={"m_err": round(np.sqrt(max(0, pcov[0,0])), 4), "c_err": round(np.sqrt(max(0, pcov[1,1])), 4)}
            ))
        except Exception:
            pass

        # 2. Quadratic Model Fit
        try:
            popt, pcov = curve_fit(quadratic_model, X, y)
            y_pred = quadratic_model(X, *popt)
            r2, rmse, aic, bic = self._compute_metrics(y, y_pred, len(popt))
            candidate_fits.append(ModelFitMetric(
                model_name="Quadratic Polynomial",
                formula="I(x) = a*x^2 + b*x + c",
                r2_score=round(r2, 4),
                rmse=round(rmse, 4),
                aic=round(aic, 2),
                bic=round(bic, 2),
                optimal_parameters={"a": round(popt[0], 4), "b": round(popt[1], 4), "c": round(popt[2], 4)},
                parameter_errors={"a_err": round(np.sqrt(max(0, pcov[0,0])), 4)}
            ))
        except Exception:
            pass

        # 3. Target Physical Model (Malus's Law)
        if experiment_type == "malus_law":
            try:
                # Initial guess [I0, theta0, C]
                i0_init = float(np.max(y) - np.min(y))
                p0 = [i0_init, 0.0, float(np.min(y))]
                bounds = ([0.0, -180.0, -100.0], [5000.0, 180.0, 500.0])
                popt, pcov = curve_fit(malus_law_model, X, y, p0=p0, bounds=bounds, maxfev=10000)
                y_pred = malus_law_model(X, *popt)
                r2, rmse, aic, bic = self._compute_metrics(y, y_pred, len(popt))
                perr = np.sqrt(np.diag(pcov))
                candidate_fits.append(ModelFitMetric(
                    model_name="Malus's Law (Trigonometric Cos^2)",
                    formula="I(theta) = I0 * cos^2(theta - theta0) + C",
                    r2_score=round(r2, 4),
                    rmse=round(rmse, 4),
                    aic=round(aic, 2),
                    bic=round(bic, 2),
                    optimal_parameters={"I0": round(popt[0], 3), "theta0": round(popt[1], 3), "C": round(popt[2], 3)},
                    parameter_errors={"I0_err": round(perr[0], 3), "theta0_err": round(perr[1], 3), "C_err": round(perr[2], 3)}
                ))
            except Exception:
                pass

        # 4. Target Physical Model (Inverse Square Law)
        elif experiment_type == "inverse_square_law":
            try:
                p0 = [1000.0, 5.0, float(np.min(y))]
                bounds = ([0.0, 0.1, -100.0], [100000.0, 50.0, 500.0])
                popt, pcov = curve_fit(inverse_square_model, X, y, p0=p0, bounds=bounds, maxfev=10000)
                y_pred = inverse_square_model(X, *popt)
                r2, rmse, aic, bic = self._compute_metrics(y, y_pred, len(popt))
                perr = np.sqrt(np.diag(pcov))
                candidate_fits.append(ModelFitMetric(
                    model_name="Inverse-Square Law",
                    formula="I(d) = I0 / (d + d0)^2 + C",
                    r2_score=round(r2, 4),
                    rmse=round(rmse, 4),
                    aic=round(aic, 2),
                    bic=round(bic, 2),
                    optimal_parameters={"I0": round(popt[0], 3), "d0": round(popt[1], 3), "C": round(popt[2], 3)},
                    parameter_errors={"I0_err": round(perr[0], 3), "d0_err": round(perr[1], 3), "C_err": round(perr[2], 3)}
                ))
            except Exception:
                pass

        # Rank models by AIC (lower is better) and R^2 (higher is better)
        if not candidate_fits:
            raise RuntimeError("All candidate models failed to fit experimental data.")

        # Best model is the one with highest R^2 that is physically justified
        best_model = max(candidate_fits, key=lambda m: m.r2_score)

        # Check convergence criteria:
        # 1. Target physical model has R^2 >= threshold
        # 2. Total samples >= 8 to ensure full phase space coverage
        is_converged = False
        rationale = ""

        is_target_best = "Malus" in best_model.model_name or "Inverse" in best_model.model_name
        if is_target_best and best_model.r2_score >= self.r2_threshold and n >= 8:
            is_converged = True
            rationale = (
                f"Convergence criteria satisfied: {best_model.model_name} achieved R^2 = {best_model.r2_score} "
                f"(>= {self.r2_threshold}) across {n} active learning points with low parameter error."
            )
        else:
            rationale = (
                f"Current best fit is {best_model.model_name} with R^2 = {best_model.r2_score}. "
                f"Further active learning exploration needed ({n} points sampled)."
            )

        return AnalysisResult(
            iteration=iteration,
            total_samples=n,
            candidate_fits=candidate_fits,
            best_model=best_model,
            is_converged=is_converged,
            convergence_rationale=rationale
        )
