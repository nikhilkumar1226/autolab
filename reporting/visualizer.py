"""
Scientific Visualization Generator.
Produces publication-quality figures showcasing experimental observations,
physical curve fits, Bayesian surrogate uncertainty bounds, and residual analyses.
"""
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive headless backend
import matplotlib.pyplot as plt

class ScienceVisualizer:
    """
    Generates academic publication-quality multi-panel experimental plots.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_discovery_figure(
        self,
        X_data: List[float],
        y_data: List[float],
        experiment_type: str,
        best_model_name: str,
        optimal_params: Dict[str, float],
        r2_score: float,
        rmse: float,
        gp_mu: Optional[np.ndarray] = None,
        gp_sigma: Optional[np.ndarray] = None,
        gp_grid: Optional[np.ndarray] = None
    ) -> Path:
        """
        Creates a comprehensive 3-panel publication figure.
        """
        X = np.array(X_data)
        y = np.array(y_data)
        
        # Sort for smooth plotting
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]

        fig = plt.figure(figsize=(15, 5), dpi=300)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        # Colors
        color_obs = "#1f77b4" # Blue
        color_fit = "#d62728" # Red
        color_gp = "#2ca02c"  # Green
        color_ci = "#98df8a"  # Light green

        x_label = "Polarizer Angle $\\theta$ (degrees)" if experiment_type == "malus_law" else "Distance $d$ (cm)"
        y_label = "Transmitted Intensity $I$ (lux)"

        # -------------------------------------------------------------
        # Panel 1: Physical Law Fit & Experimental Data
        # -------------------------------------------------------------
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.scatter(X, y, color=color_obs, s=45, alpha=0.85, edgecolors='k', zorder=5, label='Experimental Observations')

        # High-resolution theoretical curve
        x_dense = np.linspace(min(X), max(X), 300)
        if experiment_type == "malus_law" and "I0" in optimal_params:
            i0 = optimal_params.get("I0", 1000.0)
            th0 = optimal_params.get("theta0", 0.0)
            c = optimal_params.get("C", 0.0)
            rad = np.radians(x_dense - th0)
            y_fit = i0 * (np.cos(rad) ** 2) + c
            eq_str = f"$I(\\theta) = {i0:.1f} \\cos^2(\\theta - {th0:.1f}^\\circ) + {c:.1f}$"
        elif experiment_type == "inverse_square_law" and "I0" in optimal_params:
            i0 = optimal_params.get("I0", 1000.0)
            d0 = optimal_params.get("d0", 5.0)
            c = optimal_params.get("C", 0.0)
            y_fit = (i0 * 100.0) / ((x_dense + d0) ** 2) + c
            eq_str = f"$I(d) = \\frac{{{i0:.1f} \\times 100}}{{(d + {d0:.1f})^2}} + {c:.1f}$"
        else:
            y_fit = np.poly1d(np.polyfit(X, y, 2))(x_dense)
            eq_str = "Empirical Fit"

        ax1.plot(x_dense, y_fit, color=color_fit, linewidth=2.2, label=f'Fitted Model\n$R^2 = {r2_score:.4f}$\nRMSE = {rmse:.2f}')
        ax1.set_title(f"A. Physical Model: {best_model_name}", fontsize=11, fontweight='bold')
        ax1.set_xlabel(x_label, fontsize=10)
        ax1.set_ylabel(y_label, fontsize=10)
        ax1.legend(loc='best', fontsize=8, frameon=True)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # -------------------------------------------------------------
        # Panel 2: Gaussian Process Active Learning Surrogate
        # -------------------------------------------------------------
        ax2 = fig.add_subplot(1, 3, 2)
        if gp_mu is not None and gp_sigma is not None and gp_grid is not None:
            ax2.plot(gp_grid.flatten(), gp_mu, color=color_gp, linewidth=2.0, label='GP Mean $\\mu(x)$')
            ax2.fill_between(
                gp_grid.flatten(),
                gp_mu - 1.96 * gp_sigma,
                gp_mu + 1.96 * gp_sigma,
                color=color_ci,
                alpha=0.35,
                label='95% Confidence ($1.96\\sigma$)'
            )
            ax2.scatter(X, y, color=color_obs, s=40, edgecolors='k', zorder=5, label='Sampled Points')
            ax2.set_title("B. Active Learning Surrogate (GP)", fontsize=11, fontweight='bold')
            ax2.set_xlabel(x_label, fontsize=10)
            ax2.set_ylabel(y_label, fontsize=10)
            ax2.legend(loc='best', fontsize=8, frameon=True)
            ax2.grid(True, linestyle='--', alpha=0.6)
        else:
            ax2.text(0.5, 0.5, "Surrogate model uninitialized", ha='center', va='center')

        # -------------------------------------------------------------
        # Panel 3: Residual Analysis & Error Bounds
        # -------------------------------------------------------------
        ax3 = fig.add_subplot(1, 3, 3)
        if experiment_type == "malus_law" and "I0" in optimal_params:
            rad_obs = np.radians(X - optimal_params.get("theta0", 0.0))
            y_pred_obs = optimal_params.get("I0", 1000.0) * (np.cos(rad_obs) ** 2) + optimal_params.get("C", 0.0)
            residuals = y - y_pred_obs
        elif experiment_type == "inverse_square_law" and "I0" in optimal_params:
            y_pred_obs = (optimal_params.get("I0", 1000.0) * 100.0) / ((X + optimal_params.get("d0", 5.0)) ** 2) + optimal_params.get("C", 0.0)
            residuals = y - y_pred_obs
        else:
            residuals = y - np.mean(y)

        ax3.axhline(0, color='gray', linestyle='--', linewidth=1.2)
        ax3.scatter(X, residuals, color="#9467bd", s=45, edgecolors='k', alpha=0.85)
        ax3.set_title("C. Residual Distribution $\\epsilon_i = y_i - \\hat{y}_i$", fontsize=11, fontweight='bold')
        ax3.set_xlabel(x_label, fontsize=10)
        ax3.set_ylabel("Residual Error (lux)", fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"discovery_plot_{experiment_type}_{timestamp}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return filepath
