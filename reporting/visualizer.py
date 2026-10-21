"""
Scientific Visualization Generator.
Produces publication-quality figures showcasing experimental observations,
the best-ranked model fit, Bayesian surrogate uncertainty bounds, and residuals.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive headless backend
import matplotlib.pyplot as plt

from analysis.model_library import MODEL_LIBRARY
from records.run_record import RunRecord

class ScienceVisualizer:
    """
    Generates academic publication-quality multi-panel experimental plots.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_discovery_figure(
        self,
        record: RunRecord,
        gp_grid: Optional[np.ndarray] = None,
        gp_mu: Optional[np.ndarray] = None,
        gp_sigma: Optional[np.ndarray] = None
    ) -> Path:
        """
        Creates a 3-panel figure: data + best fit, GP surrogate, residuals.
        """
        x_list, y_list = record.measured_points()
        X = np.array(x_list, dtype=float)
        y = np.array(y_list, dtype=float)

        view = record.experiment.agent_view
        x_label = f"{view.independent.name} {view.independent.symbol} ({view.independent.units})"
        y_label = f"{view.dependent.name} {view.dependent.symbol} ({view.dependent.units})"

        analysis = record.result.final_analysis if record.result else None
        best = analysis.best if analysis else None
        model = MODEL_LIBRARY[best.key] if best else None

        fig = plt.figure(figsize=(15, 5), dpi=300)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        # Colors
        color_obs = "#1f77b4" # Blue
        color_fit = "#d62728" # Red
        color_gp = "#2ca02c"  # Green
        color_ci = "#98df8a"  # Light green

        # -------------------------------------------------------------
        # Panel 1: Best model fit & experimental data
        # -------------------------------------------------------------
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.scatter(X, y, color=color_obs, s=45, alpha=0.85, edgecolors='k', zorder=5, label='Measurements')
        if model is not None and len(X):
            x_dense = np.linspace(min(X), max(X), 300)
            dbic = analysis.delta_bic_runner_up
            dbic_text = f"ΔBIC to runner-up = {dbic:.1f}" if dbic is not None else "ΔBIC: n/a"
            ax1.plot(x_dense, model.predict(x_dense, best.params), color=color_fit, linewidth=2.2,
                     label=f'{best.name}\n{dbic_text}\n$R^2$ = {best.r2:.4f}, RMSE = {best.rmse:.2f}')
            ax1.set_title(f"A. Best model by BIC: {best.name}", fontsize=11, fontweight='bold')
        else:
            ax1.set_title("A. Measurements (no model fitted)", fontsize=11, fontweight='bold')
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
        # Panel 3: Residuals of the best model
        # -------------------------------------------------------------
        ax3 = fig.add_subplot(1, 3, 3)
        if model is not None and len(X):
            residuals = y - model.predict(X, best.params)
            ax3.axhline(0, color='gray', linestyle='--', linewidth=1.2)
            ax3.scatter(X, residuals, color="#9467bd", s=45, edgecolors='k', alpha=0.85)
        else:
            ax3.text(0.5, 0.5, "No model fitted", ha='center', va='center')
        ax3.set_title("C. Residuals $\\epsilon_i = y_i - \\hat{y}_i$", fontsize=11, fontweight='bold')
        ax3.set_xlabel(x_label, fontsize=10)
        ax3.set_ylabel(f"Residual ({view.dependent.units})", fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.6)

        fig.suptitle(f"Data source: {record.data_source.value}    |    run {record.run_id}",
                     fontsize=10, fontweight='bold')
        plt.tight_layout()

        filepath = self.output_dir / f"discovery_plot_{record.run_id}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return filepath
