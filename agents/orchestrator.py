"""
Discovery Orchestrator.
Main controller managing the closed-loop autonomous scientific discovery lifecycle.
"""
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import SystemConfig, config as default_config, PLOTS_DIR, REPORTS_DIR
from hardware.base import BaseHardwareRig
from hardware.simulator import PhysicsSimulationRig
from hardware.serial_driver import SerialHardwareRig
from hardware.safety_guard import HardwareSafetyGuard
from optimization.bayesian_opt import ActiveLearningEngine, AcquisitionStrategy
from agents.models import Hypothesis, AnalysisResult, DiscoverySummary
from agents.lead_scientist import LeadScientistAgent
from agents.executor import HardwareExecutorAgent
from agents.analyst import DataAnalystAgent
from agents.tools import LabTools
from reporting.visualizer import ScienceVisualizer
from reporting.paper_generator import SciencePaperGenerator

logger = logging.getLogger(__name__)
console = Console()

class AutonomousDiscoveryEngine:
    """
    Coordinates the end-to-end closed-loop discovery:
    Hypothesize -> Actuate -> Observe -> Bayesian Optimize -> Model Selection -> Publish
    """

    def __init__(self, sys_config: Optional[SystemConfig] = None):
        self.config = sys_config or default_config
        self.rig: BaseHardwareRig = (
            PhysicsSimulationRig(self.config.simulation)
            if self.config.hardware.simulate
            else SerialHardwareRig(self.config.hardware)
        )
        self.safety_guard = HardwareSafetyGuard(self.rig, self.config.hardware)
        self.optimizer = ActiveLearningEngine(
            min_bound=self.config.hardware.min_actuator_value,
            max_bound=self.config.hardware.max_actuator_value
        )
        self.tools = LabTools(self.safety_guard, self.optimizer)

        # Agent personas
        self.scientist = LeadScientistAgent(self.config.agent.model_name)
        self.executor = HardwareExecutorAgent(self.tools)
        self.analyst = DataAnalystAgent(self.config.agent.convergence_r2_threshold)

        # Reporting
        self.visualizer = ScienceVisualizer(PLOTS_DIR)
        self.paper_generator = SciencePaperGenerator(REPORTS_DIR)

    def run_discovery_loop(self, research_goal: str) -> DiscoverySummary:
        """
        Executes the autonomous closed-loop scientific experiment.
        """
        experiment_id = f"EXP_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}"
        start_time = datetime.utcnow()

        console.print(Panel.fit(
            f"[bold cyan]AUTONOMOUS SCIENTIFIC DISCOVERY SYSTEM[/bold cyan]\n"
            f"[yellow]Experiment ID:[/yellow] {experiment_id}\n"
            f"[yellow]Research Prompt:[/yellow] {research_goal}\n"
            f"[yellow]Hardware Mode:[/yellow] {'DIGITAL TWIN SIMULATION' if self.config.hardware.simulate else 'PHYSICAL SERIAL RIG'}",
            border_style="cyan"
        ))

        # 1. Connect and Home Hardware
        console.print("[dim]Connecting and homing hardware rig...[/dim]")
        self.safety_guard.connect()
        self.safety_guard.home()

        # 2. Formulate Hypothesis
        console.print("\n[bold green]=== Step 1: Lead Scientist Formulating Hypothesis ===[/bold green]")
        hypothesis: Hypothesis = self.scientist.formulate_hypothesis(research_goal)
        console.print(f"[bold]Title:[/bold] {hypothesis.title}")
        console.print(f"[bold]Hypothesis:[/bold] {hypothesis.statement}")
        console.print(f"[bold]Target Model:[/bold] {hypothesis.target_relationship}")
        console.print(f"[bold]Parameter Domain:[/bold] [{hypothesis.suggested_range[0]}, {hypothesis.suggested_range[1]}]")

        # 3. Initial Exploratory Sampling
        console.print("\n[bold green]=== Step 2: Initial Exploratory Baseline Survey ===[/bold green]")
        init_steps = self.scientist.generate_initial_plan(hypothesis, num_samples=self.config.agent.initial_sample_count)
        for step in init_steps:
            res = self.executor.execute_step(step.target_position, step.rationale)
            console.print(f"  -> Actuated to {res['position']:>6.1f}° | Sensor: {res['sensor_value']:>7.2f} lux")

        # 4. Closed-Loop Active Learning / Bayesian Optimization Iterations
        console.print("\n[bold green]=== Step 3: Active Learning & Statistical Convergence Loop ===[/bold green]")
        analysis: Optional[AnalysisResult] = None
        iteration = len(init_steps)

        while iteration < self.config.agent.max_discovery_iterations:
            iteration += 1
            
            # Analyze current dataset
            analysis = self.analyst.analyze_dataset(
                X_data=self.tools.history_x,
                y_data=self.tools.history_y,
                experiment_type=self.config.simulation.experiment_type,
                iteration=iteration
            )

            best_m = analysis.best_model
            console.print(
                f"[cyan][Iter {iteration:02d}][/cyan] Best Fit: [bold]{best_m.model_name}[/bold] "
                f"| R² = [bold green]{best_m.r2_score:.4f}[/bold green] "
                f"| RMSE = {best_m.rmse:.2f} "
                f"| Total Samples: {len(self.tools.history_x)}"
            )

            # Check if convergence reached
            if analysis.is_converged:
                console.print(f"\n[bold green]✓ SCIENTIFIC CONVERGENCE ACHIEVED:[/bold green] {analysis.convergence_rationale}")
                break

            # Suggest next optimal point using Active Learning Gaussian Process
            suggestion = self.tools.tool_suggest_next_point(strategy="uncertainty")
            next_pos = suggestion["suggested_position"]
            max_sigma = suggestion.get("max_sigma", 0.0)

            console.print(f"  [yellow]Active Learning (Uncertainty Sampling):[/yellow] Proposing next target at [bold]{next_pos:.1f}°[/bold] (Uncertainty σ = {max_sigma:.2f})")
            
            # Execute next measurement
            res = self.executor.execute_step(next_pos, rationale="Active Learning Epistemic Variance Minimization")
            console.print(f"  -> Actuated to {res['position']:>6.1f}° | Sensor: {res['sensor_value']:>7.2f} lux")

        # Final Analysis
        final_analysis = self.analyst.analyze_dataset(
            X_data=self.tools.history_x,
            y_data=self.tools.history_y,
            experiment_type=self.config.simulation.experiment_type,
            iteration=iteration
        )
        best_model = final_analysis.best_model

        # 5. Visualizer: Generate Publication-Grade Plots
        console.print("\n[bold green]=== Step 4: Generating Publication-Grade Visualizations ===[/bold green]")
        self.optimizer.fit(np.array(self.tools.history_x), np.array(self.tools.history_y))
        gp_mu, gp_sigma = self.optimizer.predict(self.optimizer.candidate_grid)

        plot_path = self.visualizer.generate_discovery_figure(
            X_data=self.tools.history_x,
            y_data=self.tools.history_y,
            experiment_type=self.config.simulation.experiment_type,
            best_model_name=best_model.model_name,
            optimal_params=best_model.optimal_parameters,
            r2_score=best_model.r2_score,
            rmse=best_model.rmse,
            gp_mu=gp_mu,
            gp_sigma=gp_sigma,
            gp_grid=self.optimizer.candidate_grid
        )
        console.print(f"  [cyan]Figure saved to:[/cyan] {plot_path}")

        # Formulate final fitted equation string
        if self.config.simulation.experiment_type == "malus_law":
            eq = f"I(\\theta) = {best_model.optimal_parameters.get('I0', 0):.2f} \\cos^2(\\theta - {best_model.optimal_parameters.get('theta0', 0):.2f}^\\circ) + {best_model.optimal_parameters.get('C', 0):.2f}"
        else:
            eq = f"I(d) = \\frac{{{best_model.optimal_parameters.get('I0', 0):.2f} \\times 100}}{{(d + {best_model.optimal_parameters.get('d0', 0):.2f})^2}} + {best_model.optimal_parameters.get('C', 0):.2f}"

        end_time = datetime.utcnow()
        summary = DiscoverySummary(
            experiment_id=experiment_id,
            start_time=start_time,
            end_time=end_time,
            total_iterations=len(self.tools.history_x),
            hypothesis=hypothesis,
            confirmed_law=best_model.model_name,
            fitted_equation=eq,
            final_r2=best_model.r2_score,
            final_rmse=best_model.rmse,
            parameters=best_model.optimal_parameters,
            observations_x=self.tools.history_x,
            observations_y=self.tools.history_y,
            plot_filepath=str(plot_path)
        )

        # 6. Writer: Synthesize Markdown & LaTeX Paper
        console.print("\n[bold green]=== Step 5: Automated Scientific Paper Synthesis ===[/bold green]")
        md_paper_path = self.paper_generator.generate_markdown_paper(summary, final_analysis)
        tex_paper_path = self.paper_generator.generate_latex_paper(summary, final_analysis)
        summary.paper_filepath = str(md_paper_path)

        console.print(f"  [cyan]Markdown Paper saved to:[/cyan] {md_paper_path}")
        console.print(f"  [cyan]LaTeX Paper saved to:[/cyan] {tex_paper_path}")

        # Disconnect rig cleanly
        self.safety_guard.disconnect()

        # Display Final Summary Table
        table = Table(title="Autonomous Scientific Discovery Results", show_lines=True)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Discovered Result", style="green")
        table.add_row("Hypothesis", hypothesis.statement)
        table.add_row("Confirmed Physical Law", best_model.model_name)
        table.add_row("Mathematical Equation", eq)
        table.add_row("Coefficient of Determination ($R^2$)", f"{best_model.r2_score:.4f}")
        table.add_row("Root Mean Square Error (RMSE)", f"{best_model.rmse:.3f} lux")
        table.add_row("Total Measurements Conducted", str(len(self.tools.history_x)))
        table.add_row("Report Document", str(md_paper_path))
        table.add_row("Plot Artifact", str(plot_path))
        console.print("\n", table)

        return summary
