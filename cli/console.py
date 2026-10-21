"""
Rich console rendering of DiscoveryEngine events, for the command line only.
"""
from typing import Optional

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from agents.events import (
    AnalysisCompleted,
    EngineEvent,
    GPUpdated,
    HypothesisProposed,
    MeasurementTaken,
    ReportReady,
    RunStarted,
    RunStopped,
    SafetyRejected,
)
from analysis.stopping import CriterionCheck


def _mark(check: CriterionCheck) -> str:
    return "[green]✓[/green]" if check.passed else "[red]✗[/red]"


class ConsoleEventPrinter:
    """Callable passed to DiscoveryEngine(on_event=...); prints each event as it arrives."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._units_x = ""
        self._units_y = ""

    def __call__(self, event: EngineEvent) -> None:
        handler = getattr(self, f"_on_{event.type}", None)
        if handler is not None:
            handler(event)

    def _on_run_started(self, e: RunStarted) -> None:
        self.console.print(Panel.fit(
            f"[bold cyan]AUTOLAB: AUTONOMOUS DISCOVERY RUN[/bold cyan]\n"
            f"[bold black on yellow] DATA SOURCE: {e.data_source.value} [/bold black on yellow]\n"
            f"[yellow]Run ID:[/yellow] {escape(e.run_id)}\n"
            f"[yellow]Experiment:[/yellow] {escape(e.experiment)}   [dim](operator view; agents are not told this)[/dim]\n"
            f"[yellow]Question:[/yellow] {escape(e.question)}\n"
            f"[yellow]Seed:[/yellow] {e.seed}   [yellow]Budget:[/yellow] {e.budget} measurements   "
            f"[yellow]Sampler:[/yellow] {e.sampler}",
            border_style="cyan",
        ))

    def _on_hypothesis(self, e: HypothesisProposed) -> None:
        h = e.hypothesis
        self.console.print("\n[bold green]=== Hypothesis and initial design ===[/bold green]")
        self.console.print(f"[bold]Hypothesis:[/bold] {escape(h.statement)}")
        self.console.print(f"[bold]Relationship:[/bold] {escape(h.target_relationship)}")
        self.console.print(f"[bold]Initial design:[/bold] {', '.join(f'{p:g}' for p in e.initial_design)}")
        self.console.print("\n[bold green]=== Measure, analyse, decide ===[/bold green]")

    def _on_measurement(self, e: MeasurementTaken) -> None:
        self._units_x, self._units_y = e.units_x, e.units_y
        self.console.print(
            f"[cyan]{escape(f'[step {e.step:02d}]')}[/cyan] {e.source:<14} x = {e.position:>7.2f} {e.units_x}  ->  "
            f"y = {e.value:>8.2f} {e.units_y}   (n = {e.n_points})"
        )

    def _on_safety_rejection(self, e: SafetyRejected) -> None:
        self.console.print(
            f"[bold red]{escape(f'[step {e.step:02d}]')} SAFETY REJECTION[/bold red] ({e.source}): {escape(e.reason)}"
        )

    def _on_gp_update(self, e: GPUpdated) -> None:
        s = e.suggestion
        self.console.print(
            f"           [yellow]GP ({s.strategy}):[/yellow] next x = {s.position:.2f} {self._units_x}"
            f"  (max σ = {s.max_sigma:.2f})"
        )

    def _on_analysis(self, e: AnalysisCompleted) -> None:
        best = e.analysis.best
        if best is None:
            self.console.print("           [dim]analysis: no model could be fitted yet[/dim]")
            return
        c = e.stop_check
        dbic = f"{c.delta_bic.value:.1f}" if c.delta_bic.value is not None else "n/a"
        rse = f"{c.max_rel_se.value:.1%}" if c.max_rel_se.value is not None else "n/a"
        line = (
            f"           best: [bold]{best.name}[/bold] | ΔBIC {dbic} (≥{c.delta_bic.threshold:g}) {_mark(c.delta_bic)}"
            f" | max rel SE {rse} (<{c.max_rel_se.threshold:.0%}) {_mark(c.max_rel_se)}"
            f" | n {e.analysis.n_points} (≥{c.min_points.threshold:.0f}) {_mark(c.min_points)}"
        )
        if c.stable_best is not None:
            streak = f"{c.stable_best.value:.0f}" if c.stable_best.value is not None else "0"
            line += f" | stable {streak}/{c.stable_best.threshold:.0f} {_mark(c.stable_best)}"
        if c.prediction is not None:
            z = f"{c.prediction.value:.1f}σ" if c.prediction.value is not None else "n/a"
            line += f" | new point {z} (≤{c.prediction.threshold:g}σ) {_mark(c.prediction)}"
        self.console.print(line)

    def _on_stopped(self, e: RunStopped) -> None:
        d = e.decision
        reason = d.reason.value if d.reason else "unknown"
        colour = "green" if reason == "converged" else "yellow"
        self.console.print(f"\n[bold {colour}]STOPPED ({reason}):[/bold {colour}] {escape(d.message)}")

        analysis = e.result.final_analysis if e.result else None
        table = Table(title="Blind model comparison (final)", show_lines=False)
        table.add_column("Rank", justify="right")
        table.add_column("Model", style="cyan")
        table.add_column("BIC", justify="right")
        table.add_column("ΔBIC", justify="right")
        table.add_column("R²", justify="right")
        if analysis is not None:
            for rank, f in enumerate(analysis.fits, 1):
                table.add_row(str(rank), f.name, f"{f.bic:.2f}", f"{f.delta_bic:.2f}", f"{f.r2:.4f}")
            for s in analysis.skipped:
                table.add_row("-", f"[dim]{escape(s.name)}[/dim]", "[dim]skipped[/dim]", "", f"[dim]{escape(s.reason)}[/dim]")
        self.console.print(table)

        best = analysis.best if analysis else None
        if best is not None:
            params = Table(title=f"Parameters: {best.name}")
            params.add_column("Parameter", style="cyan")
            params.add_column("Value", justify="right")
            params.add_column("Std. error", justify="right")
            params.add_column("Relative SE", justify="right")
            for p, v in best.params.items():
                se = best.std_errors[p]
                rse = best.rel_std_errors[p]
                params.add_row(p, f"{v:.4g}", "n/a" if se is None else f"{se:.3g}", "n/a" if rse is None else f"{rse:.2%}")
            self.console.print(params)
            self.console.print(f"[bold]Equation:[/bold] {escape(e.result.equation_latex)}")

    def _on_report_ready(self, e: ReportReady) -> None:
        a = e.artifacts
        self.console.print("\n[bold green]=== Artifacts ===[/bold green]")
        for label, path in [("Run record", a.record), ("Figure", a.plot), ("Report (md)", a.report_md), ("Report (tex)", a.report_tex)]:
            if path:
                self.console.print(f"  [cyan]{label}:[/cyan] {escape(path)}")
