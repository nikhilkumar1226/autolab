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
    AnomalyDetected,
    EngineEvent,
    GPUpdated,
    HypothesisProposed,
    LLMCalled,
    MeasurementTaken,
    ReportReady,
    RunStarted,
    RunStopped,
    SafetyRejected,
    StrategyDecided,
)
from analysis.stopping import CriterionCheck


def _mark(check: CriterionCheck) -> str:
    return "[green]✓[/green]" if check.passed else "[red]✗[/red]"


def _by(source: str, model: Optional[str]) -> str:
    return {"llm": model or "LLM", "mock": "mock", "heuristic": "heuristic", "unavailable": "not generated"}.get(source, source)

INDENT = "           "


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
            f"[yellow]Sampler:[/yellow] {e.sampler}\n"
            + (f"[yellow]Digital twin:[/yellow] {escape(e.twin)}\n" if e.twin else "")
            + f"[yellow]Agents:[/yellow] {escape(e.llm)}"
            + ("   [yellow]Blind mode:[/yellow] on (names and apparatus hidden from the agents)" if e.blind else ""),
            border_style="cyan",
        ))

    def _on_hypothesis(self, e: HypothesisProposed) -> None:
        h = e.hypothesis
        self.console.print("\n[bold green]=== Hypothesis and initial design ===[/bold green]")
        self.console.print(f"[bold]Hypothesis[/bold] ({escape(_by(h.source, h.model))}): {escape(h.statement)}")
        self.console.print(f"[bold]Relationship:[/bold] {escape(h.target_relationship)}")
        if h.range_clipped:
            self.console.print(f"[bold yellow]CLIPPED:[/bold yellow] {escape(h.range_note)} (the safety limits are unchanged)")
        self.console.print(f"[bold]Initial design:[/bold] {', '.join(f'{p:g}' for p in e.initial_design)}")
        if h.design_rationale:
            self.console.print(f"[bold]Rationale:[/bold] {escape(h.design_rationale)}")
        self.console.print("\n[bold green]=== Measure, analyse, decide ===[/bold green]")

    def _on_measurement(self, e: MeasurementTaken) -> None:
        self._units_x, self._units_y = e.units_x, e.units_y
        self.console.print(
            f"[cyan]{escape(f'[step {e.step:02d}]')}[/cyan] {e.source:<14} x = {e.position:>7.2f} {e.units_x}  ->  "
            f"y = {e.value:>8.2f} {e.units_y}   (n = {e.n_points})"
        )

    def _on_safety_rejection(self, e: SafetyRejected) -> None:
        then = {"planner": " -> fed back to the Planner", "gp": " -> GP fallback"}.get(e.fed_back_to or "", "")
        self.console.print(
            f"[bold red]{escape(f'[step {e.step:02d}]')} SAFETY REJECTION[/bold red] ({e.source}): {escape(e.reason)}{then}"
        )

    def _on_anomaly(self, e: AnomalyDetected) -> None:
        h = e.handling
        value = "no reading" if e.value is None else f"y = {e.value:.4g} {self._units_y}"
        who = h.decided_by if h.decided_by != "analyst" else f"analyst ({_by(h.source or 'heuristic', h.model)})"
        self.console.print(
            f"[bold yellow]{escape(f'[step {e.step:02d}]')} FLAGGED[/bold yellow] ({', '.join(e.flags)}) at "
            f"x = {e.position:.4g} {self._units_x}, {value} -> {h.action} {escape(f'[{who}]')}: {escape(h.reason)}"
        )

    def _on_strategy(self, e: StrategyDecided) -> None:
        what = "GP" if e.choice == "gp" else "own point"
        self.console.print(
            f"{INDENT}[magenta]planner:[/magenta] {what}, x = {e.position:.2f} {self._units_x}  {escape(e.reason)}"
        )

    def _on_llm_call(self, e: LLMCalled) -> None:
        if not e.ok:
            self.console.print(f"{INDENT}[bold red]LLM {e.agent}/{e.purpose} failed:[/bold red] {escape(e.error or '')}")
            return
        if e.provider != "litellm":
            return  # mock calls cost nothing; the reasoning lines already show them
        cost = "cost n/a" if e.cost_usd is None else f"${e.cost_usd:.5f}"
        cached = ", cached" if e.cached else ""
        self.console.print(
            f"{INDENT}[dim]llm {e.agent}/{e.purpose}: {escape(e.model or '')}, {e.input_tokens}->{e.output_tokens} tok, "
            f"{cost}, {e.latency_s:.1f} s{cached}[/dim]"
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
        r = e.review
        if r is not None:
            self.console.print(f"{INDENT}[blue]analyst ({escape(_by(r.source, r.model))}):[/blue] {escape(r.interpretation)}")
            if r.candidate is not None:
                cand = r.candidate
                colour = "green" if cand.status == "accepted" else "yellow"
                self.console.print(
                    f"{INDENT}[{colour}]candidate model {cand.status}:[/{colour}] "
                    f"{escape(cand.spec.expression)}  ({escape(cand.reason)})"
                )

    def _on_stopped(self, e: RunStopped) -> None:
        d = e.decision
        reason = d.reason.value if d.reason else "unknown"
        colour = "green" if reason == "converged" else "yellow"
        self.console.print(f"\n[bold {colour}]STOPPED ({reason}):[/bold {colour}] {escape(d.message)}")
        r = e.rationale
        if r is not None and r.source == "unavailable":
            self.console.print(f"[bold red]{escape(r.text)}[/bold red]")
        elif r is not None:
            self.console.print(f"[bold]Planner's explanation[/bold] ({escape(_by(r.source, r.model))}): {escape(r.text)}")

        analysis = e.result.final_analysis if e.result else None
        table = Table(title="Blind model comparison (final)", show_lines=False)
        table.add_column("Rank", justify="right")
        table.add_column("Model", style="cyan")
        table.add_column("BIC", justify="right")
        table.add_column("ΔBIC", justify="right")
        table.add_column("R²", justify="right")
        if analysis is not None:
            for rank, f in enumerate(analysis.fits, 1):
                name = f.name if f.redundant is None else f"[dim]{escape(f.name)} (not counted: parsimony)[/dim]"
                table.add_row(str(rank), name, f"{f.bic:.2f}", f"{f.delta_bic:.2f}", f"{f.r2:.4f}")
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
        u = e.llm_usage
        if u is not None and u.calls:
            models = ", ".join(f"{m} x{k}" for m, k in u.calls_per_model.items())
            cost = f"estimated ${u.cost_usd:.4f}" + (f" (+{u.cost_unknown_calls} unknown)" if u.cost_unknown_calls else "")
            self.console.print(
                f"\n[bold]LLM usage:[/bold] {u.calls} calls ({u.cached_calls} cached, {u.failed_calls} failed): {escape(models)}; "
                f"{u.input_tokens} in / {u.output_tokens} out tokens; {cost}"
            )
        if e.faults:
            table = Table(title="Injected faults and outliers (scored against the twin's fault log)")
            for column in ("Fault", "Cycle", "Status", "Flags", "Then"):
                table.add_column(column)
            for f in e.faults:
                colour = {"recovered": "green", "missed": "red", "detected_not_recovered": "red"}.get(f.status, "dim")
                table.add_row(f.type + ("" if f.scheduled else " (random)"), str(f.at_cycle),
                              f"[{colour}]{f.status}[/{colour}]", ", ".join(f.flags), f.resolution or "")
            self.console.print(table)
        s = e.fault_summary
        if s is not None and s.unexplained:
            flags = ", ".join(f"{k} x{v}" for k, v in s.unexplained.items())
            self.console.print(f"[dim]Flagged readings with nothing injected (false alarms on the twin): {escape(flags)}[/dim]")
        for warning in e.warnings:
            self.console.print(f"[bold red]WARNING:[/bold red] {escape(warning)}")
        self.console.print("\n[bold green]=== Artifacts ===[/bold green]")
        for label, path in [("Run record", a.record), ("Figure", a.plot), ("Report (md)", a.report_md), ("Report (tex)", a.report_tex)]:
            if path:
                self.console.print(f"  [cyan]{label}:[/cyan] {escape(path)}")
