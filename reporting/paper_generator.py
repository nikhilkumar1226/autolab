"""
Automated Scientific Report Generator.
Turns a run record into a Markdown and a LaTeX report.

Every number comes straight from the run record. The report states the data
source and makes no claims the record does not support; the interpretive
discussion is left to the Writer agent.
"""
from pathlib import Path
from typing import List, Optional
from analysis.model_library import AnalysisResult, ModelFit
from analysis.stopping import CriterionCheck, StopDecision
from records.run_record import RunRecord

def _fmt(v: Optional[float], spec: str = ".4g") -> str:
    return "n/a" if v is None else format(v, spec)

def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.2%}"

def _tex_escape(text: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        text = text.replace(a, b)
    return text

def _sampler_text(record: RunRecord) -> str:
    if record.config.agent.sampler == "random":
        return "uniform random sampling within the safe bounds (baseline)"
    return "Gaussian-process uncertainty sampling (next point = maximum predictive standard deviation)"

def _criteria(stop: StopDecision) -> List[tuple]:
    def row(label: str, c: CriterionCheck, value: str, threshold: str) -> tuple:
        return label, value, threshold, "yes" if c.passed else "no", c.detail
    return [
        row("Delta-BIC to runner-up", stop.delta_bic, _fmt(stop.delta_bic.value, ".2f"), f">= {stop.delta_bic.threshold:g}"),
        row("Max relative SE (best model)", stop.max_rel_se, _pct(stop.max_rel_se.value), f"< {stop.max_rel_se.threshold:.0%}"),
        row("Measurements", stop.min_points, _fmt(stop.min_points.value, ".0f"), f">= {stop.min_points.threshold:.0f}"),
        row("Budget remaining", stop.budget, _fmt(stop.budget.value, ".0f"), f"< {stop.budget.threshold:.0f}"),
    ]

def _summary_sentence(record: RunRecord, analysis: Optional[AnalysisResult]) -> str:
    n = record.result.n_measurements if record.result else 0
    stop_msg = record.stop.message if record.stop else "unknown"
    if analysis is None or analysis.best is None:
        return f"{n} measurements were taken; no model could be fitted. Stop: {stop_msg}."
    best: ModelFit = analysis.best
    text = f"After {n} measurements, blind model comparison ranked **{best.name}** first (BIC = {best.bic:.2f})"
    if analysis.runner_up is not None:
        text += f", ahead of {analysis.runner_up.name} by ΔBIC = {analysis.delta_bic_runner_up:.2f}"
    return text + f". Stop: {stop_msg}."

class SciencePaperGenerator:
    """
    Synthesizes run records into Markdown and LaTeX reports.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_paper(self, record: RunRecord) -> Path:
        """
        Generates the report in Markdown format.
        """
        filepath = self.output_dir / f"report_{record.run_id}.md"
        view = record.experiment.agent_view
        analysis = record.result.final_analysis if record.result else None
        best = analysis.best if analysis else None
        stopping = record.config.stopping
        hyp = record.hypothesis

        obs_rows = "".join(
            f"| {s.index} | {s.source} | {_fmt(s.position, '.2f')} | {_fmt(s.value, '.2f')} |\n"
            for s in record.steps if s.status == "measured"
        )
        rejected = [s for s in record.steps if s.status != "measured"]
        rejected_md = "".join(
            f"- Attempt {s.index} ({s.source}) at {_fmt(s.requested_position, '.2f')}: **{s.status}**: {s.message}\n"
            for s in rejected
        ) or "None.\n"

        model_rows = ""
        skipped_md = "None.\n"
        if analysis is not None:
            model_rows = "".join(
                f"| {rank} | {f.name} | `{f.formula}` | {f.n_params} | {f.bic:.2f} | {f.delta_bic:.2f} | {f.r2:.4f} | {f.rmse:.3f} |\n"
                for rank, f in enumerate(analysis.fits, 1)
            )
            if analysis.skipped:
                skipped_md = "".join(f"- {s.name}: {s.reason}\n" for s in analysis.skipped)

        param_rows = ""
        if best is not None:
            param_rows = "".join(
                f"| `{p}` | {_fmt(v)} | {_fmt(best.std_errors[p])} | {_pct(best.rel_std_errors[p])} |\n"
                for p, v in best.params.items()
            )
        equation = f"$$\n{record.result.equation_latex}\n$$" if record.result and record.result.equation_latex else "*No model fitted.*"

        criteria_rows = "".join(
            f"| {label} | {value} | {thr} | {ok} | {detail} |\n" for label, value, thr, ok, detail in _criteria(record.stop)
        ) if record.stop else ""
        stop_reason = record.stop.reason.value if record.stop and record.stop.reason else "unknown"
        provisional = (
            "" if stop_reason == "converged" else
            "\n**The stopping criteria were not all met, so this identification is provisional.**\n"
        )

        plot_md = f"![Discovery figure]({record.artifacts.plot})" if record.artifacts.plot else "*Figure not generated.*"

        content = f"""# {hyp.title if hyp else record.question}

> **Data source: {record.data_source.value}**

**Run ID:** `{record.run_id}`
**Started (UTC):** {record.started_at:%Y-%m-%d %H:%M:%S}
**Seed:** {record.seed} (rerun with the same seed and config to reproduce)
**Generated by:** AutoLab. Every number below is taken from the run record `{record.run_id}.json`.

---

## Summary
{_summary_sentence(record, analysis)}
{provisional}
## 1. Question and hypothesis
**Research question:** {record.question}

**Apparatus (as described to the agents):** {view.description}

**Hypothesis:** {hyp.statement if hyp else "n/a"}

## 2. Method
- **Independent variable (x):** {view.independent.name} ({view.independent.units}), safe range [{view.safe_bounds[0]:g}, {view.safe_bounds[1]:g}]
- **Dependent variable (y):** {view.dependent.name} ({view.dependent.units})
- **Sampling:** an initial design of {record.config.agent.initial_sample_count} points across the safe range, then {_sampler_text(record)}. Every position passed through the hardware safety guard.
- **Blind analysis:** the analyst saw only the measured (x, y) pairs, not the experiment or its true law. It fitted every model in a fixed library of candidates by nonlinear least squares and ranked them by the Bayesian Information Criterion, BIC = n ln(SSR/n) + k ln n.
- **Stopping rule:** stop when the best model beats the runner-up by ΔBIC ≥ {stopping.delta_bic_threshold:g}, every parameter's relative standard error is below {stopping.max_rel_se:.0%}, and at least {stopping.min_points} points exist; otherwise stop at the budget of {stopping.max_measurements} measurements. Relative SE is SE/|value| for scale parameters, and SE divided by the measured x- or y-span for location and offset parameters.

## 3. Observations
| Attempt | Source | x ({view.independent.units}) | y ({view.dependent.units}) |
| :---: | :---: | :---: | :---: |
{obs_rows}
**Rejected or failed attempts:**
{rejected_md}
## 4. Model comparison
| Rank | Model | Formula | k | BIC | ΔBIC | $R^2$ | RMSE |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
{model_rows}
**Models not fitted:**
{skipped_md}
## 5. Parameters of the best-ranked model{f": {best.name}" if best else ""}
| Parameter | Value | Std. error | Relative SE |
| :--- | :---: | :---: | :---: |
{param_rows}
{equation}

## 6. Stopping decision
**Reason:** `{stop_reason}`. {record.stop.message if record.stop else ""}

| Criterion | Value | Threshold | Met | Note |
| :--- | :---: | :---: | :---: | :--- |
{criteria_rows}
## 7. Figure
{plot_md}

*Figure 1: (A) measurements and the best-ranked model; (B) Gaussian-process surrogate with 95% band; (C) residuals of the best-ranked model. Data source: {record.data_source.value}.*

## 8. Limitations
- The best model is only the best of the candidates in the library; a law outside the library cannot be identified.
- BIC is an asymptotic criterion. With few measurements it can favour the wrong model with ΔBIC above the threshold.

## References
1. R. E. Kass and A. E. Raftery, "Bayes Factors", *Journal of the American Statistical Association*, vol. 90, no. 430, pp. 773–795 (1995).
"""
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def generate_latex_paper(self, record: RunRecord) -> Path:
        """
        Generates LaTeX source for the report.
        """
        filepath = self.output_dir / f"report_{record.run_id}.tex"
        view = record.experiment.agent_view
        analysis = record.result.final_analysis if record.result else None
        best = analysis.best if analysis else None
        esc = _tex_escape

        model_rows = ""
        if analysis is not None:
            model_rows = "\n".join(
                f"{esc(f.name)} & {f.n_params} & {f.bic:.2f} & {f.delta_bic:.2f} & {f.r2:.4f} \\\\"
                for f in analysis.fits
            )
        param_rows = ""
        if best is not None:
            param_rows = "\n".join(
                f"{esc(p)} & {_fmt(v)} & {_fmt(best.std_errors[p])} & {esc(_pct(best.rel_std_errors[p]))} \\\\"
                for p, v in best.params.items()
            )
        equation = (
            f"\\begin{{equation}}\n{record.result.equation_latex}\n\\end{{equation}}"
            if record.result and record.result.equation_latex else "No model was fitted."
        )
        summary = esc(_summary_sentence(record, analysis).replace("**", ""))
        stop_reason = record.stop.reason.value if record.stop and record.stop.reason else "unknown"

        tex_content = f"""\\documentclass[10pt]{{article}}
\\usepackage[margin=0.9in]{{geometry}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{booktabs}}
\\usepackage{{newunicodechar}}
\\newunicodechar{{Δ}}{{$\\Delta$}}
\\newunicodechar{{θ}}{{$\\theta$}}

\\title{{{esc(record.hypothesis.title if record.hypothesis else record.question)}}}
\\author{{AutoLab (automated report)}}
\\date{{{record.started_at:%Y-%m-%d}}}

\\begin{{document}}
\\maketitle

\\noindent\\fbox{{\\textbf{{Data source: {esc(record.data_source.value)}}}}}\\quad Run \\texttt{{{esc(record.run_id)}}}, seed {record.seed}.

\\begin{{abstract}}
{summary}
\\end{{abstract}}

\\section{{Method}}
The independent variable is {esc(view.independent.name)} ({esc(view.independent.units)}) within the safe range
[{view.safe_bounds[0]:g}, {view.safe_bounds[1]:g}]; the dependent variable is {esc(view.dependent.name)} ({esc(view.dependent.units)}).
Every position passed through the hardware safety guard. The analyst saw only the measured $(x, y)$ pairs and ranked a fixed
library of candidate models by BIC.

\\section{{Model comparison}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Model & $k$ & BIC & $\\Delta$BIC & $R^2$ \\\\
\\midrule
{model_rows}
\\bottomrule
\\end{{tabular}}

\\section{{Best-ranked model}}
\\begin{{tabular}}{{lccc}}
\\toprule
Parameter & Value & Std.\\ error & Relative SE \\\\
\\midrule
{param_rows}
\\bottomrule
\\end{{tabular}}

{equation}

\\section{{Stopping decision}}
Reason: \\texttt{{{esc(stop_reason)}}}. {esc(record.stop.message) if record.stop else ""}

\\section{{Limitations}}
Only models in the candidate library can be identified, and BIC can favour the wrong model when few measurements are available.

\\end{{document}}
"""
        filepath.write_text(tex_content, encoding="utf-8")
        return filepath
