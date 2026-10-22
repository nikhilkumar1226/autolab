"""
Publication figures for the benchmark (PNG at 300 dpi and PDF), from the benchmark rows only.

Colours follow the method, never its rank (validated categorical palette, light surface). Each
method also has its own marker and line style, so identity never rests on colour alone. Every
figure carries its data source and the LLM provider in a footer.
"""
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark.metrics import mean_ci, wilson

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD_ORDER = ["autolab", "llm_only", "gp_only", "random", "grid"]
METHOD_STYLE = {
    "autolab": {"color": "#2a78d6", "marker": "o", "ls": "-", "label": "AutoLab (full)"},
    "llm_only": {"color": "#eb6834", "marker": "s", "ls": "--", "label": "LLM only"},
    "gp_only": {"color": "#1baf7a", "marker": "^", "ls": "-.", "label": "GP only"},
    "random": {"color": "#eda100", "marker": "D", "ls": ":", "label": "Random"},
    "grid": {"color": "#e87ba4", "marker": "v", "ls": (0, (5, 1)), "label": "Grid"},
}
STATUS = {"recovered": "#0ca30c", "detected_not_recovered": "#ec835a", "missed": "#d03b3b",
          "no_effect": "#c3c2b7", "not_reached": "#e1e0d9"}
CONDITION_LABEL = {"clean": "Clean", "nonideal": "Non-idealities on", "faults": "Faults injected",
                   "hidden_apparatus": "Apparatus hidden"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "lines.linewidth": 2.0, "axes.axisbelow": True,
    })


def _save(fig, out: Path, name: str, footer: str) -> List[Path]:
    fig.text(0.01, 0.005, footer, fontsize=7, color=MUTED, ha="left", va="bottom")
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _methods(rows: Sequence[Dict[str, Any]]) -> List[str]:
    present = {r["method"] for r in rows}
    return [m for m in METHOD_ORDER if m in present]


def param_error_vs_n(rows: List[Dict[str, Any]], experiments: Sequence[str], out: Path, footer: str) -> List[Path]:
    curves = [r for r in rows if r["kind"] == "curve" and r["status"] == "ok"]
    if not curves:
        return []
    cols = 4
    nrows = math.ceil(len(experiments) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(11, 2.6 * nrows), squeeze=False)
    for ax, experiment in zip(axes.flat, experiments):
        for method in _methods(curves):
            style = METHOD_STYLE[method]
            runs = [dict(r["curve"]) for r in curves if r["experiment"] == experiment and r["method"] == method]
            ns = sorted({n for c in runs for n in c})
            # Errors span orders of magnitude: geometric mean, with the 95% CI of the mean of log10(error).
            stats = [(n, *mean_ci([math.log10(max(c[n], 1e-6)) for c in runs if n in c])) for n in ns]
            stats = [s for s in stats if not math.isnan(s[1])]
            if not stats:
                continue
            n, mean, _, lo, hi = zip(*stats)
            mean = 10 ** np.array(mean)
            lo = np.where(np.isnan(lo), mean, 10 ** np.array(lo, dtype=float))
            hi = np.where(np.isnan(hi), mean, 10 ** np.array(hi, dtype=float))
            counts = np.array([sum(1 for c in runs if k in c) for k in n])
            band = counts >= 3  # an interval from 2 runs (t = 12.7) says nothing; no band then
            if band.any():
                ax.fill_between(np.array(n)[band], lo[band], hi[band], color=style["color"], alpha=0.12, linewidth=0)
            ax.plot(n, mean, color=style["color"], ls=style["ls"], marker=style["marker"], markersize=3,
                    markevery=4, label=style["label"])
        ax.set_yscale("log")
        ax.set_title(experiment.replace("_", " "))
        ax.set_xlabel("measurements")
    for ax in axes.flat[len(experiments):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("largest parameter error (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return _save(fig, out, "fig_param_error_vs_n",
                 footer + " Geometric mean and 95% CI over seeds (bands only with 3 or more runs); clean twin; fit of the true law.")


def _accuracy_bars(ax, rows, methods):
    xs = np.arange(len(methods))
    for i, method in enumerate(methods):
        ok = [r for r in rows if r["method"] == method and r["status"] == "ok"]
        rate, lo, hi = wilson(sum(r["correct"] for r in ok), len(ok))
        if math.isnan(rate):
            continue
        style = METHOD_STYLE[method]
        ax.bar(xs[i], rate, width=0.7, color=style["color"], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(xs[i], rate, yerr=[[rate - lo], [hi - rate]], color=INK_2, capsize=3, linewidth=1)
        ax.text(xs[i], min(hi, 1.0) + 0.03, f"{rate:.0%}\nn={len(ok)}", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(xs, [METHOD_STYLE[m]["label"] for m in methods], rotation=30, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])


def law_accuracy(rows: List[Dict[str, Any]], conditions: Sequence[str], out: Path, footer: str) -> List[Path]:
    adaptive = [r for r in rows if r["kind"] == "adaptive"]
    if not adaptive:
        return []
    methods = _methods(adaptive)
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.6 * len(conditions), 3.4), squeeze=False, sharey=True)
    for ax, condition in zip(axes.flat, conditions):
        _accuracy_bars(ax, [r for r in adaptive if r["condition"] == condition], methods)
        ax.set_title(CONDITION_LABEL.get(condition, condition))
    axes.flat[0].set_ylabel("law identified correctly (blind)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out, "fig_law_accuracy", footer + " All experiments pooled; Wilson 95% CI.")


def equal_budget_accuracy(rows: List[Dict[str, Any]], conditions: Sequence[str], out: Path, footer: str) -> List[Path]:
    fixed = [r for r in rows if r["kind"] == "fixed"]
    if not fixed:
        return []
    chosen = fixed + [r for r in rows if r["kind"] == "adaptive" and r["method"] == "autolab"]
    methods = _methods(chosen)
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.4 * len(conditions), 3.4), squeeze=False, sharey=True)
    for ax, condition in zip(axes.flat, conditions):
        _accuracy_bars(ax, [r for r in chosen if r["condition"] == condition], methods)
        ax.set_title(CONDITION_LABEL.get(condition, condition))
    axes.flat[0].set_ylabel("law identified correctly (blind)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out, "fig_equal_budget_accuracy",
                 footer + " Baselines run to exactly AutoLab's median number of measurements per experiment and condition.")


def fault_recovery(rows: List[Dict[str, Any]], fault_types: Sequence[str], out: Path, footer: str) -> List[Path]:
    faulty = [r for r in rows if r["kind"] == "adaptive" and r["condition"] == "faults" and r["status"] == "ok"]
    if not faulty:
        return []
    methods = _methods(faulty)
    order = ["recovered", "detected_not_recovered", "missed", "no_effect", "not_reached"]
    fig, axes = plt.subplots(1, len(fault_types), figsize=(3.6 * len(fault_types), 3.0), squeeze=False, sharey=True)
    for ax, fault in zip(axes.flat, fault_types):
        for i, method in enumerate(methods):
            statuses = [r["fault_outcomes"].get(fault) for r in faulty if r["method"] == method]
            statuses = [s for s in statuses if s]
            left = 0.0
            for status in order:
                share = statuses.count(status) / len(statuses) if statuses else 0.0
                if share:
                    ax.barh(i, share, left=left, color=STATUS[status], edgecolor=SURFACE, linewidth=2, height=0.7,
                            label=status.replace("_", " "))
                    left += share
        ax.set_yticks(range(len(methods)), [METHOD_STYLE[m]["label"] for m in methods])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
        ax.grid(axis="y", visible=False)
        ax.set_title(fault.replace("_", " "))
        ax.invert_yaxis()
    handles, labels = [], []
    for ax in axes.flat:
        for h, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(h)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return _save(fig, out, "fig_fault_recovery", footer + " Faults condition, all experiments; share of runs.")


def cost_per_run(rows: List[Dict[str, Any]], price: Tuple[float, float], provider: str, usd_to_inr: float,
                 out: Path, footer: str) -> List[Path]:
    adaptive = [r for r in rows if r["kind"] == "adaptive" and r["status"] == "ok"]
    if not adaptive:
        return []
    methods = _methods(adaptive)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    real = provider == "litellm"
    for ax, (title, value) in zip(axes, [
        ("LLM calls per run", lambda r: r["llm_calls"]),
        ("cost per run, USD" + ("" if real else " (estimated)"),
         lambda r: r["cost_usd"] if real else (r["est_input_tokens"] * price[0] + r["est_output_tokens"] * price[1]
                                               if price else math.nan)),
    ]):
        for i, method in enumerate(methods):
            mean, _, lo, hi = mean_ci([value(r) for r in adaptive if r["method"] == method])
            if math.isnan(mean):
                continue
            ax.bar(i, mean, width=0.7, color=METHOD_STYLE[method]["color"], edgecolor=SURFACE, linewidth=2)
            if not math.isnan(lo):
                ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], color=INK_2, capsize=3, linewidth=1)
            text = f"{mean:.1f}" if "calls" in title else f"${mean:.4f}\n(INR {mean * usd_to_inr:.2f})"
            ax.text(i, (hi if not math.isnan(hi) else mean), text, ha="center", va="bottom", fontsize=7, color=INK_2)
        ax.set_xticks(range(len(methods)), [METHOD_STYLE[m]["label"] for m in methods], rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.3 if top > 0 else 1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    note = "" if real else " Cost estimated for a real model from prompt and answer sizes (4 characters per token)."
    return _save(fig, out, "fig_cost_per_run", footer + f" INR at {usd_to_inr:g} per USD." + note)


def footer_text(args) -> str:
    """Every figure's footer: data source, who answered for the LLM methods, and the seed caveat."""
    if getattr(args, "label", None):
        return (f"Data source: DIGITAL TWIN. {args.label}.\nLLM methods: {args.llm_seeds} seeds; baselines: "
                f"{args.seeds} seeds (their clean rows stand for apparatus hidden). Differences between the two groups "
                f"mix method and seed set.")
    provider_text = ("mock provider (deterministic stand-in, not a language model)" if args.provider == "mock"
                     else "LLM via litellm")
    footer = f"Data source: DIGITAL TWIN. LLM methods: {provider_text}."
    if args.provider == "mock":
        footer += (f"\nWith mock, AutoLab = GP only on the same seeds; AutoLab and LLM only use {args.llm_seeds} seeds, "
                   f"the others {args.seeds}, so differences between them here come from the seed sets.")
    return footer


def all_figures(rows: List[Dict[str, Any]], args, price, out: Path) -> List[Path]:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    footer = footer_text(args)
    paths = []
    paths += param_error_vs_n(rows, args.experiments, out, footer)
    paths += law_accuracy(rows, args.conditions, out, footer)
    paths += equal_budget_accuracy(rows, args.conditions, out, footer)
    paths += fault_recovery(rows, ["light_spike", "stuck_actuator", "sensor_dropout"], out, footer)
    paths += cost_per_run(rows, price, args.provider, args.usd_inr, out, footer)
    return paths
