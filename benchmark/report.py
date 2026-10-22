"""
Benchmark outputs: benchmark_runs.csv (one row per run), benchmark_summary.csv (statistics per
group), figures/ and RESULTS.md (tables of the key numbers and a line on each figure).
Everything here is generated from the rows; nothing is typed in by hand.
"""
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from benchmark.figures import CONDITION_LABEL, METHOD_ORDER, METHOD_STYLE, all_figures
from benchmark.metrics import mean_ci, wilson
from benchmark.summary import price_per_token, summarise

PLAN = ["key", "kind", "method", "provider", "experiment", "condition", "seed", "budget", "fixed"]
NESTED = ["param_errors", "fault_outcomes", "models", "curve"]


def _flat(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (json.dumps(v, separators=(",", ":")) if k in NESTED else v) for k, v in row.items()}


def write_csv(path: Path, rows: List[Dict[str, Any]], first: Sequence[str] = ()) -> None:
    columns = list(first) + [k for k in (rows[0] if rows else {}) if k not in first]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()})


def _pct(rate: float, lo: float, hi: float) -> str:
    return "–" if math.isnan(rate) else f"{rate:.0%} [{lo:.0%}, {hi:.0%}]"


def _num(v: float, spec: str = ".1f") -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def _methods(rows) -> List[str]:
    present = {r["method"] for r in rows}
    return [m for m in METHOD_ORDER if m in present]


def _ok(rows, **match) -> List[Dict[str, Any]]:
    return [r for r in rows if r["status"] == "ok" and all(r[k] == v for k, v in match.items())]


def _shared_identical(rows: List[Dict[str, Any]], a: str, b: str) -> "tuple[int, int]":
    """(identical, shared): adaptive runs of methods a and b on the same experiment, condition and seed."""
    key = lambda r: (r["experiment"], r["condition"], r["seed"])
    first = {key(r): r for r in rows if r["method"] == a and r["status"] == "ok"}
    second = {key(r): r for r in rows if r["method"] == b and r["status"] == "ok"}
    shared = set(first) & set(second)
    same = sum(first[k]["n_measurements"] == second[k]["n_measurements"] and first[k]["best_model"] == second[k]["best_model"]
               and first[k]["param_err_max_pct"] == second[k]["param_err_max_pct"] for k in shared)
    return same, len(shared)


def results_md(rows: List[Dict[str, Any]], args, mode: str, figure_names: List[str], price) -> str:
    adaptive = [r for r in rows if r["kind"] == "adaptive"]
    methods = _methods(adaptive)
    conditions = args.conditions
    lines: List[str] = []
    add = lines.append
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    command = f"python -m benchmark.run --{mode} --provider {args.provider}"
    if mode == "full" and (args.seeds != 20 or args.llm_seeds != 5):
        command += f" --seeds {args.seeds} --llm-seeds {args.llm_seeds}"
    add("# AutoLab benchmark results\n")
    add(f"*Generated {when} by `{command}`. Every number below comes from `benchmark_runs.csv` in this folder.*\n")
    add("> **Data source: DIGITAL TWIN** for every run.")
    if args.provider == "mock":
        add(">\n> **LLM methods ran with the mock provider**, a deterministic stand-in, not a language model. In "
            "AutoLab-mock the Planner always hands the choice to the GP, so *AutoLab equals GP only by construction*. "
            "LLM-only-mock uses a fixed rule: the widest gap, or where the two best models disagree. These rows show the "
            "pipeline works end to end; **they say nothing about what a real LLM adds**. That needs "
            "`python -m benchmark.run --full --provider litellm`.")
        same, shared = _shared_identical(adaptive, "autolab", "gp_only")
        if shared:
            add(f">\n> Check: on the {shared} adaptive runs where both have the same experiment, condition and seed, "
                f"AutoLab-mock and GP only took identical measurements and found the same law in **{same} of {shared}**. "
                f"**Any difference between them in the tables below comes only from the seed sets** "
                f"(AutoLab uses seeds 0-{args.llm_seeds - 1}, GP only 0-{args.seeds - 1}), not from the method. "
                f"The same applies to comparisons of LLM-only (5 seeds) with the baselines (20 seeds).")
    else:
        add(">\n> LLM methods ran with a real model through litellm; the models that answered are in `benchmark_runs.csv` (`models`).")
    n_err = sum(r["status"] != "ok" for r in rows)
    add(f"\nSeeds: {args.seeds} for grid / random / GP only, {args.llm_seeds} for LLM-only and AutoLab. "
        f"Runs: {len(rows)} ({n_err} ended in an error). Experiments: {', '.join(args.experiments)}. "
        f"Every method starts from the same 4-point design and uses the same stopping rule and safety guard.\n")

    add("## 1. Law identified correctly (blind), all experiments pooled\n")
    add("| Method | " + " | ".join(CONDITION_LABEL.get(c, c) for c in conditions) + " |")
    add("|---" * (len(conditions) + 1) + "|")
    for m in methods:
        cells = []
        for c in conditions:
            ok = _ok(adaptive, method=m, condition=c)
            cells.append(_pct(*wilson(sum(r["correct"] for r in ok), len(ok))) + f" (n={len(ok)})")
        add(f"| {METHOD_STYLE[m]['label']} | " + " | ".join(cells) + " |")
    add("\nWilson 95% intervals in brackets.\n")

    add("## 2. Law identified, per experiment\n")
    for c in conditions:
        add(f"**{CONDITION_LABEL.get(c, c)}**\n")
        add("| Experiment | " + " | ".join(METHOD_STYLE[m]["label"] for m in methods) + " |")
        add("|---" * (len(methods) + 1) + "|")
        for e in args.experiments:
            cells = []
            for m in methods:
                ok = _ok(adaptive, method=m, condition=c, experiment=e)
                cells.append(f"{sum(r['correct'] for r in ok)}/{len(ok)}" if ok else "–")
            add(f"| {e} | " + " | ".join(cells) + " |")
        add("")

    add("## 3. Measurements to stop and parameter error at stop\n")
    add("Parameter error: the largest error over the true law's parameters (normalised like the stopping rule: by "
        "|true value| for a scale, by the x-span for a location, by the response span for an offset), in %, for the "
        "fit of the true law at the end of the run.\n")
    add("| Method | Condition | Measurements, median | mean [95% CI] | Parameter error, median % | Stopped by budget |")
    add("|---|---|---|---|---|---|")
    for m in methods:
        for c in conditions:
            ok = _ok(adaptive, method=m, condition=c)
            if not ok:
                continue
            n = [r["n_measurements"] for r in ok]
            mean, _, lo, hi = mean_ci(n)
            errs = [r["param_err_max_pct"] for r in ok if r["param_err_max_pct"] is not None]
            med = statistics.median(errs) if errs else math.nan
            budget = sum(r["stop_reason"] == "budget_exhausted" for r in ok)
            add(f"| {METHOD_STYLE[m]['label']} | {CONDITION_LABEL.get(c, c)} | {statistics.median(n):g} | "
                f"{_num(mean)} [{_num(lo)}, {_num(hi)}] | {_num(med, '.2f')} | {budget}/{len(ok)} |")
    add("")

    fixed = [r for r in rows if r["kind"] == "fixed"]
    if fixed:
        add("## 4. Equal budget: baselines at AutoLab's median number of measurements\n")
        add("Grid, random and GP-only were run to exactly AutoLab's median n for each experiment and condition. "
            "AutoLab's own adaptive runs are shown for reference.\n")
        add("| Method | " + " | ".join(CONDITION_LABEL.get(c, c) for c in conditions) + " |")
        add("|---" * (len(conditions) + 1) + "|")
        for m in _methods(fixed) + ["autolab"]:
            cells = []
            for c in conditions:
                ok = _ok(fixed if m != "autolab" else adaptive, method=m, condition=c)
                errs = [r["param_err_max_pct"] for r in ok if r["param_err_max_pct"] is not None]
                med = statistics.median(errs) if errs else math.nan
                cells.append(_pct(*wilson(sum(r["correct"] for r in ok), len(ok))) + f"; error {_num(med, '.2f')}%")
            add(f"| {METHOD_STYLE[m]['label']} | " + " | ".join(cells) + " |")
        budgets = sorted({(r["experiment"], r["condition"], r["budget"]) for r in fixed})
        add("\nBudgets used: " + "; ".join(f"{e} {c} {b}" for e, c, b in budgets) + ".\n")

    if "faults" in conditions:
        add("## 5. Fault detection and recovery (faults condition)\n")
        add("| Method | Faults reached | Recovered | Spike recovered | Stuck actuator recovered | Dropout recovered | False alarms per run |")
        add("|---|---|---|---|---|---|---|")
        for m in methods:
            ok = _ok(adaptive, method=m, condition="faults")
            reached = sum(r["faults_reached"] for r in ok)
            per = []
            for t in ("light_spike", "stuck_actuator", "sensor_dropout"):
                outcomes = [r["fault_outcomes"].get(t) for r in ok]
                hit = [o for o in outcomes if o not in (None, "not_reached", "no_effect")]
                per.append(f"{sum(o == 'recovered' for o in hit)}/{len(hit)}")
            add(f"| {METHOD_STYLE[m]['label']} | {reached} | "
                f"{_pct(*wilson(sum(r['faults_recovered'] for r in ok), reached))} | " + " | ".join(per)
                + f" | {_num(mean_ci([r['false_alarms'] for r in ok])[0], '.2f')} |")
        add("")
        if "nonideal" in conditions:
            ok = _ok(adaptive, condition="nonideal")
            out_n, out_d = sum(r["outliers"] for r in ok), sum(r["outliers_detected"] for r in ok)
            add(f"Random outliers (non-idealities on, all methods): {out_d} of {out_n} detected.\n")

    add("## 6. LLM use and cost per run (adaptive runs, all conditions)\n")
    model = args.estimate_model
    if args.provider == "mock":
        add(f"The mock provider costs nothing. *Estimated* cost of a real model: prompt and answer characters / 4 as "
            f"tokens, priced with litellm's bundled price map for `{model}`"
            + (f" (${price[0] * 1e6:.2f} / ${price[1] * 1e6:.2f} per million input / output tokens)." if price else
               " (not in the price map: no estimate).") + f" ₹ at {args.usd_inr:g} INR/USD (assumed rate).\n")
    add("| Method | LLM calls | Tokens in / out | Cost, USD | Cost, INR | Wall time, s |")
    add("|---|---|---|---|---|---|")
    for m in methods:
        ok = _ok(adaptive, method=m)
        calls = mean_ci([r["llm_calls"] for r in ok])[0]
        if args.provider == "mock":
            tin, tout = mean_ci([r["est_input_tokens"] for r in ok])[0], mean_ci([r["est_output_tokens"] for r in ok])[0]
            usd = (tin * price[0] + tout * price[1]) if price and calls else 0.0
            tokens, cost = f"~{tin:.0f} / ~{tout:.0f} (est.)", f"~{usd:.4f} (est.)"
            inr = f"~{usd * args.usd_inr:.3f} (est.)"
        else:
            tin, tout = mean_ci([r["input_tokens"] for r in ok])[0], mean_ci([r["output_tokens"] for r in ok])[0]
            usd = mean_ci([r["cost_usd"] for r in ok])[0]
            tokens, cost, inr = f"{tin:.0f} / {tout:.0f}", f"{usd:.4f}", f"{usd * args.usd_inr:.3f}"
        add(f"| {METHOD_STYLE[m]['label']} | {_num(calls)} | {tokens} | {cost} | {inr} | "
            f"{_num(mean_ci([r['wall_s'] for r in ok])[0], '.2f')} |")
    add("")

    add("## Figures\n")
    descriptions = {
        "fig_param_error_vs_n": "Largest parameter error of the true law's fit after each measurement (geometric mean and 95% CI over seeds, clean twin, runs to 20 points): how fast each method pins down the parameters.",
        "fig_law_accuracy": "Share of runs that identified the right law blind, per method and condition (all experiments pooled, Wilson 95% CI).",
        "fig_equal_budget_accuracy": "The same, with grid / random / GP-only run to exactly AutoLab's median number of measurements, so accuracy is compared at equal cost.",
        "fig_fault_recovery": "What became of each injected fault (recovered, missed, not reached), per method, in the faults condition.",
        "fig_cost_per_run": "LLM calls and cost per run, in USD with INR alongside" + (" (cost estimated for a real model from prompt sizes)." if args.provider == "mock" else "."),
    }
    for name in figure_names:
        add(f"- `figures/{name}.png` (and `.pdf`): {descriptions.get(name, '')}")
    add("\n## Files\n")
    add("- `benchmark_runs.csv`: one row per run (plan, outcome, parameter errors, faults, LLM usage, cost, wall time, run id).")
    add("- `benchmark_summary.csv`: mean, std and 95% CI per run kind × method × experiment × condition, and pooled over experiments (`ALL`).")
    add("- Run records: `runs/benchmark/<mode>/records/` (git-ignored); every row's `run_id` names its record.")
    return "\n".join(lines) + "\n"


def write_all(rows: List[Dict[str, Any]], out: Path, args, mode: str) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["kind"], r["method"], r["experiment"], r["condition"], r["seed"], r["budget"]))
    runs_csv, summary_csv = out / "benchmark_runs.csv", out / "benchmark_summary.csv"
    write_csv(runs_csv, [_flat(r) for r in rows], PLAN)
    model = args.estimate_model
    write_csv(summary_csv, summarise(rows, model, args.usd_inr))
    price = price_per_token(model)
    figures = all_figures(rows, args, price, out / "figures")
    names = sorted({p.stem for p in figures}, key=lambda n: [
        "fig_param_error_vs_n", "fig_law_accuracy", "fig_equal_budget_accuracy", "fig_fault_recovery",
        "fig_cost_per_run"].index(n))
    md = out / "RESULTS.md"
    md.write_text(results_md(rows, args, mode, names, price), encoding="utf-8")
    return [runs_csv, summary_csv, md, *figures]
