"""
Seeded sweep of stopping-rule configurations (Phase 1 fix).

Runs every registry experiment for a range of seeds under each stopping configuration
(digital twin, GP sampler) and writes one CSV row per run. Each row records the CLI
command that reproduces it.

    python -m scripts.stopping_sweep                 # A, B, C on seeds 0-19 -> results/phase1_stopping_sweep.csv
    python -m scripts.stopping_sweep --configs C --first-seed 20 --out results/phase1_stopping_heldout.csv

Configurations (every field explicit, so they do not depend on the current defaults):
    A  Phase 1 rule: delta-BIC >= 6, relative SE < 5%, min_points = 5
    B  A with min_points = 8
    C  A plus predictive stability: same best model in 2 consecutive analyses and the
       newest point within 3 sigma of the earlier fit's prediction
"""
import argparse
import csv
import logging
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from agents.orchestrator import DiscoveryEngine
from config import BASE_DIR, OutputConfig, SimulationConfig, StoppingConfig, SystemConfig
from experiments.registry import get_experiment, list_experiments

CONFIGS: Dict[str, StoppingConfig] = {
    "A": StoppingConfig(min_points=5, predictive_stability=False),
    "B": StoppingConfig(min_points=8, predictive_stability=False),
    "C": StoppingConfig(min_points=5, predictive_stability=True),
}
DESCRIPTIONS = {
    "A": "Phase 1 rule (min_points = 5)",
    "B": "min_points = 8",
    "C": "predictive stability (min_points = 5)",
}
COLUMNS = [
    "config", "experiment", "seed", "true_model", "best_model", "correct", "stop_reason", "n_measurements",
    "delta_bic", "max_rel_se", "stable_best", "newest_point_z", "newest_point_x", "newest_point_observed",
    "newest_point_predicted", "runner_up", "runner_up_predicted", "prediction_sigma", "discrimination_sigma",
    "run_id", "rerun",
]


def _num(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.6g}"


def run_one(name: str, cfg: StoppingConfig, experiment: str, seed: int, runs_dir: Path) -> Dict[str, str]:
    sys_config = SystemConfig(
        experiment=experiment,
        seed=seed,
        simulation=SimulationConfig(transit_delay_sec=0.0),
        stopping=cfg,
        output=OutputConfig(runs_dir=runs_dir, write_report=False),
    )
    record = DiscoveryEngine(sys_config).run()
    stop = record.stop
    best = record.result.final_analysis.best if record.result and record.result.final_analysis else None
    true_model = get_experiment(experiment).ground_truth_model

    # The newest measurement as predicted by the fit before it (recorded for every config).
    measured = [s for s in record.steps if s.status == "measured"]
    p = measured[-1].prediction if measured else None
    discrimination = None
    if p is not None and p.predicted is not None and p.runner_up_predicted is not None and p.sigma > 0:
        discrimination = abs(p.predicted - p.runner_up_predicted) / p.sigma

    stability = "--stability" if cfg.predictive_stability else "--no-stability"
    return {
        "config": name,
        "experiment": experiment,
        "seed": str(seed),
        "true_model": true_model,
        "best_model": best.key if best else "",
        "correct": str(best is not None and best.key == true_model),
        "stop_reason": stop.reason.value if stop and stop.reason else "",
        "n_measurements": str(record.result.n_measurements if record.result else len(measured)),
        "delta_bic": _num(stop.delta_bic.value if stop else None),
        "max_rel_se": _num(stop.max_rel_se.value if stop else None),
        "stable_best": _num(stop.stable_best.value if stop and stop.stable_best else None),
        "newest_point_z": _num(p.z if p else None),
        "newest_point_x": _num(p.x if p else None),
        "newest_point_observed": _num(p.observed if p else None),
        "newest_point_predicted": _num(p.predicted if p else None),
        "runner_up": (p.runner_up or "") if p else "",
        "runner_up_predicted": _num(p.runner_up_predicted if p else None),
        "prediction_sigma": _num(p.sigma if p else None),
        "discrimination_sigma": _num(discrimination),
        "run_id": record.run_id,
        "rerun": (
            f"python main.py --experiment {experiment} --seed {seed} --no-report "
            f"--min-points {cfg.min_points} {stability}"
        ),
    }


def summarise(rows: List[Dict[str, str]], name: str, experiments: List[str]) -> Dict[str, dict]:
    print(f"\n### Config {name}: {DESCRIPTIONS[name]}\n")
    print("| Experiment | Correct law | Measurements at stop (min / median / max) | Stop reasons |")
    print("|---|---|---|---|")
    summary = {}
    for experiment in experiments:
        sub = [r for r in rows if r["config"] == name and r["experiment"] == experiment]
        n = [int(r["n_measurements"]) for r in sub]
        correct = sum(r["correct"] == "True" for r in sub)
        reasons = Counter(r["stop_reason"] for r in sub)
        summary[experiment] = {"correct": correct, "runs": len(sub), "n": n}
        print(
            f"| {experiment} | {correct}/{len(sub)} | {min(n)} / {statistics.median(n):g} / {max(n)} | "
            + ", ".join(f"{count} {reason}" for reason, count in reasons.most_common()) + " |"
        )
    wrong = [r for r in rows if r["config"] == name and r["correct"] != "True"]
    for r in wrong:
        print(
            f"- wrong law: {r['experiment']} seed {r['seed']}: {r['best_model']} at n = {r['n_measurements']} "
            f"({r['stop_reason']}); newest point x = {r['newest_point_x']}, best vs runner-up "
            f"({r['runner_up']}) predictions {r['discrimination_sigma'] or 'n/a'} sigma apart"
        )
    return summary


def choose(summaries: Dict[str, Dict[str, dict]]) -> Optional[str]:
    """Configs with >= 95% correct (19/20) on every experiment; fewest pooled median measurements wins."""
    qualified = []
    for name, per_exp in summaries.items():
        if all(s["correct"] >= math.ceil(0.95 * s["runs"]) for s in per_exp.values()):
            pooled = [n for s in per_exp.values() for n in s["n"]]
            qualified.append((statistics.median(pooled), statistics.mean(pooled), name))
    print("\n### Selection (>= 19/20 correct on every experiment, then fewest median measurements)\n")
    for name, per_exp in summaries.items():
        pooled = [n for s in per_exp.values() for n in s["n"]]
        ok = any(q[2] == name for q in qualified)
        print(
            f"- {name}: {'qualifies' if ok else 'does not qualify'}; pooled median "
            f"{statistics.median(pooled):g}, mean {statistics.mean(pooled):.2f} measurements"
        )
    if not qualified:
        print("\nNo configuration qualifies; keep the current default.")
        return None
    chosen = min(qualified)[2]
    print(f"\nChosen: {chosen} ({DESCRIPTIONS[chosen]})")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Seeded sweep of stopping-rule configurations")
    parser.add_argument("--configs", default="A,B,C", help="Comma-separated subset of A,B,C (default: all)")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--out", type=Path, default=BASE_DIR / "results" / "phase1_stopping_sweep.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = set(names) - set(CONFIGS)
    if unknown:
        parser.error(f"unknown config(s): {', '.join(sorted(unknown))}")
    experiments = list_experiments()
    seeds = range(args.first_seed, args.first_seed + args.n_seeds)

    rows: List[Dict[str, str]] = []
    for name in names:
        runs_dir = BASE_DIR / "runs" / args.out.stem / name
        for experiment in experiments:
            for seed in seeds:
                rows.append(run_one(name, CONFIGS[name], experiment, seed, runs_dir))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs (seeds {seeds.start}-{seeds.stop - 1}) to {args.out}")

    summaries = {name: summarise(rows, name, experiments) for name in names}
    if set(names) == set(CONFIGS):
        choose(summaries)


if __name__ == "__main__":
    main()
