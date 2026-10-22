"""
Seeded sweep of the digital twin's conditions (Phase 3).

Runs every registry experiment for a range of seeds under three conditions, with the
default stopping rule, GP sampling and no LLM (provider none):
    clean     non-idealities off, no faults
    faults    the default fault schedule (config.DEFAULT_FAULTS)
    nonideal  the default fault schedule and every non-ideality on
and writes one CSV row per run with the law found, the stop, the outcome of each injected
fault, the random outliers detected or missed, and the false alarms.

    python -m scripts.twin_sweep                      # seeds 0-19 -> results/phase3_twin_sweep.csv
    python -m scripts.twin_sweep --n-seeds 5 --out /tmp/quick.csv
"""
import argparse
import csv
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from agents.orchestrator import DiscoveryEngine
from config import (
    BASE_DIR, DEFAULT_FAULTS, LLMConfig, NonIdealityConfig, OutputConfig, SimulationConfig, SystemConfig,
)
from experiments.registry import get_experiment, list_experiments

CONDITIONS: Dict[str, SimulationConfig] = {
    "clean": SimulationConfig(transit_delay_sec=0.0),
    "faults": SimulationConfig(transit_delay_sec=0.0, faults=list(DEFAULT_FAULTS)),
    "nonideal": SimulationConfig(transit_delay_sec=0.0, faults=list(DEFAULT_FAULTS),
                                 nonideal=NonIdealityConfig(enabled=True)),
}
FAULT_TYPES = [f.type for f in DEFAULT_FAULTS]
COLUMNS = ["condition", "experiment", "seed", "true_model", "best_model", "correct", "stop_reason", "n_measurements",
           "attempts", *FAULT_TYPES, "outliers", "outliers_detected", "outliers_missed", "false_alarms", "run_id", "rerun"]


def run_one(condition: str, experiment: str, seed: int, runs_dir: Path) -> Dict[str, str]:
    cfg = SystemConfig(
        experiment=experiment, seed=seed, simulation=CONDITIONS[condition],
        llm=LLMConfig(provider="none"), output=OutputConfig(runs_dir=runs_dir, write_report=False),
    )
    record = DiscoveryEngine(cfg).run()
    best = record.result.final_analysis.best if record.result and record.result.final_analysis else None
    true_model = get_experiment(experiment).ground_truth_model
    statuses = {f.type: f.status for f in record.faults if f.scheduled}
    s = record.fault_summary
    flags = {"clean": "", "faults": " --faults default", "nonideal": " --faults default --nonideal"}[condition]
    return {
        "condition": condition, "experiment": experiment, "seed": str(seed), "true_model": true_model,
        "best_model": best.key if best else "", "correct": str(int(best is not None and best.key == true_model)),
        "stop_reason": record.stop.reason.value if record.stop and record.stop.reason else "",
        "n_measurements": str(record.result.n_measurements if record.result else 0),
        "attempts": str(len(record.steps)),
        **{t: statuses.get(t, "") for t in FAULT_TYPES},
        "outliers": str(s.outliers if s else 0), "outliers_detected": str(s.outliers_detected if s else 0),
        "outliers_missed": str(s.outliers_missed if s else 0),
        "false_alarms": str(sum(s.unexplained.values()) if s else 0),
        "run_id": record.run_id,
        "rerun": f"python main.py --experiment {experiment} --seed {seed} --provider none --no-report{flags}",
    }


def summarise(rows: List[Dict[str, str]], experiments: List[str]) -> None:
    for condition in CONDITIONS:
        print(f"\n### {condition}\n")
        print("| Experiment | Correct law | Measurements (min / median / max) | Stop reasons | "
              + " | ".join(FAULT_TYPES) + " | Random outliers detected | False alarms |")
        print("|---" * (6 + len(FAULT_TYPES)) + "|")
        for experiment in experiments:
            sub = [r for r in rows if r["condition"] == condition and r["experiment"] == experiment]
            if not sub:
                continue
            n = [int(r["n_measurements"]) for r in sub]
            stops = Counter(r["stop_reason"] for r in sub)
            faults = []
            for t in FAULT_TYPES:
                c = Counter(r[t] for r in sub if r[t])
                faults.append(", ".join(f"{k} {v}" for k, v in sorted(c.items())) or "-")
            out, det = sum(int(r["outliers"]) for r in sub), sum(int(r["outliers_detected"]) for r in sub)
            print(f"| {experiment} | {sum(int(r['correct']) for r in sub)}/{len(sub)} | "
                  f"{min(n)} / {statistics.median(n):g} / {max(n)} | "
                  f"{', '.join(f'{v} {k}' for k, v in stops.most_common())} | " + " | ".join(faults)
                  + f" | {det}/{out} | {sum(int(r['false_alarms']) for r in sub)} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seeded sweep of the digital twin's conditions")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--out", type=Path, default=BASE_DIR / "results" / "phase3_twin_sweep.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    experiments = list_experiments()
    rows: List[Dict[str, str]] = []
    for condition in conditions:
        runs_dir = BASE_DIR / "runs" / args.out.stem / condition
        for experiment in experiments:
            for seed in range(args.first_seed, args.first_seed + args.n_seeds):
                rows.append(run_one(condition, experiment, seed, runs_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.out}")
    summarise(rows, experiments)


if __name__ == "__main__":
    main()
