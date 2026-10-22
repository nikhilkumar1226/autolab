"""
Benchmark harness.

    python -m benchmark.run --quick                     # 2 seeds, mock LLM, all experiments  -> results/quick/
    python -m benchmark.run --full --provider mock      # 20 seeds (5 for LLM methods), mock  -> results/
    python -m benchmark.run --full --provider litellm   # the same with the real LLM (costs money; see --dry-run)
    python -m benchmark.run --full --dry-run            # print the plan and the estimated LLM calls, run nothing

Methods (all start from the same 4-point initial design and use the same stopping rule):
    grid      coarse-to-fine grid (no LLM)
    random    uniform random points (no LLM)
    gp_only   GP uncertainty sampling (provider none)
    llm_only  the Planner chooses every point, no GP (LLM provider)
    autolab   the full system: Planner + GP + Analyst + Writer (LLM provider)
Conditions: clean; nonideal (every non-ideality on); faults (the default fault schedule).
Run kinds:
    adaptive  the normal stopping rule (budget 30)
    fixed     grid / random / gp_only run to exactly the median n of autolab for that experiment
              and condition, so accuracy can be compared at an equal number of measurements
    curve     every method, clean, run to exactly 20 points: the parameter error after each point

Resume-safe: each finished run is stored in runs/benchmark/<mode>/rows/; a rerun skips them.
Non-LLM runs do not depend on the provider, so switching --provider reruns only the LLM methods.
"""
import argparse
import json
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from config import (
    BASE_DIR, DEFAULT_FAULTS, AgentConfig, LLMConfig, NonIdealityConfig, OutputConfig, SimulationConfig,
    StoppingConfig, SystemConfig,
)
from experiments.registry import list_experiments

METHODS: Dict[str, Dict[str, Any]] = {
    "grid": {"sampler": "grid", "llm": False},
    "random": {"sampler": "random", "llm": False},
    "gp_only": {"sampler": "gp", "llm": False},
    "llm_only": {"sampler": "llm", "llm": True},
    "autolab": {"sampler": "gp", "llm": True},
}
CONDITIONS = ["clean", "nonideal", "faults"]
FIXED_METHODS = ["grid", "random", "gp_only"]
CURVE_BUDGET = 20
DEFAULT_USD_TO_INR = 88.0  # assumed exchange rate (October 2026); override with --usd-inr


def simulation(condition: str) -> SimulationConfig:
    if condition == "nonideal":
        return SimulationConfig(transit_delay_sec=0.0, nonideal=NonIdealityConfig(enabled=True))
    if condition == "faults":
        return SimulationConfig(transit_delay_sec=0.0, faults=list(DEFAULT_FAULTS))
    return SimulationConfig(transit_delay_sec=0.0)


class Task(BaseModel):
    kind: str
    method: str
    provider: str
    experiment: str
    condition: str
    seed: int
    budget: int
    fixed: bool = False

    @property
    def key(self) -> str:
        return f"{self.kind}-{self.method}-{self.provider}-{self.experiment}-{self.condition}-s{self.seed}-b{self.budget}"


def make_config(task: Task, records_dir: Path) -> SystemConfig:
    method = METHODS[task.method]
    llm = LLMConfig() if task.provider == "litellm" else LLMConfig(provider=task.provider, cache=False)
    llm = llm.model_copy(update={"provider": task.provider})
    stopping = StoppingConfig(max_measurements=task.budget)
    if task.fixed:
        stopping = stopping.model_copy(update={"min_points": task.budget})
    return SystemConfig(
        experiment=task.experiment, seed=task.seed, simulation=simulation(task.condition),
        agent=AgentConfig(sampler=method["sampler"]), llm=llm, stopping=stopping,
        output=OutputConfig(runs_dir=records_dir, write_report=False),
    )


def execute(task_json: str, records_dir: str, usd_to_inr: float) -> Dict[str, Any]:
    """Run one task in a worker process and return its row."""
    logging.disable(logging.CRITICAL)
    from agents.orchestrator import DiscoveryEngine
    from benchmark.metrics import run_metrics
    from experiments.registry import get_experiment
    from llm.client import LLMUnavailableError
    from records.run_record import load_run_record, record_path

    task = Task.model_validate_json(task_json)
    started: Dict[str, str] = {}

    def remember(event):
        if event.type == "run_started":
            started["run_id"] = event.run_id

    engine = DiscoveryEngine(make_config(task, Path(records_dir)), on_event=remember)
    try:
        record = engine.run()
    except LLMUnavailableError:
        record = load_run_record(record_path(started["run_id"], Path(records_dir)))
    row = task.model_dump()
    row["key"] = task.key
    row.update(run_metrics(record, get_experiment(task.experiment), usd_to_inr))
    return row


class Store:
    """Finished rows, one JSON file per task, so an interrupted benchmark resumes where it stopped."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, task: Task) -> Path:
        return self.directory / f"{task.key}.json"

    def get(self, task: Task) -> Optional[Dict[str, Any]]:
        p = self.path(task)
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    def put(self, row: Dict[str, Any]) -> None:
        task = Task.model_validate(row)
        self.path(task).write_text(json.dumps(row, default=float), encoding="utf-8")


def plan_stage1(args) -> List[Task]:
    tasks = []
    for method, spec in METHODS.items():
        if method not in args.methods:
            continue
        provider = args.provider if spec["llm"] else "none"
        seeds = range(args.llm_seeds if spec["llm"] else args.seeds)
        for experiment in args.experiments:
            for condition in args.conditions:
                for seed in seeds:
                    tasks.append(Task(kind="adaptive", method=method, provider=provider, experiment=experiment,
                                      condition=condition, seed=seed, budget=30))
            if "clean" in args.conditions and not args.no_curves:
                for seed in seeds:
                    tasks.append(Task(kind="curve", method=method, provider=provider, experiment=experiment,
                                      condition="clean", seed=seed, budget=CURVE_BUDGET, fixed=True))
    return tasks


def plan_stage2(args, rows: List[Dict[str, Any]]) -> List[Task]:
    """Equal-budget runs: each non-LLM method at autolab's median n for that experiment and condition."""
    tasks = []
    for experiment in args.experiments:
        for condition in args.conditions:
            ns = [r["n_measurements"] for r in rows if r["kind"] == "adaptive" and r["method"] == "autolab"
                  and r["experiment"] == experiment and r["condition"] == condition and r["status"] == "ok"]
            if not ns:
                continue
            budget = max(5, int(round(statistics.median(ns))))
            for method in FIXED_METHODS:
                if method not in args.methods:
                    continue
                for seed in range(args.seeds):
                    tasks.append(Task(kind="fixed", method=method, provider="none", experiment=experiment,
                                      condition=condition, seed=seed, budget=budget, fixed=True))
    return tasks


def run_tasks(tasks: List[Task], store: Store, records_dir: Path, workers: int, usd_to_inr: float) -> List[Dict[str, Any]]:
    rows, pending = [], []
    for task in tasks:
        row = store.get(task)
        (rows.append(row) if row is not None else pending.append(task))
    if not pending:
        return rows
    print(f"  {len(pending)} run(s) to do, {len(rows)} already done (resumed)", flush=True)
    t0, done = time.perf_counter(), 0
    if workers <= 1:  # in this process (tests, debugging)
        for task in pending:
            row = execute(task.model_dump_json(), str(records_dir), usd_to_inr)
            store.put(row)
            rows.append(row)
        return rows
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, t.model_dump_json(), str(records_dir), usd_to_inr): t for t in pending}
        for future in as_completed(futures):
            row = future.result()
            store.put(row)
            rows.append(row)
            done += 1
            if done % 50 == 0 or done == len(pending):
                print(f"  {done}/{len(pending)} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    return rows


def estimate_llm_calls(tasks: List[Task]) -> int:
    """Rough plan-time estimate: about 2 calls per measurement after the initial design, plus 3 per run."""
    return sum(3 + 2 * (12 if t.kind == "adaptive" else t.budget - 4) for t in tasks if METHODS[t.method]["llm"])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AutoLab benchmark (see module docstring)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="2 seeds, mock LLM: a few minutes")
    mode.add_argument("--full", action="store_true", help="20 seeds (5 for LLM methods)")
    parser.add_argument("--provider", choices=["mock", "litellm"], default=None,
                        help="LLM provider for llm_only and autolab (default: mock for --quick, LLM_PROVIDER for --full)")
    parser.add_argument("--seeds", type=int, default=None, help="Seeds for non-LLM methods (default 2 quick / 20 full)")
    parser.add_argument("--llm-seeds", type=int, default=None, help="Seeds for LLM methods (default 2 quick / 5 full)")
    parser.add_argument("--experiments", default=",".join(list_experiments()))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--no-curves", action="store_true", help="Skip the 20-point learning-curve runs")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--usd-inr", type=float, default=DEFAULT_USD_TO_INR, help="Exchange rate for the INR columns")
    parser.add_argument("--estimate-model", default="anthropic/claude-haiku-5-5",
                        help="With the mock provider: the real model whose prices estimate the cost of LLM methods")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default results/quick or results)")
    parser.add_argument("--work", type=Path, default=None,
                        help="Folder for finished rows and run records (default runs/benchmark/<quick|full>)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and the estimated LLM calls; run nothing")
    args = parser.parse_args(argv)

    mode_name = "quick" if args.quick else "full"
    args.seeds = args.seeds if args.seeds is not None else (2 if args.quick else 20)
    args.llm_seeds = args.llm_seeds if args.llm_seeds is not None else (2 if args.quick else 5)
    if args.provider is None:
        args.provider = "mock" if args.quick else (os.getenv("LLM_PROVIDER", "mock").strip() or "mock")
        if args.provider not in ("mock", "litellm"):
            args.provider = "mock"
    args.experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    args.conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out = args.out or (BASE_DIR / "results" / ("quick" if args.quick else ""))
    work = args.work or (BASE_DIR / "runs" / "benchmark" / mode_name)
    store, records_dir = Store(work / "rows"), work / "records"

    stage1 = plan_stage1(args)
    llm_tasks = [t for t in stage1 if METHODS[t.method]["llm"]]
    print(f"Benchmark ({mode_name}): {len(stage1)} runs in stage 1 ({len(llm_tasks)} with the LLM provider "
          f"'{args.provider}'), then equal-budget runs for {', '.join(m for m in FIXED_METHODS if m in args.methods)}.")
    if args.provider == "litellm":
        todo = [t for t in llm_tasks if store.get(t) is None]
        print(f"  Real LLM: about {estimate_llm_calls(todo)} calls for {len(todo)} runs not yet done "
              f"(answers already in llm_cache/ cost nothing).")
    if args.dry_run:
        return 0

    t0 = time.perf_counter()
    print("Stage 1: adaptive and learning-curve runs", flush=True)
    rows = run_tasks(stage1, store, records_dir, args.workers, args.usd_inr)
    print("Stage 2: equal-budget runs", flush=True)
    rows += run_tasks(plan_stage2(args, rows), store, records_dir, args.workers, args.usd_inr)

    from benchmark.report import write_all
    paths = write_all(rows, out, args, mode_name)
    print(f"Done in {time.perf_counter() - t0:.0f} s. Wrote:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
