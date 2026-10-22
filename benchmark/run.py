"""
Benchmark harness.

    python -m benchmark.run --quick                     # 2 seeds, mock LLM, all experiments  -> results/quick/
    python -m benchmark.run --full --provider mock      # 20 seeds (5 for LLM methods), mock  -> results/
    python -m benchmark.run --subset                    # reduced subset with a real free-tier LLM -> results/subset/
    python -m benchmark.run --subset --dry-run          # the plan, and the estimated calls and tokens; runs nothing
    python -m benchmark.run --full --provider litellm   # the full benchmark with a real LLM (paid credit)

Methods (all start from the same 4-point initial design and use the same stopping rule and safety guard):
    grid      coarse-to-fine grid (no LLM)
    random    uniform random points (no LLM)
    gp_only   GP uncertainty sampling (provider none)
    llm_only  the Planner chooses every point, no GP (LLM provider)
    autolab   the full system: Planner + GP + Analyst + Writer (LLM provider)
Conditions: clean; nonideal (every non-ideality on); faults (the default fault schedule);
hidden_apparatus (clean physics, but the agents get only the variable names and units, no
description of the apparatus).
Run kinds:
    adaptive  the normal stopping rule (budget 30)
    fixed     grid / random / gp_only run to exactly the median n of autolab for that experiment
              and condition, so accuracy can be compared at an equal number of measurements
    curve     every method, clean, run to exactly 20 points: the parameter error after each point

--subset (free tier): malus_law, inverse_square_law, rc_charging and diode_iv; clean, faults and
hidden_apparatus; llm_only and autolab with the real provider (LLM_MODEL, then LLM_FALLBACKS from
.env), 3 seeds. The non-LLM baselines are the full benchmark's rows (reused; computed if missing,
which costs nothing). Non-LLM methods never read the apparatus description, so their clean rows
stand for hidden_apparatus. Real-LLM runs go one at a time, paced to --rpm requests per minute.

Resume-safe: each finished run is stored in runs/benchmark/<quick|full>/rows/ (the subset shares
the full benchmark's store). A run that ends because every model in the chain is unavailable
(e.g. a daily limit) is not stored, and after 2 such runs in a row the benchmark stops; run the
same command again later to continue. Answers already in llm_cache/ replay for free, so an
interrupted run picks up where it stopped. Rows of a real LLM carry a fingerprint of the model
chain, so runs with different models are never mixed.
"""
import argparse
import hashlib
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
ALL_CONDITIONS = [*CONDITIONS, "hidden_apparatus"]
FIXED_METHODS = ["grid", "random", "gp_only"]
BASELINE_METHODS = FIXED_METHODS
LLM_METHODS = ["llm_only", "autolab"]
SUBSET_EXPERIMENTS = ["malus_law", "inverse_square_law", "rc_charging", "diode_iv"]
SUBSET_CONDITIONS = ["clean", "faults", "hidden_apparatus"]
SUBSET_SEEDS = 3
SUBSET_RPM = 10.0
STOP_AFTER_UNAVAILABLE = 2
CURVE_BUDGET = 20
DEFAULT_USD_TO_INR = 88.0  # assumed exchange rate (October 2026); override with --usd-inr


def simulation(condition: str) -> SimulationConfig:
    if condition == "nonideal":
        return SimulationConfig(transit_delay_sec=0.0, nonideal=NonIdealityConfig(enabled=True))
    if condition == "faults":
        return SimulationConfig(transit_delay_sec=0.0, faults=list(DEFAULT_FAULTS))
    return SimulationConfig(transit_delay_sec=0.0)  # clean and hidden_apparatus


def chain_fingerprint(chain: List[str]) -> str:
    return hashlib.sha1(",".join(chain).encode()).hexdigest()[:8]


class Task(BaseModel):
    kind: str
    method: str
    provider: str
    experiment: str
    condition: str
    seed: int
    budget: int
    fixed: bool = False
    chain: str = ""  # real LLM only: fingerprint of the model chain, so different models never share rows

    @property
    def key(self) -> str:
        key = f"{self.kind}-{self.method}-{self.provider}-{self.experiment}-{self.condition}-s{self.seed}-b{self.budget}"
        return key + (f"-c{self.chain}" if self.chain else "")


def make_config(task: Task, records_dir: Path, rpm: float = 0.0) -> SystemConfig:
    method = METHODS[task.method]
    llm = LLMConfig() if task.provider == "litellm" else LLMConfig(provider=task.provider, cache=False)
    llm = llm.model_copy(update={"provider": task.provider,
                                 "min_request_interval_s": 60.0 / rpm if rpm and task.provider == "litellm" else 0.0})
    stopping = StoppingConfig(max_measurements=task.budget)
    if task.fixed:
        stopping = stopping.model_copy(update={"min_points": task.budget})
    return SystemConfig(
        experiment=task.experiment, seed=task.seed, simulation=simulation(task.condition),
        agent=AgentConfig(sampler=method["sampler"], describe_apparatus=task.condition != "hidden_apparatus"),
        llm=llm, stopping=stopping, output=OutputConfig(runs_dir=records_dir, write_report=False),
    )


def execute(task_json: str, records_dir: str, usd_to_inr: float, rpm: float = 0.0) -> Dict[str, Any]:
    """Run one task (in a worker process, or in this one for paced real-LLM runs) and return its row."""
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

    engine = DiscoveryEngine(make_config(task, Path(records_dir), rpm), on_event=remember)
    unavailable = False
    try:
        record = engine.run()
    except LLMUnavailableError:
        unavailable = True
        record = load_run_record(record_path(started["run_id"], Path(records_dir)))
    row = task.model_dump()
    row["key"] = task.key
    row.update(run_metrics(record, get_experiment(task.experiment), usd_to_inr))
    row["llm_unavailable"] = unavailable
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

    def put(self, row: Dict[str, Any]) -> bool:
        """Store a finished row. A run cut short because no model was available is not stored: it is retried."""
        if row.get("llm_unavailable"):
            return False
        task = Task.model_validate(row)
        self.path(task).write_text(json.dumps(row, default=float), encoding="utf-8")
        return True


class ChainUnavailable(RuntimeError):
    """Several runs in a row found every model unavailable (usually a daily limit): stop, resume later."""


def plan_stage1(args) -> List[Task]:
    tasks = []
    for method, spec in METHODS.items():
        if method not in args.methods:
            continue
        provider = args.provider if spec["llm"] else "none"
        chain = args.chain if spec["llm"] and provider == "litellm" else ""
        seeds = range(args.llm_seeds if spec["llm"] else args.seeds)
        conditions = args.conditions if spec["llm"] else [c for c in args.conditions if c != "hidden_apparatus"]
        for experiment in args.experiments:
            for condition in conditions:
                for seed in seeds:
                    tasks.append(Task(kind="adaptive", method=method, provider=provider, experiment=experiment,
                                      condition=condition, seed=seed, budget=30, chain=chain))
            if "clean" in args.conditions and not args.no_curves:
                for seed in seeds:
                    tasks.append(Task(kind="curve", method=method, provider=provider, experiment=experiment,
                                      condition="clean", seed=seed, budget=CURVE_BUDGET, fixed=True, chain=chain))
    return tasks


def plan_stage2(args, rows: List[Dict[str, Any]]) -> List[Task]:
    """Equal-budget runs: each non-LLM method at autolab's median n for that experiment and condition."""
    if args.subset:
        return []
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


def run_tasks(tasks: List[Task], store: Store, records_dir: Path, workers: int, usd_to_inr: float,
              rpm: float = 0.0) -> List[Dict[str, Any]]:
    """
    Stored rows are reused. Runs with a paced real LLM go one at a time in this process (the pacing
    is process-wide); everything else runs in parallel workers (or here with --workers 1).
    """
    rows, pending = [], []
    for task in tasks:
        row = store.get(task)
        (rows.append(row) if row is not None else pending.append(task))
    if not pending:
        return rows
    print(f"  {len(pending)} run(s) to do, {len(rows)} already done (resumed)", flush=True)
    paced = [t for t in pending if t.provider == "litellm" and rpm > 0]
    parallel = [t for t in pending if t not in paced]
    t0, done = time.perf_counter(), 0
    if parallel and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute, t.model_dump_json(), str(records_dir), usd_to_inr): t for t in parallel}
            for future in as_completed(futures):
                row = future.result()
                if store.put(row):
                    rows.append(row)
                done += 1
                if done % 50 == 0 or done == len(parallel):
                    print(f"  {done}/{len(parallel)} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    else:
        paced = parallel + paced
    unavailable_in_a_row = 0
    for i, task in enumerate(paced, 1):
        row = execute(task.model_dump_json(), str(records_dir), usd_to_inr, rpm)
        if store.put(row):
            rows.append(row)
            unavailable_in_a_row = 0
        else:
            unavailable_in_a_row += 1
        if task.provider == "litellm":
            models = ", ".join(f"{m} x{k}" for m, k in row["models"].items()) or "no model answered"
            state = "STOPPED: no model available" if row.get("llm_unavailable") else f"{row['n_measurements']} measurements"
            print(f"  [{i}/{len(paced)}] {task.method} {task.experiment} {task.condition} seed {task.seed}: "
                  f"{state}; {row['llm_calls']} LLM calls ({models}); {time.perf_counter() - t0:.0f} s", flush=True)
        if unavailable_in_a_row >= STOP_AFTER_UNAVAILABLE:
            left = len(paced) - i + unavailable_in_a_row
            raise ChainUnavailable(
                f"{unavailable_in_a_row} runs in a row found no model available (a daily or per-minute limit, most "
                f"likely). {len(rows)} runs are done and stored; {left} remain. Run the same command again later to "
                f"continue (cached answers replay for free). Last error: {row.get('error', '')[:300]}"
            )
    return rows


def estimate(tasks: List[Task], store: Store, rpm: float) -> Dict[str, Any]:
    """
    Calls and tokens the real-LLM tasks will need, from the mock rows of the same method,
    experiment and condition (hidden_apparatus uses clean) where those exist, else a rule of thumb
    (2 calls per measurement after the initial design, plus 3 per run; 700 tokens in, 60 out per call).
    """
    total = {"runs": 0, "calls": 0.0, "tokens_in": 0.0, "tokens_out": 0.0, "from_mock_rows": 0}
    for t in tasks:
        if not METHODS[t.method]["llm"] or store.get(t) is not None:
            continue
        twins = []
        for seed in range(20):
            mock = t.model_copy(update={"provider": "mock", "chain": "", "seed": seed,
                                        "condition": "clean" if t.condition == "hidden_apparatus" else t.condition})
            row = store.get(mock)
            if row is not None:
                twins.append(row)
        if twins:
            calls = statistics.mean(r["llm_calls"] for r in twins)
            tin = statistics.mean(r["est_input_tokens"] for r in twins)
            tout = statistics.mean(r["est_output_tokens"] for r in twins)
            total["from_mock_rows"] += 1
        else:
            calls = 3 + 2 * (12 if t.kind == "adaptive" else t.budget - 4)
            tin, tout = 700 * calls, 60 * calls
        total["runs"] += 1
        total["calls"] += calls
        total["tokens_in"] += tin
        total["tokens_out"] += tout
    total["minutes_at_pace"] = total["calls"] / rpm if rpm else None
    return total


def models_used(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    used: Dict[str, int] = {}
    for r in rows:
        if r.get("provider") == "litellm":
            for model, n in (r.get("models") or {}).items():
                used[model] = used.get(model, 0) + n
    return dict(sorted(used.items(), key=lambda kv: -kv[1]))


def subset_label(rows: List[Dict[str, Any]], provider: str) -> str:
    if provider != "litellm":
        return "Reduced subset, mock provider (a rehearsal: not a real LLM)"
    used = models_used(rows)
    names = ", ".join(f"{m} ({n} calls)" for m, n in used.items()) or "none yet"
    return f"Reduced subset, real LLM: {names}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AutoLab benchmark (see module docstring)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="2 seeds, mock LLM: a few minutes")
    mode.add_argument("--full", action="store_true", help="20 seeds (5 for LLM methods)")
    mode.add_argument("--subset", action="store_true",
                      help="Reduced subset with a real free-tier LLM: 4 experiments x 3 conditions x 3 seeds")
    parser.add_argument("--provider", choices=["mock", "litellm"], default=None,
                        help="LLM provider for llm_only and autolab (default: mock for --quick, litellm for --subset, "
                             "LLM_PROVIDER for --full)")
    parser.add_argument("--seeds", type=int, default=None, help="Seeds for non-LLM methods (default 2 quick / 20 full)")
    parser.add_argument("--llm-seeds", type=int, default=None, help="Seeds for LLM methods (default 2 quick / 5 full / 3 subset)")
    parser.add_argument("--experiments", default=None)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--no-curves", action="store_true", help="Skip the 20-point learning-curve runs")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--rpm", type=float, default=None,
                        help="Real LLM: at most this many requests per minute, one run at a time "
                             f"(default {SUBSET_RPM:g} for --subset, unpaced otherwise)")
    parser.add_argument("--usd-inr", type=float, default=DEFAULT_USD_TO_INR, help="Exchange rate for the INR columns")
    parser.add_argument("--estimate-model", default="anthropic/claude-haiku-5-5",
                        help="With the mock provider: the real model whose prices estimate the cost of LLM methods")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default results/quick, results/subset or results)")
    parser.add_argument("--work", type=Path, default=None,
                        help="Folder for finished rows and run records (default runs/benchmark/quick, or "
                             "runs/benchmark/full for --full and --subset)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and the estimated LLM calls; run nothing")
    args = parser.parse_args(argv)

    mode_name = "quick" if args.quick else "subset" if args.subset else "full"
    if args.subset:
        args.seeds = args.seeds if args.seeds is not None else 20
        args.llm_seeds = args.llm_seeds if args.llm_seeds is not None else SUBSET_SEEDS
        args.provider = args.provider or "litellm"
        args.no_curves = True
        args.rpm = SUBSET_RPM if args.rpm is None else args.rpm
        default_experiments, default_conditions = SUBSET_EXPERIMENTS, SUBSET_CONDITIONS
        default_methods = [*BASELINE_METHODS, *LLM_METHODS]
    else:
        args.seeds = args.seeds if args.seeds is not None else (2 if args.quick else 20)
        args.llm_seeds = args.llm_seeds if args.llm_seeds is not None else (2 if args.quick else 5)
        if args.provider is None:
            args.provider = "mock" if args.quick else (os.getenv("LLM_PROVIDER", "mock").strip() or "mock")
            if args.provider not in ("mock", "litellm"):
                args.provider = "mock"
        args.rpm = args.rpm or 0.0
        default_experiments, default_conditions, default_methods = list_experiments(), CONDITIONS, list(METHODS)
    split = lambda text, default: [x.strip() for x in text.split(",") if x.strip()] if text else list(default)
    args.experiments = split(args.experiments, default_experiments)
    args.conditions = split(args.conditions, default_conditions)
    args.methods = split(args.methods, default_methods)
    args.chain = chain_fingerprint(LLMConfig().chain) if args.provider == "litellm" else ""
    out = args.out or (BASE_DIR / "results" / ("quick" if args.quick else "subset" if args.subset else ""))
    work = args.work or (BASE_DIR / "runs" / "benchmark" / ("quick" if args.quick else "full"))
    store, records_dir = Store(work / "rows"), work / "records"

    stage1 = plan_stage1(args)
    llm_tasks = [t for t in stage1 if METHODS[t.method]["llm"]]
    if args.subset:
        print(f"Benchmark (reduced subset): {len(llm_tasks)} runs with the LLM provider '{args.provider}' "
              f"({', '.join(args.experiments)}; {', '.join(args.conditions)}; {args.llm_seeds} seeds), "
              f"plus {len(stage1) - len(llm_tasks)} non-LLM baseline runs (reused if already done; free).")
    else:
        print(f"Benchmark ({mode_name}): {len(stage1)} runs in stage 1 ({len(llm_tasks)} with the LLM provider "
              f"'{args.provider}'), then equal-budget runs for {', '.join(m for m in FIXED_METHODS if m in args.methods)}.")
    if args.provider == "litellm":
        chain = LLMConfig().chain
        e = estimate(llm_tasks, store, args.rpm)
        done = len(llm_tasks) - e["runs"]
        print(f"  Model chain (LLM_MODEL, then LLM_FALLBACKS): {' -> '.join(chain)}")
        print(f"  Real LLM: {e['runs']} runs to do ({done} already done). Estimated {e['calls']:.0f} calls, "
              f"~{e['tokens_in'] / 1000:.0f}k input and ~{e['tokens_out'] / 1000:.0f}k output tokens "
              f"({e['from_mock_rows']} runs estimated from the mock benchmark's rows, the rest by rule of thumb; "
              f"output tokens exclude hidden reasoning tokens, which reasoning models such as gpt-oss add).")
        if e["minutes_at_pace"] is not None:
            print(f"  At {args.rpm:g} requests per minute: about {e['minutes_at_pace'] / 60:.1f} h of requests. "
                  f"Check today's free-tier limits for each model; if a daily limit is reached the benchmark stops "
                  f"and the same command resumes it. Cached answers cost nothing.")
    if args.dry_run:
        return 0

    t0 = time.perf_counter()
    try:
        print("Stage 1: adaptive runs" + ("" if args.no_curves else " and learning curves"), flush=True)
        rows = run_tasks(stage1, store, records_dir, args.workers, args.usd_inr, args.rpm)
        stage2 = plan_stage2(args, rows)
        if stage2:
            print("Stage 2: equal-budget runs", flush=True)
            rows += run_tasks(stage2, store, records_dir, args.workers, args.usd_inr, args.rpm)
    except ChainUnavailable as exc:
        print(f"Stopped: {exc}")
        return 3
    except KeyboardInterrupt:
        print("\nInterrupted. Finished runs are stored; run the same command again to continue.")
        return 130

    if args.subset:
        args.label = subset_label(rows, args.provider)
    from benchmark.report import write_all
    paths = write_all(rows, out, args, mode_name)
    print(f"Done in {time.perf_counter() - t0:.0f} s. Wrote:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
