"""
Main entrypoint for the Autonomous Scientific Discovery System.
Run autonomous scientific discovery loops on digital twin simulation or real physical hardware.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from rich.markup import escape

from config import (
    config, DEFAULT_FAULTS, AgentConfig, FaultSpec, HardwareConfig, LLMConfig, NonIdealityConfig, OutputConfig,
    SimulationConfig, StoppingConfig, SystemConfig,
)
from experiments.registry import list_experiments
from agents.orchestrator import DiscoveryEngine
from cli.console import ConsoleEventPrinter
from llm.client import LLMUnavailableError
from records.run_record import list_run_records

def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Autonomous Scientific Discovery System - LLM Agents for Hardware-Grounded Experimental Design"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="malus_law",
        choices=list_experiments(),
        help="Experiment from the registry (default: malus_law)"
    )
    parser.add_argument(
        "--question", "--prompt",
        dest="question",
        type=str,
        default=None,
        help="Research question for the planner (default: a neutral question built from the variable names)"
    )
    parser.add_argument(
        "--provider",
        choices=["litellm", "mock", "none"],
        default=None,
        help="LLM provider: litellm (real model), mock (offline, no key) or none (no LLM; baseline). "
             f"Default: LLM_PROVIDER from .env (now: {LLMConfig().provider})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="First model in the litellm chain, e.g. anthropic/claude-haiku-5-5 (default: LLM_MODEL from .env); "
             "LLM_FALLBACKS still follow it"
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Hide the apparatus description and the variable names from every LLM agent (units and bounds stay)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not serve LLM answers from llm_cache/ (and do not add to it)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for every random draw in the run (default: drawn at random, then recorded)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=StoppingConfig().max_measurements,
        help=f"Maximum number of measurements (default: {StoppingConfig().max_measurements})"
    )
    parser.add_argument(
        "--rse-threshold",
        type=float,
        default=StoppingConfig().max_rel_se,
        help=f"Max relative standard error per parameter to stop (default: {StoppingConfig().max_rel_se})"
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=StoppingConfig().min_points,
        help=f"Never stop on evidence with fewer measurements than this (default: {StoppingConfig().min_points})"
    )
    parser.add_argument(
        "--stability",
        action=argparse.BooleanOptionalAction,
        default=StoppingConfig().predictive_stability,
        help="Require the same best model in consecutive analyses and a correctly predicted newest point "
             f"(default: {'on' if StoppingConfig().predictive_stability else 'off'})"
    )
    parser.add_argument(
        "--sampler",
        choices=["gp", "random", "grid", "llm"],
        default="gp",
        help="How to choose points after the initial design: gp (GP uncertainty sampling; the Planner may choose its "
             "own point), random, grid (baselines), llm (the Planner chooses every point, no GP). Default: gp"
    )
    parser.add_argument(
        "--nonideal",
        action="store_true",
        help="Digital twin: turn on the non-idealities (drift, saturation, quantisation, position error, backlash, "
             "outliers, polariser extinction ratio)"
    )
    parser.add_argument(
        "--faults",
        type=str,
        default="none",
        help="Digital twin: injected faults. 'none' (default), 'default' (a light spike, a stuck actuator and a "
             "2-cycle sensor dropout), a JSON list such as '[{\"type\": \"light_spike\", \"at_step\": 6}]', "
             "or a path to a JSON file with such a list"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip the figure and Markdown/LaTeX report (the JSON run record is always written)"
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="Run on physical hardware via PySerial instead of Digital Twin simulation"
    )
    parser.add_argument(
        "--port",
        type=str,
        default=config.hardware.port,
        help=f"Serial port for physical microcontroller (default: {config.hardware.port})"
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=config.hardware.baud_rate,
        help=f"Baud rate for serial connection (default: {config.hardware.baud_rate})"
    )
    return parser.parse_args(argv)

def parse_faults(text: str) -> List[FaultSpec]:
    """--faults: none, default, a JSON list, or a JSON file."""
    text = text.strip()
    if text.lower() in ("", "none"):
        return []
    if text.lower() == "default":
        return [f.model_copy() for f in DEFAULT_FAULTS]
    path = Path(text)
    raw = path.read_text(encoding="utf-8") if not text.startswith("[") and path.is_file() else text
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--faults: not 'none', 'default', a JSON list or a JSON file ({exc.msg})")
    if not isinstance(items, list):
        raise SystemExit("--faults: expected a JSON list of faults")
    return [FaultSpec.model_validate(item) for item in items]

def build_config(args) -> SystemConfig:
    llm = LLMConfig()
    updates = {"provider": args.provider, "model": args.model}
    llm = llm.model_copy(update={k: v for k, v in updates.items() if v is not None})
    if args.no_cache:
        llm = llm.model_copy(update={"cache": False})
    return SystemConfig(
        experiment=args.experiment,
        seed=args.seed,
        hardware=HardwareConfig(
            simulate=not args.hardware,
            port=args.port,
            baud_rate=args.baud,
        ),
        simulation=SimulationConfig(
            nonideal=NonIdealityConfig(enabled=args.nonideal),
            faults=parse_faults(args.faults),
        ),
        agent=AgentConfig(sampler=args.sampler, blind=args.blind),
        llm=LLMConfig.model_validate(llm.model_dump()),
        stopping=StoppingConfig(
            max_measurements=args.budget,
            max_rel_se=args.rse_threshold,
            min_points=args.min_points,
            predictive_stability=args.stability,
        ),
        output=OutputConfig(write_report=not args.no_report),
    )

def rerun_command(args, sys_config: SystemConfig, seed: int) -> str:
    rerun = f"python main.py --experiment {args.experiment} --seed {seed} --provider {sys_config.llm.provider}"
    if args.model:
        rerun += f" --model {args.model}"
    if args.blind:
        rerun += " --blind"
    if args.budget != StoppingConfig().max_measurements:
        rerun += f" --budget {args.budget}"
    if args.rse_threshold != StoppingConfig().max_rel_se:
        rerun += f" --rse-threshold {args.rse_threshold}"
    if args.min_points != StoppingConfig().min_points:
        rerun += f" --min-points {args.min_points}"
    if args.stability != StoppingConfig().predictive_stability:
        rerun += " --stability" if args.stability else " --no-stability"
    if args.sampler != "gp":
        rerun += f" --sampler {args.sampler}"
    if args.nonideal:
        rerun += " --nonideal"
    if args.faults.strip().lower() not in ("", "none"):
        faults = args.faults if args.faults.strip().lower() == "default" else json.dumps(
            [f.model_dump(exclude_defaults=True) for f in sys_config.simulation.faults], separators=(",", ":"))
        rerun += f" --faults '{faults}'" if faults.startswith("[") else f" --faults {faults}"
    return rerun

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    sys_config = build_config(args)
    printer = ConsoleEventPrinter()
    if args.blind and args.question:
        printer.console.print("[yellow]Blind mode: your --question is passed to the agents unchanged and may reveal the apparatus.[/yellow]")
    engine = DiscoveryEngine(sys_config, on_event=printer)
    try:
        record = engine.run(question=args.question)
    except LLMUnavailableError as exc:
        printer.console.print(f"\n[bold red]Run stopped: the LLM is unavailable.[/bold red] {escape(str(exc))}")
        records = list_run_records(sys_config.output.runs_dir)
        if records:
            printer.console.print(f"The run record (with every attempt) is in {escape(str(records[-1]))}")
        printer.console.print("Test the keys with [cyan]python -m llm.check[/cyan], or run with --provider mock.")
        return 1

    printer.console.print(f"\n[bold]Done.[/bold] Reproduce this run with: [cyan]{rerun_command(args, sys_config, record.seed)}[/cyan]")
    if sys_config.llm.provider == "litellm":
        printer.console.print("[dim]With litellm, a rerun replays the same LLM answers from llm_cache/ (unless --no-cache).[/dim]")
    return 0

if __name__ == "__main__":
    sys.exit(main())
