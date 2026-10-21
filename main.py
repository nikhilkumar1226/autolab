"""
Main entrypoint for the Autonomous Scientific Discovery System.
Run autonomous scientific discovery loops on digital twin simulation or real physical hardware.
"""
import argparse
import logging
from config import config, SystemConfig, HardwareConfig, AgentConfig, StoppingConfig, OutputConfig
from experiments.registry import list_experiments
from agents.orchestrator import DiscoveryEngine
from cli.console import ConsoleEventPrinter

def parse_args():
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
        choices=["gp", "random"],
        default="gp",
        help="How to choose points after the initial design (default: gp uncertainty sampling)"
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
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    sys_config = SystemConfig(
        experiment=args.experiment,
        seed=args.seed,
        hardware=HardwareConfig(
            simulate=not args.hardware,
            port=args.port,
            baud_rate=args.baud,
        ),
        agent=AgentConfig(sampler=args.sampler),
        stopping=StoppingConfig(
            max_measurements=args.budget,
            max_rel_se=args.rse_threshold,
            min_points=args.min_points,
            predictive_stability=args.stability,
        ),
        output=OutputConfig(write_report=not args.no_report),
    )

    printer = ConsoleEventPrinter()
    engine = DiscoveryEngine(sys_config, on_event=printer)
    record = engine.run(question=args.question)

    rerun = f"python main.py --experiment {args.experiment} --seed {record.seed}"
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
    printer.console.print(f"\n[bold]Done.[/bold] Reproduce this run with: [cyan]{rerun}[/cyan]")

if __name__ == "__main__":
    main()
