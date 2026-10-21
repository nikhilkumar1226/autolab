"""
Main entrypoint for the Autonomous Scientific Discovery System.
Run autonomous scientific discovery loops on digital twin simulation or real physical hardware.
"""
import sys
import argparse
from config import config, SystemConfig, HardwareConfig, SimulationConfig, AgentConfig
from agents.orchestrator import AutonomousDiscoveryEngine

def parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous Scientific Discovery System - LLM Agents for Hardware-Grounded Experimental Design"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="malus_law",
        choices=["malus_law", "inverse_square_law"],
        help="Scientific experiment to run (default: malus_law)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Investigate the mathematical relationship between polarizer angle and transmitted light intensity",
        help="High-level scientific inquiry prompt for the Lead Scientist Agent"
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
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=12,
        help="Maximum discovery iterations before termination (default: 12)"
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.985,
        help="R^2 threshold for scientific convergence (default: 0.985)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Configure system
    sim_mode = not args.hardware
    
    # Adjust prompt if running inverse square law with default prompt
    prompt = args.prompt
    if args.experiment == "inverse_square_law" and "polarizer" in prompt:
        prompt = "Investigate how light radiant intensity decays as distance increases from a stationary source"

    sys_config = SystemConfig(
        hardware=HardwareConfig(
            simulate=sim_mode,
            port=args.port,
            baud_rate=args.baud,
            min_actuator_value=0.0 if args.experiment == "malus_law" else 5.0,
            max_actuator_value=180.0 if args.experiment == "malus_law" else 100.0,
        ),
        simulation=SimulationConfig(
            experiment_type=args.experiment,
            i0_intensity=1000.0,
            theta_offset=0.0,
            noise_std=10.0
        ),
        agent=AgentConfig(
            max_discovery_iterations=args.max_iterations,
            convergence_r2_threshold=args.r2_threshold
        )
    )

    engine = AutonomousDiscoveryEngine(sys_config)
    summary = engine.run_discovery_loop(research_goal=prompt)
    
    print("\nDiscovery completed successfully!")
    print(f"Summary: Discovered {summary.confirmed_law} with R^2 = {summary.final_r2:.4f}")
    print(f"Paper report: {summary.paper_filepath}")
    print(f"Visualization: {summary.plot_filepath}")

if __name__ == "__main__":
    main()
