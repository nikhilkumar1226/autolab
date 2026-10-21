"""
End-to-end integration test for the Autonomous Scientific Discovery loop.
"""
from pathlib import Path
from config import SystemConfig, HardwareConfig, SimulationConfig, AgentConfig
from agents.orchestrator import AutonomousDiscoveryEngine

def test_autonomous_discovery_loop_end_to_end(tmp_path):
    sys_config = SystemConfig(
        hardware=HardwareConfig(
            simulate=True,
            min_actuator_value=0.0,
            max_actuator_value=180.0
        ),
        simulation=SimulationConfig(
            experiment_type="malus_law",
            i0_intensity=1000.0,
            theta_offset=0.0,
            ambient_noise_level=10.0,
            noise_std=5.0,
            transit_delay_sec=0.0
        ),
        agent=AgentConfig(
            max_discovery_iterations=10,
            convergence_r2_threshold=0.98,
            initial_sample_count=4
        )
    )

    engine = AutonomousDiscoveryEngine(sys_config)
    summary = engine.run_discovery_loop(
        research_goal="Investigate polarized light transmission as a function of filter rotation"
    )

    assert summary.confirmed_law == "Malus's Law (Trigonometric Cos^2)"
    assert summary.final_r2 >= 0.98
    assert summary.paper_filepath is not None
    assert summary.plot_filepath is not None
    assert Path(summary.paper_filepath).exists()
    assert Path(summary.plot_filepath).exists()
