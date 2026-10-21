"""
Reproducibility: the same seed must give an identical run.
"""
import re
from pathlib import Path

import numpy as np

from agents.orchestrator import DiscoveryEngine
from config import AgentConfig, SimulationConfig
from experiments.registry import get_experiment
from hardware.simulator import PhysicsSimulationRig
from optimization.baselines import RandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def scientific_content(record):
    """Everything in a record except IDs, timestamps and timings."""
    steps = [s.model_dump(exclude={"duration_s"}) for s in record.steps]
    return steps, record.stop.model_dump(), record.result.model_dump()


def test_same_seed_gives_identical_runs(make_config):
    first = DiscoveryEngine(make_config(experiment="inverse_square_law", seed=7)).run()
    second = DiscoveryEngine(make_config(experiment="inverse_square_law", seed=7)).run()
    assert first.run_id != second.run_id
    assert scientific_content(first) == scientific_content(second)


def test_engine_can_be_reused_without_leaking_state(make_config):
    engine = DiscoveryEngine(make_config(seed=5))
    assert scientific_content(engine.run()) == scientific_content(engine.run())


def test_different_seeds_give_different_data(make_config):
    a = DiscoveryEngine(make_config(seed=1)).run()
    b = DiscoveryEngine(make_config(seed=2)).run()
    assert a.measured_points()[1] != b.measured_points()[1]


def test_random_sampler_runs_are_reproducible(make_config):
    cfg = make_config(experiment="beer_lambert_law", seed=3, agent=AgentConfig(sampler="random"))
    first, second = DiscoveryEngine(cfg).run(), DiscoveryEngine(cfg).run()
    assert scientific_content(first) == scientific_content(second)
    assert {s.source for s in first.steps} <= {"initial_design", "random"}
    assert any(s.source == "random" for s in first.steps)


def test_simulator_noise_comes_from_the_generator():
    spec = get_experiment("malus_law")

    def readings(seed):
        rig = PhysicsSimulationRig(spec, np.random.default_rng(seed), SimulationConfig(transit_delay_sec=0.0))
        rig.connect()
        rig.move_actuator(45.0)
        return [rig.read_sensor().sensor_value for _ in range(5)]

    assert readings(11) == readings(11)
    assert readings(11) != readings(12)


def test_random_sampler_is_seeded_and_in_bounds():
    def draws(seed):
        sampler = RandomSampler(5.0, 100.0, np.random.default_rng(seed))
        return [sampler.suggest() for _ in range(200)]

    assert all(5.0 <= v <= 100.0 for v in draws(0))
    assert draws(0) == draws(0)
    assert draws(0) != draws(1)


def test_no_global_numpy_random_calls():
    """Every draw must come from a Generator created from the run seed."""
    allowed = {"default_rng", "SeedSequence", "Generator"}
    pattern = re.compile(r"\b(?:np|numpy)\.random\.(\w+)\s*\(")
    offenders = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in {".venv", "venv", "tests"} for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for name in pattern.findall(line):
                if name not in allowed:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}: np.random.{name}")
    assert offenders == []
