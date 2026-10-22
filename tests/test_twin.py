"""
Digital-twin non-idealities and fault injection, tested on the rig itself (no engine).
"""
import math

import numpy as np
import pytest

from config import FaultSpec, NonIdealityConfig, SimulationConfig
from experiments.registry import get_experiment
from hardware.simulator import PhysicsSimulationRig

# Every non-ideality off except the ones a test turns on.
QUIET = dict(enabled=True, extinction_ratio=math.inf, drift_step_sigma=0.0, saturation=False, quantisation=False,
             position_error_rel=0.0, backlash_rel=0.0, outlier_prob=0.0)


def rig(name="malus_law", seed=0, noise_std=0.0, faults=(), spec=None, **nonideal) -> PhysicsSimulationRig:
    ni = NonIdealityConfig(**{**QUIET, **nonideal}) if nonideal else NonIdealityConfig()
    sim = SimulationConfig(noise_std=noise_std, transit_delay_sec=0.0, nonideal=ni, faults=list(faults))
    r = PhysicsSimulationRig(spec or get_experiment(name), np.random.default_rng(seed), sim)
    r.connect()
    return r


def read_at(r: PhysicsSimulationRig, x: float):
    r.move_actuator(x)
    return r.read_sensor()


# --- non-idealities off: the Phase 1 twin ---------------------------------------------------

def test_without_non_idealities_the_rig_draws_one_normal_per_reading():
    spec = get_experiment("malus_law")
    r = rig(noise_std=10.0)
    expected_rng = np.random.default_rng(0)
    for x in [0.0, 45.0, 110.0, 110.0, 170.0]:
        expected = max(0.0, spec.true_response(x) + expected_rng.normal(0.0, 10.0))
        assert read_at(r, x).sensor_value == expected
    assert r.fault_log == []


# --- each non-ideality ----------------------------------------------------------------------

def test_finite_extinction_ratio_leaks_light_through_crossed_polarisers():
    spec = get_experiment("malus_law")
    A, x0, C = (spec.true_params[k] for k in ("A", "x0", "C"))
    r = rig(extinction_ratio=100.0)
    assert read_at(r, x0 + 90.0).sensor_value == pytest.approx(C + A / 100.0)
    assert read_at(r, x0).sensor_value == pytest.approx(C + A)
    # Only polariser experiments are affected.
    beer = rig("beer_lambert_law", extinction_ratio=100.0)
    assert read_at(beer, 0.0).sensor_value == pytest.approx(get_experiment("beer_lambert_law").true_response(0.0))


def test_drift_is_a_seeded_random_walk_of_the_set_size():
    def drift(seed):
        r = rig(seed=seed, noise_std=0.5, drift_step_sigma=4.0)  # walk step std = 4 x 0.5 = 2
        r.move_actuator(45.0)
        return np.array([r.read_sensor().sensor_value for _ in range(400)])

    a = drift(1)
    steps = np.diff(a)
    # Consecutive differences: one walk step plus the difference of two noise draws.
    assert np.std(steps) == pytest.approx(np.sqrt(2.0 ** 2 + 2 * 0.5 ** 2), rel=0.1)
    assert abs(a[-1] - a[0]) > 5.0  # it wanders, unlike noise around a fixed value
    assert np.array_equal(a, drift(1)) and not np.array_equal(a, drift(2))


def test_saturation_clips_at_full_scale_only_when_enabled():
    low_range = get_experiment("beer_lambert_law").model_copy(update={"sensor_range": (0.0, 1000.0)})
    assert read_at(rig(spec=low_range), 0.0).sensor_value == pytest.approx(1015.0)  # ideal sensor
    assert read_at(rig(spec=low_range, saturation=True), 0.0).sensor_value == 1000.0


def test_quantisation_rounds_to_the_adc_step():
    spec = get_experiment("rc_charging")
    r = rig("rc_charging", noise_std=0.01, quantisation=True)
    for x in [3.0, 17.0, 60.0, 140.0]:
        value = read_at(r, x).sensor_value
        steps = value / spec.sensor_resolution
        assert steps == pytest.approx(round(steps), abs=1e-9)


def test_position_error_reaches_the_physics_but_the_rig_reports_the_command():
    spec = get_experiment("hookes_law")
    r = rig("hookes_law", seed=3, position_error_rel=0.15)
    errors = []
    for i in range(300):
        target = 2.0 + (i % 2) * 10.0
        m = read_at(r, target)
        assert m.actuator_position == target
        actual = (m.sensor_value - 50.0) / 4.0  # invert the noise-free law: L = 50 + 4 F
        errors.append(actual - target)
    assert np.std(errors) == pytest.approx(0.15 * spec.position_tolerance, rel=0.15)
    assert abs(np.mean(errors)) < 0.05 * spec.position_tolerance


def test_backlash_depends_on_the_direction_of_approach():
    spec = get_experiment("hookes_law")
    r = rig("hookes_law", backlash_rel=0.4)
    from_below = (read_at(r, 10.0).sensor_value - 50.0) / 4.0
    read_at(r, 15.0)
    from_above = (read_at(r, 10.0).sensor_value - 50.0) / 4.0
    assert from_below - from_above == pytest.approx(0.4 * spec.position_tolerance)


def test_outlier_spikes_are_logged_as_ground_truth():
    r = rig(noise_std=10.0, outlier_prob=1.0, outlier_sigma=8.0)
    ideal = get_experiment("malus_law").true_response(45.0)
    values = [read_at(r, 45.0).sensor_value for _ in range(20)]
    assert all(abs(v - ideal) > 40.0 for v in values)  # 80 lux spike, 10 lux noise
    assert [e.type for e in r.fault_log] == ["outlier"] * 20
    assert all(e.fault_index is None and abs(e.detail["size"]) == 80.0 for e in r.fault_log)


def test_every_non_ideality_on_is_still_reproducible_from_the_seed():
    def run(seed):
        r = rig(seed=seed, noise_std=10.0, **{k: v for k, v in NonIdealityConfig(enabled=True).model_dump().items()})
        return [(m.sensor_value, m.metadata["position_feedback"]) for m in (read_at(r, x) for x in np.linspace(0, 180, 30))]

    assert run(4) == run(4) and run(4) != run(5)


# --- faults ---------------------------------------------------------------------------------

def test_a_light_spike_is_added_at_its_cycle_only():
    r = rig("beer_lambert_law", faults=[FaultSpec(type="light_spike", at_step=3, magnitude=0.25)])
    spec = get_experiment("beer_lambert_law")
    values = [read_at(r, 50.0).sensor_value for _ in range(4)]
    clean = spec.true_response(50.0)
    assert values[2] == pytest.approx(clean + 0.25 * r.response_span)
    assert values[0] == values[1] == values[3] == pytest.approx(clean)
    [event] = r.fault_log
    assert (event.type, event.cycle, event.fault_index) == ("light_spike", 3, 0)


def test_a_sensor_dropout_returns_nan_or_times_out_for_its_duration():
    nan = rig(faults=[FaultSpec(type="sensor_dropout", at_step=2, duration=2)])
    values = [read_at(nan, 45.0).sensor_value for _ in range(4)]
    assert math.isfinite(values[0]) and math.isnan(values[1]) and math.isnan(values[2]) and math.isfinite(values[3])

    timeout = rig(faults=[FaultSpec(type="sensor_dropout", at_step=1, mode="timeout")])
    timeout.move_actuator(45.0)
    with pytest.raises(TimeoutError):
        timeout.read_sensor()
    assert math.isfinite(timeout.read_sensor().sensor_value)


def test_a_stuck_actuator_acknowledges_the_move_but_stays_put():
    r = rig(faults=[FaultSpec(type="stuck_actuator", at_step=2)])
    first = read_at(r, 30.0)
    stuck = read_at(r, 120.0)
    assert stuck.actuator_position == 120.0 and stuck.metadata["position_feedback"] == 30.0
    assert stuck.sensor_value == pytest.approx(first.sensor_value)  # still measuring at 30 deg
    assert r.get_status().current_position == 120.0
    [event] = r.fault_log
    assert event.type == "stuck_actuator" and event.detail == {"commanded": 120.0, "actual": 30.0}
    moved = read_at(r, 120.0)  # next cycle: the command is executed
    assert moved.metadata["position_feedback"] == 120.0


def test_a_pause_is_a_cycle_and_a_dropout_during_it_affects_no_reading():
    r = rig(faults=[FaultSpec(type="sensor_dropout", at_step=2)])
    read_at(r, 45.0)
    r.wait(1.0)
    assert math.isfinite(read_at(r, 45.0).sensor_value)
    assert r.get_status().telemetry["cycle"] == 3
    [event] = r.fault_log
    assert event.cycle == 2 and event.read is False
