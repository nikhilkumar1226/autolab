"""
Phase 6: the serial driver against a fake serial port that behaves like the firmware, the
firmware's own safety rules, a full AutoLab run over "Wokwi" (rfc2217), and checks that the
firmware source, the driver and the digital twin agree.
"""
import math
import re
from pathlib import Path

import numpy as np
import pytest

import main
from agents.orchestrator import DiscoveryEngine
from analysis.stopping import StopReason
from config import HardwareConfig, SafetyConfig
from experiments.registry import get_experiment
from hardware.base import DataSource
from hardware.safety_guard import HardwareSafetyGuard, SafetyLimits, SafetyViolationError
from hardware.serial_driver import COMMANDS, ERROR_CODES, FirmwareError, SerialHardwareRig, parse_fields
from tests.fake_firmware import FakeBoard, port_factory

ROOT = Path(__file__).resolve().parent.parent
FIRMWARE = (ROOT / "firmware" / "arduino_rig" / "arduino_rig.ino").read_text(encoding="utf-8")
WOKWI = "rfc2217://localhost:4000"


def rig(board, port=WOKWI, sleeps=None):
    r = SerialHardwareRig(HardwareConfig(port=port, timeout_seconds=0.1), port_factory=port_factory(board),
                          sleep=(sleeps.append if sleeps is not None else (lambda s: None)))
    assert r.connect()
    return r


# --- connection and labels -------------------------------------------------------------------

def test_wokwi_connection_reads_the_firmware_version_and_is_labelled_firmware_in_the_loop():
    sleeps = []
    r = rig(FakeBoard(), sleeps=sleeps)
    assert r.data_source == DataSource.FIRMWARE_IN_THE_LOOP
    assert r.firmware["VERSION"] == "2.0.0" and r.firmware["SENSOR"] == "EMULATED"
    assert sleeps == []  # no bootloader wait for a simulated board


def test_a_usb_board_is_a_physical_rig_and_gets_its_bootloader_wait():
    sleeps = []
    r = rig(FakeBoard(), port="/dev/ttyUSB0", sleeps=sleeps)
    assert r.data_source == DataSource.PHYSICAL_RIG and sleeps == [2.0]
    assert SerialHardwareRig(HardwareConfig(port="COM3")).data_source == DataSource.PHYSICAL_RIG


def test_a_silent_port_does_not_connect():
    board = FakeBoard()
    board.silent = 10
    r = SerialHardwareRig(HardwareConfig(port=WOKWI, timeout_seconds=0.1), port_factory=port_factory(board))
    assert not r.connect() and not r.is_connected()


# --- commands ----------------------------------------------------------------------------------

def test_home_move_and_read_follow_the_emulated_malus_law():
    board = FakeBoard(seed=1)
    r = rig(board)
    assert r.home() and r.get_status().current_position == 0.0
    readings = {}
    for angle in (20.0, 65.0, 110.0):
        assert r.move_actuator(angle)
        readings[angle] = np.mean([r.read_sensor().sensor_value for _ in range(30)])
    assert readings[20.0] == pytest.approx(1015.0, abs=10.0)   # maximum at the phase angle
    assert readings[65.0] == pytest.approx(515.0, abs=10.0)    # half way at 45 degrees from it
    assert readings[110.0] == pytest.approx(15.0, abs=10.0)    # crossed polarisers
    m = r.read_sensor()
    assert m.actuator_position == 110.0 and m.units_actuator == "deg" and m.metadata["firmware"] == "2.0.0"
    assert board.commands[-1] == "READ" and "MOVE 110.00" in board.commands


@pytest.mark.parametrize("line, code", [
    ("MOVE 200", "E_BOUNDS"), ("MOVE -1", "E_BOUNDS"), ("MOVE abc", "E_SYNTAX"), ("MOVE", "E_SYNTAX"),
    ("MOVE 1e5", "E_SYNTAX"), ("MOVE 90;HOME", "E_SYNTAX"), ("MOVE nan", "E_SYNTAX"), ("FLY", "E_UNKNOWN"),
    ("MOVE " + "9" * 40, "E_SYNTAX"),
])
def test_the_firmware_rejects_bad_commands_with_a_code(line, code):
    r = rig(FakeBoard())
    r.home()
    with pytest.raises(FirmwareError) as info:
        r._command(line)
    assert info.value.code == code
    assert r.get_status().current_position == 0.0


def test_event_lines_and_noise_are_skipped():
    board = FakeBoard()
    r = rig(board)
    board.pending += ["EVENT: something happened", "garbage line"]
    assert r._command("PING") == "PONG"
    assert r.events == ["EVENT: something happened"]


def test_a_lost_reply_is_a_timeout_and_the_guard_flags_it():
    board = FakeBoard()
    r = rig(board)
    r.home()
    board.silent = 1
    with pytest.raises(TimeoutError):
        r.read_sensor()
    spec = get_experiment("malus_law")
    guard = HardwareSafetyGuard(r, SafetyLimits.from_spec(spec, SafetyConfig()))
    board.silent = 1
    result = guard.safe_read()
    assert result.flags == ["timeout"] and not result.is_valid


def test_a_dropped_link_reconnects_once_and_repeats_the_command():
    board = FakeBoard()
    r = rig(board)
    r.home()
    board.fail_writes = 1
    assert r.move_actuator(45.0)
    assert board.ports_opened == 2 and board.angle == 45.0
    board.fail_writes = 2
    with pytest.raises(OSError):
        r.move_actuator(50.0)


# --- E-stop and watchdog: both layers -----------------------------------------------------------

def test_the_guards_estop_latches_the_firmware_too_and_the_reset_clears_both():
    board = FakeBoard()
    r = rig(board)
    guard = HardwareSafetyGuard(r, SafetyLimits.from_spec(get_experiment("malus_law"), SafetyConfig()))
    guard.home()
    guard.safe_move(30.0)
    guard.emergency_stop("dashboard")
    assert board.estop and board.reason == "operator" and not board.attached  # servo released
    with pytest.raises(FirmwareError, match="E_ESTOP"):
        r.move_actuator(40.0)  # even a caller that bypassed the guard is refused by the firmware
    guard.reset_estop(operator="Nikhil")
    assert not board.estop and "RESET" in board.commands
    guard.safe_move(40.0)
    assert board.angle == 40.0


def test_the_firmware_watchdog_releases_the_servo_when_the_host_goes_silent():
    board = FakeBoard()
    r = rig(board)
    r.home()
    r.move_actuator(90.0)
    board.advance(301.0)
    assert board.estop and board.reason == "watchdog" and not board.attached
    with pytest.raises(FirmwareError, match="E_ESTOP"):
        r.move_actuator(80.0)
    assert any("WATCHDOG" in e for e in r.events)
    r.reset_emergency_stop()
    assert r.move_actuator(80.0)


# --- a full run over the firmware ---------------------------------------------------------------

def test_a_full_autolab_run_over_the_firmware_link(make_config, monkeypatch):
    import serial
    board = FakeBoard(seed=3)
    monkeypatch.setattr(serial, "serial_for_url", port_factory(board))
    cfg = make_config(experiment="malus_law", seed=0, write_report=True,
                      hardware=HardwareConfig(simulate=False, port=WOKWI, timeout_seconds=0.1),
                      safety=SafetyConfig(max_moves_per_s=1000))  # the fake answers instantly
    record = DiscoveryEngine(cfg).run()
    assert record.data_source == DataSource.FIRMWARE_IN_THE_LOOP
    assert record.stop.reason == StopReason.CONVERGED
    assert record.result.final_analysis.best.key == "cos2_phase"
    params = record.result.final_analysis.best.params
    assert params["x0"] == pytest.approx(20.0, abs=3.0) and params["A"] == pytest.approx(1000.0, rel=0.05)
    assert record.experiment.noise_std is None and record.faults == []
    moves = [c for c in board.commands if c.startswith("MOVE")]
    assert len(moves) == len([s for s in record.steps if s.status != "rejected"])
    markdown = Path(record.artifacts.report_md).read_text(encoding="utf-8")
    assert "Data source: FIRMWARE-IN-THE-LOOP (Wokwi)" in markdown and record.audit_intact()


def test_hardware_mode_only_drives_the_polariser_experiment(capsys):
    assert main.main(["--experiment", "diode_iv", "--hardware", "--port", WOKWI]) == 2
    assert "supports --experiment malus_law only" in capsys.readouterr().err


# --- the firmware source agrees with the driver and the twin -----------------------------------

def test_every_protocol_command_and_error_code_is_in_the_firmware():
    for word in COMMANDS:
        assert f'"{word}"' in FIRMWARE, word
    for code in ERROR_CODES:
        assert f'"{code}"' in FIRMWARE, code
    for name in ("SENSOR_EMULATED", "SENSOR_LDR", "SENSOR_TSL2591"):
        assert f"#ifdef {name}" in FIRMWARE or f"defined({name})" in FIRMWARE
    assert "const int SERVO_PIN = 9;" in FIRMWARE and "const int LDR_PIN = A0;" in FIRMWARE
    assert "stepper" not in FIRMWARE.lower() and "MLX90614" not in FIRMWARE


def test_the_firmware_limits_and_emulated_optics_match_the_digital_twin():
    def const(name):
        return float(re.search(rf"const \w+ {name} = ([-\d.]+)", FIRMWARE).group(1))

    spec = get_experiment("malus_law")
    assert (const("MIN_ANGLE"), const("MAX_ANGLE")) == spec.safe_bounds
    assert const("MAX_STEP_DEG") >= spec.max_step_delta
    assert const("EMU_AMPLITUDE") == spec.true_params["A"] and const("EMU_PHASE_DEG") == spec.true_params["x0"]
    assert const("EMU_AMBIENT") == spec.true_params["C"] and const("EMU_NOISE_SD") == spec.noise.std


def test_the_wokwi_project_points_at_the_built_firmware_and_wires_the_parts():
    import json
    import tomllib
    toml = tomllib.loads((ROOT / "wokwi.toml").read_text())["wokwi"]
    assert toml["rfc2217ServerPort"] == 4000
    assert (ROOT / toml["firmware"]).exists() and (ROOT / toml["elf"]).exists()
    diagram = json.loads((ROOT / "diagram.json").read_text())
    parts = {p["type"] for p in diagram["parts"]}
    assert {"wokwi-arduino-uno", "wokwi-servo", "wokwi-photoresistor-sensor"} <= parts
    wires = {tuple(sorted(c[:2])) for c in diagram["connections"]}
    assert ("servo:PWM", "uno:9") in wires and ("ldr:AO", "uno:A0") in wires


def test_parse_fields():
    assert parse_fields("STATUS: OK, POS: 12.00, HOMED: 1") == {"STATUS": "OK", "POS": "12.00", "HOMED": "1"}
    assert parse_fields("STATUS: ERROR, CODE: E_BOUNDS, MSG: angle outside [0, 180]")["CODE"] == "E_BOUNDS"
