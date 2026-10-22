"""
Serial driver for the AutoLab rig firmware (firmware/arduino_rig/arduino_rig.ino, v2).

The same driver talks to a board on USB (COM3, /dev/ttyUSB0) and to the firmware running in the
Wokwi simulator (rfc2217://localhost:4000): ports are opened with serial.serial_for_url. The
data-source label follows the port: rfc2217 means FIRMWARE-IN-THE-LOOP (Wokwi), anything else
PHYSICAL RIG.

Protocol (one command per line, one reply line each; see the firmware header):
    PING -> PONG            VERSION -> VERSION: ...       STATUS -> STATUS: OK, POS: .., HOMED: .., ESTOP: ..
    HOME -> STATUS: OK, POS: 0.00                         MOVE <deg> -> STATUS: OK, POS: <deg>
    READ -> STATUS: OK, DATA: <lux>                       ESTOP / RESET -> STATUS: OK, ESTOP: 1 / 0
    errors: STATUS: ERROR, CODE: <E_...>, MSG: <text>
Lines starting with "EVENT:" (boot banner, watchdog trip) are unsolicited; they are logged and
skipped while waiting for a reply. A firmware error becomes FirmwareError with its code. If the
link drops, the driver reconnects once and repeats the command (every command is idempotent:
MOVE is an absolute position). No reply within the timeout raises TimeoutError, which the
safety guard turns into a flagged reading.
"""
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from hardware.base import BaseHardwareRig, DataSource, MeasurementResult, RigStatus
from config import HardwareConfig

logger = logging.getLogger(__name__)

COMMANDS = ("PING", "VERSION", "STATUS", "HOME", "MOVE", "READ", "ESTOP", "RESET")
ERROR_CODES = ("E_SYNTAX", "E_BOUNDS", "E_STEP", "E_ESTOP", "E_SENSOR", "E_UNKNOWN")
REPLY_PREFIXES = {"PING": ("PONG",), "VERSION": ("VERSION:",)}
MAX_SKIPPED_LINES = 20


class FirmwareError(RuntimeError):
    """The firmware refused a command (its own safety checks, or a bad command)."""

    def __init__(self, code: str, message: str, reply: str):
        super().__init__(f"firmware {code}: {message}")
        self.code, self.message, self.reply = code, message, reply


def parse_fields(reply: str) -> Dict[str, str]:
    """'STATUS: OK, POS: 12.00, HOMED: 1' -> {'STATUS': 'OK', 'POS': '12.00', 'HOMED': '1'}."""
    fields: Dict[str, str] = {}
    for part in reply.split(","):
        if ":" in part:
            key, value = part.split(":", 1)
            fields[key.strip().upper()] = value.strip()
    return fields


def _number(fields: Dict[str, str], key: str, reply: str) -> float:
    try:
        value = float(fields[key])
    except (KeyError, ValueError):
        raise ValueError(f"no number for {key} in firmware reply {reply!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite in firmware reply {reply!r}")
    return value


class SerialHardwareRig(BaseHardwareRig):
    """The AutoLab rig over a serial port (USB) or RFC2217 (Wokwi)."""

    def __init__(
        self,
        hw_config: Optional[HardwareConfig] = None,
        units_actuator: str = "deg",
        units_sensor: str = "lux",
        port_factory: Optional[Callable[..., Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = hw_config or HardwareConfig()
        self.units_actuator = units_actuator
        self.units_sensor = units_sensor
        self._port_factory = port_factory
        self._sleep = sleep
        self._serial = None
        self._connected = False
        self._is_homed = False
        self._current_position = 0.0
        self._is_busy = False
        self.firmware: Dict[str, str] = {}
        self.last_error = ""  # why the last connect() failed, for the run's error message
        self.events: list = []

    @property
    def is_simulated_firmware(self) -> bool:
        return self.config.port.lower().startswith("rfc2217://")

    @property
    def data_source(self) -> DataSource:
        # Wokwi forwards the simulated Arduino's serial port over RFC2217.
        return DataSource.FIRMWARE_IN_THE_LOOP if self.is_simulated_firmware else DataSource.PHYSICAL_RIG

    # --- connection -----------------------------------------------------------------
    def _open(self) -> None:
        factory = self._port_factory
        if factory is None:
            try:
                import serial
            except ImportError:
                raise ImportError("pyserial is required for hardware. Run `pip install pyserial`.") from None
            factory = serial.serial_for_url
        options = {"baudrate": self.config.baud_rate, "timeout": self.config.timeout_seconds}
        if not self.is_simulated_firmware:
            # pySerial's rfc2217 port (Wokwi) raises NotImplementedError on open if write_timeout is set.
            options["write_timeout"] = self.config.timeout_seconds
        self._serial = factory(self.config.port, **options)
        if not self.is_simulated_firmware:
            self._sleep(2.0)  # a USB Arduino resets when the port opens (DTR); wait for the bootloader
        for method in ("reset_input_buffer", "reset_output_buffer"):
            if hasattr(self._serial, method):
                getattr(self._serial, method)()

    def connect(self) -> bool:
        """Open the port, check the firmware answers PING, and read its VERSION line."""
        try:
            self._open()
            for attempt in range(3):
                try:
                    if self._command("PING", retry=False) == "PONG":
                        break
                except TimeoutError:
                    if attempt == 2:
                        raise
            self.firmware = parse_fields(self._command("VERSION", retry=False))
            self._connected = True
            self.last_error = ""
            logger.info("Connected to %s: firmware %s", self.config.port, self.firmware)
            return True
        except Exception as exc:
            logger.error("Failed to connect to the rig on %s: %s", self.config.port, exc)
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.disconnect()
            return False

    def disconnect(self) -> None:
        if self._serial is not None and getattr(self._serial, "is_open", False):
            try:
                self._serial.close()
            except Exception as exc:
                logger.error("Error closing serial port: %s", exc)
        self._connected = False

    def is_connected(self) -> bool:
        return bool(self._serial is not None and getattr(self._serial, "is_open", False) and self._connected)

    # --- one command, one reply ------------------------------------------------------
    def _exchange(self, cmd: str) -> str:
        if self._serial is None or not getattr(self._serial, "is_open", False):
            raise ConnectionError("Hardware serial port is not open.")
        verb = cmd.split()[0]
        expected = REPLY_PREFIXES.get(verb, ("STATUS:",))
        self._serial.write((cmd.strip() + "\n").encode("ascii"))
        self._serial.flush()
        for _ in range(MAX_SKIPPED_LINES):
            raw = self._serial.readline()
            if not raw:
                raise TimeoutError(f"no reply to {verb} within {self.config.timeout_seconds:g} s")
            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue
            if line.startswith("EVENT:"):
                logger.warning("Rig event: %s", line)
                self.events.append(line)
                continue
            if line.startswith(expected):
                return line
            if line.startswith("STATUS: ERROR"):
                return line
            logger.warning("Skipping unexpected line from the rig: %r", line)
        raise TimeoutError(f"no valid reply to {verb} after {MAX_SKIPPED_LINES} lines")

    def _command(self, cmd: str, retry: bool = True) -> str:
        """Send a command and return its reply line. Raises FirmwareError for STATUS: ERROR."""
        self._is_busy = True
        try:
            try:
                reply = self._exchange(cmd)
            except TimeoutError:
                raise  # no reply is not a broken link (TimeoutError is an OSError): the caller flags it
            except (OSError, ConnectionError) as exc:  # serial.SerialException is an OSError
                if not retry:
                    raise
                logger.warning("Serial link error (%s); reconnecting once and repeating %s", exc, cmd.split()[0])
                self.disconnect()
                self._open()
                reply = self._exchange(cmd)
        finally:
            self._is_busy = False
        if reply.startswith("STATUS: ERROR"):
            fields = parse_fields(reply)
            raise FirmwareError(fields.get("CODE", "E_UNKNOWN"), fields.get("MSG", reply), reply)
        return reply

    # --- rig interface ---------------------------------------------------------------------
    def home(self) -> bool:
        fields = parse_fields(self._command("HOME"))
        self._current_position = _number(fields, "POS", str(fields))
        self._is_homed = True
        return True

    def move_actuator(self, position: float) -> bool:
        reply = self._command(f"MOVE {position:.2f}")
        reached = _number(parse_fields(reply), "POS", reply)
        if abs(reached - round(position, 2)) > 0.011:
            raise RuntimeError(f"firmware moved to {reached}, not the requested {position:.2f}")
        self._current_position = reached
        return True

    def read_sensor(self) -> MeasurementResult:
        reply = self._command("READ")
        value = _number(parse_fields(reply), "DATA", reply)
        return MeasurementResult(
            timestamp=datetime.now(timezone.utc),
            actuator_position=self._current_position,
            sensor_value=value,
            units_actuator=self.units_actuator,
            units_sensor=self.units_sensor,
            is_valid=True,
            metadata={"raw_response": reply, "firmware": self.firmware.get("VERSION", "")},
        )

    def emergency_stop(self) -> None:
        """Latch the firmware's own E-stop too (servo released). Best effort: the host guard is latched already."""
        try:
            self._command("ESTOP", retry=False)
        except Exception as exc:
            logger.error("Could not send ESTOP to the rig: %s", exc)

    def reset_emergency_stop(self) -> None:
        self._command("RESET")

    def get_status(self) -> RigStatus:
        return RigStatus(
            is_connected=self.is_connected(),
            is_homed=self._is_homed,
            current_position=self._current_position,
            is_busy=self._is_busy,
            telemetry={"port": self.config.port, "baud": self.config.baud_rate, "firmware": dict(self.firmware)},
        )
