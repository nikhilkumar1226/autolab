"""
A fake serial port that behaves like firmware/arduino_rig/arduino_rig.ino (SENSOR_EMULATED), for
testing the serial driver without a board or Wokwi. FakeBoard holds the firmware state and
outlives connections; FakePort is one open connection to it (a reconnect makes a new port).

It mirrors the firmware's rules: strict decimal arguments, bounds, step limit, latched E-stop,
the command watchdog (on a simulated clock), the 32-character line limit, EVENT lines.
"""
import math
import random
from typing import List, Optional

MIN_ANGLE, MAX_ANGLE, MAX_STEP = 0.0, 180.0, 180.0
WATCHDOG_S = 300
LINE_MAX = 32
A, PHASE, AMBIENT, NOISE = 1000.0, 20.0, 15.0, 10.0


def _plain_decimal(text: str) -> Optional[float]:
    t = text.strip()
    body = t[1:] if t[:1] in "+-" else t
    digits = sum(c.isdigit() for c in body)
    if not body or any(not (c.isdigit() or c == ".") for c in body) or body.count(".") > 1 or digits == 0 or digits > 8:
        return None
    return float(t)


class FakeBoard:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.angle = 0.0
        self.homed = False
        self.estop = False
        self.reason = "none"
        self.attached = False
        self.now = 0.0                 # simulated seconds
        self.last_command = 0.0
        self.pending: List[str] = []   # unsolicited lines waiting to be read
        self.commands: List[str] = []  # every line received, for assertions
        self.silent = 0                # replies to swallow (simulates a lost reply)
        self.fail_writes = 0           # writes that raise OSError (a dropped USB link)
        self.ports_opened = 0

    def advance(self, seconds: float) -> None:
        self.now += seconds
        if not self.estop and self.homed and self.now - self.last_command > WATCHDOG_S:
            self.estop, self.reason, self.attached = True, "watchdog", False
            self.pending.append(f"EVENT: WATCHDOG, MSG: no command for {WATCHDOG_S} s; servo released; send RESET")

    def handle(self, line: str) -> Optional[str]:
        self.commands.append(line)
        self.last_command = self.now
        if len(line) > LINE_MAX:
            return "STATUS: ERROR, CODE: E_SYNTAX, MSG: line too long"
        verb, _, arg = line.strip().partition(" ")
        verb, arg = verb.upper(), arg.strip()
        if verb == "PING":
            return "PONG"
        if verb == "VERSION":
            return (f"VERSION: 2.0.0, SENSOR: EMULATED, MIN: {MIN_ANGLE:.2f}, MAX: {MAX_ANGLE:.2f}, "
                    f"MAX_STEP: {MAX_STEP:.2f}, WATCHDOG_S: {WATCHDOG_S}")
        if verb == "STATUS":
            return (f"STATUS: OK, POS: {self.angle:.2f}, HOMED: {int(self.homed)}, ESTOP: {int(self.estop)}, "
                    f"REASON: {self.reason}")
        if verb == "HOME":
            if self.estop:
                return "STATUS: ERROR, CODE: E_ESTOP, MSG: E-stop latched; send RESET"
            self.angle, self.homed, self.attached = MIN_ANGLE, True, True
            return f"STATUS: OK, POS: {self.angle:.2f}"
        if verb == "MOVE":
            target = _plain_decimal(arg) if arg else None
            if target is None:
                return "STATUS: ERROR, CODE: E_SYNTAX, MSG: MOVE needs a plain decimal angle"
            if self.estop:
                return "STATUS: ERROR, CODE: E_ESTOP, MSG: E-stop latched; send RESET"
            if not MIN_ANGLE <= target <= MAX_ANGLE:
                return "STATUS: ERROR, CODE: E_BOUNDS, MSG: angle outside [0, 180]"
            if abs(target - self.angle) > MAX_STEP:
                return "STATUS: ERROR, CODE: E_STEP, MSG: step larger than MAX_STEP"
            self.now += 0.1 + abs(target - self.angle) * 0.002
            self.angle, self.attached = target, True
            return f"STATUS: OK, POS: {self.angle:.2f}"
        if verb == "READ":
            self.now += 0.02
            value = A * math.cos(math.radians(self.angle - PHASE)) ** 2 + AMBIENT + self.rng.gauss(0.0, NOISE)
            return f"STATUS: OK, DATA: {max(0.0, value):.2f}"
        if verb == "ESTOP":
            self.estop, self.reason, self.attached = True, "operator", False
            return "STATUS: OK, ESTOP: 1"
        if verb == "RESET":
            self.estop, self.reason, self.attached = False, "none", True
            return f"STATUS: OK, ESTOP: 0, POS: {self.angle:.2f}"
        return "STATUS: ERROR, CODE: E_UNKNOWN, MSG: unknown command"


class FakePort:
    """The subset of a pyserial port the driver uses."""

    def __init__(self, board: FakeBoard, url: str, **kwargs):
        self.board, self.url, self.kwargs = board, url, kwargs
        self.is_open = True
        self._out: List[bytes] = [b"EVENT: READY, VERSION: 2.0.0, SENSOR: EMULATED\n"]
        board.ports_opened += 1

    def write(self, data: bytes) -> int:
        if self.board.fail_writes:
            self.board.fail_writes -= 1
            raise OSError("device disconnected")
        for raw in data.decode("ascii").split("\n"):
            if not raw.strip():
                continue
            reply = self.board.handle(raw)
            self._out += [(line + "\n").encode() for line in self.board.pending]
            self.board.pending.clear()
            if self.board.silent:
                self.board.silent -= 1
            elif reply is not None:
                self._out.append((reply + "\n").encode())
        return len(data)

    def flush(self) -> None:
        pass

    def readline(self) -> bytes:
        return self._out.pop(0) if self._out else b""  # b"" = the read timed out

    def reset_input_buffer(self) -> None:
        self._out.clear()

    def reset_output_buffer(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False


def port_factory(board: FakeBoard):
    """A stand-in for serial.serial_for_url that connects to `board`."""
    def open_port(url, **kwargs):
        return FakePort(board, url, **kwargs)
    return open_port
