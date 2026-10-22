"""
A spy between the safety guard and a rig: it records every command that reaches the hardware,
and `unsafe_commands` checks those commands against the safe envelope independently of the
guard's own code.

A command is unsafe if it moves the actuator to a value that is not a finite real number,
outside the safe bounds, by more than the step limit, faster than the rate limit, more times in
a row without a reading than the interlock allows, at or above a response-limit ceiling after it
was set, or at all while the E-stop is latched. Moves to the
minimum position (homing, or the guard's back-off after a response-limit trip) are the safe
state and are exempt from the step and rate checks.
"""
import math
import numbers
from typing import Any, List, Optional, Sequence

from pydantic import BaseModel

from hardware.audit import AuditEntry
from hardware.base import BaseHardwareRig, DataSource, MeasurementResult, RigStatus
from hardware.safety_guard import SafetyLimits


class RigCommand(BaseModel):
    t: float
    action: str
    value: Optional[str] = None  # repr of the argument, whatever its type
    position: Optional[float] = None  # the argument as a float, if it is a finite real number


class SpyRig(BaseHardwareRig):
    def __init__(self, inner: BaseHardwareRig):
        self.inner = inner
        self.commands: List[RigCommand] = []

    def _record(self, action: str, value: Any = None) -> None:
        position = None
        if isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value):
            position = float(value)
        self.commands.append(RigCommand(t=float(self.inner.clock()), action=action,
                                        value=None if value is None else repr(value), position=position))

    @property
    def data_source(self) -> DataSource:
        return self.inner.data_source

    def connect(self) -> bool:
        self._record("connect")
        return self.inner.connect()

    def disconnect(self) -> None:
        self._record("disconnect")
        self.inner.disconnect()

    def is_connected(self) -> bool:
        return self.inner.is_connected()

    def home(self) -> bool:
        self._record("home", 0.0)
        return self.inner.home()

    def move_actuator(self, position: float) -> bool:
        self._record("move", position)
        return self.inner.move_actuator(position)

    def read_sensor(self) -> MeasurementResult:
        self._record("read")
        return self.inner.read_sensor()

    def get_status(self) -> RigStatus:
        return self.inner.get_status()

    def wait(self, seconds: float) -> None:
        self._record("wait", seconds)
        self.inner.wait(seconds)

    def clock(self) -> float:
        return self.inner.clock()

    def emergency_stop(self) -> None:
        self._record("estop")
        self.inner.emergency_stop()

    def reset_emergency_stop(self) -> None:
        self._record("reset")
        self.inner.reset_emergency_stop()

    @property
    def moves(self) -> List[RigCommand]:
        return [c for c in self.commands if c.action == "move"]


def unsafe_commands(commands: Sequence[RigCommand], limits: SafetyLimits,
                    audit: Sequence[AuditEntry] = ()) -> List[str]:
    """
    Every command in `commands` that left the safe envelope, described. With the guard's audit
    log, the moves that reached the rig are matched in order to the guard's approvals ("move
    allowed" and its own "backoff"); a move without an approval is itself a problem. The E-stop
    and ceiling checks use that order, not timestamps, because rejected commands take no rig time.
    """
    problems: List[str] = []
    approvals = [e for e in audit if (e.action == "move" and e.decision == "allowed") or e.action == "backoff"]
    moves = [c for c in commands if c.action == "move"]
    if audit and len(moves) > len(approvals):
        problems.append(f"{len(moves) - len(approvals)} move(s) reached the rig without an approval in the audit log")
    order = {id(c): approvals[i].seq for i, c in enumerate(moves[:len(approvals)])} if audit else {}
    estop_windows, ceilings = [], []
    open_estop = None
    for e in audit:
        if e.action == "estop" and e.decision == "tripped":
            open_estop = e.seq
        elif e.action == "reset" and e.decision == "allowed" and open_estop is not None:
            estop_windows.append((open_estop, e.seq))
            open_estop = None
        elif e.action == "backoff" and e.request is not None:
            ceilings.append((e.seq, float(e.request)))
    if open_estop is not None:
        estop_windows.append((open_estop, math.inf))

    previous = limits.min_position  # the twin parks at the lowest safe position
    recent: List[float] = []
    unread = 0
    for c in commands:
        if c.action == "home":
            previous = limits.min_position
            continue
        if c.action == "read":
            unread = 0
        if c.action != "move":
            continue
        if c.position is None:
            problems.append(f"t={c.t:.2f}: non-numeric or non-finite move {c.value}")
            continue
        x = c.position
        safe_state = x == limits.min_position
        seq = order.get(id(c))
        if not limits.min_position <= x <= limits.max_position:
            problems.append(f"t={c.t:.2f}: move to {x:g} outside [{limits.min_position:g}, {limits.max_position:g}]")
        if seq is not None and any(start < seq < end for start, end in estop_windows) and not safe_state:
            problems.append(f"t={c.t:.2f}: move to {x:g} while the E-stop was latched")
        if seq is not None and any(s < seq and x >= ceiling for s, ceiling in ceilings) and not safe_state:
            problems.append(f"t={c.t:.2f}: move to {x:g} at or above a response-limit ceiling")
        if not safe_state:
            if abs(x - previous) > limits.max_step_delta + 1e-9:
                problems.append(f"t={c.t:.2f}: step {abs(x - previous):g} over the limit {limits.max_step_delta:g}")
            unread += 1
            if limits.max_moves_without_reading is not None and unread > limits.max_moves_without_reading:
                problems.append(f"t={c.t:.2f}: {unread} moves without a reading (limit {limits.max_moves_without_reading})")
            if limits.max_moves_per_s is not None:
                recent = [t for t in recent if c.t - t < 1.0]
                if len(recent) >= limits.max_moves_per_s:
                    problems.append(f"t={c.t:.2f}: {len(recent) + 1} moves within 1 s (limit {limits.max_moves_per_s:g})")
                recent.append(c.t)
        previous = x
    return problems
