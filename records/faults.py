"""
Fault scoring: what became of each injected fault and each random outlier (digital twin only).

The twin's fault log is ground truth, like the true law: it is read after the run has stopped
and compared with what the safety guard flagged. Nothing here feeds back into a run.

For each fault, the affected readings are those the fault actually changed (a spike, a dropout,
or a reading taken while a stuck actuator was off target by more than the position tolerance).
- detected: at least one affected reading was flagged;
- missed: at least one affected reading entered the data unflagged;
- recovered: detected, not missed, and the run carried on afterwards (it measured again, or
  ended by converging or by its budget rather than aborting).
Status: not_reached (the run ended before the fault's cycle), no_effect (reached but no
reading was changed, e.g. a dropout during a pause or a stuck actuator asked to move less than
the tolerance), recovered, missed, or detected_not_recovered.
Anomalies at cycles where nothing was injected are counted by flag as `unexplained`: on a clean
run these are false alarms.
"""
from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

FaultStatus = Literal["not_reached", "no_effect", "recovered", "missed", "detected_not_recovered"]


class FaultOutcome(BaseModel):
    type: str = Field(description="light_spike, sensor_dropout, stuck_actuator, or outlier (a non-ideality)")
    scheduled: bool = Field(description="True for a fault from the schedule, False for a random outlier")
    index: Optional[int] = Field(default=None, description="Position in the fault schedule")
    at_cycle: int
    duration: int = 1
    status: FaultStatus
    detected: bool = False
    recovered: bool = False
    missed: bool = False
    affected_steps: List[int] = Field(default_factory=list, description="Attempt indices of the readings it changed")
    flags: List[str] = Field(default_factory=list, description="Flags raised on those readings")
    resolution: Optional[str] = Field(default=None, description="Action after the last affected reading")


class FaultSummary(BaseModel):
    scheduled: int = 0
    reached: int = 0
    detected: int = 0
    recovered: int = 0
    missed: int = 0
    no_effect: int = 0
    outliers: int = 0
    outliers_detected: int = 0
    outliers_missed: int = 0
    unexplained: Dict[str, int] = Field(default_factory=dict, description="Anomaly flags at cycles with nothing injected")


def _score(kind: str, scheduled: bool, index: Optional[int], at_cycle: int, duration: int, cycles: Sequence[int],
           reached: bool, steps_by_cycle: Dict[int, Any], steps: Sequence[Any], run_ok: bool) -> FaultOutcome:
    base = dict(type=kind, scheduled=scheduled, index=index, at_cycle=at_cycle, duration=duration)
    if not reached:
        return FaultOutcome(status="not_reached", **base)
    affected = [steps_by_cycle[c] for c in cycles if c in steps_by_cycle]
    if not affected:
        return FaultOutcome(status="no_effect", **base)
    detected = any(s.status == "anomaly" for s in affected)
    missed = any(s.status == "measured" for s in affected)
    last = max(s.index for s in affected)
    carried_on = run_ok or any(s.status == "measured" and s.index > last for s in steps)
    recovered = detected and not missed and carried_on
    status: FaultStatus = "missed" if missed else "recovered" if recovered else "detected_not_recovered"
    final = max(affected, key=lambda s: s.index)
    return FaultOutcome(
        status=status, detected=detected, recovered=recovered, missed=missed,
        affected_steps=sorted(s.index for s in affected),
        flags=sorted({f for s in affected for f in s.flags}),
        resolution=final.anomaly.action if final.anomaly is not None else None,
        **base,
    )


def score_faults(steps: Sequence[Any], fault_log: Sequence[Any], schedule: Sequence[Any], last_cycle: int,
                 run_ok: bool) -> "tuple[List[FaultOutcome], FaultSummary]":
    """
    `steps`: the run record's steps; `fault_log`: the twin's FaultEvents; `schedule`: the configured
    FaultSpecs; `last_cycle`: the rig's final cycle count; `run_ok`: the run converged or used its budget.
    """
    steps_by_cycle = {s.rig_cycle: s for s in steps if s.rig_cycle is not None}
    outcomes: List[FaultOutcome] = []
    for i, f in enumerate(schedule):
        cycles = [e.cycle for e in fault_log if e.fault_index == i and e.read]
        outcomes.append(_score(f.type, True, i, f.at_step, f.duration, cycles, last_cycle >= f.at_step,
                               steps_by_cycle, steps, run_ok))
    for e in fault_log:
        if e.fault_index is None:
            outcomes.append(_score(e.type, False, None, e.cycle, 1, [e.cycle], True, steps_by_cycle, steps, run_ok))

    injected_cycles = {e.cycle for e in fault_log if e.read}
    unexplained = Counter(
        flag for s in steps if s.status == "anomaly" and s.rig_cycle not in injected_cycles for flag in s.flags
    )
    sched = [o for o in outcomes if o.scheduled]
    outliers = [o for o in outcomes if not o.scheduled]
    summary = FaultSummary(
        scheduled=len(sched),
        reached=sum(o.status != "not_reached" for o in sched),
        detected=sum(o.detected for o in sched),
        recovered=sum(o.recovered for o in sched),
        missed=sum(o.missed for o in sched),
        no_effect=sum(o.status == "no_effect" for o in sched),
        outliers=len(outliers),
        outliers_detected=sum(o.detected for o in outliers),
        outliers_missed=sum(o.missed for o in outliers),
        unexplained=dict(unexplained),
    )
    return outcomes, summary
