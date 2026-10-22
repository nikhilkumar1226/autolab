"""
Append-only audit log of the safety guard: every request, the decision and the reason.

Entries can only be appended. Each entry stores the SHA-256 of the previous entry and its own
hash over its content, so a deleted, reordered or edited entry breaks the chain, and verify()
reports where. The log is saved in every run record.
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

GENESIS = "0" * 64

Decision = Literal["allowed", "rejected", "flagged", "tripped", "info"]


class AuditEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int = Field(description="0-based position in the log")
    t_rig: float = Field(description="Rig clock (simulated seconds on the twin, wall time on hardware)")
    timestamp: datetime
    action: str = Field(description="connect, home, move, read, pause, estop, reset, backoff, ...")
    request: Optional[str] = Field(default=None, description="The requested value, as repr() (any type)")
    decision: Decision
    reason: str = ""
    prev_hash: str
    hash: str

    def body(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude={"hash"})


def _digest(body: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only: there is no method to remove or change an entry; `entries` returns a copy."""

    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []

    def append(self, t_rig: float, action: str, decision: Decision, reason: str = "", request: Any = None) -> AuditEntry:
        prev = self._entries[-1].hash if self._entries else GENESIS
        body = {
            "seq": len(self._entries),
            "t_rig": float(t_rig),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "request": None if request is None else repr(request)[:200],
            "decision": decision,
            "reason": reason,
            "prev_hash": prev,
        }
        # Hash what pydantic stores (e.g. the normalised timestamp), so verify() sees the same body.
        entry = AuditEntry(**body, hash="")
        entry = entry.model_copy(update={"hash": _digest(entry.body())})
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> Tuple[AuditEntry, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


def verify_chain(entries: Sequence[AuditEntry]) -> Tuple[bool, str]:
    """(True, "") if the chain is intact, else (False, the first problem)."""
    prev = GENESIS
    for i, e in enumerate(entries):
        if e.seq != i:
            return False, f"entry {i} has seq {e.seq} (missing or reordered entries)"
        if e.prev_hash != prev:
            return False, f"entry {i} does not follow entry {i - 1}"
        if _digest(e.body()) != e.hash:
            return False, f"entry {i} was changed after it was written"
        prev = e.hash
    return True, ""
