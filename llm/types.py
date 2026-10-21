"""
Records of LLM calls, saved in the run record so every prompt and answer can be audited.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

AttemptOutcome = Literal[
    "ok", "cached", "auth", "credit", "rate_limit", "timeout", "server", "not_found",
    "bad_request", "connection", "invalid_json", "missing_key", "error",
]


class ModelAttempt(BaseModel):
    """One request (or cache lookup) to one model."""
    model: str
    outcome: AttemptOutcome
    error: Optional[str] = Field(default=None, description="Redacted provider error or validation error")
    latency_s: float = 0.0


class LLMCallRecord(BaseModel):
    """One complete_json call: the prompt, every model tried, and the answer that was used."""
    id: int = Field(description="Position in RunRecord.llm_calls")
    agent: str = Field(description="planner, analyst or writer")
    purpose: str = Field(description="What the call was for, e.g. plan, strategy, review, stop_rationale, discussion")
    step: Optional[int] = Field(default=None, description="Attempt index the call belongs to, if any")
    provider: Literal["litellm", "mock"]
    prompt_version: str = ""
    schema_name: str
    model: Optional[str] = Field(default=None, description="The model whose answer was used; None if every model failed")
    attempts: List[ModelAttempt] = Field(default_factory=list)
    system: str
    user: str
    raw_response: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = Field(default_factory=list, description="Invalid answers that were fed back")
    json_retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = Field(default=0.0, description="Estimated (litellm price map); 0 for cache hits and mock; None if unknown")
    latency_s: float = 0.0
    cached: bool = False
    ok: bool = True
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LLMUsage(BaseModel):
    """Totals over a run's LLM calls."""
    calls: int = 0
    failed_calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_unknown_calls: int = Field(default=0, description="Calls whose cost litellm could not estimate")
    calls_per_model: Dict[str, int] = Field(default_factory=dict, description="Which model answered, and how often")

    @classmethod
    def from_calls(cls, calls: Sequence[LLMCallRecord]) -> "LLMUsage":
        usage = cls()
        for c in calls:
            usage.calls += 1
            usage.failed_calls += not c.ok
            usage.cached_calls += c.cached
            usage.input_tokens += c.input_tokens
            usage.output_tokens += c.output_tokens
            if c.cost_usd is None:
                usage.cost_unknown_calls += 1
            else:
                usage.cost_usd += c.cost_usd
            if c.model is not None:
                usage.calls_per_model[c.model] = usage.calls_per_model.get(c.model, 0) + 1
        return usage
