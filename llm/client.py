"""
LLM client: the single entry point, complete_json, through which every agent asks a model.

Providers (LLMConfig.provider, env LLM_PROVIDER):
- "litellm": a real model through litellm. The chain is LLM_MODEL then LLM_FALLBACKS.
  Each model gets JSON-schema (or JSON) mode and a timeout. An invalid answer is re-asked
  up to max_json_retries times with the validation error fed back. HTTP 429 is retried with
  jittered exponential backoff. On auth errors, "credit balance too low", rate limits,
  timeouts, 5xx, connection errors, unknown models or persistently invalid JSON the next
  model is tried. If every model fails, LLMUnavailableError is raised: a real run never
  falls back to the mock. Validated answers are cached on disk.
- "mock": deterministic and offline. The calling agent supplies the policy; it receives the
  same JSON payload the real model would see, and its answer goes through the same
  validation as a real one.
- "none": no LLM at all. complete_json raises NotAvailable and agents use their heuristics
  (the ablation baseline).

Every call is reported to `on_call` as an LLMCallRecord (prompt, raw answer, parsed answer,
models tried, tokens, estimated cost, latency) so the run record holds all of it.
"""
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError

from config import LLMConfig
from llm.cache import DiskCache, cache_key
from llm.types import AttemptOutcome, LLMCallRecord, ModelAttempt

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
MockPolicy = Callable[[Dict[str, Any]], Any]
CallSink = Callable[[LLMCallRecord], None]

FEEDBACK = (
    "Your previous reply was not valid: {error}. Reply again with only a JSON object "
    "that has exactly the required fields."
)


class NotAvailable(RuntimeError):
    """Provider 'none': there is no LLM; the agent must use its heuristic."""


class LLMUnavailableError(RuntimeError):
    """Every model in the chain failed (or none can be called). Never caught by agents."""

    def __init__(self, message: str, call: Optional[LLMCallRecord] = None):
        super().__init__(message)
        self.call = call


_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{6,}|AIza[0-9A-Za-z_\-]{10,}|gsk_[A-Za-z0-9]{10,})")


def redact(text: str, limit: int = 400) -> str:
    """Remove anything that looks like an API key, and shorten."""
    text = _SECRET.sub("[REDACTED]", str(text)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def extract_json(raw: str) -> Any:
    """Parse a model answer as JSON, tolerating code fences and text around the object."""
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start:end + 1])
        raise


def _describe(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'answer'}: {e['msg']}" for e in exc.errors()[:6]
        )
    if isinstance(exc, json.JSONDecodeError):
        return f"not valid JSON ({exc.msg} at character {exc.pos})"
    return f"{type(exc).__name__}: {exc}"


def _litellm():
    """Import litellm on first use: the import is slow, and offline runs never need it."""
    # The bundled price map is enough for cost estimates; never fetch it over the network.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm
    litellm.suppress_debug_info = True
    return litellm


def missing_keys(model: str) -> List[str]:
    """API-key variables the model needs that are unset or empty (litellm counts empty as set)."""
    empty = {k: v for k, v in os.environ.items() if k.endswith("_API_KEY") and not v.strip()}
    for k in empty:
        del os.environ[k]
    try:
        return list(_litellm().validate_environment(model=model).get("missing_keys", []))
    except Exception as exc:  # unknown provider prefix: let the request itself fail and be recorded
        logger.debug("validate_environment(%s) failed: %s", model, exc)
        return []
    finally:
        os.environ.update(empty)


def classify(exc: Exception) -> AttemptOutcome:
    """Map a provider error to the outcome recorded in the run record."""
    litellm = _litellm()
    message = str(exc).lower()
    if "credit balance" in message or isinstance(exc, litellm.exceptions.PaymentRequiredError):
        return "credit"
    for cls, outcome in [
        (litellm.RateLimitError, "rate_limit"),
        (litellm.AuthenticationError, "auth"),
        (litellm.PermissionDeniedError, "auth"),
        (litellm.Timeout, "timeout"),
        (litellm.NotFoundError, "not_found"),
        (litellm.BadRequestError, "bad_request"),
        (litellm.APIConnectionError, "connection"),
        (litellm.InternalServerError, "server"),
        (litellm.ServiceUnavailableError, "server"),
        (litellm.BadGatewayError, "server"),
    ]:
        if isinstance(exc, cls):
            return outcome
    status = getattr(exc, "status_code", None)
    return "server" if isinstance(status, int) and status >= 500 else "error"


def _retry_after(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "litellm_response_headers", None) or {}
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


class LLMClient:
    def __init__(
        self,
        cfg: LLMConfig,
        rng: Optional[np.random.Generator] = None,
        on_call: Optional[CallSink] = None,
        sleep: Callable[[float], None] = time.sleep,
        completion: Optional[Callable[..., Any]] = None,
    ):
        self.cfg = cfg
        # A stand-in for litellm.completion (the red-team harness uses a scripted adversary). It
        # replaces only the network request; parsing, validation, retries and the chain are unchanged.
        self._completion = completion
        # Only used for backoff jitter; the engine passes a Generator derived from the run seed.
        self._rng = rng if rng is not None else np.random.default_rng()
        self._on_call = on_call
        self._sleep = sleep
        self._cache = DiskCache(cfg.cache_dir) if cfg.cache else None
        self.calls: List[LLMCallRecord] = []

    @property
    def provider(self) -> str:
        return self.cfg.provider

    def describe(self) -> str:
        """Short label for displays and reports."""
        if self.provider == "litellm":
            return "litellm: " + " -> ".join(self.cfg.chain)
        if self.provider == "mock":
            return "mock (deterministic offline stand-in, not a language model)"
        return "none (no LLM; heuristic baseline)"

    def assert_ready(self) -> None:
        """Fail fast, before any hardware moves, if no model in the chain can be called."""
        if self.provider != "litellm" or self._completion is not None:
            return
        problems = {m: missing_keys(m) for m in self.cfg.chain}
        if all(problems.values()):
            details = "; ".join(f"{m} needs {', '.join(keys)}" for m, keys in problems.items())
            raise LLMUnavailableError(
                f"No model in the LLM chain can be called ({details}). Put the keys in .env "
                f"(see .env.example), or run with --provider mock or --provider none."
            )

    # ------------------------------------------------------------------
    def complete_json(
        self,
        system: str,
        user: str,
        schema: Type[T],
        *,
        agent: str,
        purpose: str,
        step: Optional[int] = None,
        mock: Optional[MockPolicy] = None,
        prompt_version: str = "",
    ) -> T:
        """Ask the configured provider and return its answer validated against `schema`."""
        if self.provider == "none":
            raise NotAvailable(f"provider 'none': no LLM for {agent}/{purpose}")

        record = LLMCallRecord(
            id=len(self.calls), agent=agent, purpose=purpose, step=step,
            provider=self.provider, prompt_version=prompt_version, schema_name=schema.__name__,
            system=system, user=user,
        )
        self.calls.append(record)
        t0 = time.perf_counter()
        try:
            if self.provider == "mock":
                return self._mock(record, user, schema, mock)
            return self._chain(record, system, user, schema)
        except LLMUnavailableError as exc:
            record.ok, record.error = False, str(exc)
            exc.call = record
            raise
        except Exception as exc:  # e.g. a mock policy returning an invalid answer: a bug, recorded and raised
            record.ok, record.error = False, redact(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record.latency_s = time.perf_counter() - t0
            if self._on_call is not None:
                self._on_call(record)

    # ------------------------------------------------------------------
    def _mock(self, record: LLMCallRecord, user: str, schema: Type[T], mock: Optional[MockPolicy]) -> T:
        if mock is None:
            raise ValueError(f"no mock policy supplied for {record.agent}/{record.purpose}")
        answer = mock(json.loads(user))
        raw = answer.model_dump_json() if isinstance(answer, BaseModel) else json.dumps(answer)
        parsed = schema.model_validate(extract_json(raw))  # a mock bug fails loudly, not retried
        record.model = "mock"
        record.attempts.append(ModelAttempt(model="mock", outcome="ok"))
        record.raw_response = raw
        record.parsed = parsed.model_dump(mode="json")
        return parsed

    def _mode(self, model: str) -> str:
        if self.cfg.structured_output != "auto":
            return self.cfg.structured_output
        try:
            return "json_schema" if _litellm().supports_response_schema(model=model) else "json_object"
        except Exception:
            return "json_object"

    def _chain(self, record: LLMCallRecord, system: str, user: str, schema: Type[T]) -> T:
        base = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        schema_json = schema.model_json_schema()
        chain = self.cfg.chain
        keys = {m: cache_key(m, base, schema_json, self.cfg.temperature, self.cfg.max_output_tokens, self._mode(m))
                for m in chain}

        # 1. Cache, in priority order: a rerun replays the answer that was used last time.
        if self._cache is not None:
            for model in chain:
                hit = self._cache.get(keys[model])
                if hit is None:
                    continue
                try:
                    parsed = schema.model_validate(extract_json(hit.raw))
                except (ValueError, ValidationError):
                    continue
                record.attempts.append(ModelAttempt(model=model, outcome="cached"))
                record.model, record.cached = model, True
                record.raw_response, record.parsed = hit.raw, parsed.model_dump(mode="json")
                record.input_tokens, record.output_tokens, record.cost_usd = hit.input_tokens, hit.output_tokens, 0.0
                return parsed

        # 2. The network, model by model.
        record.cost_usd = 0.0
        for model in chain:
            missing = missing_keys(model) if self._completion is None else []
            if missing:
                record.attempts.append(ModelAttempt(
                    model=model, outcome="missing_key", error=f"missing {', '.join(missing)}; no request sent",
                ))
                continue
            parsed = self._try_model(record, model, base, schema, schema_json)
            if parsed is not None:
                if self._cache is not None:
                    self._cache.put(keys[model], model, record.raw_response, record.input_tokens,
                                    record.output_tokens, record.cost_usd)
                return parsed

        summary = "; ".join(f"{a.model}: {a.outcome}" + (f" ({a.error})" if a.error else "")
                            for a in record.attempts)
        raise LLMUnavailableError(f"every model in the LLM chain failed for {record.agent}/{record.purpose}: {summary}")

    def _try_model(self, record: LLMCallRecord, model: str, base: List[Dict[str, str]],
                   schema: Type[T], schema_json: Dict[str, Any]) -> Optional[T]:
        """All attempts at one model. Returns the validated answer, or None to move to the next model."""
        litellm = _litellm()
        import openai  # installed with litellm; every litellm provider error derives from openai.APIError

        mode = self._mode(model)
        response_format = (
            {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema_json, "strict": False}}
            if mode == "json_schema" else {"type": "json_object"}
        )
        messages = list(base)
        rate_retries = json_retries = 0
        while True:
            t0 = time.perf_counter()
            try:
                response = (self._completion or litellm.completion)(
                    model=model, messages=messages, temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_output_tokens, timeout=self.cfg.timeout_s,
                    response_format=response_format, num_retries=0,
                )
            except openai.APIError as exc:
                latency = time.perf_counter() - t0
                outcome = classify(exc)
                error = redact(f"{type(exc).__name__}: {exc}")
                if outcome == "rate_limit" and rate_retries < self.cfg.rate_limit_retries:
                    delay = self._backoff(rate_retries, exc)
                    record.attempts.append(ModelAttempt(
                        model=model, outcome=outcome, error=f"{error} (retrying in {delay:.1f} s)", latency_s=latency,
                    ))
                    rate_retries += 1
                    self._sleep(delay)
                    continue
                logger.warning("LLM %s failed for %s/%s: %s", model, record.agent, record.purpose, error)
                record.attempts.append(ModelAttempt(model=model, outcome=outcome, error=error, latency_s=latency))
                return None
            latency = time.perf_counter() - t0
            self._account(record, model, response)

            raw = (response.choices[0].message.content or "") if response.choices else ""
            try:
                parsed = schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                error = _describe(exc)
                record.validation_errors.append(f"{model}: {error}")
                record.attempts.append(ModelAttempt(
                    model=model, outcome="invalid_json", error=redact(f"{error} | answer began: {raw[:160]!r}"),
                    latency_s=latency,
                ))
                if json_retries < self.cfg.max_json_retries:
                    json_retries += 1
                    record.json_retries += 1
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": FEEDBACK.format(error=error)},
                    ]
                    continue
                return None

            record.attempts.append(ModelAttempt(model=model, outcome="ok", latency_s=latency))
            record.model, record.raw_response = model, raw
            record.parsed = parsed.model_dump(mode="json")
            return parsed

    def _account(self, record: LLMCallRecord, model: str, response: Any) -> None:
        """Add a response's tokens and estimated cost (every answered request is billed)."""
        usage = getattr(response, "usage", None)
        record.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        record.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        try:
            cost = float(_litellm().completion_cost(completion_response=response, model=model))
        except Exception:
            cost = None
        if cost is None or record.cost_usd is None:
            record.cost_usd = None
        else:
            record.cost_usd += cost

    def _backoff(self, attempt: int, exc: Exception) -> float:
        delay = min(self.cfg.backoff_max_s, self.cfg.backoff_base_s * 2 ** attempt) * (0.5 + float(self._rng.random()))
        hint = _retry_after(exc)
        if hint is not None:
            delay = max(delay, hint)
        return min(delay, self.cfg.backoff_max_s)
