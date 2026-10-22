"""
Tests for the LLM client: providers, validation retries, the fallback chain, backoff,
the disk cache, redaction and `python -m llm.check`. Offline: litellm.completion is scripted.
"""
import io
import json

import numpy as np
import pytest
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console

from config import LLMConfig
from llm import check
from llm.client import LLMClient, LLMUnavailableError, NotAvailable, extract_json, redact
from llm.client import _litellm

SYSTEM = "Return the value and a label."
USER = json.dumps({"question": "what is two?"})
VALID = '{"value": 2, "label": "two"}'


class Answer(BaseModel):
    value: float
    label: str = Field(min_length=1)


def make_client(cfg: LLMConfig):
    calls, sleeps = [], []
    client = LLMClient(cfg, rng=np.random.default_rng(0), on_call=calls.append, sleep=sleeps.append)
    return client, calls, sleeps


def ask(client: LLMClient) -> Answer:
    return client.complete_json(SYSTEM, USER, Answer, agent="test", purpose="unit", prompt_version="t1")


def error(name: str, message: str = "boom"):
    litellm = _litellm()
    cls = getattr(litellm, name)
    if name in ("BadRequestError", "NotFoundError", "Timeout"):
        return cls(message, model="m", llm_provider="anthropic")
    return cls(message, llm_provider="anthropic", model="m")


# --- mock and none -----------------------------------------------------------

def test_mock_provider_answers_from_the_payload_and_logs_the_call(tmp_path):
    client, calls, _ = make_client(LLMConfig(provider="mock", cache_dir=tmp_path))
    seen = []

    def policy(payload):
        seen.append(payload)
        return {"value": 2.0, "label": "two"}

    answer = client.complete_json(SYSTEM, USER, Answer, agent="test", purpose="unit", mock=policy)
    assert answer == Answer(value=2.0, label="two")
    assert seen == [json.loads(USER)]  # the mock sees exactly what a real model would
    [call] = calls
    assert (call.provider, call.model, call.agent, call.purpose) == ("mock", "mock", "test", "unit")
    assert call.system == SYSTEM and call.user == USER and call.parsed == {"value": 2.0, "label": "two"}
    assert call.input_tokens == call.output_tokens == 0 and call.cost_usd == 0.0 and call.ok


def test_mock_answers_are_validated_like_real_ones(tmp_path):
    client, calls, _ = make_client(LLMConfig(provider="mock", cache_dir=tmp_path))
    with pytest.raises(ValidationError):
        client.complete_json(SYSTEM, USER, Answer, agent="test", purpose="unit", mock=lambda p: {"value": "x"})
    assert not calls[0].ok and "ValidationError" in calls[0].error


def test_none_provider_raises_not_available(tmp_path):
    client, calls, _ = make_client(LLMConfig(provider="none", cache_dir=tmp_path))
    with pytest.raises(NotAvailable):
        ask(client)
    assert calls == [] and client.describe().startswith("none")


# --- litellm: one model --------------------------------------------------------

def test_valid_answer_records_model_tokens_cost_and_request(fake_litellm, litellm_config):
    fake_litellm.queue(VALID)
    client, calls, _ = make_client(litellm_config())

    assert ask(client) == Answer(value=2, label="two")
    [call] = calls
    assert call.model == "anthropic/claude-haiku-5-5" and call.ok and not call.cached
    assert (call.input_tokens, call.output_tokens) == (100, 20)
    assert call.cost_usd == pytest.approx(100 * 1e-7 + 20 * 5e-7)  # litellm's local price map
    assert call.raw_response == VALID and call.parsed == {"value": 2.0, "label": "two"}
    assert call.prompt_version == "t1" and call.schema_name == "Answer"

    [request] = fake_litellm.calls
    assert request["messages"] == [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["schema"] == Answer.model_json_schema()
    assert request["timeout"] == 30.0 and request["num_retries"] == 0 and request["max_tokens"] == 1024


def test_code_fenced_json_is_accepted(fake_litellm, litellm_config):
    fake_litellm.queue(f"```json\n{VALID}\n```")
    client, _, _ = make_client(litellm_config())
    assert ask(client).label == "two"


def test_invalid_json_is_retried_with_the_validation_error_fed_back(fake_litellm, litellm_config):
    fake_litellm.queue("Sure! The answer is two.", '{"value": "two", "label": "two"}', VALID)
    client, calls, _ = make_client(litellm_config())

    assert ask(client).value == 2
    assert fake_litellm.models == ["anthropic/claude-haiku-5-5"] * 3
    second, third = fake_litellm.calls[1]["messages"], fake_litellm.calls[2]["messages"]
    assert second[2] == {"role": "assistant", "content": "Sure! The answer is two."}
    assert second[3]["role"] == "user" and "not valid JSON" in second[3]["content"]
    assert "value" in third[-1]["content"] and "valid number" in third[-1]["content"]
    [call] = calls
    assert call.json_retries == 2 and len(call.validation_errors) == 2
    assert [a.outcome for a in call.attempts] == ["invalid_json", "invalid_json", "ok"]
    assert (call.input_tokens, call.output_tokens) == (300, 60)  # every answered request is billed


def test_persistently_invalid_json_moves_to_the_next_model(fake_litellm, litellm_config):
    fake_litellm.queue("no", "still no", "never", VALID)
    client, calls, _ = make_client(litellm_config())

    assert ask(client).value == 2
    assert fake_litellm.models == ["anthropic/claude-haiku-5-5"] * 3 + ["gemini/gemini-3.6-flash"]
    assert calls[0].model == "gemini/gemini-3.6-flash"
    assert [a.outcome for a in calls[0].attempts] == ["invalid_json"] * 3 + ["ok"]


# --- litellm: fallback chain ---------------------------------------------------

@pytest.mark.parametrize("exc_name, message, outcome", [
    ("AuthenticationError", "invalid x-api-key", "auth"),
    ("BadRequestError", "Your credit balance is too low to access the Anthropic API.", "credit"),
    ("Timeout", "Request timed out", "timeout"),
    ("InternalServerError", "Overloaded", "server"),
    ("ServiceUnavailableError", "Service unavailable", "server"),
    ("NotFoundError", "model not found", "not_found"),
    ("APIConnectionError", "connection refused", "connection"),
])
def test_fallback_chain_moves_to_the_next_model(fake_litellm, litellm_config, exc_name, message, outcome):
    fake_litellm.queue(error(exc_name, message), VALID)
    client, calls, sleeps = make_client(litellm_config())

    assert ask(client).value == 2
    assert fake_litellm.models == ["anthropic/claude-haiku-5-5", "gemini/gemini-3.6-flash"]
    [call] = calls
    assert call.model == "gemini/gemini-3.6-flash"  # the model that actually answered is recorded
    assert [a.outcome for a in call.attempts] == [outcome, "ok"]
    assert message.split()[0] in call.attempts[0].error
    assert sleeps == []  # only rate limits are retried on the same model


def test_rate_limit_backs_off_with_jitter_then_falls_back(fake_litellm, litellm_config):
    fake_litellm.queue(*(error("RateLimitError", "429 Too Many Requests") for _ in range(3)), VALID)
    client, calls, sleeps = make_client(litellm_config())

    assert ask(client).value == 2
    assert fake_litellm.models == ["anthropic/claude-haiku-5-5"] * 3 + ["gemini/gemini-3.6-flash"]
    assert len(sleeps) == 2
    assert 1.0 <= sleeps[0] <= 3.0 and 2.0 <= sleeps[1] <= 6.0  # base 2 s x 2^i, jitter 50-150 %
    assert [a.outcome for a in calls[0].attempts] == ["rate_limit"] * 3 + ["ok"]


def test_backoff_jitter_is_reproducible_from_the_generator(fake_litellm, litellm_config):
    delays = []
    for _ in range(2):
        fake_litellm.queue(error("RateLimitError"), VALID)
        client, _, sleeps = make_client(litellm_config(cache=False))
        ask(client)
        delays.append(sleeps)
    assert delays[0] == delays[1]


def test_every_model_failing_raises_a_clear_error(fake_litellm, litellm_config):
    fake_litellm.queue(
        error("BadRequestError", "Your credit balance is too low"),
        error("RateLimitError"), error("RateLimitError"), error("RateLimitError"),
        error("Timeout", "Request timed out"),
    )
    client, calls, _ = make_client(litellm_config())

    with pytest.raises(LLMUnavailableError) as info:
        ask(client)
    message = str(info.value)
    for text in ["anthropic/claude-haiku-5-5: credit", "gemini/gemini-3.6-flash: rate_limit",
                 "gemini/gemini-3.5-flash-lite: timeout"]:
        assert text in message
    [call] = calls
    assert not call.ok and call.model is None and call.error == message
    assert info.value.call is call
    assert "mock" not in message and all(a.model != "mock" for a in call.attempts)


def test_missing_key_is_skipped_without_a_request(fake_litellm, litellm_config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # an empty key in .env counts as missing
    fake_litellm.queue(VALID)
    client, calls, _ = make_client(litellm_config())

    assert ask(client).value == 2
    assert fake_litellm.models == ["gemini/gemini-3.6-flash"]
    assert calls[0].attempts[0].outcome == "missing_key"
    assert "ANTHROPIC_API_KEY" in calls[0].attempts[0].error


def test_assert_ready_fails_fast_when_no_model_has_a_key(fake_litellm, litellm_config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client, _, _ = make_client(litellm_config())
    with pytest.raises(LLMUnavailableError, match="No model in the LLM chain can be called"):
        client.assert_ready()
    assert fake_litellm.calls == []


# --- cache ---------------------------------------------------------------------

def test_identical_prompt_is_served_from_the_cache(fake_litellm, litellm_config):
    cfg = litellm_config()
    fake_litellm.queue(VALID)
    first, _, _ = make_client(cfg)
    ask(first)

    second, calls, _ = make_client(cfg)
    assert ask(second) == Answer(value=2, label="two")
    assert len(fake_litellm.calls) == 1  # no second request
    [call] = calls
    assert call.cached and call.cost_usd == 0.0 and call.model == "anthropic/claude-haiku-5-5"
    assert [a.outcome for a in call.attempts] == ["cached"]


def test_cache_replays_the_fallback_answer_without_retrying_the_first_model(fake_litellm, litellm_config):
    cfg = litellm_config()
    fake_litellm.queue(error("AuthenticationError"), VALID)
    ask(make_client(cfg)[0])

    client, calls, _ = make_client(cfg)
    ask(client)
    assert len(fake_litellm.calls) == 2  # both from the first run
    assert calls[0].cached and calls[0].model == "gemini/gemini-3.6-flash"


def test_a_different_prompt_is_not_a_cache_hit(fake_litellm, litellm_config):
    cfg = litellm_config()
    fake_litellm.queue(VALID, VALID)
    client, _, _ = make_client(cfg)
    ask(client)
    client.complete_json(SYSTEM, json.dumps({"question": "what is three?"}), Answer, agent="test", purpose="unit")
    assert len(fake_litellm.calls) == 2


def test_cache_can_be_disabled(fake_litellm, litellm_config):
    cfg = litellm_config(cache=False)
    fake_litellm.queue(VALID, VALID)
    ask(make_client(cfg)[0])
    ask(make_client(cfg)[0])
    assert len(fake_litellm.calls) == 2 and not cfg.cache_dir.exists()


# --- helpers -----------------------------------------------------------------

def test_redact_hides_api_keys():
    text = redact("Incorrect API key sk-ant-api03-AbCdEf123456 or AIzaSyA1b2C3d4E5f6G7h8 or gsk_abcdefghij1234")
    assert "sk-ant" not in text and "AIza" not in text and "gsk_" not in text
    assert text.count("[REDACTED]") == 3


def test_extract_json_tolerates_fences_and_surrounding_text():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": 1} hope it helps') == {"a": 1}
    with pytest.raises(json.JSONDecodeError):
        extract_json("no json here")


def test_chain_has_no_duplicates():
    cfg = LLMConfig(provider="litellm", model="a/x", fallbacks=["b/y", "a/x", "c/z"])
    assert cfg.chain == ["a/x", "b/y", "c/z"]


# --- python -m llm.check -------------------------------------------------------

def test_llm_check_reports_each_model(fake_litellm, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-haiku-5-5")
    monkeypatch.setenv("LLM_FALLBACKS", "gemini/gemini-3.6-flash,gemini/gemini-3.5-flash-lite")
    fake_litellm.queue(error("AuthenticationError", "invalid x-api-key sk-ant-api03-SECRETSECRET"), '{"ok": true}',
                       error("RateLimitError", "quota exceeded"))
    out = io.StringIO()

    code = check.main([], console=Console(file=out, width=200))
    text = out.getvalue()
    assert code == 0
    assert "ERROR anthropic/claude-haiku-5-5  auth" in text
    assert "OK    gemini/gemini-3.6-flash" in text
    assert "ERROR gemini/gemini-3.5-flash-lite  rate_limit" in text
    assert "SECRETSECRET" not in text and "test-anthropic-key" not in text
    assert "first working model is gemini/gemini-3.6-flash" in text
    # The ping uses the agents' output-token cap, not a special tiny one.
    assert all(c["max_tokens"] == LLMConfig().max_output_tokens for c in fake_litellm.calls) and len(fake_litellm.calls) == 3


def test_llm_check_fails_when_no_model_works(fake_litellm, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    out = io.StringIO()
    fake_litellm.queue(error("Timeout", "Request timed out"))
    code = check.main(["anthropic/claude-haiku-5-5", "gemini/gemini-3.6-flash"], console=Console(file=out, width=200))
    text = out.getvalue()
    assert code == 1
    assert "missing_key: missing ANTHROPIC_API_KEY; no request sent" in text
    assert "ERROR gemini/gemini-3.6-flash  timeout" in text
    assert "Chain usable: no" in text


# --- Groq JSON mode, pacing and cool-downs (Phase 5.1) ---------------------------------

GROQ = "groq/openai/gpt-oss-120b"


def groq_json_failure():
    """What Groq sends when its JSON mode rejects the model's output: a 400 with an empty generation."""
    return _litellm().BadRequestError(
        message='GroqException - {"error":{"message":"Failed to validate JSON. Please adjust your prompt.",'
                '"type":"invalid_request_error","code":"json_validate_failed","failed_generation":""}}',
        model="openai/gpt-oss-120b", llm_provider="groq",
    )


def test_llm_check_pings_groq_like_the_agents_do(fake_litellm, monkeypatch):
    """A reasoning model given a tiny token cap returns nothing, and Groq reports json_validate_failed."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    fake_litellm.responder = lambda kw: groq_json_failure() if kw["max_tokens"] < 256 else {"ok": True}
    call = check.check_model(GROQ, LLMConfig())
    assert call.ok and call.model == GROQ
    [request] = fake_litellm.calls
    agent_like = LLMClient(LLMConfig(provider="litellm", model=GROQ, fallbacks=[]))
    assert request["max_tokens"] == LLMConfig().max_output_tokens
    assert request["response_format"]["type"] == ("json_schema" if agent_like._mode(GROQ) == "json_schema" else "json_object")


def test_a_json_validate_failed_from_groq_is_re_asked_then_reported(fake_litellm, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    fake_litellm.queue(groq_json_failure(), VALID)
    client, _, _ = make_client(LLMConfig(provider="litellm", model=GROQ, fallbacks=[], cache=False))
    out = client.complete_json("s", "u", Answer, agent="a", purpose="p")
    assert out.label == "two"
    record = client.calls[-1]
    assert [a.outcome for a in record.attempts] == ["invalid_json", "ok"] and record.json_retries == 1
    assert "json_validate_failed" in record.validation_errors[0]

    fake_litellm.script.clear()
    fake_litellm.responder = lambda kw: groq_json_failure()
    out_text = io.StringIO()
    code = check.main([GROQ], console=Console(file=out_text, width=250))
    text = out_text.getvalue()
    assert code == 1 and f"ERROR {GROQ}  invalid_json: the provider rejected the model's output in JSON mode" in text
    assert len(fake_litellm.calls) == 2 + 1 + LLMConfig().max_json_retries  # the check re-asks like a run


def test_requests_are_paced_across_clients(fake_litellm, monkeypatch):
    slept = []
    fake_litellm.responder = lambda kw: VALID
    cfg = LLMConfig(provider="litellm", model="anthropic/claude-haiku-5-5", fallbacks=[], cache=False,
                    min_request_interval_s=6.0)
    for _ in range(3):  # a new client per run, as in the benchmark: the pacing is process-wide
        client = LLMClient(cfg, sleep=slept.append)
        client.complete_json("s", "u", Answer, agent="a", purpose="p")
    assert len(slept) == 2 and all(5.0 < s <= 6.0 for s in slept)


def test_a_rate_limited_model_is_skipped_while_it_cools_off(fake_litellm, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    fake_litellm.responder = lambda kw: (error("RateLimitError", "quota exceeded for the day")
                                         if kw["model"].startswith("gemini") else VALID)
    cfg = LLMConfig(provider="litellm", model="gemini/gemini-3.5-flash-lite", fallbacks=[GROQ], cache=False,
                    rate_limit_retries=1)
    first, _, _ = make_client(cfg)
    first.complete_json("s", "u", Answer, agent="a", purpose="p")
    gemini_requests = sum(m.startswith("gemini") for m in fake_litellm.models)
    assert gemini_requests == 2  # one try and one backoff retry
    second, _, _ = make_client(cfg)
    second.complete_json("s", "u", Answer, agent="a", purpose="p")
    assert sum(m.startswith("gemini") for m in fake_litellm.models) == gemini_requests  # skipped, no request
    attempt = second.calls[-1].attempts[0]
    assert attempt.outcome == "rate_limit" and "cooling off" in attempt.error
    assert second.calls[-1].model == GROQ
