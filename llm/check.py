"""
Key and model check: `python -m llm.check`.

Sends one tiny JSON request to every model in the chain (LLM_MODEL, then LLM_FALLBACKS)
and prints OK or the error for each. The ping takes the same path as the agents: the
agents' output-token cap, structured-output mode, temperature, timeout and JSON re-asks.
An earlier version capped the ping at 32 output tokens; a reasoning model (gpt-oss on Groq)
spends those on reasoning, returns nothing, and Groq's JSON mode then reports
"json_validate_failed", although real runs work. Only three things differ from a run: the
chain is this one model, the cache is off, and a rate limit is reported at once instead of
waited out. Keys are never printed, only the names of missing ones. Exit code 0 if at least
one model answers.
"""
import argparse
import logging
import sys
from typing import List, Optional

from pydantic import BaseModel
from rich.console import Console
from rich.markup import escape

from config import LLMConfig
from llm.client import LLMClient, LLMUnavailableError
from llm.types import LLMCallRecord


class Ping(BaseModel):
    ok: bool


SYSTEM = 'You are a connectivity check. Reply with the JSON object {"ok": true} and nothing else.'
USER = '{"request": "ping"}'


def check_model(model: str, base: LLMConfig) -> LLMCallRecord:
    """One ping to one model through LLMClient.complete_json with the agents' settings (see the module docstring)."""
    cfg = base.model_copy(update={
        "provider": "litellm", "model": model, "fallbacks": [], "cache": False,
        "rate_limit_retries": 0, "rate_limit_cooldown_s": 0.0, "min_request_interval_s": 0.0,
    })
    client = LLMClient(cfg)
    try:
        client.complete_json(SYSTEM, USER, Ping, agent="check", purpose="ping")
    except LLMUnavailableError:
        pass
    return client.calls[-1]


def main(argv: Optional[List[str]] = None, console: Optional[Console] = None) -> int:
    parser = argparse.ArgumentParser(description="Send one tiny request to every configured LLM and report OK / error.")
    parser.add_argument("models", nargs="*", help="Models to check (default: LLM_MODEL then LLM_FALLBACKS from .env)")
    args = parser.parse_args(argv)
    console = console or Console()

    base = LLMConfig()
    models = args.models or base.chain
    console.print(f"Checking {len(models)} model(s), in priority order:")
    working = []
    for i, model in enumerate(models, 1):
        call = check_model(model, base)
        if call.ok:
            working.append(model)
            cost = "cost unknown" if call.cost_usd is None else f"${call.cost_usd:.6f}"
            console.print(
                f"  {i}. [green]OK   [/green] {escape(model)}  {call.latency_s:.2f} s, "
                f"{call.input_tokens} in / {call.output_tokens} out tokens, {cost}"
            )
        else:
            last = call.attempts[-1] if call.attempts else None
            reason = f"{last.outcome}: {last.error}" if last is not None else (call.error or "unknown error")
            console.print(f"  {i}. [red]ERROR[/red] {escape(model)}  {escape(reason)}")
    if working:
        console.print(f"Chain usable: [green]yes[/green]. The first working model is {escape(working[0])}.")
        return 0
    console.print("Chain usable: [red]no[/red]. Check the keys in .env (see .env.example), or use --provider mock.")
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)  # the per-model lines already say what failed
    sys.exit(main())
