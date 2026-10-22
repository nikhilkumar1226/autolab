"""
Disk cache for LLM answers (llm_cache/, git-ignored).

An answer is cached only after it passed schema validation, keyed by everything that
determines it: model, messages, response schema, temperature, token cap and output
mode. Rerunning a seeded run therefore replays the same answers at no cost.
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CachedAnswer(BaseModel):
    model: str
    raw: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None
    created_at: datetime


def cache_key(model: str, messages: List[Dict[str, str]], schema: Dict[str, Any],
              temperature: float, max_tokens: int, mode: str) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "schema": schema, "temperature": temperature,
         "max_tokens": max_tokens, "mode": mode},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[CachedAnswer]:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return CachedAnswer.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            logger.warning("Ignoring unreadable LLM cache entry %s", path)
            return None

    def put(self, key: str, model: str, raw: str, input_tokens: int, output_tokens: int,
            cost_usd: Optional[float]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = CachedAnswer(
            model=model, raw=raw, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd, created_at=datetime.now(timezone.utc),
        )
        path.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
