"""A cache for LLM calls, in SQLite, with a lifetime.

An identical request (same model, effort, prompt version, system text, prompt and schema) gets
the stored answer instead of a new call. Streaming is never cached. The key includes everything
that can change the answer, so changing the model or the reasoning setting cannot return a stale
one.
"""

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from llm import ToolTurn
from observability import current_trace_id  # noqa: F401  (kept for symmetry with other packages)

T = TypeVar("T", bound=BaseModel)


class LlmCache:
    def __init__(self, path: str, ttl_s: int, clock=time.time) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._ttl = ttl_s
        self._clock = clock
        self.hits = 0
        self.misses = 0
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                "expires_at REAL NOT NULL)"
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM llm_cache WHERE key=?", (key,)
            ).fetchone()
        if row is None or row[1] < self._clock():
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, value: Any) -> None:
        if self._ttl <= 0:
            return
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?,?,?)",
                (key, json.dumps(value), self._clock() + self._ttl),
            )

    def purge_expired(self) -> int:
        with self._lock, self._conn:
            return self._conn.execute(
                "DELETE FROM llm_cache WHERE expires_at < ?", (self._clock(),)
            ).rowcount

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]


def make_key(kind: str, namespace: str, **parts: Any) -> str:
    text = json.dumps({"kind": kind, "ns": namespace, **parts}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


class CachingLLM:
    """Wraps any `LLM`. `namespace` should change whenever something outside the request changes
    the answer: the model, the reasoning setting, the prompt version."""

    def __init__(self, inner, cache: LlmCache, namespace: "str | Callable[[], str]") -> None:
        self._inner = inner
        self._cache = cache
        self._ns = namespace if callable(namespace) else (lambda: namespace)

    def complete(self, prompt: str, *, system: str | None = None, max_output_tokens=None) -> str:
        key = make_key("complete", self._ns(), prompt=prompt, system=system)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = self._inner.complete(prompt, system=system, max_output_tokens=max_output_tokens)
        self._cache.put(key, out)
        return out

    def complete_json(
        self, prompt: str, schema: type[T], *, system: str | None = None, max_output_tokens=None
    ) -> T:
        key = make_key("json", self._ns(), prompt=prompt, system=system, schema=schema.__name__)
        hit = self._cache.get(key)
        if hit is not None:
            return schema.model_validate(hit)
        out = self._inner.complete_json(
            prompt, schema, system=system, max_output_tokens=max_output_tokens
        )
        self._cache.put(key, out.model_dump())
        return out

    def complete_with_tools(
        self, items, tools, *, system: str | None = None, max_output_tokens=None
    ) -> ToolTurn:
        key = make_key("tools", self._ns(), items=items, tools=tools, system=system)
        hit = self._cache.get(key)
        if hit is not None:
            return ToolTurn.model_validate(hit)
        out = self._inner.complete_with_tools(
            items, tools, system=system, max_output_tokens=max_output_tokens
        )
        self._cache.put(key, out.model_dump())
        return out

    def stream(self, prompt: str, *, system: str | None = None, max_output_tokens=None):
        return self._inner.stream(prompt, system=system, max_output_tokens=max_output_tokens)
