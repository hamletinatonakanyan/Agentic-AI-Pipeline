"""Targets: anything that answers a question and returns contexts.

- `CallableTarget` wraps a Python function. e2e uses it for in-process presets, such as retrieval,
  so `evals` never imports `retrieval`.
- `HttpTarget` POSTs `{"question": ...}` to a URL and reads the target contract back.
"""

from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from common import AppError, Environment
from evals.models import Context, HttpTargetConfig, TargetResponse
from evals.protocols import Target
from evals.settings import EvalSettings


class TargetError(Exception):
    """One question could not be answered. The run records it and carries on."""


class CallableTarget:
    def __init__(self, fn: Callable[[str], TargetResponse]) -> None:
        self._fn = fn

    def ask(self, question: str) -> TargetResponse:
        try:
            return self._fn(question)
        except AppError as exc:
            raise TargetError(f"{exc.code}: {exc.message}") from exc


class HttpTarget:
    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        timeout_s: float,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = url
        self._headers = headers
        self._timeout = timeout_s
        self._http = client or httpx.Client(timeout=timeout_s)

    def ask(self, question: str) -> TargetResponse:
        try:
            response = self._http.post(
                self._url, json={"question": question}, headers=self._headers
            )
        except httpx.TimeoutException as exc:
            raise TargetError(f"the target did not answer within {self._timeout:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise TargetError(f"cannot reach the target ({type(exc).__name__})") from exc
        if response.status_code >= 400:
            raise TargetError(f"the target returned HTTP {response.status_code}")
        try:
            return TargetResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise TargetError("the target's reply does not follow the contract") from exc


def build_http_target(
    config: HttpTargetConfig,
    settings: EvalSettings,
    client: httpx.Client | None = None,
    env: dict[str, str] | None = None,
) -> Target:
    """Checks the URL host and the header references against the allow-lists before anything
    is sent. Headers come from environment variables by NAME, and only allow-listed names."""
    parsed = urlparse(config.url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise AppError("invalid_target", "The target URL must be http or https.", 422)
    if parsed.hostname not in settings.allowed_target_hosts:
        raise AppError(
            "target_not_allowed",
            f"Host '{parsed.hostname}' is not in EVAL_ALLOWED_TARGET_HOSTS "
            f"({', '.join(settings.allowed_target_hosts) or 'empty'}).",
            403,
        )
    environment = Environment() if env is None else env
    headers: dict[str, str] = {}
    for header, ref in config.headers_refs.items():
        if ref not in settings.allowed_header_refs:
            raise AppError(
                "target_not_allowed",
                f"'{ref}' is not in EVAL_ALLOWED_HEADER_REFS, so it cannot be sent as a header.",
                403,
            )
        value = environment.get(ref)
        if not value:
            raise AppError("invalid_target", f"The variable {ref} is not set.", 422)
        headers[header] = value
    return HttpTarget(
        config.url, headers, config.timeout_s or settings.target_timeout_s, client=client
    )


def describe(config: Any) -> str:
    """A short label for lists and the config snapshot. Never includes header values."""
    if config.kind == "http":
        return f"http {urlparse(config.url).hostname}{urlparse(config.url).path}"
    options = ", ".join(
        f"{k}={v}" for k, v in sorted(config.options.items()) if v not in (None, "")
    )
    return f"{config.name}" + (f" ({options})" if options else "")


# ----- a ready-made adapter for the retrieval service -------------------------------

DEFAULT_RETRIEVAL_URL = "http://localhost:8002/api/retrieval/retrieve"
_RETRIEVAL_OPTIONS = (
    "collection", "mode", "top_k", "candidates", "filters", "rewrite", "rerank", "rerank_model",
)  # fmt: skip


class RetrievalHttpTarget:
    """Calls a retrieval service's `/retrieve` and maps the hits to the target contract.
    Retrieval returns no answer, so only retrieval metrics are computed for it."""

    def __init__(self, inner: HttpTarget, url: str, options: dict[str, Any]) -> None:
        self._http = inner._http
        self._headers = inner._headers
        self._timeout = inner._timeout
        self._url = url
        self._options = options

    def ask(self, question: str) -> TargetResponse:
        try:
            response = self._http.post(
                self._url,
                json={"query": question, **self._options},
                headers=self._headers,
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise TargetError(f"retrieval did not answer within {self._timeout:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise TargetError(f"cannot reach retrieval ({type(exc).__name__})") from exc
        if response.status_code >= 400:
            try:
                detail = response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                detail = f"HTTP {response.status_code}"
            raise TargetError(f"retrieval failed: {detail}"[:300])
        try:
            body = response.json()
            hits = body["hits"]
            return TargetResponse(
                contexts=[Context(chunk_id=h["chunk_id"], text=h["text"]) for h in hits],
                retrieved_chunk_ids=[h["chunk_id"] for h in hits],
                timings={"retrieval_ms": float(body["timings"]["total_ms"])},
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise TargetError("retrieval's reply does not look like /retrieve") from exc


def build_retrieval_http_target(
    options: dict[str, Any],
    settings: EvalSettings,
    client: httpx.Client | None = None,
    env: dict[str, str] | None = None,
) -> Target:
    unknown = set(options) - set(_RETRIEVAL_OPTIONS) - {"url"}
    if unknown:
        raise AppError(
            "invalid_target", f"Unknown retrieval options: {', '.join(sorted(unknown))}.", 422
        )
    url = str(options.get("url") or DEFAULT_RETRIEVAL_URL)
    config = HttpTargetConfig(kind="http", url=url, headers_refs={"X-API-Key": "APP_API_KEY"})
    inner = build_http_target(config, settings, client=client, env=env)
    return RetrievalHttpTarget(
        inner, url, {k: options[k] for k in _RETRIEVAL_OPTIONS if k in options}
    )
