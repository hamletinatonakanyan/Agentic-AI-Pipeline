"""The tracing API that other packages import.

    with span("ingest.file", {"source": name}):
        with span("chunk"):          # becomes a child automatically
            ...

    @traced("tool.call")
    def call(...): ...

Safe by default: until `init_observability()` runs, or when `OBS_ENABLED=false`, every call
here is a cheap no-op, so packages work with observability off.
"""

import functools
import inspect
import logging
import random
import secrets
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from observability import attrs
from observability.models import SpanData, SpanEvent, SpanStatus
from observability.protocols import SpanExporter, TraceStore
from observability.redact import Redactor
from observability.settings import ObsSettings
from observability.store import SqliteTraceStore

_log = logging.getLogger("observability")


class Span:
    """A span that is still open. Recording spans collect data. The no-op span ignores it."""

    def __init__(
        self,
        name: str = "",
        trace_id: str = "",
        span_id: str = "",
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        recording: bool = True,
    ) -> None:
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.recording = recording
        self.attributes: dict[str, Any] = dict(attributes or {})
        self.events: list[SpanEvent] = []
        self.status: SpanStatus = "unset"
        self.status_message = ""
        self._start_ts = time.time()
        self._t0 = time.perf_counter()

    def set_attribute(self, key: str, value: Any) -> None:
        if self.recording:
            self.attributes[key] = value

    def set_attributes(self, values: dict[str, Any]) -> None:
        if self.recording:
            self.attributes.update(values)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        if self.recording:
            self.events.append(SpanEvent(name=name, ts=time.time(), attributes=attributes or {}))

    def set_status(self, status: SpanStatus, message: str = "") -> None:
        if self.recording:
            self.status = status
            self.status_message = message[:500]

    def record_exception(self, exc: BaseException) -> None:
        if not self.recording:
            return
        message = f"{type(exc).__name__}: {exc}"
        self.set_status("error", message)
        self.add_event("exception", {"type": type(exc).__name__, "message": str(exc)[:500]})

    def _finish(self, service: str) -> SpanData:
        duration_ms = (time.perf_counter() - self._t0) * 1000
        return SpanData(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_id=self.parent_id,
            name=self.name,
            service=service,
            start_ts=self._start_ts,
            end_ts=self._start_ts + duration_ms / 1000,
            duration_ms=duration_ms,
            status="ok" if self.status == "unset" else self.status,
            status_message=self.status_message,
            attributes=self.attributes,
            events=self.events,
        )


# Shared "do nothing" span. Also the marker for "this trace was not sampled".
NOOP_SPAN = Span(recording=False)

_current: ContextVar[Span | None] = ContextVar("observability_current_span", default=None)


@dataclass
class Observability:
    """What `init_observability` returns: the live configuration."""

    settings: ObsSettings
    exporters: list[SpanExporter] = field(default_factory=list)
    store: TraceStore | None = None
    redactor: Redactor = field(default_factory=lambda: Redactor([]))
    rng: random.Random = field(default_factory=random.Random)

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def shutdown(self) -> None:
        for exporter in self.exporters:
            try:
                exporter.shutdown()
            except Exception:
                _log.warning("exporter shutdown failed", exc_info=True)


_state: Observability | None = None


def init_observability(
    settings: ObsSettings | None = None,
    *,
    exporters: list[SpanExporter] | None = None,
    force: bool = False,
) -> Observability:
    """Idempotent: a second call returns the first result. `force=True` rebuilds (tests).

    `exporters` replaces the ones named in settings. Tests pass an `InMemoryExporter`.
    """
    global _state
    if _state is not None and not force:
        return _state
    if _state is not None:
        _state.shutdown()

    settings = settings or ObsSettings()
    handle = Observability(settings=settings, redactor=Redactor(settings.redact_patterns))
    if settings.enabled:
        if exporters is not None:
            handle.exporters = list(exporters)
        else:
            handle.exporters = _build_exporters(settings, handle)
    _state = handle
    return handle


def _build_exporters(settings: ObsSettings, handle: Observability) -> list[SpanExporter]:
    from observability.exporters import ConsoleExporter

    built: list[SpanExporter] = []
    for name in settings.exporter_names:
        if name == "local":
            store = SqliteTraceStore(settings.local_db_path)
            handle.store = store
            built.append(store)
        elif name == "console":
            built.append(ConsoleExporter())
    return built


def shutdown_observability() -> None:
    """Close exporters and go back to the no-op state."""
    global _state
    if _state is not None:
        _state.shutdown()
    _state = None


def _new_trace_id() -> str:
    return secrets.token_hex(16)


def _new_span_id() -> str:
    return secrets.token_hex(8)


def _export(state: Observability, span: Span) -> None:
    """Redact, then hand to every exporter. Failures here never reach the caller."""
    try:
        data = state.redactor.span(span._finish(state.settings.service_name))
    except Exception:
        _log.warning("could not build span data", exc_info=True)
        return
    for exporter in state.exporters:
        try:
            exporter.export(data)
        except Exception:
            _log.warning("span export failed", exc_info=True)


def _reset(token: Token) -> None:
    try:
        _current.reset(token)
    except ValueError:
        # The span closed in a different context than it opened in. Nothing to restore.
        pass


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Span]:
    state = _state
    if state is None or not state.enabled:
        yield NOOP_SPAN
        return

    parent = _current.get()
    if parent is not None and not parent.recording:
        yield NOOP_SPAN  # inside an unsampled trace
        return

    if parent is None and state.settings.sample_rate < 1.0:
        if state.rng.random() >= state.settings.sample_rate:
            token = _current.set(NOOP_SPAN)  # children see it and stay silent
            try:
                yield NOOP_SPAN
            finally:
                _reset(token)
            return

    current = Span(
        name=name,
        trace_id=parent.trace_id if parent else _new_trace_id(),
        span_id=_new_span_id(),
        parent_id=parent.span_id if parent else None,
        attributes=attributes,
    )
    token = _current.set(current)
    try:
        yield current
    except BaseException as exc:
        current.record_exception(exc)
        raise
    finally:
        _reset(token)
        _export(state, current)


def traced(name: str) -> Callable:
    """Decorator form of `span`. Works on normal and `async def` functions."""

    def decorate(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with span(name):
                    return await fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorate


def current_span() -> Span:
    """The active span, or the no-op span. Safe to call anywhere."""
    found = _current.get()
    return found if found is not None else NOOP_SPAN


def current_trace_id() -> str | None:
    """For responses: the trace id of the active span, or None when nothing is recording."""
    found = _current.get()
    return found.trace_id if found is not None and found.recording else None


def record_llm_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    ttft_ms: float | None = None,
    total_ms: float | None = None,
    cost_usd: float | None = None,
) -> None:
    """Put LLM usage on the active span. With no active span, record a tiny `llm.call` span."""
    values: dict[str, Any] = {
        attrs.MODEL: model,
        attrs.INPUT_TOKENS: input_tokens,
        attrs.OUTPUT_TOKENS: output_tokens,
    }
    if ttft_ms is not None:
        values[attrs.TTFT_MS] = ttft_ms
    if total_ms is not None:
        values[attrs.TOTAL_MS] = total_ms
    if cost_usd is not None:
        values[attrs.COST_USD] = cost_usd

    active = _current.get()
    if active is not None and active.recording:
        active.set_attributes(values)
        return
    with span("llm.call", values):
        pass


def record_error(exc: BaseException, target: Span | None = None) -> None:
    """Mark a span as failed without raising. Defaults to the active span."""
    (target or current_span()).record_exception(exc)
