from typing import Protocol

from observability.models import SpanData, TraceFilter, TraceSummary, TraceTree


class SpanExporter(Protocol):
    """Receives each finished span. Must not raise: failures are caught and logged anyway."""

    def export(self, span: SpanData) -> None: ...

    def shutdown(self) -> None: ...


class TraceStore(Protocol):
    """Read side of a trace store (the SQLite store also implements `SpanExporter`)."""

    def list_traces(
        self, filters: TraceFilter, limit: int = 50, offset: int = 0
    ) -> list[TraceSummary]: ...

    def get_trace(self, trace_id: str) -> TraceTree | None: ...
