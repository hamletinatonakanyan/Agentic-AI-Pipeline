"""Simple exporters. The SQLite one lives in `store.py`."""

import logging

from observability.models import SpanData

_log = logging.getLogger("observability.spans")


class ConsoleExporter:
    """One JSON log line per finished span (through the shared JSON logging setup)."""

    def export(self, span: SpanData) -> None:
        _log.info(
            span.name,
            extra={
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_id": span.parent_id,
                "duration_ms": round(span.duration_ms, 2),
                "status": span.status,
                "attributes": span.attributes,
            },
        )

    def shutdown(self) -> None:
        pass


class InMemoryExporter:
    """Keeps spans in a list. For tests in any package."""

    def __init__(self) -> None:
        self.spans: list[SpanData] = []

    def export(self, span: SpanData) -> None:
        self.spans.append(span)

    def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        self.spans.clear()

    def names(self) -> list[str]:
        return [s.name for s in self.spans]
