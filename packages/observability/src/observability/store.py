"""Local trace store in SQLite. It is both an exporter (writes) and a TraceStore (reads)."""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from observability import attrs
from observability.models import (
    SpanData,
    SpanEvent,
    SpanNode,
    TraceFilter,
    TraceSummary,
    TraceTree,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    span_id        TEXT PRIMARY KEY,
    trace_id       TEXT NOT NULL,
    parent_id      TEXT,
    name           TEXT NOT NULL,
    service        TEXT NOT NULL,
    start_ts       REAL NOT NULL,
    end_ts         REAL NOT NULL,
    duration_ms    REAL NOT NULL,
    status         TEXT NOT NULL,
    status_message TEXT NOT NULL,
    attributes     TEXT NOT NULL,
    events         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_start ON spans(start_ts);
"""

# json_extract paths need the dotted attribute name quoted.
_IN_PATH = f'$."{attrs.INPUT_TOKENS}"'
_OUT_PATH = f'$."{attrs.OUTPUT_TOKENS}"'
_COST_PATH = f'$."{attrs.COST_USD}"'
_SESSION_PATH = f'$."{attrs.SESSION_ID}"'
_INPUT_PATH = f'$."{attrs.INPUT_VALUE}"'
_PREVIEW_CHARS = 140


def _preview(text: Any) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= _PREVIEW_CHARS else text[: _PREVIEW_CHARS - 1] + "…"


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class SqliteTraceStore:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One shared connection, guarded by a lock, because spans finish on many threads.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    # ----- exporter side -----------------------------------------------------------------

    def export(self, span: SpanData) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    span.span_id,
                    span.trace_id,
                    span.parent_id,
                    span.name,
                    span.service,
                    span.start_ts,
                    span.end_ts,
                    span.duration_ms,
                    span.status,
                    span.status_message,
                    json.dumps(span.attributes, default=str),
                    json.dumps([e.model_dump() for e in span.events], default=str),
                ),
            )

    def shutdown(self) -> None:
        with self._lock:
            self._conn.close()

    # ----- reader side -------------------------------------------------------------------

    def list_traces(
        self, filters: TraceFilter, limit: int = 50, offset: int = 0
    ) -> list[TraceSummary]:
        """Newest first. A trace appears once its root span has finished."""
        errors = "(SELECT COUNT(*) FROM spans s WHERE s.trace_id = r.trace_id AND s.status='error')"
        where = ["r.parent_id IS NULL"]
        params: dict[str, Any] = {
            "in_path": _IN_PATH,
            "out_path": _OUT_PATH,
            "cost_path": _COST_PATH,
            "session_path": _SESSION_PATH,
            "input_path": _INPUT_PATH,
            "limit": limit,
            "offset": offset,
        }
        if filters.name:
            where.append("r.name LIKE :name ESCAPE '\\'")
            params["name"] = f"%{_like_escape(filters.name)}%"
        if filters.status == "error":
            where.append(f"{errors} > 0")
        elif filters.status == "ok":
            where.append(f"{errors} = 0")
        if filters.since is not None:
            where.append("r.start_ts >= :since")
            params["since"] = filters.since
        if filters.until is not None:
            where.append("r.start_ts <= :until")
            params["until"] = filters.until
        if filters.min_duration_ms is not None:
            where.append("r.duration_ms >= :min_dur")
            params["min_dur"] = filters.min_duration_ms
        if filters.session_id:
            where.append("json_extract(r.attributes, :session_path) = :session_id")
            params["session_id"] = filters.session_id

        sql = f"""
            SELECT r.trace_id, r.name AS root_name, r.start_ts, r.duration_ms,
              (SELECT COUNT(*) FROM spans s WHERE s.trace_id = r.trace_id) AS span_count,
              {errors} AS errors,
              (SELECT COALESCE(SUM(json_extract(s.attributes, :in_path)), 0)
                 FROM spans s WHERE s.trace_id = r.trace_id) AS in_tokens,
              (SELECT COALESCE(SUM(json_extract(s.attributes, :out_path)), 0)
                 FROM spans s WHERE s.trace_id = r.trace_id) AS out_tokens,
              (SELECT COUNT(*) FROM spans s WHERE s.trace_id = r.trace_id
                 AND json_extract(s.attributes, :in_path) IS NOT NULL) AS llm_calls,
              (SELECT SUM(json_extract(s.attributes, :cost_path))
                 FROM spans s WHERE s.trace_id = r.trace_id) AS cost,
              json_extract(r.attributes, :session_path) AS session_id,
              json_extract(r.attributes, :input_path) AS input_value
            FROM spans r
            WHERE {" AND ".join(where)}
            ORDER BY r.start_ts DESC
            LIMIT :limit OFFSET :offset
        """
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            TraceSummary(
                trace_id=r["trace_id"],
                root_name=r["root_name"],
                start_ts=r["start_ts"],
                duration_ms=r["duration_ms"],
                status="error" if r["errors"] else "ok",
                span_count=r["span_count"],
                input_tokens=int(r["in_tokens"] or 0),
                output_tokens=int(r["out_tokens"] or 0),
                llm_calls=int(r["llm_calls"] or 0),
                cost_usd=r["cost"],
                session_id=r["session_id"],
                input_preview=_preview(r["input_value"]),
            )
            for r in rows
        ]

    def get_trace(self, trace_id: str) -> TraceTree | None:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_ts, span_id", (trace_id,)
            ).fetchall()
        if not rows:
            return None
        nodes = {r["span_id"]: self._node(r) for r in rows}
        roots: list[SpanNode] = []
        for node in nodes.values():
            parent = nodes.get(node.parent_id) if node.parent_id else None
            (parent.children if parent else roots).append(node)

        first = roots[0]
        ints = [int(n.attributes.get(attrs.INPUT_TOKENS) or 0) for n in nodes.values()]
        outs = [int(n.attributes.get(attrs.OUTPUT_TOKENS) or 0) for n in nodes.values()]
        costs = [
            float(n.attributes[attrs.COST_USD])
            for n in nodes.values()
            if n.attributes.get(attrs.COST_USD) is not None
        ]
        summary = TraceSummary(
            trace_id=trace_id,
            root_name=first.name,
            start_ts=first.start_ts,
            duration_ms=first.duration_ms,
            status="error" if any(n.status == "error" for n in nodes.values()) else "ok",
            span_count=len(nodes),
            input_tokens=sum(ints),
            output_tokens=sum(outs),
            llm_calls=sum(
                1 for n in nodes.values() if n.attributes.get(attrs.INPUT_TOKENS) is not None
            ),
            cost_usd=sum(costs) if costs else None,
            session_id=first.attributes.get(attrs.SESSION_ID),
            input_preview=_preview(first.attributes.get(attrs.INPUT_VALUE)),
        )
        return TraceTree(summary=summary, roots=roots)

    @staticmethod
    def _node(row: sqlite3.Row) -> SpanNode:
        return SpanNode(
            trace_id=row["trace_id"],
            span_id=row["span_id"],
            parent_id=row["parent_id"],
            name=row["name"],
            service=row["service"],
            start_ts=row["start_ts"],
            end_ts=row["end_ts"],
            duration_ms=row["duration_ms"],
            status=row["status"],
            status_message=row["status_message"],
            attributes=json.loads(row["attributes"]),
            events=[SpanEvent(**e) for e in json.loads(row["events"])],
        )
