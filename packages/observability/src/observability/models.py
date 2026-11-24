"""The contract: what a finished span, a trace summary and a trace tree look like."""

from typing import Any, Literal

from pydantic import BaseModel, Field

SpanStatus = Literal["unset", "ok", "error"]


class SpanEvent(BaseModel):
    name: str
    ts: float
    attributes: dict[str, Any] = Field(default_factory=dict)


class SpanData(BaseModel):
    """A finished span, as exported and stored."""

    trace_id: str
    span_id: str
    parent_id: str | None = None
    name: str
    service: str = ""
    start_ts: float  # unix seconds
    end_ts: float
    duration_ms: float
    status: SpanStatus = "ok"
    status_message: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)


class TraceSummary(BaseModel):
    trace_id: str
    root_name: str
    start_ts: float
    duration_ms: float
    status: Literal["ok", "error"]
    span_count: int
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0  # spans that recorded token usage
    cost_usd: float | None = None  # estimated; None when no span carried a cost
    session_id: str | None = None  # the conversation this run belongs to (root `session.id`)
    input_preview: str = ""  # the start of the root span's `input.value`


class SpanNode(SpanData):
    children: list["SpanNode"] = Field(default_factory=list)


class TraceTree(BaseModel):
    summary: TraceSummary
    # Normally one root. More than one means a parent span is missing from the store.
    roots: list[SpanNode]


class TraceFilter(BaseModel):
    name: str | None = None  # substring of the root span name
    status: Literal["ok", "error"] | None = None
    since: float | None = None  # unix seconds
    until: float | None = None
    min_duration_ms: float | None = None
    session_id: str | None = None  # root span attribute `session.id`


class ObsStatus(BaseModel):
    enabled: bool
    exporters: list[str]
    sample_rate: float
    store: str  # "sqlite", or "none" when no local store is active
    db_path: str = ""
