# observability

A small tracing API, a local SQLite trace store, and a UI to explore traces. Standalone on port
8005; in the combined app at `/ui/observability` and `/api/obs`.

## Quick start

```bash
make run-observability    # http://localhost:8005
```

The list shows recent traces with duration, status, model calls, tokens and estimated cost. Click
one to open it. A search box takes a **trace id** or a **conversation id** (the latter lists
every run of that conversation). In the combined app, each chat answer links straight to its
trace.

For a trace whose root is an agent run the page shows the **run**: the question, the answer,
usage tiles (time, model calls, tokens in and out, estimated cost, tool calls, retries), and
every step as a colored card with a duration bar. Click a step for its details, such as the
passages given to the model, tool arguments and results, or the self-check verdict. **All
spans** shows the plain waterfall for any trace.

## Use it from another package

```python
from observability import span, traced, record_llm_call, current_trace_id

with span("ingest.file", {"source": name}):
    with span("chunk"):          # child of ingest.file, automatically
        ...

@traced("tool.call")
def call_tool(...): ...

record_llm_call("gpt-6-luna", input_tokens=10, output_tokens=5, ttft_ms=120, total_ms=800)
```

Until `init_observability()` runs, or when `OBS_ENABLED=false`, all of this is a no-op, so
packages work with observability off. Exporter failures never reach the caller.

## What is recorded

A span has a name, times, a parent, attributes, events and a status (`ok` or `error`). Attribute
names are in `attrs.py` (OpenTelemetry GenAI names where they exist). Before export, values are
scrubbed with `OBS_REDACT_PATTERNS`, and attributes whose key looks like `api_key`, `secret`,
`password` or `authorization` are replaced entirely.

## Conversations, content and cost

Whoever starts a trace can set three conventional attributes on its root span: `session.id` (the
conversation it belongs to), `input.value` and `output.value`. Then:

- `GET /api/obs/traces?session_id=<id>` lists every run of one conversation, and each summary
  has an input preview.
- A trace summary counts model calls (spans with token usage) and adds up `llm.cost_usd` into an
  **estimated** cost. It is `null` when no prices are configured: unknown, never zero.
- Deep links: `/ui/observability/#trace=<id>` and `#conversation=<id>`.

## HTTP API

| Path | Auth | Purpose |
|---|---|---|
| `GET /health` | none | liveness |
| `GET /status` | key | enabled, exporters, store |
| `GET /api/obs/traces` | key | list: `name`, `status`, `since`, `until`, `min_duration_ms`, `session_id`, `limit`, `offset`. Each row has `session_id`, `input_preview`, `llm_calls`, `cost_usd` |
| `GET /api/obs/traces/{trace_id}` | key | the span tree |

Inside the combined app the health and status routes are under `/api/obs` as well.

## Configuration

`OBS_ENABLED`, `OBS_EXPORTERS` (`local` and/or `console`), `OBS_LOCAL_DB_PATH`,
`OBS_SAMPLE_RATE`, `OBS_REDACT_PATTERNS`. See `.env.example`.

## Scope

Not included: Prometheus metrics and OTLP export.

## Tests

```bash
uv run pytest packages/observability
```
