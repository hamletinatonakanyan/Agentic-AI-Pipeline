# mcp_tools

Two halves: **our own MCP server** (a calculator, corpus search, table queries, a resource and a
prompt), and a **safe MCP client** that discovers and calls tools on any registered server.
Standalone on port 8004; in the combined app at `/ui/mcp`, `/api/mcp` and `/mcp`. Built on the
official `mcp` SDK 2.x (`MCPServer`, called `FastMCP` in 1.x).

## Quick start

```bash
make run-mcp          # UI and /mcp on http://localhost:8004
make inspect-mcp      # the official MCP Inspector on our stdio server (needs Node)
```

In the UI: **Servers** lists registered servers and lets you register another (a stdio command
or an HTTP URL) and choose which of its tools are allowed. **Tools** lists a server's tools with
their schemas and lets you call one with generated form fields. **Our server** shows what this
package exposes.

For HTTP clients, point them at `http://localhost:8004/mcp` (or `:8000/mcp` in the combined app)
and send the `X-API-Key` header. For the Inspector over HTTP, add that header there too.

## Our server

| Kind | Name | What it does |
|---|---|---|
| tool | `calculate(expression)` | arithmetic through a syntax tree. **Never `eval`.** Names, calls, attributes and strings are refused |
| tool | `search_corpus(query, top_k, collection)` | hybrid search over the ingested documents, with sources and pages |
| tool | `list_tables(collection)` | the tables found in a collection (csv, xlsx, tables in documents): columns, row counts, a sample row |
| tool | `query_table(table, aggregate, column, filters, group_by, ...)` | count, sum, average, min/max, distinct count, or matching rows, over **all** rows of a table |
| resource | `corpus://sources`, `corpus://sources/{collection}` | what has been ingested |
| prompt | `summarize_with_citations(topic)` | a reusable instruction |

- **stdio**: `python -m mcp_tools.server` (a client launches it). Stdout is the protocol, so it
  logs only to stderr.
- **HTTP**: `/mcp` on the app, behind the same `X-API-Key` as every route. It answers only to the
  Host names in `MCP_HTTP_ALLOWED_HOSTS` (protection against DNS rebinding). The app's lifespan
  must run the SDK's session manager; `app.py` shows how.

## Table questions

Search returns a few passages, so it cannot answer "how many P1 tickets are there?". The table
tools read every row instead. This is **not text-to-SQL**: the model never writes a query. It
fills fixed fields (a table, filters of the form `column / op / value`, one aggregate, an
optional `group_by`) that are checked against the table's real header, and the answer is
computed in plain Python: no SQL, no `eval`, nothing is written.

```json
{"table": "tickets.csv",
 "filters": [{"column": "priority", "op": "=", "value": "P1"}],
 "aggregate": "count"}
```

- Operators: `= != < <= > >= contains`. Numbers compare as numbers (so 9 < 10), text ignores
  case, and an empty cell is never smaller or larger.
- Aggregates: `count`, `count_distinct`, `sum`, `avg`, `min`, `max` (over `column`), or `rows`
  (list the matches, with `select` and `limit`). `group_by` takes up to two columns.
- Mistakes (unknown table or column, `sum` without a column) come back as errors that list the
  valid names.
- Rows come from the stored chunks of the source: ingest splits a table by rows without overlap
  and repeats the header, so the table is rebuilt exactly. Tables inside documents are found too,
  and a source with several tables gets names such as `doc.md#1` and `doc.md#2`.
- Limits: `MCP_TABLE_MAX_CHUNKS`, `MCP_TABLE_MAX_ROWS`, `MCP_TABLE_MAX_GROUPS`. If a limit cuts the
  read short, the result says it is incomplete.

## The client: `MCPClientManager`

`register / unregister / list_tools / check / call_tool / llm_tools`. A connection is opened for
each operation and closed afterwards (simple, no leaked processes), with a short cache of tool
lists. Before anything is sent, `call_tool` checks in order: the server exists and is enabled,
the tool is on the allow-list, and the arguments match the tool's JSON Schema. Then it applies a
timeout and cuts long output (`truncated=true`). Every failure comes back as
`ToolResult(is_error=True, error_kind=...)`, never an exception, so an agent can recover.

`llm_tools()` returns the allowed tools in the **OpenAI function format**, named
`<server>__<tool>` (cleaned to `[a-zA-Z0-9_-]{1,64}` and unique), with a lookup back to the real
server and tool and each tool's read-only hint. Descriptions are written by whoever runs a
server, so they are treated as untrusted text: control characters stripped, length capped.

## Security rules

A registration is a request to run a command or contact a URL on your behalf, so it is limited:

- A **stdio command** must equal an entry of `MCP_ALLOWED_COMMANDS` (default: only our own
  server). `python -c ...`, `bash`, or an extra argument are refused.
- An **HTTP URL** host must be in `MCP_ALLOWED_HOSTS` (default: localhost).
- **Secrets are referenced by name** (`env_refs`, `headers_refs`), only names in
  `MCP_ALLOWED_ENV_REFS` / `MCP_ALLOWED_HEADER_REFS`, and never stored. The value is read at
  connect time from the environment, then `.env`.
- A stdio tool server gets **only the variables its registration names**, plus a few safe
  defaults. Your `OPENAI_API_KEY` and others stay behind.
- The rules are checked when saving **and again** when connecting.
- Tool output is untrusted: a malicious server can put instructions in it. It is capped, and the
  agent treats it as data.
- Our own server's allow-list defaults to all four tools. A stored list that is still an earlier
  version's untouched default is upgraded on startup; a list you changed is kept.

## HTTP API (under `/api/mcp`)

`GET/POST /servers`, `DELETE /servers/{name}`, `PATCH /servers/{name}/allowlist`,
`PATCH /servers/{name}/enabled`, `POST /servers/{name}/check`, `GET /servers/{name}/tools`,
`POST /servers/{name}/tools/{tool}/call`, `GET /llm-tools`, `GET /exposed`, `GET /status`. A
refused call is an HTTP error (not allowed 403, unknown 404, bad arguments 422, disabled 409). A
tool that ran and failed is a normal 200 with `is_error: true`.

## Configuration

`MCP_SERVER_NAME`, `MCP_RETRIEVAL_URL`, `MCP_INGEST_URL`, `MCP_DEFAULT_COLLECTION`,
`MCP_DB_PATH`, `MCP_CALL_TIMEOUT_S`, `MCP_MAX_OUTPUT_CHARS`, the allow-lists
(`MCP_ALLOWED_COMMANDS`, `MCP_ALLOWED_HOSTS`, `MCP_ALLOWED_ENV_REFS`, `MCP_ALLOWED_HEADER_REFS`)
and the table limits. See `.env.example`.

## Tests

```bash
make test        # offline, including real stdio subprocesses and a real HTTP server
make test-live   # the official Inspector against our server
```
