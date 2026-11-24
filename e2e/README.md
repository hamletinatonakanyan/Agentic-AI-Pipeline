# e2e

The combined application: one process (port 8000) that mounts every package's API and UI, plus a
chat that runs the whole pipeline.

```bash
make up          # Qdrant
make run-e2e     # http://localhost:8000
```

## What is in it

| Where | What |
|---|---|
| `/` | the shell: access key, domain, Status, **Chat**, and one tab per enabled package |
| `/api/status` | each component's state: enabled, configured, reachable, what is missing |
| `/api/ingest`, `/api/retrieval`, `/api/evals`, `/api/mcp`, `/api/obs`, `/api/agent` | each package's API |
| `/ui/<package>` | each package's own UI (the tabs are these pages in frames) |
| `/mcp` | our MCP server over HTTP, behind the same key |

Only `GET /health`, every `/api/<package>/health`, and the static UI files are public. Everything
else needs `X-API-Key`.

## Using it

1. Open http://localhost:8000 and enter `APP_API_KEY`. It is kept for the browser session and
   never shown again.
2. **Status** shows a card per component: enabled, configured, reachable, and which setting is
   missing if any.
3. Set the **Domain** (the active collection), ingest files in **Ingest**, and ask in **Chat**.

## The chat

The Chat tab talks to `/api/agent`; there is no separate chat backend. It offers streaming
answers, `[n]` citations you can click to see the passage, tool calls with their arguments and
results, an Allow or Decline prompt for tools that need approval, per-answer timings, and a link
to the answer's trace. Conversations are listed in the sidebar and keep their sources and tool
calls when reopened. The options panel sets the search mode, reranking, passage count, tools,
and the global *Use reasoning* switch.

The chat tells you what is missing (agent off, `OPENAI_API_KEY`, Qdrant unreachable, retrieval
off) instead of failing on send. The **Agent** tab keeps the debugging view: graph, state,
history and fork.

## Domains

The **Domain** box at the top is the active collection, one per company or project. Chat,
Ingest, Retrieval and Evals all default to it, and a conversation remembers its domain. Delete a
file, or a whole collection, from the Ingest tab.

## Observability

Each answer links to its trace and to every run of its conversation. Search a trace id or a
conversation id in the Observability tab to see the run: the question, every step with its
details, the answer, tokens and an estimated cost (set `LLM_PRICE_INPUT_PER_M` and
`LLM_PRICE_OUTPUT_PER_M` for the estimate).

## Wiring

This package is the one place that knows several packages. `wiring.py` holds the in-process
adapters: retrieval and tools for the agent, the corpus for our MCP server, and two eval targets
(`retrieval`, and the whole `agent`). Turn a component off with `E2E_ENABLE_<NAME>=false`
(`INGEST`, `RETRIEVAL`, `EVALS`, `MCP`, `OBS`, `AGENT`).

## Tests

```bash
uv run pytest e2e     # offline: in-memory Qdrant, scripted model, no keys, no downloads
```
