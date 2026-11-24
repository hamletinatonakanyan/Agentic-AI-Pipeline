# agent_graph

The agent: a [LangGraph](https://langchain-ai.github.io/langgraph/) graph that decides what to do
with each message, searches the documents or calls tools, writes an answer with citations, checks
the answer against its sources, and remembers facts between conversations. Standalone on port
8006; in the combined app at `/ui/agent` and `/api/agent`.

## Quick start

```bash
make up                # Qdrant with ingested documents
make run-agent         # http://localhost:8006 (needs retrieval on 8002 and the MCP app on 8004)
```

Inside the combined app the agent uses retrieval and the tools in-process, so there is nothing
else to start (`make run-e2e`). The UI has four tabs:

- **Chat**: send messages with streaming answers, options per request, approvals, and the live
  list of graph events.
- **Graph**: the graph drawn as SVG; nodes light up as a run goes through them.
- **State & history**: the saved state of the conversation and every checkpoint, with **Fork from
  here** to branch from a past step.
- **Memory**: the long-term facts for a user id, with add, delete and search by meaning.

Over HTTP:

```bash
curl -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d '{}' \
     http://localhost:8006/api/agent/threads                        # {"thread_id": "<id>"}
curl -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     http://localhost:8006/api/agent/threads/<id>/messages \
     -d '{"message": "What is the refund window?", "options": {"retrieval": {"collection": "docs"}}}'
```

## The graph

```
begin → recall → load_tools → route ─┬→ retrieve ───────────────┐
                                     ├→ plan_tools ⇄ (approve → run_tool × N → collect_tools)
                                     └→ (direct) ───────────────┤
                                                                ▼
                              answer = generate ⇄ check  →  respond → remember
```

| Node | What it does |
|---|---|
| `begin` | resets everything that belongs to one turn (the conversation itself is kept) |
| `recall` | long-term memory: the facts about this user closest in meaning to the question |
| `route` | one JSON model call: `retrieve`, `tool` or `direct`, plus the question rewritten to stand alone. With neither search nor tools enabled it is skipped |
| `retrieve` | calls the retrieval service (cached) |
| `plan_tools` | asks the model which tools to call, or whether it is done (at most `AGENT_MAX_TOOL_ROUNDS`) |
| `approve` | pauses for a human on any tool its server did not mark read-only |
| `run_tool` | one per call, all in parallel (LangGraph `Send`) |
| `answer` | a subgraph: `generate` (streams tokens) and `check` (lists claims the sources do not support, regenerates up to `AGENT_MAX_CHECK_RETRIES` times) |
| `respond` | turns `[1]`, `[2]` in the answer into citations and drops numbers that do not exist |
| `remember` | optionally saves facts the user stated (off by default) |

Direct answers skip the self-check. If the check still fails after its retries, the answer is
delivered with the unsupported claims listed.

## How it works

- **State and reducers**: nodes return small updates; reducers (`add_messages`,
  `operator.add`) say how an update merges. Per-turn fields are reset by `begin`.
- **Checkpointer and `thread_id`**: every step is saved (SQLite), so a conversation survives a
  restart and can be resumed or forked. One run per thread at a time (409 `thread_busy`).
- **Interrupt and resume**: a run pauses for approval, is saved, and continues from the
  approval node when you answer: `{"approve": true}` or `{"decisions": {call_id: bool}}`. A
  missing decision means no.
- **Fork**: start a new branch from any past checkpoint *in the same thread*. The old branch
  stays in the history. Passing a new message edits the question and regenerates.
- **Long-term memory**: a LangGraph store with semantic search, namespaced per `user_id`. Saving
  facts from chats is opt-in because it stores personal statements. Recall applies a minimum
  score.
- **Native tool calling**: tools are offered to the model in the Responses API function format;
  malformed arguments are reported back to the model instead of being run.
- **Caches** (on by default, 10-minute TTL, a request can bypass them): model responses, and the
  search node. The search cache key includes a data version, so ingesting or deleting documents
  never lets stale passages come back.

## Domains and tools

The active collection comes from `options.retrieval.collection`. The agent tells the model which
collection is active and **pins** it on any tool that has a `collection` argument, whatever the
model asked for, so a conversation cannot read another domain's data.

For "how many" or "total" questions over a table, the router chooses the tool route and the
planner calls `list_tables` then `query_table` from `mcp_tools`.

## API (everything needs `X-API-Key` except `/health`)

| Route | |
|---|---|
| `POST /api/agent/threads`, `GET .../threads`, `GET/DELETE .../threads/{id}` | conversations |
| `POST .../threads/{id}/messages` (or `/messages/stream`) | send a message |
| `POST .../threads/{id}/resume` (or `/resume/stream`) | answer an approval |
| `POST .../threads/{id}/fork` (or `/fork/stream`) | `{"checkpoint_id", "message"?}` |
| `GET .../threads/{id}/state`, `.../history`, `GET .../graph` | look inside |
| `GET/POST/DELETE .../memory/{user_id}` | long-term facts (`?query=` searches by meaning) |
| `GET/PUT .../reasoning` | the global "Use reasoning" switch (lasts until restart) |

Streaming routes send Server-Sent Events: `run_started`, `node_start`, `node_end`, `token`,
`answer_replaced` (a failed self-check is being retried), `tool_call`, `citation`, `interrupt`,
`done`, `error`. Options per request: `retrieval` (mode, rewrite, rerank, collection, top_k),
`tools`, `check`, `use_cache`, `remember`.

## Traces, history and cost

- A run is one trace: `agent.run` (with `session.id` set to the thread id, `input.value` and
  `output.value`) and one span per step holding what it saw and decided (route and reason, the
  query, each passage with source, page, score and text, tool arguments and results, check
  verdicts, the answer). `AGENT_TRACE_CONTENT=false` keeps only names, counts and timings. Text
  is clipped and redacted.
- Each answer's sources, tool calls, timings, trace id and options are saved (`agent.sqlite`), so
  reopening a conversation shows them again. A conversation remembers its domain.

## Safety

- Document text, tool output and memory are **data**: they go to the model in marked blocks and
  the model is told not to obey instructions inside them.
- A tool that is not read-only, invented, declined, or sent malformed arguments is never run
  silently.
- Runs, tool rounds and message length are capped.
- This package imports only `common`, `llm` and `observability`. Retrieval and tools are
  `Protocol`s, supplied by the combined app in-process or by HTTP adapters when standalone.

## Configuration

`AGENT_DATA_DIR`, `AGENT_CHECKPOINT_BACKEND` (`sqlite` or `memory`), `AGENT_MAX_CHECK_RETRIES`,
`AGENT_MAX_TOOL_ROUNDS`, `AGENT_APPROVAL` (`sensitive`, `all`, `none`), `AGENT_USE_CACHE`,
`AGENT_CACHE_TTL_S`, `AGENT_AUTO_REMEMBER`, `AGENT_MEMORY_EMBED_MODEL`, `AGENT_TRACE_CONTENT`,
and, standalone only, `AGENT_RETRIEVAL_URL` and `AGENT_MCP_URL`. The model comes from `LLM_*`.

## Tests

```bash
uv run pytest packages/agent_graph    # offline: scripted model, fake embedder, in-memory persistence
make test-live                         # real model and real embedding model
```
