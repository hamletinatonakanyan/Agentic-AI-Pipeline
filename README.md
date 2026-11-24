# Agentic Blocks

A toolkit of independent, reusable building blocks for retrieval-augmented and agentic AI
systems: document ingestion, hybrid retrieval, evaluation, MCP tools, tracing, and a
[LangGraph](https://langchain-ai.github.io/langgraph/) agent. Each block runs on its own with
its own API and UI. Together they form one application with a chat that answers from your
documents, calls tools, shows its sources, and records every step it took.

## Contents

- [The idea](#the-idea)
- [Architecture](#architecture)
- [Components](#components)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Using the app end to end](#using-the-app-end-to-end)
- [Configuration](#configuration)
- [Domains](#domains)
- [Security model](#security-model)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

## The idea

Most retrieval-augmented systems are one tangled application. Here the pieces are separate,
so each can be understood, tested, replaced and reused:

- **Ingest** turns files into searchable chunks, and never does the same work twice.
- **Retrieval** finds the best chunks for a question: dense, keyword or hybrid search, with
  optional query rewriting and reranking.
- **Evals** measures retrieval and answer quality against a golden dataset, so a change to any
  setting can be compared with numbers instead of impressions.
- **MCP tools** gives the agent a safe way to use tools, including its own tools for
  arithmetic and for counting or summing the rows of a table.
- **The agent** decides what to do with each message: search, use tools, or answer directly. It
  checks its answer against its sources, pauses for approval before risky tool calls, and can be
  resumed or forked after a restart.
- **Observability** records every run as a trace, so any answer can be traced back to the
  passages, tool calls and model calls behind it.

Packages never import each other, apart from three shared ones (`common`, `llm`,
`observability`). Anything outside a package (Qdrant, OpenAI, MCP servers, other packages) sits
behind a `Protocol`, so tests use fakes and the end-to-end app supplies the real
implementations. Because of that, any block can be lifted into another project.

## Architecture

```
                         ┌──────────────────────────────┐
   browser ─────────────▶│  e2e app  (port 8000)        │
   (chat, tabs)          │  one process, one access key │
                         └──────────────┬───────────────┘
                                        │ mounts
   ┌──────────┬──────────┬──────────┬───┴──────┬───────────┬──────────────┐
   ▼          ▼          ▼          ▼          ▼           ▼              ▼
 ingest   retrieval    evals    mcp_tools  agent_graph  observability   common / llm
 files →   question →  datasets  own tools  LangGraph    traces,         (shared:
 chunks    best chunks runs      + safe     agent,       runs, cost      settings,
                       compare   MCP client memory                       errors, OpenAI)
   │          │                      │          │
   └──────────┴─────────┬────────────┘          │
                        ▼                       ▼
                     Qdrant                  OpenAI
                 (vector store)             (gpt-6-luna)
```

A question in the chat flows like this:

```
question → recall memory → route ─┬→ search documents ──────────────┐
                                  ├→ call tools (approve if risky) ─┤→ write answer → check against sources
                                  └→ answer directly ───────────────┘        → citations → (save facts)
```

Every arrow above is a span in one trace.

## Components

| Package | Role | Standalone | In the app |
|---|---|---|---|
| [`common`](packages/common) | settings base, error envelope, access-key check, logging, shared UI kit | library | everywhere |
| [`llm`](packages/llm) | the one OpenAI adapter: completion, JSON output, streaming, tool calling, cost estimate | library | everywhere |
| [`observability`](packages/observability) | span API, local SQLite trace store, trace UI | `:8005` | `/ui/observability` |
| [`ingest`](packages/ingest) | parse, chunk, embed and store files in Qdrant, idempotently | `:8001` | `/ui/ingest` |
| [`retrieval`](packages/retrieval) | dense, sparse and hybrid search, query rewriting, reranking | `:8002` | `/ui/retrieval` |
| [`evals`](packages/evals) | golden datasets, metrics, runs, run comparison | `:8003` | `/ui/evals` |
| [`mcp_tools`](packages/mcp_tools) | our MCP server (calculator, search, table queries) and a safe MCP client | `:8004` | `/ui/mcp`, `/mcp` |
| [`agent_graph`](packages/agent_graph) | the LangGraph agent, threads, memory, approvals | `:8006` | `/ui/agent` |
| [`e2e`](e2e) | the combined app: all of the above plus the chat | `:8000` | `/` |

Each package has its own `pyproject.toml`, `README.md`, `.env.example`, tests, and (where it has
an API) a plain HTML and JavaScript UI with no build step.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Docker, for Qdrant (`make up`)
- An OpenAI API key for anything that calls a model (chat, query rewriting, eval judging)
- Optional: a LlamaCloud key to parse scanned PDF pages; Node.js only for the MCP Inspector

The first ingest downloads the embedding model (`google/embeddinggemma-300m`, about 1.2 GB).
Reranking downloads a second model (about 0.6 GB) the first time you tick "Rerank".

## Quick start

```bash
uv sync                  # creates the project's own .venv
cp .env.example .env     # then edit .env: set APP_API_KEY and OPENAI_API_KEY
make up                  # starts Qdrant in Docker
make run-e2e             # the full app on http://localhost:8000
```

Open http://localhost:8000, enter your `APP_API_KEY` in the access-key field, and open the
**Status** tab. Every card should be green except those whose key you have not set yet.

The project's commands (running a single component, tests, lint, stopping Qdrant) are in the
`Makefile`; run `make help` to list them.

**Bring your own data.** Put your documents (pdf, md, docx, csv, xlsx, txt) in a folder anywhere
on disk. In the **Ingest** tab either upload the files, or enter the folder's path under
**Ingest path**. For the second option, add that folder to `INGEST_ALLOWED_PATHS` in `.env`
(comma-separated) and restart the app.

## Using the app end to end

### 1. Choose a domain

The **Domain** box at the top is the active collection, for example `my-project`. Chat, Ingest,
Retrieval and Evals all default to it. A new name is created by its first ingest.

### 2. Ingest documents (Ingest tab)

Upload your files, or enter a server folder under **Ingest path** (it must be inside
`INGEST_ALLOWED_PATHS`).

- **Collection** is the domain the chunks are stored in.
- **Chunk strategy**: `recursive` splits at paragraphs, then sentences; `markdown` also never
  crosses a `#` heading; `fixed` cuts every N tokens. Tables are always split by rows, with the
  header repeated in every chunk.
- **Metadata** (optional): a JSON object such as `{"client": "acme", "type": "faq"}`, added to
  every chunk and usable as a search filter.
- Use **Preview chunks** to see how a file will be cut before writing anything.

Ingesting is idempotent. Re-running the same files adds nothing; editing a file re-embeds only
the chunks whose text changed. You can delete one file, or a whole collection, from the same tab.

### 3. Try the search (Retrieval tab)

Ask a question and compare settings: dense, sparse or hybrid, query rewriting, reranking,
`top_k`, filters. **Compare two settings** puts two configurations side by side on the same
query and marks the hits that only one of them found.

### 4. Chat (Chat tab)

Ask a question your documents answer. The answer streams in with `[1]`-style citations; click
one to see the passage it came from. Questions about whole tables ("how many rows have status
open?", "average amount per region") are answered with the table tools instead of a search.

The options panel controls the search mode, reranking, tools, and the global *Use reasoning*
switch. Tools that are not read-only pause the run and ask for your approval first.

### 5. Inspect a run (Observability tab)

Every answer links to its trace. Open it, or search a trace id or a conversation id. An agent
run shows the question, the answer, tokens and an estimated cost, and each step as a card
(route, search, tools, answer, check, citations) with its details: the passages given to the
model, tool arguments and results, the self-check verdict.

### 6. Measure quality (Evals tab)

1. **Datasets**: upload a golden dataset (JSONL), or draft questions from your chunks with the
   LLM and review them.
2. **Runs**: pick a dataset, a target (`retrieval` or the whole `agent`), the metrics to compute,
   and start. Run the two targets separately, then open **Compare** to see which questions
   improved or regressed.

### 7. Add your own tools (MCP tab)

Register another MCP server (stdio command or HTTP URL), choose which of its tools are allowed,
and the agent can use them. Registrations are checked against allow-lists, see
[Security model](#security-model).

## Configuration

All settings are environment variables, read from `.env` through
[pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) with a prefix
per package. The root [`.env.example`](.env.example) lists every variable; each package's own
`.env.example` lists the ones it reads. The ones you will touch first:

| Variable | Purpose |
|---|---|
| `APP_API_KEY` | the one access key for every API route (sent as `X-API-Key`) |
| `OPENAI_API_KEY` | model calls: chat, rewriting, judging, question drafting |
| `QDRANT_URL` | where Qdrant runs (default `http://localhost:6333`) |
| `LLM_USE_REASONING` | `true` sends `reasoning.effort=low`, `false` sends `none` |
| `LLM_PRICE_INPUT_PER_M`, `LLM_PRICE_OUTPUT_PER_M` | USD per million tokens, only to estimate cost in traces |
| `LLAMA_CLOUD_API_KEY` | parsing scanned PDF pages (optional) |
| `INGEST_ALLOWED_PATHS` | server folders that "Ingest path" may read |
| `AGENT_APPROVAL` | `sensitive` (default), `all` or `none`: which tool calls ask first |
| `AGENT_TRACE_CONTENT` | put passages, tool results and answers in traces (default on) |
| `E2E_ENABLE_<NAME>` | turn a component of the combined app off |

Secrets live only in `.env`, which is git-ignored. The `.env.example` files hold names with
empty values.

## Domains

A domain is a Qdrant collection. Use one per company or project to keep their documents apart.
The agent's search, its table tools and the evals all stay inside the active domain: when a
conversation has a domain, the agent pins it on every tool that takes a `collection`, whatever
the model asks for. A collection holds one embedding model and one keyword language, so mixing
is refused instead of silently degrading search.

## Security model

- **One access key** protects every route except `/health` and the static UI files, which hold
  no secrets. Anyone with `APP_API_KEY` has full access, so do not expose the port publicly
  without putting real authentication in front of it.
- **Allow-lists**: anything that makes the server run a command, call a URL or send a secret on
  a user's behalf (MCP registrations, HTTP eval targets) is checked against allow-lists from
  settings by exact match, when it is saved and again when it is used. Secrets are referenced by
  name and never stored.
- **Untrusted text**: document passages, tool output and memory go to the model inside marked
  blocks, and the model is told not to follow instructions found in them. Tool descriptions
  from other servers are cleaned and length-capped.
- **Approvals**: tools their server does not mark read-only pause the run for a human decision.
- **No code execution**: the calculator parses a syntax tree and the table tools use fixed
  operations. Neither uses `eval` or SQL.
- **Redaction**: traces redact key-like strings and attributes named like secrets.

## Project layout

```
packages/
  common/          settings base, errors, auth, logging, UI kit
  llm/             OpenAI adapter
  observability/   tracing API, SQLite store, trace UI
  ingest/          parse, chunk, embed, store
  retrieval/       search, rewrite, rerank
  evals/           datasets, metrics, runs
  mcp_tools/       MCP server and client
  agent_graph/     the LangGraph agent
e2e/               the combined app and the chat
infra/             docker-compose.yml (Qdrant)
Makefile           the project's commands (`make help`)
.env.example       every setting, names only
```

Each package follows the same layout: `src/<name>/` with `settings`, `models`, `service`,
`protocols`, `api`, `app` and a `ui/` folder, plus `tests/`, `README.md` and `.env.example`.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `401 Missing or invalid X-API-Key` | the key in the page or request does not match `APP_API_KEY` |
| `503 not_configured` on every route | `APP_API_KEY` is not set in `.env` |
| Status shows Qdrant unreachable | run `make up`, and check `QDRANT_URL` |
| Chat is disabled, status says `OPENAI_API_KEY` | add the key to `.env` and restart |
| First ingest or first memory use is slow | the embedding model is downloading (about 1.2 GB) |
| "Ingest path" is hidden | set `INGEST_ALLOWED_PATHS` and restart |
| Retrieval metrics are 0 in evals | the dataset's `relevant_chunk_ids` do not match what was ingested (chunk ids depend on the file path and chunking used at ingest) |
| An answer cites nothing | the question went to a tool or was answered directly; open its trace to see the route |
