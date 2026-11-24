# retrieval

Finds the best chunks for a question. Standalone on port 8002; in the combined app at
`/ui/retrieval` and `/api/retrieval`.

```
question → rewrite (optional, LLM) → search each query → merge → rerank (optional) → top_k
```

## Quick start

```bash
make up              # Qdrant, with some ingested documents
make run-retrieval   # http://localhost:8002
```

In the UI: choose a collection, type a question, and press **Search**. Each hit shows its rank,
scores, source and page. Tick **Compare two settings** to run two configurations on the same
query side by side; hits are marked "also in the other" or "only here".

Over HTTP:

```bash
curl -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     http://localhost:8002/api/retrieval/retrieve \
     -d '{"query": "refund window", "collection": "docs", "mode": "hybrid", "top_k": 5}'
```

## Search modes

- `dense`: meaning, via the collection's embedding model.
- `sparse`: exact words, via BM25.
- `hybrid`: both, fused on the server with **RRF**, `score = sum of 1 / (60 + rank)`. Each search
  first fetches `candidates` results, so a chunk that is good at both keywords and meaning is
  found in both lists.

Queries are encoded from the **collection's own spec** (`ingest_spec` in its metadata): its
embedding model, its query prefix (Gemma needs `task: search result | query: `) and its BM25
language. This package does not import `ingest`: the payload and the spec are the contract.

## Rewriting (needs `OPENAI_API_KEY`)

| Strategy | What it does |
|---|---|
| `multi_query` | a few other phrasings, merged with RRF |
| `hyde` | searches with a passage the model writes as a hypothetical answer |
| `step_back` | also searches a more general version of the question |

The original query is always searched too. Without a key, searches without rewriting still work.

## Reranking

A cross-encoder reads the question and one chunk together. It runs on the top `candidates` and
keeps `top_k`. Default model: `BAAI/bge-reranker-v2-m3-int8` (multilingual, 0.57 GB, downloads on
first use). Other models come from fastembed's list (`GET /strategies`). Check a model's licence
before relying on it.

## Filters

All must match: `{"source_name": "a.pdf", "page": [1, 2], "page": {"gte": 3}, "metadata.year": 2025}`.
A list matches any of its values; `gte`, `gt`, `lte`, `lt` express ranges.

## Reading stored chunks

`GET /collections/{name}/chunks?source_name=&after=&limit=` returns stored chunks a page at a
time (follow `next`). It is read-only, in storage order rather than document order (sort by
`page`, `chunk_index`), and a missing collection is a 404, never an empty list. The table tools
in `mcp_tools` use it to rebuild whole tables.

## HTTP API (under `/api/retrieval`)

| Method | Path | Notes |
|---|---|---|
| POST | `/retrieve` | `query` plus options. Returns hits, rewritten queries and timings per stage. |
| POST | `/rewrite` | only the rewriting, to inspect it |
| POST | `/rerank` | rerank candidates you supply, to isolate the reranker |
| GET | `/strategies`, `/collections`, `/status`, `/health` | |
| GET | `/collections/{name}/chunks` | stored chunks, paged |

## Configuration

`RETRIEVAL_COLLECTION`, `RETRIEVAL_TOP_K`, `RETRIEVAL_CANDIDATES`, `RETRIEVAL_RERANK_MODEL`,
`QDRANT_URL`. See `.env.example`.

## Tests

```bash
make test        # offline: in-memory Qdrant and fakes
make test-live   # downloads the embedding and reranker models
```
