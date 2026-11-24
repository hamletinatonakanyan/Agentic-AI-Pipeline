# evals

Measures how good retrieval and answers are against a golden dataset, and compares runs.
Standalone on port 8003; in the combined app at `/ui/evals` and `/api/evals`.

```
dataset (accepted questions) → target answers each → score → report → compare two runs
```

## Quick start

```bash
make run-evals    # http://localhost:8003
```

1. **Datasets**: upload a golden dataset, or draft questions from your chunks.
2. **Runs**: choose the dataset, a target, the k values and the metrics, then **Start run**.
3. Open the finished run for per-metric summaries and a per-question drill-down.
4. **Compare**: pick two runs to see the metric deltas and the questions that changed.

## Datasets

Upload **JSONL**, one object per line:

```json
{"id": "refund", "question": "What is the refund window?", "relevant_chunk_ids": ["<chunk id>"], "reference_answer": "14 days."}
```

Only `question` is required. Uploaded questions are accepted immediately.

Or have the LLM **draft** questions from a collection: it samples chunks, makes one model call
per chunk, and links each draft to its chunk. Drafts must be reviewed in the UI (accept, edit
or reject); **only accepted items are used in runs**. Drafts are answered by a single chunk, so
they cannot test questions that need several chunks or whole-table answers; write those by
hand. In the review you can edit each question's reference answer and relevant chunk ids.

## Metrics

`GET /api/evals/metrics` lists every metric with the dataset fields it needs and what the
target must return.

| Metric | Needs from the dataset | Needs from the target | How |
|---|---|---|---|
| recall@k, precision@k, hit rate@k, MRR, nDCG@k | `relevant_chunk_ids` | `retrieved_chunk_ids` | plain maths in `metrics.py` |
| faithfulness | nothing extra | `answer`, `contexts` | LLM judge: split the answer into claims and check each against the retrieved passages |
| completeness | `reference_answer` | `answer` | LLM judge: which points of the reference the answer covers |
| answer accuracy | `reference_answer` | `answer` | LLM judge, 0 to 1 with a reason |
| source accuracy | `relevant_chunk_ids` | `cited_chunk_ids` | deterministic F1, no LLM |

Worked example: relevant chunk `c7`, retrieved `[c2, c7, c9]` gives recall@1 = 0, recall@3 = 1,
precision@3 = 1/3, MRR = 1/2, nDCG@3 = 0.631.

A run takes `config.metrics` (a list of ids; omit it for all). Choosing fewer judged metrics
means fewer model calls. The UI shows, for the chosen dataset, how many questions have each
needed field, and warns before a run about metrics it cannot compute.

A metric the target cannot support is **not computed, with the reason** (for example "the
target returned no answer"), never a silent zero. Judge failures are recorded per question. A
missing `OPENAI_API_KEY` stops the run at once instead of failing every question.

The judge's model and prompt version are pinned and stored in each run. An LLM judge has biases
(length, confident tone, its own style). The prompts say what to ignore, but spot-check the
per-question reasons in the UI.

## Targets

A target is anything that answers a question and returns the contract:

```
{answer, contexts[{chunk_id, text}], retrieved_chunk_ids, cited_chunk_ids, timings{retrieval_ms, ttft_ms, total_ms}}
```

Any field it cannot provide is left out.

| Target | Where | Options |
|---|---|---|
| `retrieval` | in the combined app | the retrieval options: `mode`, `top_k`, `rerank`, `rewrite`, `collection`, `filters` |
| `agent` | in the combined app | `collection`, `mode`, `top_k`, `rerank`, `rewrite`, `tools`, `check`, `max_check_retries` |
| `retrieval_http` | standalone | a running retrieval service over HTTP |
| HTTP URL | anywhere | POSTs `{"question": ...}` and reads the contract back |

The `agent` target scores the whole pipeline: retrieval, the answer and its citations. Every
question runs in a thread of its own with all caches off, so runs are comparable, and a question
that needs a tool approved is recorded as an error (an eval cannot approve). Set
`{"tools": true}` to allow table and calculator tools.

**Safety:** an HTTP target may only call hosts in `EVAL_ALLOWED_TARGET_HOSTS` (default
localhost) and may only send environment variables in `EVAL_ALLOWED_HEADER_REFS` (default
`APP_API_KEY`) as headers, by name. Without this, anyone with the UI could point a target at
their own server and receive your keys.

## Runs

A run is a job (`202` with `job_id`; poll `/jobs/{id}`). Before starting, the UI shows how many
judge calls it will make. The report has mean, standard deviation and count per metric, latency
p50 and p95 (retrieval, time to first token, total), the metrics that were not computed, and a
per-question drill-down. The config snapshot records the target (header names only), k values,
metrics, judge model and prompt version. A run is one trace (`eval.run`) with a span per
question.

## HTTP API (under `/api/evals`)

`POST /datasets` (upload), `POST /datasets/generate`, `GET /datasets[/{id}]`,
`PATCH /datasets/{id}/items/{item}`, `DELETE /datasets/{id}`, `GET /metrics`,
`POST /runs/estimate`, `POST /runs`, `GET /runs[/{id}]`, `GET /runs/compare?a=&b=`,
`GET /targets`, `GET /collections`, `GET /jobs/{id}`, `GET /status`, `GET /health`.

## Configuration

`EVAL_DB_PATH`, `EVAL_JUDGE_MODEL` (empty means `LLM_MODEL`), `EVAL_DEFAULT_KS`,
`EVAL_ALLOWED_TARGET_HOSTS`, `EVAL_ALLOWED_HEADER_REFS`, `EVAL_MAX_GENERATE`. See `.env.example`.

## Tests

```bash
make test        # offline: scripted targets, a fake judge, in-memory SQLite and Qdrant
make test-live   # the judge prompts against the real model
```
