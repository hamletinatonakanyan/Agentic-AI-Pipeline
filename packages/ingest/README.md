# ingest

Turns files into searchable chunks in Qdrant, idempotently. Standalone on port 8001; in the
combined app at `/ui/ingest` and `/api/ingest`.

```
file → parse (Markdown pages) → chunk → embed (dense + BM25) → upsert → delete stale chunks
```

## Quick start

```bash
make up            # Qdrant
make run-ingest    # http://localhost:8001
```

In the UI: pick a **collection** (the domain the chunks go into), choose a chunking strategy,
drop files in, use **Preview chunks** to check the cuts, then **Ingest**. The report lists what
was added, skipped or failed, how many chunks were embedded and how many were reused. The
**Collections and sources** section lists what is stored and lets you delete a file or a whole
collection.

Over HTTP:

```bash
curl -H "X-API-Key: $KEY" -F "files=@handbook.md" -F 'options={"collection":"docs"}' \
     http://localhost:8001/api/ingest/files        # 202 {"job_id": "..."}
curl -H "X-API-Key: $KEY" http://localhost:8001/api/ingest/jobs/<job_id>
```

## Formats

`.pdf .txt .md .docx .csv .xlsx`. Tables become **Markdown pipe tables** everywhere. Large
tables are split by rows and the header row is repeated in every chunk, with no overlap, so
every row is stored exactly once and a table can be rebuilt from its chunks. In an xlsx each
visible sheet is one "page" and the sheet name is kept as the label. HTML and ODF are not
supported.

## Chunking

| Strategy | Behaviour |
|---|---|
| `recursive` (default) | splits at paragraphs, then lines, sentences and words, merging neighbours up to the chunk size, with overlap |
| `markdown` | like `recursive`, but never crosses a `#` heading |
| `fixed` | every N tokens, stepping N minus the overlap |

Size and overlap are in tokens (`INGEST_CHUNK_SIZE`, default 512; `INGEST_CHUNK_OVERLAP`,
default 64). Every chunk keeps exact character offsets into its page, its page number, a label
and its section path.

**Metadata** is an optional JSON object (`{"client": "acme", "year": 2025}`) stored on every
chunk of the run. Retrieval can filter on it (`metadata.year`).

## PDFs, page by page

Each page's text layer is read locally. A page with fewer than `INGEST_SCANNED_TEXT_THRESHOLD`
characters is treated as scanned and sent, with the other scanned pages of that file, to
LlamaParse. Results are cached by file hash, page and tier, so a page is sent once. Without
`LLAMA_CLOUD_API_KEY`, or with `INGEST_PDF_PARSER=local`, scanned pages are skipped with a
warning.

The local text layer is plain text, so tables inside a text-layer PDF are flattened. Only
LlamaParse output is turned into Markdown tables.

## Idempotency, per chunk

- A chunk's id is a UUIDv5 of its source, the vector settings (model, prefix, BM25 language) and
  **its own text**. The same text always gets the same id.
- Edit a file and re-ingest: only chunks with new text are embedded. Unchanged chunks keep their
  id and vectors (their position and metadata are refreshed in place). Chunks that no longer
  exist are deleted last, so search never sees a gap. The report shows `chunks embedded` and
  `chunks reused`.
- A file is skipped outright when its hash and config are unchanged and its chunks are still
  stored.
- Changing the embedding model, prefix or language changes every id, because the old vectors are
  no longer valid. Changing chunk size or metadata re-chunks, but any chunk whose text is the
  same is reused.
- Reuse depends on the strategy. `recursive` and `markdown` cut at paragraph and sentence edges,
  so boundaries stay put and an edit touches the chunk around it. `fixed` windows shift when
  text is inserted, so they reuse little.
- A crash midway is repaired by re-running: what was already written is reused.

A SQLite manifest (`INGEST_MANIFEST_PATH`) records what was ingested and caches LlamaParse
results.

## Collections and domains

One embedding model and one BM25 language per collection. The collection's own metadata stores
`ingest_spec` (model, dimensions, language, query and document prefixes) so retrieval can encode
a query the same way. Mixing is refused with a 409. The payload fields are listed in
`models.PAYLOAD_FIELDS`.

A **domain** is a collection: use one per company or project to keep their documents apart. A
collection is created by its first ingest. You can delete one file
(`DELETE /collections/{name}/sources/{source_id}`) or a whole collection
(`DELETE /collections/{name}?confirm={name}`, where the name must be repeated). Deleting a
collection also clears the manifest, so the same files can be ingested again. Ingest refuses to
delete a collection it did not create.

Embedding models download from Hugging Face on first real use. BM25 has bundled stemmers for 18
languages; for any other language use `none`. BM25 splits on spaces, so for Chinese or Japanese
the dense vectors carry most of the retrieval.

## Server-side paths

`POST /path` ingests a file or folder on the machine running the server instead of an upload.
The path must lie inside one of the folders in `INGEST_ALLOWED_PATHS` (comma-separated);
without it the feature is off. Unsupported file types in a folder are skipped.

## HTTP API (under `/api/ingest`)

| Method | Path | Notes |
|---|---|---|
| POST | `/files` | multipart `files` plus `options` (JSON). `202` with `job_id`. Optional `Idempotency-Key`. |
| POST | `/path` | `{path, options}`. `202`. |
| POST | `/chunks/preview` | one file, no write, never calls LlamaParse (cached pages are used). |
| GET | `/collections`, `/collections/{name}/sources` | |
| DELETE | `/collections/{name}` | needs `?confirm={name}` |
| DELETE | `/collections/{name}/sources/{source_id}` | |
| GET | `/jobs/{id}`, `/status`, `/health` | also at the root when standalone |

## Configuration

See `.env.example`. Main variables: `INGEST_EMBED_MODEL`, `INGEST_SPARSE_LANGUAGE`,
`INGEST_CHUNK_STRATEGY`, `INGEST_CHUNK_SIZE`, `INGEST_CHUNK_OVERLAP`, `INGEST_DEFAULT_COLLECTION`,
`INGEST_PDF_PARSER`, `INGEST_LLAMAPARSE_TIER`, `INGEST_ALLOWED_PATHS`, `INGEST_MAX_UPLOAD_MB`,
`QDRANT_URL`.

## Tests

```bash
make test              # offline: fakes and in-memory Qdrant
make test-integration  # needs Qdrant running
make test-live         # downloads models and spends LlamaParse credits
```
