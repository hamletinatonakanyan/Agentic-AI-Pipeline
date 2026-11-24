"""IngestService: files in, searchable chunks out. Pure Python, no HTTP.

Per file: unchanged? -> skip. Otherwise parse -> chunk -> embed the NEW chunks -> upsert them ->
refresh the position fields of unchanged chunks -> delete this source's stale points. Upsert comes
before delete, so search never sees a gap, and a crash in the middle leaves the old data intact
(the manifest is only updated at the end, so a re-run repairs, and reuses what was written).

Idempotency is per chunk: a point's id is a UUIDv5 of the source, the vector settings (model,
prefix, BM25 language) and the chunk's own text. The same text always gets the same id, so an
unchanged chunk is neither re-embedded nor duplicated, and editing a file re-embeds only the
chunks whose text changed.
"""

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common import AppError
from ingest.chunking import chunk_document
from ingest.errors import ParseError
from ingest.manifest import Manifest, SourceRecord
from ingest.models import (
    SUPPORTED_EXTENSIONS,
    ChunkOptions,
    ChunkPreview,
    CollectionInfo,
    CollectionSpec,
    DeleteCollectionReport,
    DeleteReport,
    FileResult,
    IngestOptions,
    IngestReport,
    IngestStatus,
    PointRecord,
    PreviewResponse,
    SourceInfo,
    UploadedFile,
)
from ingest.parsers import FileParser, clean_name, extension
from ingest.protocols import DenseEmbedder, SparseEmbedder, Tokenizer, VectorStore
from ingest.settings import BM25_LANGUAGES, NO_LANGUAGE, IngestSettings
from ingest.tokenizer import ApproxTokenizer
from observability import current_trace_id, record_error, span

_log = logging.getLogger("ingest")

CHUNKER_VERSION = "1"
MAX_PREVIEW_CHUNKS = 500
_NAMESPACE = uuid.UUID("6f9a0c1e-3b7d-4c2a-9e51-0d4b8a7c2f13")

# (display name, source key, function that reads the bytes)
Item = tuple[str, str, Callable[[], bytes]]


@dataclass(frozen=True)
class Resolved:
    """Options with every default filled in."""

    collection: str
    chunking: ChunkOptions
    language: str
    metadata: dict[str, Any]


class IngestService:
    def __init__(
        self,
        settings: IngestSettings,
        store: VectorStore,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        parser: FileParser,
        manifest: Manifest,
        tokenizer: Tokenizer | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._s = settings
        self._store = store
        self._dense = dense
        self._sparse = sparse
        self._parser = parser
        self._manifest = manifest
        self._tok = tokenizer or ApproxTokenizer()
        self._clock = clock

    @property
    def max_upload_mb(self) -> int:
        return self._s.max_upload_mb

    # ----- running ingest -----------------------------------------------------------------------

    def ingest_files(self, files: list[UploadedFile], opts: IngestOptions) -> IngestReport:
        items: list[Item] = [
            (clean_name(f.name), f"upload/{clean_name(f.name)}", (lambda d=f.data: d))
            for f in files
        ]
        return self._run(items, opts)

    def ingest_path(self, path: str, opts: IngestOptions) -> IngestReport:
        files = self.collect_paths(path)
        items: list[Item] = [(f.name, f"path/{f}", f.read_bytes) for f in files]
        return self._run(items, opts)

    def _run(self, items: list[Item], opts: IngestOptions) -> IngestReport:
        started = time.perf_counter()
        r = self._resolve(opts)
        report = IngestReport(run_id=uuid.uuid4().hex[:12], collection=r.collection)
        with span(
            "ingest.run", {"ingest.collection": r.collection, "ingest.files": len(items)}
        ) as run:
            report.trace_id = current_trace_id()
            spec = self._spec(r)
            self._store.ensure_collection(r.collection, spec)
            seen: set[str] = set()
            for name, key, read in items:
                if key in seen:
                    self._fail(report, name, "duplicate file name in this run; skipped")
                    continue
                seen.add(key)
                self._ingest_one(name, key, read, r, report)
            run.set_attributes(
                {
                    "ingest.added": report.added,
                    "ingest.updated": report.updated,
                    "ingest.skipped": report.skipped,
                    "ingest.failed": len(report.failed),
                    "ingest.chunks_written": report.chunks_written,
                    "ingest.chunks_reused": report.chunks_reused,
                }
            )
        report.duration_ms = (time.perf_counter() - started) * 1000
        return report

    def _ingest_one(
        self, name: str, key: str, read: Callable[[], bytes], r: Resolved, report: IngestReport
    ) -> None:
        with span("ingest.file", {"source": name}) as sp:
            try:
                result = self._process(name, key, read, r, report)
            except AppError:
                raise  # the store is unreachable or the collection is wrong: stop the whole run
            except ParseError as exc:
                record_error(exc, sp)
                self._fail(report, name, str(exc))
                return
            except Exception as exc:  # one bad file must not stop the others
                _log.exception("ingest failed for %s", name)
                record_error(exc, sp)
                self._fail(report, name, f"unexpected error: {type(exc).__name__}: {exc}")
                return
            sp.set_attributes({"ingest.status": result.status, "ingest.chunks": result.chunks})
        report.files.append(result)

    def _process(
        self, name: str, key: str, read: Callable[[], bytes], r: Resolved, report: IngestReport
    ) -> FileResult:
        data = read()
        limit = self._s.max_upload_mb * 1024 * 1024
        if len(data) > limit:
            raise ParseError(f"file is larger than INGEST_MAX_UPLOAD_MB ({self._s.max_upload_mb})")

        collection = r.collection
        source_id = str(uuid.uuid5(_NAMESPACE, f"{collection}\n{key}"))
        content_hash = hashlib.sha256(data).hexdigest()
        config_hash = self._config_hash(r)
        existing_ids = self._store.list_point_ids(collection, source_id)
        record = self._manifest.get_source(collection, source_id)

        if (
            record
            and record.content_hash == content_hash
            and record.config_hash == config_hash
            and record.chunk_count == len(existing_ids)
        ):
            report.skipped += 1
            return FileResult(source=name, status="skipped", chunks=len(existing_ids))

        with span("ingest.parse") as sp:
            doc = self._parser.parse(UploadedFile(name=name, data=data))
            sp.set_attributes(
                {
                    "pages": len(doc.pages),
                    "pages_local": doc.stats.pages_local,
                    "pages_llamaparse": doc.stats.pages_llamaparse,
                    "pages_cached": doc.stats.pages_cached,
                    "pages_skipped": doc.stats.pages_skipped,
                }
            )
        report.pages_local += doc.stats.pages_local
        report.pages_llamaparse += doc.stats.pages_llamaparse
        report.pages_cached += doc.stats.pages_cached
        report.pages_skipped += doc.stats.pages_skipped
        report.warnings += [f"{name}: {w}" for w in doc.warnings]

        with span("ingest.chunk") as sp:
            chunks = chunk_document(doc, r.chunking, self._tok)
            sp.set_attribute("chunks", len(chunks))

        if not chunks:
            if record or existing_ids:
                report.deleted += self._store.delete_by_source(collection, source_id)
                self._manifest.delete_source(collection, source_id)
                report.updated += 1
                report.warnings.append(f"{name}: no text found; its earlier chunks were removed")
                return FileResult(source=name, status="updated", chunks=0)
            report.skipped += 1
            report.warnings.append(f"{name}: no text found; nothing to store")
            return FileResult(source=name, status="skipped")

        # Which chunks are new? A chunk's id depends on its own text, so an unchanged chunk keeps
        # its id and its stored vectors, and only changed chunks are embedded.
        ids = self._chunk_ids(source_id, self._vector_hash(r), chunks)
        new = [i for i, pid in enumerate(ids) if pid not in existing_ids]
        reused = [i for i, pid in enumerate(ids) if pid in existing_ids]

        ingested_at = datetime.fromtimestamp(self._clock(), UTC).isoformat()

        def payload(i: int) -> dict[str, Any]:
            c = chunks[i]
            return {
                "text": c.text,
                "source_id": source_id,
                "source_name": name,
                "content_hash": content_hash,
                "page": c.page_no,
                "label": c.label,
                "section_path": c.section_path,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "chunk_index": i,
                "chunker_version": CHUNKER_VERSION,
                "embedding_model": self._dense.model_name,
                "sparse_language": r.language,
                "parser": c.parser,
                "metadata": r.metadata,
            }

        points: list[PointRecord] = []
        with span(
            "ingest.embed",
            {"chunks": len(new), "reused": len(reused), "model": self._dense.model_name},
        ):
            if new:
                texts = [chunks[i].text for i in new]
                dense = self._dense.embed_documents(texts)
                sparse = self._sparse.embed_documents(texts, r.language)
                points = [
                    PointRecord(
                        id=ids[i],
                        dense=dense[n],
                        sparse=sparse[n],
                        payload={**payload(i), "ingested_at": ingested_at},
                    )
                    for n, i in enumerate(new)
                ]

        with span("ingest.upsert", {"points": len(points), "refreshed": len(reused)}):
            if points:
                self._store.upsert(collection, points)
            if reused:
                # Same text, same vectors: only where the chunk sits (and the metadata) may have
                # changed. `ingested_at` stays as it was: that is when it was embedded.
                self._store.set_payloads(collection, {ids[i]: payload(i) for i in reused})
            # Only now remove what this source had before and no longer has.
            report.deleted += self._store.delete_by_source(collection, source_id, keep_ids=set(ids))

        parsers: dict[str, int] = {}
        for page in doc.pages:
            parsers[page.parser] = parsers.get(page.parser, 0) + 1
        self._manifest.put_source(
            SourceRecord(
                collection=collection,
                source_id=source_id,
                source_name=name,
                content_hash=content_hash,
                config_hash=config_hash,
                chunk_count=len(ids),
                pages=len(doc.pages),
                parsers=parsers,
                metadata=r.metadata,
                last_ingested_at=self._clock(),
            )
        )

        report.chunks_written += len(new)
        report.chunks_reused += len(reused)
        if existing_ids:
            report.updated += 1
            status = "updated"
        else:
            report.added += 1
            status = "added"
        return FileResult(source=name, status=status, chunks=len(ids), reused=len(reused))

    @staticmethod
    def _fail(report: IngestReport, source: str, error: str) -> None:
        report.failed.append({"source": source, "error": error})
        report.files.append(FileResult(source=source, status="failed", error=error))

    # ----- preview, listing, deleting -------------------------------------------------------------

    def preview_chunks(self, file: UploadedFile, opts: IngestOptions) -> PreviewResponse:
        """Parse and chunk one file. Writes nothing and never calls LlamaParser."""
        r = self._resolve(opts)
        with span("ingest.preview", {"source": clean_name(file.name)}):
            trace_id = current_trace_id()
            doc = self._parser.parse(file, allow_remote=False)
            chunks = chunk_document(doc, r.chunking, self._tok)

        warnings = list(doc.warnings)
        if len(chunks) > MAX_PREVIEW_CHUNKS:
            warnings.append(f"showing the first {MAX_PREVIEW_CHUNKS} of {len(chunks)} chunks")
        return PreviewResponse(
            source_name=doc.source_name,
            trace_id=trace_id,
            warnings=warnings,
            stats=doc.stats,
            chunks=[
                ChunkPreview(index=i, **c.model_dump())
                for i, c in enumerate(chunks[:MAX_PREVIEW_CHUNKS])
            ],
        )

    def list_collections(self) -> list[CollectionInfo]:
        return self._store.list_collections()

    def list_sources(self, collection: str) -> list[SourceInfo]:
        return self._manifest.list_sources(collection)

    def delete_source(self, collection: str, source_id: str) -> DeleteReport:
        deleted = 0
        if self._store.get_collection(collection) is not None:
            deleted = self._store.delete_by_source(collection, source_id)
        removed = self._manifest.delete_source(collection, source_id)
        if not removed and deleted == 0:
            raise AppError("source_not_found", f"No source {source_id} in '{collection}'.", 404)
        return DeleteReport(source_id=source_id, chunks_deleted=deleted)

    def delete_collection(self, name: str, confirm: str | None) -> DeleteCollectionReport:
        """Delete a whole collection (a "domain") and forget what was ingested into it.

        Only collections ingest created are touched: a collection that holds something else in
        the same Qdrant is refused. The caller must repeat the name as `confirm`."""
        if confirm != name:
            raise AppError(
                "confirmation_required",
                f"To delete '{name}', send confirm={name} (type the name to confirm).",
                422,
            )
        info = self._store.get_collection(name)
        if info is not None and info.spec is None:
            raise AppError(
                "collection_not_managed",
                f"Collection '{name}' was not created by ingest, so ingest will not delete it.",
                409,
            )
        chunks = info.points if info is not None else 0
        forgotten = self._manifest.delete_collection(name)
        if info is None and forgotten == 0:
            raise AppError("collection_not_found", f"No collection '{name}'.", 404)
        if info is not None:
            self._store.delete_collection(name)
        return DeleteCollectionReport(
            collection=name, chunks_deleted=chunks, sources_forgotten=forgotten
        )

    def status(self) -> IngestStatus:
        s = self._s
        return IngestStatus(
            pdf_parser=s.pdf_parser,
            llamaparse_configured=bool(s.llama_cloud_api_key),
            embed_model=s.embed_model,
            sparse_language=s.sparse_language,
            languages=[*BM25_LANGUAGES, NO_LANGUAGE],
            qdrant_url=s.qdrant_url,
            manifest_path=s.manifest_path,
            allowed_paths=s.allowed_paths,
            supported_extensions=list(SUPPORTED_EXTENSIONS),
            max_upload_mb=s.max_upload_mb,
            default_collection=s.default_collection,
            chunk_strategy=s.chunk_strategy,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
        )

    # ----- server-side paths ------------------------------------------------------------

    def collect_paths(self, path: str) -> list[Path]:
        """The supported files at `path`, which must lie inside INGEST_ALLOWED_PATHS."""
        roots = [Path(p).expanduser().resolve() for p in self._s.allowed_paths]
        if not roots:
            raise AppError(
                "path_not_allowed",
                "Server-side paths are off: INGEST_ALLOWED_PATHS is empty.",
                403,
            )

        def inside(p: Path) -> bool:
            return any(p.is_relative_to(root) for root in roots)

        target = Path(path).expanduser().resolve()  # resolves symlinks and ".."
        if not inside(target):
            raise AppError("path_not_allowed", "That path is outside INGEST_ALLOWED_PATHS.", 403)
        if not target.exists():
            raise AppError("path_not_found", "No such file or folder.", 404)

        if target.is_file():
            if extension(target.name) not in SUPPORTED_EXTENSIONS:
                raise AppError(
                    "unsupported_format",
                    f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
                    415,
                )
            return [target]

        found = []
        for p in sorted(target.rglob("*")):
            relative = p.relative_to(target).parts
            if any(part.startswith(".") for part in relative) or not p.is_file():
                continue
            if extension(p.name) in SUPPORTED_EXTENSIONS and inside(p.resolve()):
                found.append(p.resolve())
        if not found:
            raise AppError(
                "no_supported_files",
                f"No supported files in that folder. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
                404,
            )
        return found

    # ----- options -----------------------------------------------------------------------

    def _resolve(self, opts: IngestOptions) -> Resolved:
        s = self._s
        return Resolved(
            collection=opts.collection or s.default_collection,
            chunking=opts.chunking
            or ChunkOptions(
                strategy=s.chunk_strategy, chunk_size=s.chunk_size, overlap=s.chunk_overlap
            ),
            language=opts.sparse_language or s.sparse_language,
            metadata=opts.metadata,
        )

    def _spec(self, r: Resolved) -> CollectionSpec:
        return CollectionSpec(
            embedding_model=self._dense.model_name,
            dense_dim=self._dense.dim,
            sparse_model=self._sparse.model_name,
            sparse_language=r.language,
            doc_prefix=self._dense.doc_prefix,
            query_prefix=self._dense.query_prefix,
        )

    def _vector_hash(self, r: Resolved) -> str:
        """What a stored vector depends on besides the text. If any of this changes, old vectors
        are no longer valid, so chunk ids change and everything is embedded again."""
        payload = {
            "embedding_model": self._dense.model_name,
            "doc_prefix": self._dense.doc_prefix,
            "sparse_model": self._sparse.model_name,
            "language": r.language,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]

    @staticmethod
    def _chunk_ids(source_id: str, vector_hash: str, chunks: list) -> list[str]:
        """One id per chunk, from the chunk's own text. The same text twice in one file gets
        a counter, so ids stay unique and stay the same from run to run."""
        seen: dict[str, int] = {}
        ids = []
        for c in chunks:
            text_hash = hashlib.sha256(c.text.encode()).hexdigest()
            n = seen.get(text_hash, 0)
            seen[text_hash] = n + 1
            ids.append(str(uuid.uuid5(_NAMESPACE, f"{source_id}:{vector_hash}:{text_hash}:{n}")))
        return ids

    def _config_hash(self, r: Resolved) -> str:
        """Anything that changes what is stored must change this, so changed settings re-ingest."""
        payload = {
            "chunker": CHUNKER_VERSION,
            "chunking": r.chunking.model_dump(),
            "embedding_model": self._dense.model_name,
            "doc_prefix": self._dense.doc_prefix,
            "sparse_model": self._sparse.model_name,
            "language": r.language,
            "metadata": r.metadata,
        }
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:32]
