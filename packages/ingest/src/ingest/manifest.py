"""The manifest: SQLite memory of what was ingested.

Three jobs, one file:
- `sources`: what each source looked like when it was last ingested, so unchanged files are skipped.
- `parse_cache`: LlamaParser results by file hash, page number and tier (implements `ParseCache`).
- `jobs`: long-running runs, so a client can poll `GET /jobs/{id}`.
"""

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ingest.models import IngestReport, JobStatus, SourceInfo

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    collection       TEXT NOT NULL,
    source_id        TEXT NOT NULL,
    source_name      TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    config_hash      TEXT NOT NULL,
    chunk_count      INTEGER NOT NULL,
    pages            INTEGER NOT NULL,
    parsers          TEXT NOT NULL,
    metadata         TEXT NOT NULL,
    last_ingested_at REAL NOT NULL,
    PRIMARY KEY (collection, source_id)
);
CREATE TABLE IF NOT EXISTS parse_cache (
    content_hash TEXT NOT NULL,
    page_no      INTEGER NOT NULL,
    tier         TEXT NOT NULL,
    text         TEXT NOT NULL,
    created_at   REAL NOT NULL,
    PRIMARY KEY (content_hash, page_no, tier)
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE,
    state           TEXT NOT NULL,
    created_at      REAL NOT NULL,
    finished_at     REAL,
    report          TEXT,
    error           TEXT NOT NULL DEFAULT ''
);
"""


@dataclass
class SourceRecord:
    collection: str
    source_id: str
    source_name: str
    content_hash: str
    config_hash: str
    chunk_count: int
    pages: int
    parsers: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_ingested_at: float = 0.0


class Manifest:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----- sources -------------------------------------------------------------------------

    def get_source(self, collection: str, source_id: str) -> SourceRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sources WHERE collection=? AND source_id=?", (collection, source_id)
            ).fetchone()
        return self._record(row) if row else None

    def put_source(self, rec: SourceRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.collection,
                    rec.source_id,
                    rec.source_name,
                    rec.content_hash,
                    rec.config_hash,
                    rec.chunk_count,
                    rec.pages,
                    json.dumps(rec.parsers),
                    json.dumps(rec.metadata, default=str),
                    rec.last_ingested_at or time.time(),
                ),
            )

    def delete_source(self, collection: str, source_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM sources WHERE collection=? AND source_id=?", (collection, source_id)
            )
        return cur.rowcount > 0

    def delete_collection(self, collection: str) -> int:
        """Forget every source of a collection, so the same files can be ingested again."""
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM sources WHERE collection=?", (collection,))
        return cur.rowcount

    def list_sources(self, collection: str) -> list[SourceInfo]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sources WHERE collection=? ORDER BY source_name", (collection,)
            ).fetchall()
        return [
            SourceInfo(
                source_id=r["source_id"],
                source_name=r["source_name"],
                content_hash=r["content_hash"],
                chunk_count=r["chunk_count"],
                pages=r["pages"],
                parsers=json.loads(r["parsers"]),
                last_ingested_at=r["last_ingested_at"],
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]

    @staticmethod
    def _record(row: sqlite3.Row) -> SourceRecord:
        return SourceRecord(
            collection=row["collection"],
            source_id=row["source_id"],
            source_name=row["source_name"],
            content_hash=row["content_hash"],
            config_hash=row["config_hash"],
            chunk_count=row["chunk_count"],
            pages=row["pages"],
            parsers=json.loads(row["parsers"]),
            metadata=json.loads(row["metadata"]),
            last_ingested_at=row["last_ingested_at"],
        )

    # ----- parse cache (ParseCache) ------------------------------------------------------------

    def get(self, content_hash: str, page_no: int, tier: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT text FROM parse_cache WHERE content_hash=? AND page_no=? AND tier=?",
                (content_hash, page_no, tier),
            ).fetchone()
        return row["text"] if row else None

    def put(self, content_hash: str, page_no: int, tier: str, text: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO parse_cache VALUES (?,?,?,?,?)",
                (content_hash, page_no, tier, text, time.time()),
            )

    # ----- jobs ----------------------------------------------------------------------------------

    def create_job(self, idempotency_key: str | None = None) -> tuple[JobStatus, bool]:
        """Returns (job, created). Reusing an idempotency key returns the original job."""
        with self._lock, self._conn:
            if idempotency_key:
                row = self._conn.execute(
                    "SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                if row:
                    return self._job(row), False
            job_id = uuid.uuid4().hex
            now = time.time()
            self._conn.execute(
                "INSERT INTO jobs (job_id, idempotency_key, state, created_at) VALUES (?,?,?,?)",
                (job_id, idempotency_key, "queued", now),
            )
        return JobStatus(job_id=job_id, state="queued", created_at=now), True

    def update_job(
        self,
        job_id: str,
        state: str,
        report: IngestReport | None = None,
        error: str = "",
    ) -> None:
        finished = time.time() if state in ("completed", "failed") else None
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE jobs SET state=?, finished_at=COALESCE(?, finished_at), report=?, error=? "
                "WHERE job_id=?",
                (state, finished, report.model_dump_json() if report else None, error, job_id),
            )

    def get_job(self, job_id: str) -> JobStatus | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._job(row) if row else None

    def fail_unfinished_jobs(self) -> int:
        """At startup: jobs left queued or running by a stopped process will never finish."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE jobs SET state='failed', finished_at=?, error='interrupted by a restart' "
                "WHERE state IN ('queued', 'running')",
                (time.time(),),
            )
        return cur.rowcount

    @staticmethod
    def _job(row: sqlite3.Row) -> JobStatus:
        return JobStatus(
            job_id=row["job_id"],
            state=row["state"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
            report=IngestReport.model_validate_json(row["report"]) if row["report"] else None,
            error=row["error"],
        )
