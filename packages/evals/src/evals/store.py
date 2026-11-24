"""SQLite: datasets with their items, runs with their per-question results, and jobs."""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from evals.models import (
    Dataset,
    DatasetItem,
    DatasetSummary,
    ItemPatch,
    JobStatus,
    QuestionResult,
    Run,
    RunReport,
    RunSummary,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, collection TEXT, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    dataset_id TEXT NOT NULL, id TEXT NOT NULL, position INTEGER NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (dataset_id, id)
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL, created_at REAL NOT NULL,
    finished_at REAL, progress INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0,
    result TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, dataset_name TEXT NOT NULL,
    created_at REAL NOT NULL, started_at REAL, config TEXT NOT NULL, label TEXT NOT NULL,
    report TEXT
);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL, position INTEGER NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (run_id, position)
);
"""


class EvalStore:
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

    # ----- datasets -------------------------------------------------------------------------

    def create_dataset(
        self, name: str, items: list[DatasetItem], collection: str | None = None
    ) -> str:
        dataset_id = uuid.uuid4().hex[:12]
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO datasets VALUES (?,?,?,?)", (dataset_id, name, collection, time.time())
            )
            self._conn.executemany(
                "INSERT INTO items VALUES (?,?,?,?)",
                [(dataset_id, it.id, i, it.model_dump_json()) for i, it in enumerate(items)],
            )
        return dataset_id

    def _summary(self, row: sqlite3.Row, items: list[DatasetItem]) -> DatasetSummary:
        counts: dict[str, int] = {"draft": 0, "accepted": 0, "rejected": 0}
        for it in items:
            counts[it.status] += 1
        return DatasetSummary(
            id=row["id"],
            name=row["name"],
            status="draft" if counts["draft"] else "ready",
            created_at=row["created_at"],
            collection=row["collection"],
            counts=counts,
        )

    def _items(self, dataset_id: str) -> list[DatasetItem]:
        rows = self._conn.execute(
            "SELECT data FROM items WHERE dataset_id=? ORDER BY position", (dataset_id,)
        ).fetchall()
        return [DatasetItem.model_validate_json(r["data"]) for r in rows]

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
            if not row:
                return None
            items = self._items(dataset_id)
        return Dataset(**self._summary(row, items).model_dump(), items=items)

    def list_datasets(self) -> list[DatasetSummary]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
            return [self._summary(r, self._items(r["id"])) for r in rows]

    def patch_item(self, dataset_id: str, item_id: str, patch: ItemPatch) -> DatasetItem | None:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT data FROM items WHERE dataset_id=? AND id=?", (dataset_id, item_id)
            ).fetchone()
            if not row:
                return None
            item = DatasetItem.model_validate_json(row["data"])
            update = patch.model_dump(exclude_none=True)
            item = item.model_copy(update=update)
            self._conn.execute(
                "UPDATE items SET data=? WHERE dataset_id=? AND id=?",
                (item.model_dump_json(), dataset_id, item_id),
            )
        return item

    def delete_dataset(self, dataset_id: str) -> bool:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM items WHERE dataset_id=?", (dataset_id,))
            return self._conn.execute("DELETE FROM datasets WHERE id=?", (dataset_id,)).rowcount > 0

    # ----- jobs (used by runs and by question generation) -------------------------------------

    def create_job(
        self, kind: str, total: int = 0, idempotency_key: str | None = None
    ) -> tuple[JobStatus, bool]:
        with self._lock, self._conn:
            if idempotency_key:
                row = self._conn.execute(
                    "SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                if row:
                    return self._job(row), False
            job_id = uuid.uuid4().hex[:12]
            now = time.time()
            self._conn.execute(
                "INSERT INTO jobs (id, kind, state, created_at, total, idempotency_key) "
                "VALUES (?,?,?,?,?,?)",
                (job_id, kind, "queued", now, total, idempotency_key),
            )
        return JobStatus(
            job_id=job_id, kind=kind, state="queued", created_at=now, total=total
        ), True

    def update_job(
        self,
        job_id: str,
        *,
        state: str | None = None,
        progress: int | None = None,
        total: int | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        sets, args = [], []
        for column, value in (
            ("state", state),
            ("progress", progress),
            ("total", total),
            ("error", error),
        ):
            if value is not None:
                sets.append(f"{column}=?")
                args.append(value)
        if result is not None:
            sets.append("result=?")
            args.append(json.dumps(result))
        if state in ("completed", "failed"):
            sets.append("finished_at=?")
            args.append(time.time())
        if not sets:
            return
        with self._lock, self._conn:
            self._conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", (*args, job_id))

    def get_job(self, job_id: str) -> JobStatus | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job(row) if row else None

    def fail_unfinished_jobs(self) -> int:
        with self._lock, self._conn:
            return self._conn.execute(
                "UPDATE jobs SET state='failed', finished_at=?, error='interrupted by a restart' "
                "WHERE state IN ('queued', 'running')",
                (time.time(),),
            ).rowcount

    @staticmethod
    def _job(row: sqlite3.Row) -> JobStatus:
        return JobStatus(
            job_id=row["id"],
            kind=row["kind"],
            state=row["state"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
            progress=row["progress"],
            total=row["total"],
            result=json.loads(row["result"]),
            error=row["error"],
        )

    # ----- runs -------------------------------------------------------------------------------

    def create_run(
        self, run_id: str, dataset_id: str, dataset_name: str, config: dict[str, Any], label: str
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO runs (id, dataset_id, dataset_name, created_at, config, label) "
                "VALUES (?,?,?,?,?,?)",
                (run_id, dataset_id, dataset_name, time.time(), json.dumps(config), label),
            )

    def mark_run_started(self, run_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE runs SET started_at=? WHERE id=?", (time.time(), run_id))

    def add_result(self, run_id: str, position: int, result: QuestionResult) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO results VALUES (?,?,?)",
                (run_id, position, result.model_dump_json()),
            )

    def save_report(self, run_id: str, report: RunReport) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET report=? WHERE id=?", (report.model_dump_json(), run_id)
            )

    def _run_summary(self, row: sqlite3.Row, job: sqlite3.Row | None) -> RunSummary:
        return RunSummary(
            id=row["id"],
            dataset_id=row["dataset_id"],
            dataset_name=row["dataset_name"],
            state=job["state"] if job else "failed",
            created_at=row["created_at"],
            finished_at=job["finished_at"] if job else None,
            progress=job["progress"] if job else 0,
            total=job["total"] if job else 0,
            target=row["label"],
            error=job["error"] if job else "",
        )

    def get_run(self, run_id: str, with_results: bool = True) -> Run | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                return None
            job = self._conn.execute("SELECT * FROM jobs WHERE id=?", (run_id,)).fetchone()
            results = []
            if with_results:
                rows = self._conn.execute(
                    "SELECT data FROM results WHERE run_id=? ORDER BY position", (run_id,)
                ).fetchall()
                results = [QuestionResult.model_validate_json(r["data"]) for r in rows]
        return Run(
            **self._run_summary(row, job).model_dump(),
            started_at=row["started_at"],
            config=json.loads(row["config"]),
            report=RunReport.model_validate_json(row["report"]) if row["report"] else None,
            results=results,
        )

    def list_runs(self) -> list[RunSummary]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
            jobs = {j["id"]: j for j in self._conn.execute("SELECT * FROM jobs WHERE kind='run'")}
            return [self._run_summary(r, jobs.get(r["id"])) for r in rows]
