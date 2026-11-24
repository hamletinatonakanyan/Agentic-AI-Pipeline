"""Long-running runs: `submit` returns at once, a worker thread does the work, clients poll.

One worker, so runs go one after another (embedding is CPU-heavy and the manifest is simple).
`inline=True` runs the work in the caller's thread, for deterministic tests.
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from common import AppError
from ingest.manifest import Manifest
from ingest.models import IngestReport, JobStatus
from observability import record_error, span

_log = logging.getLogger("ingest.jobs")


class JobRunner:
    def __init__(self, manifest: Manifest, inline: bool = False) -> None:
        self._manifest = manifest
        self._inline = inline
        self._pool = (
            None if inline else ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        )
        self._manifest.fail_unfinished_jobs()

    def submit(
        self, work: Callable[[], IngestReport], idempotency_key: str | None = None
    ) -> JobStatus:
        job, created = self._manifest.create_job(idempotency_key)
        if not created:
            return job  # same Idempotency-Key: hand back the original job, do not run again
        if self._pool is None:
            self._execute(job.job_id, work)
        else:
            self._pool.submit(self._execute, job.job_id, work)
        return self._manifest.get_job(job.job_id) or job

    def get(self, job_id: str) -> JobStatus | None:
        return self._manifest.get_job(job_id)

    def _execute(self, job_id: str, work: Callable[[], IngestReport]) -> None:
        self._manifest.update_job(job_id, "running")
        try:
            report = work()
        except AppError as exc:
            self._manifest.update_job(job_id, "failed", error=f"{exc.code}: {exc.message}")
        except Exception as exc:
            _log.exception("job %s failed", job_id)
            with span("ingest.job_failed") as sp:
                record_error(exc, sp)
            self._manifest.update_job(job_id, "failed", error=f"{type(exc).__name__}: {exc}"[:500])
        else:
            self._manifest.update_job(job_id, "completed", report=report)

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
