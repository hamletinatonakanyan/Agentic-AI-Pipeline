import time

from common import AppError
from ingest.jobs import JobRunner
from ingest.manifest import Manifest, SourceRecord
from ingest.models import IngestReport


def record(**kw) -> SourceRecord:
    base = dict(
        collection="docs", source_id="s1", source_name="a.txt", content_hash="h", config_hash="c",
        chunk_count=3, pages=1, parsers={"local": 1}, metadata={"k": "v"}, last_ingested_at=5.0,
    )  # fmt: skip
    return SourceRecord(**{**base, **kw})


def test_source_roundtrip_replace_list_and_delete():
    m = Manifest(":memory:")
    assert m.get_source("docs", "s1") is None
    m.put_source(record())
    assert m.get_source("docs", "s1") == record()
    m.put_source(record(chunk_count=9))
    assert m.get_source("docs", "s1").chunk_count == 9
    m.put_source(record(source_id="s2", source_name="b.txt"))
    m.put_source(record(collection="other", source_id="s3"))
    assert [s.source_name for s in m.list_sources("docs")] == ["a.txt", "b.txt"]
    assert m.list_sources("docs")[0].parsers == {"local": 1}
    assert m.delete_source("docs", "s1") is True
    assert m.delete_source("docs", "s1") is False


def test_parse_cache_is_keyed_by_hash_page_and_tier_and_keeps_empty_text():
    m = Manifest(":memory:")
    assert m.get("h", 1, "fast") is None
    m.put("h", 1, "fast", "text")
    m.put("h", 2, "fast", "")
    assert m.get("h", 1, "fast") == "text"
    assert m.get("h", 2, "fast") == ""  # an empty result is a hit, not a miss
    assert m.get("h", 1, "agentic") is None
    assert m.get("other", 1, "fast") is None


def test_manifest_persists_in_a_file(tmp_path):
    path = str(tmp_path / "data" / "m.sqlite")
    first = Manifest(path)
    first.put_source(record())
    first.put("h", 1, "fast", "cached")
    first.close()
    second = Manifest(path)
    assert second.get_source("docs", "s1").chunk_count == 3
    assert second.get("h", 1, "fast") == "cached"


def report() -> IngestReport:
    return IngestReport(run_id="r", collection="docs", added=1)


def test_job_lifecycle_inline():
    m = Manifest(":memory:")
    jobs = JobRunner(m, inline=True)
    job = jobs.submit(report)
    assert job.state == "completed" and job.report.added == 1
    assert jobs.get(job.job_id).finished_at is not None
    assert jobs.get("nope") is None


def test_failing_job_is_recorded_not_raised():
    jobs = JobRunner(Manifest(":memory:"), inline=True)

    def boom():
        raise RuntimeError("disk on fire")

    job = jobs.submit(boom)
    assert job.state == "failed" and "disk on fire" in job.error

    def refused():
        raise AppError("vector_store_unavailable", "Cannot reach Qdrant", 503)

    assert "vector_store_unavailable" in jobs.submit(refused).error


def test_same_idempotency_key_returns_the_first_job_and_runs_once():
    jobs = JobRunner(Manifest(":memory:"), inline=True)
    runs = []

    def work():
        runs.append(1)
        return report()

    first = jobs.submit(work, idempotency_key="k1")
    second = jobs.submit(work, idempotency_key="k1")
    other = jobs.submit(work, idempotency_key="k2")
    assert first.job_id == second.job_id != other.job_id
    assert len(runs) == 2


def test_unfinished_jobs_are_failed_on_restart():
    m = Manifest(":memory:")
    job, _ = m.create_job()
    m.update_job(job.job_id, "running")
    JobRunner(m, inline=True)  # a new process starts
    got = m.get_job(job.job_id)
    assert got.state == "failed" and "restart" in got.error


def test_background_worker_completes_the_job():
    jobs = JobRunner(Manifest(":memory:"))
    try:
        job = jobs.submit(report)
        for _ in range(100):
            if jobs.get(job.job_id).state == "completed":
                break
            time.sleep(0.02)
        assert jobs.get(job.job_id).state == "completed"
    finally:
        jobs.shutdown()
