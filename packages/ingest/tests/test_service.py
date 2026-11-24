import pytest

from common import AppError
from ingest.models import ChunkOptions, IngestOptions, UploadedFile
from observability import init_observability, shutdown_observability
from observability.exporters import InMemoryExporter
from observability.settings import ObsSettings

TEXT_A = "\n\n".join(f"Paragraph {i} of document A talks about topic {i}." * 3 for i in range(8))
TEXT_B = "\n\n".join(f"Paragraph {i} of document B covers subject {i}." * 3 for i in range(8))


def files(**named):
    return [
        UploadedFile(name=n, data=d.encode() if isinstance(d, str) else d) for n, d in named.items()
    ]


def ingest(rig, opts=None, **named):
    return rig.service.ingest_files(files(**named), opts or IngestOptions())


@pytest.fixture(autouse=True)
def _obs_off():
    shutdown_observability()
    yield
    shutdown_observability()


def test_new_file_is_added_with_the_documented_payload(rig):
    report = ingest(rig, **{"a.txt": TEXT_A})
    assert (report.added, report.updated, report.skipped) == (1, 0, 0)
    assert report.chunks_written > 1
    assert report.failed == [] and report.collection == "docs"

    points = list(rig.store.points("docs").values())
    assert len(points) == report.chunks_written
    p = points[0]
    assert set(p.payload) >= {
        "text", "source_id", "source_name", "content_hash", "page", "label", "section_path",
        "char_start", "char_end", "chunk_index", "chunker_version", "embedding_model",
        "sparse_language", "parser", "ingested_at", "metadata",
    }  # fmt: skip
    assert p.payload["source_name"] == "a.txt"
    assert p.payload["embedding_model"] == "fake-dense"
    assert p.payload["ingested_at"].startswith("2023-11-14")
    assert len(p.dense) == 4 and p.sparse.indices


def test_rerunning_the_same_files_changes_nothing(rig):
    first = ingest(rig, **{"a.txt": TEXT_A, "b.txt": TEXT_B})
    ids = set(rig.store.points("docs"))
    embedded = rig.dense.texts_embedded

    second = ingest(rig, **{"a.txt": TEXT_A, "b.txt": TEXT_B})
    assert (second.added, second.updated, second.skipped) == (0, 0, 2)
    assert second.chunks_written == 0 and second.deleted == 0
    assert set(rig.store.points("docs")) == ids
    assert rig.dense.texts_embedded == embedded  # nothing was embedded again
    assert first.chunks_written == len(ids)


def test_changing_one_file_replaces_only_its_chunks(rig):
    ingest(rig, **{"a.txt": TEXT_A, "b.txt": TEXT_B})
    by_source = lambda name: {  # noqa: E731
        i for i, p in rig.store.points("docs").items() if p.payload["source_name"] == name
    }
    a_before, b_before = by_source("a.txt"), by_source("b.txt")

    report = ingest(
        rig, **{"a.txt": TEXT_A + "\n\nA brand new closing paragraph.", "b.txt": TEXT_B}
    )
    assert (report.added, report.updated, report.skipped) == (0, 1, 1)
    assert by_source("b.txt") == b_before  # untouched
    assert len(a_before & by_source("a.txt")) >= len(a_before) - 2  # the old chunks survive
    assert report.chunks_written >= 1 and report.chunks_reused > 0  # only the new tail is embedded
    sources = {s.source_name: s for s in rig.service.list_sources("docs")}
    assert sources["a.txt"].chunk_count == len(by_source("a.txt"))
    assert rig.store.count("docs") == len(by_source("a.txt")) + len(b_before)  # no duplicates


def test_point_ids_are_deterministic_across_fresh_installs(rig_factory):
    ids = []
    for _ in range(2):
        r = rig_factory()
        ingest(r, **{"a.txt": TEXT_A})
        ids.append(sorted(r.store.points("docs")))
    assert ids[0] == ids[1]


def test_changing_chunk_options_re_ingests_without_duplicates(rig):
    ingest(rig, **{"a.txt": TEXT_A})
    small = IngestOptions(chunking=ChunkOptions(chunk_size=20, overlap=0))
    report = ingest(rig, small, **{"a.txt": TEXT_A})
    assert report.updated == 1
    assert rig.store.count("docs") == report.chunks_written + report.chunks_reused


def test_changing_metadata_updates_the_payload_without_embedding_anything(rig):
    ingest(rig, **{"a.txt": TEXT_A})
    embedded = rig.dense.texts_embedded
    report = ingest(rig, IngestOptions(metadata={"year": 2025}), **{"a.txt": TEXT_A})
    assert report.updated == 1
    assert report.chunks_written == 0 and report.chunks_reused > 0
    assert rig.dense.texts_embedded == embedded  # same text, same vectors: nothing re-embedded
    assert all(p.payload["metadata"] == {"year": 2025} for p in rig.store.points("docs").values())


def test_a_wiped_collection_is_repaired_on_the_next_run(rig):
    ingest(rig, **{"a.txt": TEXT_A})
    rig.store.collections["docs"].clear()  # someone emptied it behind our back
    report = ingest(rig, **{"a.txt": TEXT_A})
    assert report.added == 1 and rig.store.count("docs") > 0


def test_one_bad_file_does_not_stop_the_others(rig):
    report = ingest(rig, **{"bad.docx": b"not a docx", "a.txt": TEXT_A})
    assert report.added == 1
    assert [f["source"] for f in report.failed] == ["bad.docx"]
    assert "could not open docx" in report.failed[0]["error"]
    assert {f.status for f in report.files} == {"added", "failed"}


def test_unsupported_type_is_a_failed_file_not_a_crash(rig):
    report = ingest(rig, **{"page.html": "<p>hi</p>"})
    assert "unsupported" in report.failed[0]["error"]


def test_empty_file_is_skipped_with_a_warning(rig):
    report = ingest(rig, **{"empty.txt": "   "})
    assert report.skipped == 1 and rig.store.count("docs") == 0
    assert any("empty.txt" in w for w in report.warnings)


def test_a_file_that_becomes_empty_loses_its_old_chunks(rig):
    ingest(rig, **{"a.txt": TEXT_A})
    report = ingest(rig, **{"a.txt": " "})
    assert report.updated == 1 and report.deleted > 0
    assert rig.store.count("docs") == 0 and rig.service.list_sources("docs") == []


def test_duplicate_names_in_one_run_are_reported(rig):
    report = ingest(rig, **{"a.txt": TEXT_A})
    both = rig.service.ingest_files(
        [UploadedFile("x.txt", b"one"), UploadedFile("x.txt", b"two")], IngestOptions()
    )
    assert both.added == 1 and "duplicate" in both.failed[0]["error"]
    assert report.added == 1


def test_oversized_file_fails(rig_factory):
    r = rig_factory(max_upload_mb=1)
    report = r.service.ingest_files(
        [UploadedFile("big.txt", b"x" * (1024 * 1024 + 1))], IngestOptions()
    )
    assert "INGEST_MAX_UPLOAD_MB" in report.failed[0]["error"]


def test_a_second_collection_is_independent(rig):
    ingest(rig, **{"a.txt": TEXT_A})
    ingest(rig, IngestOptions(collection="other"), **{"a.txt": TEXT_A})
    assert rig.store.count("docs") == rig.store.count("other") > 0


def test_store_errors_abort_the_run_instead_of_marking_every_file_failed(rig):
    rig.store.ensure_collection("docs", rig.store.specs.get("docs") or _spec())
    with pytest.raises(AppError, match="different settings"):
        ingest(rig, IngestOptions(sparse_language="german"), **{"a.txt": TEXT_A})


def _spec():
    from ingest.models import CollectionSpec

    return CollectionSpec(
        embedding_model="other", dense_dim=4, sparse_model="Qdrant/bm25", sparse_language="english"
    )


# ----- PDFs through the whole service ---------------------------------------------------------


def test_scanned_pdf_pages_are_sent_once_across_re_ingests(rig_factory, remote_cls, make_pdf):
    remote = remote_cls()
    r = rig_factory(remote=remote)
    pdf = make_pdf(["A page with a real text layer, long enough to count as text for sure.", None])

    first = r.service.ingest_files([UploadedFile("scan.pdf", pdf)], IngestOptions())
    assert (first.pages_local, first.pages_llamaparse, first.pages_cached) == (1, 1, 0)
    assert remote.calls == [[2]]
    parsers = {p.payload["parser"] for p in r.store.points("docs").values()}
    assert parsers == {"local", "llamaparse"}

    # Same file and options: skipped, so the parser is not even consulted.
    again = r.service.ingest_files([UploadedFile("scan.pdf", pdf)], IngestOptions())
    assert again.skipped == 1 and remote.calls == [[2]]

    # New chunking forces a re-parse, and the scanned page comes from the cache.
    rechunk = r.service.ingest_files(
        [UploadedFile("scan.pdf", pdf)],
        IngestOptions(chunking=ChunkOptions(chunk_size=20, overlap=0)),
    )
    assert rechunk.updated == 1
    assert (rechunk.pages_llamaparse, rechunk.pages_cached) == (0, 1)
    assert remote.calls == [[2]]  # still only one paid send, ever
    assert r.service.list_sources("docs")[0].parsers == {"local": 1, "llamaparse": 1}


def test_local_only_mode_reports_skipped_scanned_pages(rig_factory, make_pdf):
    r = rig_factory(mode="local")
    pdf = make_pdf(["A page with a real text layer, long enough to count as text for sure.", None])
    report = r.service.ingest_files([UploadedFile("scan.pdf", pdf)], IngestOptions())
    assert report.pages_skipped == 1 and report.added == 1
    assert any("INGEST_PDF_PARSER=local" in w for w in report.warnings)


# ----- preview ---------------------------------------------------------------------------------------


def test_preview_returns_chunks_and_writes_nothing(rig):
    resp = rig.service.preview_chunks(UploadedFile("a.txt", TEXT_A.encode()), IngestOptions())
    assert len(resp.chunks) > 1
    assert [c.index for c in resp.chunks] == list(range(len(resp.chunks)))
    assert resp.chunks[0].tokens <= 40
    assert rig.store.collections == {} and rig.dense.calls == 0
    assert rig.service.list_sources("docs") == []


def test_preview_never_calls_the_remote_parser(rig_factory, remote_cls, make_pdf):
    remote = remote_cls()
    r = rig_factory(remote=remote)
    resp = r.service.preview_chunks(UploadedFile("s.pdf", make_pdf([None])), IngestOptions())
    assert remote.calls == [] and resp.stats.pages_skipped == 1
    assert "not previewed" in resp.warnings[0]


def test_preview_is_capped(rig):
    huge = "word " * 40_000
    resp = rig.service.preview_chunks(
        UploadedFile("h.txt", huge.encode()),
        IngestOptions(chunking=ChunkOptions(chunk_size=16, overlap=0)),
    )
    assert len(resp.chunks) == 500 and "first 500" in resp.warnings[-1]


# ----- sources ---------------------------------------------------------------------------------------


def test_delete_source_removes_chunks_and_manifest_row(rig):
    ingest(rig, **{"a.txt": TEXT_A, "b.txt": TEXT_B})
    a = next(s for s in rig.service.list_sources("docs") if s.source_name == "a.txt")
    done = rig.service.delete_source("docs", a.source_id)
    assert done.chunks_deleted == a.chunk_count
    assert [s.source_name for s in rig.service.list_sources("docs")] == ["b.txt"]
    with pytest.raises(AppError) as err:
        rig.service.delete_source("docs", a.source_id)
    assert err.value.status_code == 404


def test_deleted_source_can_be_ingested_again(rig):
    ingest(rig, **{"a.txt": TEXT_A})
    rig.service.delete_source("docs", rig.service.list_sources("docs")[0].source_id)
    assert ingest(rig, **{"a.txt": TEXT_A}).added == 1


# ----- server paths ----------------------------------------------------------------------------------


def test_ingest_path_reads_allowed_folders_only(rig_factory, tmp_path):
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text(TEXT_A)
    (root / "sub" / "b.md").write_text("# Title\n\n" + TEXT_B)
    (root / ".hidden.txt").write_text("secret")
    (root / "image.png").write_bytes(b"png")
    (tmp_path / "outside.txt").write_text("not allowed")

    r = rig_factory(allowed_paths=str(root))
    report = r.service.ingest_path(str(root), IngestOptions())
    assert report.added == 2
    assert {s.source_name for s in r.service.list_sources("docs")} == {"a.txt", "b.md"}

    for bad in (str(tmp_path / "outside.txt"), str(root / ".." / "outside.txt"), "/etc/hosts"):
        with pytest.raises(AppError) as err:
            r.service.ingest_path(bad, IngestOptions())
        assert err.value.code == "path_not_allowed" and err.value.status_code == 403


def test_symlink_escaping_the_allowed_folder_is_refused(rig_factory, tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("top secret")
    (root / "link.txt").symlink_to(tmp_path / "secret.txt")
    r = rig_factory(allowed_paths=str(root))
    with pytest.raises(AppError, match="outside INGEST_ALLOWED_PATHS"):
        r.service.ingest_path(str(root / "link.txt"), IngestOptions())
    with pytest.raises(AppError) as err:  # as part of a folder, it is simply left out
        r.service.ingest_path(str(root), IngestOptions())
    assert err.value.code == "no_supported_files"


def test_paths_are_off_when_no_allowed_folder_is_configured(rig, tmp_path):
    with pytest.raises(AppError, match="INGEST_ALLOWED_PATHS is empty"):
        rig.service.ingest_path(str(tmp_path), IngestOptions())


def test_missing_and_unsupported_paths(rig_factory, tmp_path):
    (tmp_path / "x.png").write_bytes(b"png")
    r = rig_factory(allowed_paths=str(tmp_path))
    with pytest.raises(AppError) as err:
        r.service.ingest_path(str(tmp_path / "nope.txt"), IngestOptions())
    assert err.value.status_code == 404
    with pytest.raises(AppError) as err:
        r.service.ingest_path(str(tmp_path / "x.png"), IngestOptions())
    assert err.value.status_code == 415


def test_the_same_file_by_path_is_idempotent(rig_factory, tmp_path):
    (tmp_path / "a.txt").write_text(TEXT_A)
    r = rig_factory(allowed_paths=str(tmp_path))
    assert r.service.ingest_path(str(tmp_path), IngestOptions()).added == 1
    assert r.service.ingest_path(str(tmp_path), IngestOptions()).skipped == 1


# ----- tracing ---------------------------------------------------------------------------------------


def test_a_run_produces_one_trace_with_a_span_per_step(rig):
    exporter = InMemoryExporter()
    init_observability(ObsSettings(_env_file=None), exporters=[exporter])
    report = ingest(rig, **{"a.txt": TEXT_A})

    spans = {s.name: s for s in exporter.spans}
    assert {"ingest.run", "ingest.file", "ingest.parse", "ingest.chunk", "ingest.embed",
            "ingest.upsert"} <= set(spans)  # fmt: skip
    assert len({s.trace_id for s in exporter.spans}) == 1
    assert report.trace_id == spans["ingest.run"].trace_id
    assert spans["ingest.file"].parent_id == spans["ingest.run"].span_id
    assert spans["ingest.embed"].parent_id == spans["ingest.file"].span_id
    assert spans["ingest.run"].attributes["ingest.added"] == 1
    assert spans["ingest.chunk"].attributes["chunks"] == report.chunks_written


def test_a_failed_file_marks_its_span_as_an_error(rig):
    exporter = InMemoryExporter()
    init_observability(ObsSettings(_env_file=None), exporters=[exporter])
    ingest(rig, **{"bad.docx": b"nope"})
    file_span = next(s for s in exporter.spans if s.name == "ingest.file")
    assert file_span.status == "error"
    assert next(s for s in exporter.spans if s.name == "ingest.run").status == "ok"


def test_ingest_works_with_observability_off(rig):
    assert ingest(rig, **{"a.txt": TEXT_A}).trace_id is None


# ----- per-chunk idempotency -------------------------------------------------------------------------

PARAS = [
    f"Paragraph {i}: " + f"this paragraph is about subject number {i} in some detail. " * 3
    for i in range(10)
]


def doc(paras) -> str:
    return "\n\n".join(paras)


def point_ids(rig):
    return set(rig.store.points("docs"))


def test_editing_one_paragraph_embeds_only_the_changed_chunks(rig):
    first = ingest(rig, **{"a.txt": doc(PARAS)})
    total = first.chunks_written
    assert total >= 8
    ids_before = point_ids(rig)
    embedded_before = rig.dense.texts_embedded

    edited = list(PARAS)
    edited[5] = "Paragraph 5 was rewritten completely with brand new content about something else."
    report = ingest(rig, **{"a.txt": doc(edited)})

    assert report.updated == 1
    assert 1 <= report.chunks_written <= 3  # the edited chunk, maybe a neighbour through overlap
    assert report.chunks_reused == total - report.chunks_written + (len(point_ids(rig)) - total)
    assert rig.dense.texts_embedded - embedded_before == report.chunks_written  # only those
    assert len(ids_before & point_ids(rig)) >= total - 5  # almost every id is unchanged
    assert report.deleted >= 1
    assert rig.store.count("docs") == rig.service.list_sources("docs")[0].chunk_count
    texts = " ".join(p.payload["text"] for p in rig.store.points("docs").values())
    assert "rewritten completely" in texts and "subject number 5" not in texts


def test_unchanged_chunks_keep_their_vectors_and_get_fresh_positions(rig):
    ingest(rig, **{"a.txt": doc(PARAS)})
    by_text = {p.payload["text"]: p for p in rig.store.points("docs").values()}
    last = max(by_text.values(), key=lambda p: p.payload["chunk_index"])
    old_vectors = (last.dense, last.sparse.indices)
    old_ingested_at = last.payload["ingested_at"]
    old_start = last.payload["char_start"]

    # Insert text at the top: every later chunk keeps its text but moves.
    ingest(rig, **{"a.txt": doc(["A new opening paragraph that was not there before."] + PARAS)})
    same = next(
        p for p in rig.store.points("docs").values() if p.payload["text"] == last.payload["text"]
    )
    assert same.id == last.id  # identity follows the text
    assert (same.dense, same.sparse.indices) == old_vectors  # not re-embedded
    assert same.payload["ingested_at"] == old_ingested_at  # still when it was embedded
    assert same.payload["char_start"] > old_start  # but its position is current
    assert same.payload["chunk_index"] > last.payload["chunk_index"] - 1


def test_deleting_a_paragraph_removes_only_what_no_longer_exists(rig):
    first = ingest(rig, **{"a.txt": doc(PARAS)})
    report = ingest(rig, **{"a.txt": doc(PARAS[:4] + PARAS[6:])})
    # Removing paragraphs leaves every remaining chunk intact: nothing is embedded, 6 are deleted.
    assert report.chunks_written == 0
    assert report.chunks_reused == rig.store.count("docs") == first.chunks_written - report.deleted
    assert report.deleted >= 1
    texts = " ".join(p.payload["text"] for p in rig.store.points("docs").values())
    assert "subject number 4" not in texts and "subject number 3" in texts


def test_fixed_windows_shift_so_little_is_reused(rig):
    """Honest limit: fixed-size windows move when text is inserted, so few chunks survive."""
    fixed = IngestOptions(chunking=ChunkOptions(strategy="fixed", chunk_size=40, overlap=8))
    ingest(rig, fixed, **{"a.txt": doc(PARAS)})
    report = ingest(rig, fixed, **{"a.txt": doc(["Extra opening words here."] + PARAS)})
    assert report.chunks_reused <= 1 and report.chunks_written > 5


def test_recursive_realigns_after_an_insertion(rig):
    ingest(rig, **{"a.txt": doc(PARAS)})
    report = ingest(
        rig, **{"a.txt": doc(PARAS[:3] + ["An inserted paragraph in the middle."] + PARAS[3:])}
    )
    assert report.chunks_reused > report.chunks_written


def test_the_same_text_twice_in_one_file_gets_distinct_stable_ids(rig_factory):
    repeated = doc(["Repeated paragraph, exactly the same each time. " * 3] * 3 + PARAS[:2])
    ids = []
    for _ in range(2):
        r = rig_factory()
        report = ingest(r, **{"a.txt": repeated})
        assert r.store.count("docs") == report.chunks_written  # no id collisions lost a chunk
        ids.append(sorted(r.store.points("docs")))
    assert ids[0] == ids[1]


def test_a_failure_midway_is_repaired_and_reuses_what_was_written(rig):
    ingest(rig, **{"a.txt": doc(PARAS)})
    real_delete = rig.store.delete_by_source

    def crash(*args, **kwargs):
        raise RuntimeError("process killed")

    rig.store.delete_by_source = crash
    edited = list(PARAS)
    edited[2] = "Paragraph 2 changed before the crash happened in this run."
    failed = ingest(rig, **{"a.txt": doc(edited)})
    assert failed.failed and "process killed" in failed.failed[0]["error"]
    # New chunks were written before the crash, and the old ones are still there: no data lost.
    assert rig.store.count("docs") > rig.service.list_sources("docs")[0].chunk_count

    rig.store.delete_by_source = real_delete
    embedded = rig.dense.texts_embedded
    retry = ingest(rig, **{"a.txt": doc(edited)})
    assert retry.updated == 1 and retry.chunks_written == 0  # everything was already written
    assert rig.dense.texts_embedded == embedded
    assert rig.store.count("docs") == rig.service.list_sources("docs")[0].chunk_count


def test_different_vector_settings_never_share_ids(rig_factory):
    a, b = rig_factory(), rig_factory()
    b.dense.doc_prefix = "passage: "
    ingest(a, **{"a.txt": doc(PARAS[:2])})
    ingest(b, **{"a.txt": doc(PARAS[:2])})
    assert point_ids(a).isdisjoint(point_ids(b))


def test_run_report_and_file_result_show_reuse(rig):
    ingest(rig, **{"a.txt": doc(PARAS)})
    report = ingest(rig, **{"a.txt": doc(PARAS + ["One more paragraph at the very end."])})
    result = report.files[0]
    assert result.status == "updated" and result.reused == report.chunks_reused > 0
    assert result.chunks == report.chunks_written + report.chunks_reused
