"""The real Qdrant adapter, against qdrant-client's in-memory mode (no Docker)."""

import pytest
from qdrant_client import QdrantClient, models

from common import AppError
from ingest.models import CollectionSpec, PointRecord, SparseVec
from ingest.store import DENSE, SPARSE, QdrantVectorStore

SPEC = CollectionSpec(
    embedding_model="fake", dense_dim=3, sparse_model="Qdrant/bm25", sparse_language="english",
    doc_prefix="D: ", query_prefix="Q: ",
)  # fmt: skip


def point(i: int, source: str = "s1") -> PointRecord:
    return PointRecord(
        id=f"00000000-0000-0000-0000-{i:012d}",
        dense=[1.0, 0.0, float(i)],
        sparse=SparseVec(indices=[1, 2 + i], values=[1.0, 0.5]),
        payload={"source_id": source, "text": f"chunk {i}"},
    )


@pytest.fixture
def store() -> QdrantVectorStore:
    return QdrantVectorStore(QdrantClient(":memory:"))


def test_create_collection_and_read_its_spec_back(store):
    assert store.get_collection("docs") is None
    store.ensure_collection("docs", SPEC)
    info = store.get_collection("docs")
    assert info.spec == SPEC and info.points == 0
    assert [c.name for c in store.list_collections()] == ["docs"]


def test_ensure_is_idempotent_for_the_same_spec(store):
    store.ensure_collection("docs", SPEC)
    store.ensure_collection("docs", SPEC)


@pytest.mark.parametrize(
    "change",
    [{"embedding_model": "other"}, {"dense_dim": 8}, {"sparse_language": "german"}],
)
def test_mixing_models_or_languages_is_refused(store, change):
    store.ensure_collection("docs", SPEC)
    with pytest.raises(AppError) as err:
        store.ensure_collection("docs", SPEC.model_copy(update=change))
    assert err.value.code == "collection_mismatch" and err.value.status_code == 409
    assert list(change)[0] in err.value.details["differences"]


def test_a_collection_not_made_by_ingest_is_refused(store):
    store._client.create_collection(
        "foreign", vectors_config=models.VectorParams(size=3, distance="Cosine")
    )
    with pytest.raises(AppError) as err:
        store.ensure_collection("foreign", SPEC)
    assert err.value.code == "collection_not_managed"


def test_upsert_is_idempotent_and_counts_by_source(store):
    store.ensure_collection("docs", SPEC)
    pts = [point(1), point(2), point(3, "s2")]
    assert store.upsert("docs", pts) == 3
    store.upsert("docs", pts)  # same ids: overwritten, not duplicated
    assert store.count("docs") == 3
    assert store.count("docs", "s1") == 2 and store.count("docs", "s2") == 1
    assert store.get_collection("docs").points == 3


def test_delete_by_source_can_keep_the_new_points(store):
    store.ensure_collection("docs", SPEC)
    store.upsert("docs", [point(1), point(2), point(3), point(4, "s2")])
    keep = {point(1).id, point(2).id}
    assert store.delete_by_source("docs", "s1", keep_ids=keep) == 1
    assert store.count("docs", "s1") == 2 and store.count("docs", "s2") == 1
    assert store.delete_by_source("docs", "s1") == 2
    assert store.count("docs") == 1


def test_large_upserts_are_batched(store):
    store.ensure_collection("docs", SPEC)
    store.upsert("docs", [point(i) for i in range(1, 150)])
    assert store.count("docs") == 149


def test_stored_points_work_for_dense_sparse_and_hybrid_search(store):
    """What ingest writes is what retrieval needs to query."""
    store.ensure_collection("docs", SPEC)
    store.upsert("docs", [point(i) for i in range(1, 4)])
    c = store._client
    dense = c.query_points("docs", query=[1.0, 0.0, 2.0], using=DENSE, limit=2)
    sparse = c.query_points(
        "docs", query=models.SparseVector(indices=[3], values=[1.0]), using=SPARSE, limit=2
    )
    hybrid = c.query_points(
        "docs",
        prefetch=[
            models.Prefetch(query=[1.0, 0.0, 2.0], using=DENSE, limit=5),
            models.Prefetch(
                query=models.SparseVector(indices=[3], values=[1.0]), using=SPARSE, limit=5
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=3,
    )
    assert dense.points and sparse.points and len(hybrid.points) == 3
    assert hybrid.points[0].payload["text"].startswith("chunk")


def test_unreachable_qdrant_becomes_a_clear_503():
    store = QdrantVectorStore(
        QdrantClient(url="http://127.0.0.1:9", timeout=1, check_compatibility=False)
    )
    with pytest.raises(AppError) as err:
        store.count("docs")
    assert err.value.status_code == 503 and err.value.code == "vector_store_unavailable"
    assert "make up" in err.value.message


@pytest.mark.integration
def test_against_a_real_qdrant_server(rig_factory, make_pdf):
    """Run with `make up` then `make test-integration`. Uses a throwaway collection."""
    import uuid

    from ingest.models import IngestOptions, UploadedFile

    real = QdrantVectorStore(QdrantClient(url="http://localhost:6333", check_compatibility=False))
    rig = rig_factory()
    rig.service._store = real
    name = f"it-{uuid.uuid4().hex[:8]}"
    opts = IngestOptions(collection=name)
    files = [UploadedFile("a.txt", b"Integration text. " * 40)]
    try:
        first = rig.service.ingest_files(files, opts)
        again = rig.service.ingest_files(files, opts)
        assert first.added == 1 and again.skipped == 1
        assert real.count(name) == first.chunks_written
    finally:
        real._client.delete_collection(name)


def test_list_point_ids_covers_more_than_one_scroll_page(store):
    store.ensure_collection("docs", SPEC)
    store.upsert("docs", [point(i) for i in range(1, 1201)])
    store.upsert("docs", [point(5000, "other")])
    ids = store.list_point_ids("docs", "s1")
    assert len(ids) == 1200 and point(5000).id not in ids
    assert store.list_point_ids("docs", "nobody") == set()


def test_set_payloads_merges_fields_and_leaves_vectors_alone(store):
    store.ensure_collection("docs", SPEC)
    store.upsert("docs", [point(1), point(2)])
    before = store._client.retrieve("docs", [point(1).id], with_vectors=True)[0].vector
    store.set_payloads("docs", {point(1).id: {"page": 7, "text": "edited"}})
    got = store._client.retrieve("docs", [point(1).id, point(2).id], with_vectors=True)
    first = next(p for p in got if p.id == point(1).id)
    second = next(p for p in got if p.id == point(2).id)
    assert first.payload == {"source_id": "s1", "text": "edited", "page": 7}  # merged, not replaced
    assert first.vector == before  # vectors untouched
    assert second.payload["text"] == "chunk 2"  # other points untouched


def test_set_payloads_handles_many_points(store):
    store.ensure_collection("docs", SPEC)
    store.upsert("docs", [point(i) for i in range(1, 201)])
    store.set_payloads("docs", {point(i).id: {"n": i} for i in range(1, 201)})
    got = store._client.retrieve("docs", [point(150).id])
    assert got[0].payload["n"] == 150
