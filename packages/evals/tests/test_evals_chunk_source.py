import pytest
from qdrant_client import QdrantClient, models

from common import AppError
from evals.chunk_source import QdrantChunkSource


@pytest.fixture
def source() -> QdrantChunkSource:
    client = QdrantClient(":memory:")
    client.create_collection(
        "docs",
        vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.COSINE)},
    )
    client.upsert(
        "docs",
        [
            models.PointStruct(
                id=f"00000000-0000-0000-0000-{i:012d}",
                vector={"dense": [1.0, float(i)]},
                payload={
                    "text": "word " * (10 * i),
                    "source_name": f"s{i % 2}.txt",
                    "page": i,
                    "other": 1,
                },
            )
            for i in range(1, 11)
        ],
    )
    return QdrantChunkSource(client)


def test_samples_chunks_with_their_ids_and_attribution(source):
    chunks = source.sample("docs", 3, seed=1, min_chars=0)
    assert len(chunks) == 3 and len({c.chunk_id for c in chunks}) == 3
    c = chunks[0]
    assert c.chunk_id.startswith("00000000-0000-0000-0000-") and c.text.startswith("word")
    assert c.source_name.endswith(".txt") and c.page in range(1, 11)


def test_the_same_seed_gives_the_same_sample_and_another_seed_a_different_one(source):
    one = [c.chunk_id for c in source.sample("docs", 4, seed=7, min_chars=0)]
    again = [c.chunk_id for c in source.sample("docs", 4, seed=7, min_chars=0)]
    other = [c.chunk_id for c in source.sample("docs", 4, seed=8, min_chars=0)]
    assert one == again and one != other


def test_short_chunks_are_skipped(source):
    chunks = source.sample("docs", 10, seed=0, min_chars=200)  # "word " * 10i is 50i characters
    assert chunks and all(len(c.text) >= 200 for c in chunks) and len(chunks) == 7


def test_asking_for_more_than_exist_returns_what_exists(source):
    assert len(source.sample("docs", 99, seed=0, min_chars=0)) == 10


def test_unknown_collection_is_404_and_listing_works(source):
    with pytest.raises(AppError) as err:
        source.sample("nope", 1, 0, 0)
    assert err.value.code == "collection_not_found" and err.value.status_code == 404
    assert source.list_collections() == ["docs"]


def test_qdrant_down_is_a_clear_503():
    down = QdrantChunkSource(
        QdrantClient(url="http://127.0.0.1:9", timeout=1, check_compatibility=False)
    )
    for call in (lambda: down.list_collections(), lambda: down.sample("docs", 1, 0, 0)):
        with pytest.raises(AppError) as err:
            call()
        assert err.value.code == "vector_store_unavailable" and "make up" in err.value.message
