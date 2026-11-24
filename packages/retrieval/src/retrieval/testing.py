"""Fakes for tests in any package. `FakeEncoder` uses the same hashing as `ingest.testing`, so text
ingested with the ingest fakes can be found again with these (the cross-package contract test)."""

import hashlib
from collections.abc import Callable

from qdrant_client import QdrantClient, models

from retrieval.models import CollectionSpec, RerankModelInfo, RewriteStrategy, SparseVec
from retrieval.store import DENSE, SPARSE, SPEC_KEY

PUNCT = ".,;:!?\"'()"  # identical in ingest.testing and retrieval.testing


class FakeEncoder:
    def __init__(self) -> None:
        self.dense_calls: list[tuple[str, CollectionSpec]] = []
        self.sparse_calls: list[tuple[str, CollectionSpec]] = []

    def encode_dense(self, query: str, spec: CollectionSpec) -> list[float]:
        self.dense_calls.append((query, spec))
        return [b / 255 for b in hashlib.sha256(query.encode()).digest()[:4]]

    def encode_sparse(self, query: str, spec: CollectionSpec) -> SparseVec:
        self.sparse_calls.append((query, spec))
        words = sorted({w.strip(PUNCT).lower() for w in query.split()} - {""})
        return SparseVec(
            indices=[int(hashlib.md5(w.encode()).hexdigest()[:6], 16) for w in words],
            values=[1.0] * len(words),
        )


class FakeReranker:
    """Scores by how many query words a text contains, unless given a function."""

    def __init__(self, scorer: Callable[[str, str], float] | None = None) -> None:
        self._scorer = scorer or (
            lambda q, t: float(sum(w.lower() in t.lower() for w in q.split()))
        )
        self.calls: list[tuple[str, list[str], str | None]] = []

    def score(self, query: str, texts: list[str], model: str | None = None) -> list[float]:
        self.calls.append((query, list(texts), model))
        return [self._scorer(query, t) for t in texts]

    def models(self) -> list[RerankModelInfo]:
        return [RerankModelInfo(name="fake-reranker", size_gb=0.0, description="for tests")]


class FakeRewriter:
    def __init__(self, extra: list[str] | None = None) -> None:
        self.extra = extra or ["a second phrasing", "a third phrasing"]
        self.calls: list[tuple[str, RewriteStrategy]] = []

    def rewrite(self, query: str, strategy: RewriteStrategy) -> list[str]:
        self.calls.append((query, strategy))
        return [query] if strategy == "none" else [query, *self.extra]


# ----- a small seeded collection, in the layout ingest writes ----------------------------------

SPEC = CollectionSpec(
    embedding_model="fake-dense", dense_dim=4, sparse_model="Qdrant/bm25",
    sparse_language="english", doc_prefix="", query_prefix="Q: ",
)  # fmt: skip

# (source, page, year, text)
DOCS = [
    ("cats.txt", 1, 2023, "Cats are small domestic animals that purr when content."),
    ("cats.txt", 2, 2023, "A cat sleeps most of the day and hunts at night."),
    ("dogs.txt", 1, 2024, "Dogs are loyal companions that bark at strangers."),
    ("dogs.txt", 2, 2024, "A dog needs a daily walk and plenty of water."),
    ("birds.txt", 1, 2025, "Birds have feathers and most of them can fly high."),
    ("birds.txt", 3, 2025, "The zebrafinch is a small bird from Australia."),
    ("fish.txt", 1, 2025, "Fish breathe through gills and live in water."),
    ("space.txt", 4, 2022, "Jupiter is the largest planet in the solar system."),
]


def point_id(i: int) -> str:
    return f"00000000-0000-0000-0000-{i:012d}"


def seed(client: QdrantClient, name: str = "docs", spec: CollectionSpec | None = SPEC) -> None:
    metadata = {SPEC_KEY: spec.model_dump()} if spec else None
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        metadata=metadata,
    )
    enc = FakeEncoder()
    points = []
    for i, (source, page, year, text) in enumerate(DOCS, start=1):
        sparse = enc.encode_sparse(text, SPEC)
        points.append(
            models.PointStruct(
                id=point_id(i),
                vector={
                    DENSE: enc.encode_dense(text, SPEC),
                    SPARSE: models.SparseVector(indices=sparse.indices, values=sparse.values),
                },
                payload={
                    "text": text, "source_name": source, "source_id": source, "page": page,
                    "metadata": {"year": year}, "chunk_index": i,
                },
            )
        )  # fmt: skip
    client.upsert(name, points)
