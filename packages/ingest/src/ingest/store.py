"""The Qdrant adapter. Dense and BM25 sparse vectors live in one collection.

Collection layout (the contract with retrieval):
- named dense vector `dense` (cosine) and named sparse vector `sparse` (IDF modifier, for BM25)
- collection metadata `ingest_spec`: the embedding model, dimension, sparse model, language and the
  query/document prefixes, so retrieval can encode a query the same way the documents were encoded
- payload per point: see `ingest.models.PAYLOAD_FIELDS`
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from common import AppError
from ingest.models import CollectionInfo, CollectionSpec, PointRecord

DENSE = "dense"
SPARSE = "sparse"
SPEC_KEY = "ingest_spec"
_BATCH = 64


@contextmanager
def _guard() -> Iterator[None]:
    """Turn 'cannot reach Qdrant' into a clear 503 instead of a stack trace."""
    try:
        yield
    except ResponseHandlingException as exc:
        raise AppError(
            "vector_store_unavailable",
            "Cannot reach Qdrant. Is it running (`make up`) and is QDRANT_URL correct?",
            503,
            {"reason": type(exc.source).__name__ if getattr(exc, "source", None) else "unknown"},
        ) from exc
    except UnexpectedResponse as exc:
        raise AppError(
            "vector_store_error", f"Qdrant returned HTTP {exc.status_code}.", 502
        ) from exc


def _source_filter(source_id: str, keep_ids: set[str] | None = None) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id))],
        must_not=[models.HasIdCondition(has_id=sorted(keep_ids))] if keep_ids else None,
    )


class QdrantVectorStore:
    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def get_collection(self, name: str) -> CollectionInfo | None:
        with _guard():
            if not self._client.collection_exists(name):
                return None
            info = self._client.get_collection(name)
        raw = (info.config.metadata or {}).get(SPEC_KEY)
        return CollectionInfo(
            name=name,
            points=info.points_count or 0,
            spec=CollectionSpec.model_validate(raw) if raw else None,
        )

    def list_collections(self) -> list[CollectionInfo]:
        with _guard():
            names = sorted(c.name for c in self._client.get_collections().collections)
        return [info for n in names if (info := self.get_collection(n)) is not None]

    def ensure_collection(self, name: str, spec: CollectionSpec) -> None:
        existing = self.get_collection(name)
        if existing is None:
            with _guard():
                self._client.create_collection(
                    collection_name=name,
                    vectors_config={
                        DENSE: models.VectorParams(
                            size=spec.dense_dim, distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                    },
                    metadata={SPEC_KEY: spec.model_dump()},
                )
                self._client.create_payload_index(
                    name, "source_id", models.PayloadSchemaType.KEYWORD
                )
            return

        if existing.spec is None:
            raise AppError(
                "collection_not_managed",
                f"Collection '{name}' exists but was not created by ingest. Use another name.",
                409,
            )
        old = existing.spec
        differences = {
            field: [getattr(old, field), getattr(spec, field)]
            for field in ("embedding_model", "dense_dim", "sparse_model", "sparse_language")
            if getattr(old, field) != getattr(spec, field)
        }
        if differences:
            raise AppError(
                "collection_mismatch",
                f"Collection '{name}' was built with different settings, and vectors from "
                "different models or languages cannot be mixed. Use a new collection name.",
                409,
                {"differences": differences},
            )

    def delete_collection(self, name: str) -> None:
        with _guard():
            self._client.delete_collection(name)

    def upsert(self, collection: str, points: list[PointRecord]) -> int:
        with _guard():
            for i in range(0, len(points), _BATCH):
                batch = points[i : i + _BATCH]
                self._client.upsert(
                    collection_name=collection,
                    points=[
                        models.PointStruct(
                            id=p.id,
                            vector={
                                DENSE: p.dense,
                                SPARSE: models.SparseVector(
                                    indices=p.sparse.indices, values=p.sparse.values
                                ),
                            },
                            payload=p.payload,
                        )
                        for p in batch
                    ],
                    wait=True,
                )
        return len(points)

    def delete_by_source(
        self, collection: str, source_id: str, keep_ids: set[str] | None = None
    ) -> int:
        before = self.count(collection, source_id)
        with _guard():
            self._client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(filter=_source_filter(source_id, keep_ids)),
                wait=True,
            )
        return before - self.count(collection, source_id)

    def list_point_ids(self, collection: str, source_id: str) -> set[str]:
        ids: set[str] = set()
        offset = None
        with _guard():
            while True:
                points, offset = self._client.scroll(
                    collection_name=collection,
                    scroll_filter=_source_filter(source_id),
                    limit=1000,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                ids.update(str(p.id) for p in points)
                if offset is None:
                    return ids

    def set_payloads(self, collection: str, updates: dict[str, dict[str, Any]]) -> None:
        operations = [
            models.SetPayloadOperation(set_payload=models.SetPayload(payload=payload, points=[pid]))
            for pid, payload in updates.items()
        ]
        with _guard():
            for i in range(0, len(operations), _BATCH):
                self._client.batch_update_points(
                    collection_name=collection,
                    update_operations=operations[i : i + _BATCH],
                    wait=True,
                )

    def count(self, collection: str, source_id: str | None = None) -> int:
        with _guard():
            result = self._client.count(
                collection_name=collection,
                count_filter=_source_filter(source_id) if source_id else None,
                exact=True,
            )
        return result.count
