"""The Qdrant search adapter. Reads the layout ingest writes (documented in ingest/store.py):
named vectors `dense` and `sparse`, and collection metadata `ingest_spec`."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from common import AppError
from retrieval.filters import build_filter
from retrieval.models import ChunkItem, CollectionInfo, CollectionSpec, RawHit, SparseVec

# The contract with ingest. A test in e2e checks these match ingest's.
DENSE = "dense"
SPARSE = "sparse"
SPEC_KEY = "ingest_spec"


@contextmanager
def _guard() -> Iterator[None]:
    try:
        yield
    except ResponseHandlingException as exc:
        raise AppError(
            "vector_store_unavailable",
            "Cannot reach Qdrant. Is it running (`make up`) and is QDRANT_URL correct?",
            503,
        ) from exc
    except UnexpectedResponse as exc:
        raise AppError(
            "vector_store_error", f"Qdrant returned HTTP {exc.status_code}.", 502
        ) from exc


class QdrantSearchStore:
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

    def scroll(
        self, collection: str, *, source_name: str | None, after: str | None, limit: int
    ) -> tuple[list[ChunkItem], str | None]:
        flt = (
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_name", match=models.MatchValue(value=source_name)
                    )
                ]
            )
            if source_name
            else None
        )
        with _guard():
            points, next_offset = self._client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=limit,
                offset=after,
                with_payload=True,
                with_vectors=False,
            )
        chunks = []
        for p in points:
            payload = p.payload or {}
            chunks.append(
                ChunkItem(
                    chunk_id=str(p.id),
                    source_id=str(payload.get("source_id", "")),
                    source_name=str(payload.get("source_name", "")),
                    page=payload.get("page"),
                    chunk_index=int(payload.get("chunk_index") or 0),
                    text=str(payload.get("text", "")),
                )
            )
        return chunks, (str(next_offset) if next_offset is not None else None)

    def search(
        self,
        collection: str,
        *,
        dense: list[float] | None,
        sparse: SparseVec | None,
        limit: int,
        filters: dict[str, Any],
        prefetch_limit: int | None = None,
    ) -> list[RawHit]:
        flt = build_filter(filters)
        wide = max(prefetch_limit or limit, limit)
        sparse_query = (
            models.SparseVector(indices=sparse.indices, values=sparse.values) if sparse else None
        )
        with _guard():
            if dense is not None and sparse_query is not None:
                # Hybrid: both searches run on the server and are fused there with RRF.
                response = self._client.query_points(
                    collection_name=collection,
                    prefetch=[
                        models.Prefetch(query=dense, using=DENSE, limit=wide, filter=flt),
                        models.Prefetch(query=sparse_query, using=SPARSE, limit=wide, filter=flt),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit,
                    with_payload=True,
                )
            elif dense is not None:
                response = self._client.query_points(
                    collection_name=collection, query=dense, using=DENSE, limit=limit,
                    query_filter=flt, with_payload=True,
                )  # fmt: skip
            elif sparse_query is not None:
                response = self._client.query_points(
                    collection_name=collection, query=sparse_query, using=SPARSE, limit=limit,
                    query_filter=flt, with_payload=True,
                )  # fmt: skip
            else:
                return []
        return [
            RawHit(id=str(p.id), score=float(p.score), payload=dict(p.payload or {}))
            for p in response.points
        ]
