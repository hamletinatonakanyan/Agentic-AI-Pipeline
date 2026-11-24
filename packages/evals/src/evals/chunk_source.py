"""Reads chunks straight from Qdrant (the payload ingest wrote), so evals never imports ingest."""

import random

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from common import AppError
from evals.protocols import ChunkRef

MAX_SCAN = 50_000  # chunks read to sample from


class QdrantChunkSource:
    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def _unavailable(self, exc: Exception) -> AppError:
        if isinstance(exc, UnexpectedResponse):
            return AppError("vector_store_error", f"Qdrant returned HTTP {exc.status_code}.", 502)
        return AppError(
            "vector_store_unavailable",
            "Cannot reach Qdrant. Is it running (`make up`) and is QDRANT_URL correct?",
            503,
        )

    def list_collections(self) -> list[str]:
        try:
            return sorted(c.name for c in self._client.get_collections().collections)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise self._unavailable(exc) from exc

    def sample(self, collection: str, n: int, seed: int, min_chars: int) -> list[ChunkRef]:
        try:
            if not self._client.collection_exists(collection):
                raise AppError("collection_not_found", f"No collection '{collection}'.", 404)
            chunks: list[ChunkRef] = []
            offset = None
            while len(chunks) < MAX_SCAN:
                points, offset = self._client.scroll(
                    collection,
                    limit=1000,
                    offset=offset,
                    with_payload=["text", "source_name", "page"],
                    with_vectors=False,
                )
                for p in points:
                    payload = p.payload or {}
                    text = str(payload.get("text", ""))
                    if len(text) >= min_chars:
                        chunks.append(
                            ChunkRef(
                                chunk_id=str(p.id),
                                text=text,
                                source_name=str(payload.get("source_name", "")),
                                page=payload.get("page"),
                            )
                        )
                if offset is None:
                    break
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise self._unavailable(exc) from exc
        rng = random.Random(seed)
        return rng.sample(chunks, min(n, len(chunks)))
