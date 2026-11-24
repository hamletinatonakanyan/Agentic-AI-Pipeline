"""Interfaces for everything outside this package. Tests use fakes for all of them."""

from typing import Any, Protocol

from retrieval.models import (
    ChunkItem,
    CollectionInfo,
    CollectionSpec,
    RawHit,
    RewriteStrategy,
    SparseVec,
)


class SearchStore(Protocol):
    def get_collection(self, name: str) -> CollectionInfo | None: ...

    def list_collections(self) -> list[CollectionInfo]: ...

    def scroll(
        self, collection: str, *, source_name: str | None, after: str | None, limit: int
    ) -> tuple[list[ChunkItem], str | None]:
        """Stored chunks in a stable order, optionally of one source, `limit` at a time. The
        second value is the cursor for the next page, or None at the end."""
        ...

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
        """Dense only, sparse only, or both (fused on the server with RRF).

        For both, each search first fetches `prefetch_limit` candidates (default `limit`), and
        the fusion keeps the best `limit`. A wide prefetch matters: a chunk that is great for
        keywords but mediocre for meaning only gets credit from both lists if it is in both."""
        ...
        ...


class QueryEncoder(Protocol):
    def encode_dense(self, query: str, spec: CollectionSpec) -> list[float]:
        """Encode like the collection's documents were: its model, and its query prefix."""
        ...

    def encode_sparse(self, query: str, spec: CollectionSpec) -> SparseVec:
        """BM25 with the collection's language."""
        ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str], model: str | None = None) -> list[float]:
        """One relevance score per text, higher is better."""
        ...


class QueryRewriter(Protocol):
    def rewrite(self, query: str, strategy: RewriteStrategy) -> list[str]:
        """The queries to search. The original query always comes first."""
        ...
