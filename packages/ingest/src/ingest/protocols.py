"""Interfaces for everything outside this package. Tests use fakes for all of them."""

from typing import Any, Protocol

from ingest.models import CollectionInfo, CollectionSpec, PointRecord, SparseVec


class Tokenizer(Protocol):
    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character range (start, end) of each token, in order."""
        ...


class DenseEmbedder(Protocol):
    model_name: str
    doc_prefix: str
    query_prefix: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class SparseEmbedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str], language: str) -> list[SparseVec]: ...


class VectorStore(Protocol):
    def get_collection(self, name: str) -> CollectionInfo | None: ...

    def ensure_collection(self, name: str, spec: CollectionSpec) -> None:
        """Create it, or check an existing one was built with the same model and language."""
        ...

    def list_collections(self) -> list[CollectionInfo]: ...

    def upsert(self, collection: str, points: list[PointRecord]) -> int: ...

    def delete_by_source(
        self, collection: str, source_id: str, keep_ids: set[str] | None = None
    ) -> int:
        """Delete a source's points, except those whose id is in `keep_ids`."""
        ...

    def delete_collection(self, name: str) -> None:
        """Drop the whole collection with all its points."""
        ...

    def count(self, collection: str, source_id: str | None = None) -> int: ...

    def list_point_ids(self, collection: str, source_id: str) -> set[str]:
        """Ids of every point this source currently has."""
        ...

    def set_payloads(self, collection: str, updates: dict[str, dict[str, Any]]) -> None:
        """Merge new payload fields into existing points. Vectors are not touched."""
        ...


class PdfParser(Protocol):
    """A paid or remote parser for scanned pages."""

    def parse(self, pdf: bytes, pages: list[int]) -> dict[int, str]:
        """Parse the given 1-based pages. Returns markdown text by page number."""
        ...


class ParseCache(Protocol):
    def get(self, content_hash: str, page_no: int, tier: str) -> str | None: ...

    def put(self, content_hash: str, page_no: int, tier: str, text: str) -> None: ...
