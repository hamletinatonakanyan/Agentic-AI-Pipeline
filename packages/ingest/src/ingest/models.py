"""The contract: option, document, chunk, report and API models.

The Qdrant payload (see `PAYLOAD_FIELDS`) is the contract with retrieval. Retrieval reads it
without importing this package.
"""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ingest.settings import check_language

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx", ".csv", ".xlsx")

ChunkStrategy = Literal["fixed", "recursive", "markdown"]
ParserName = Literal["local", "llamaparse"]

# Fields every point carries in its Qdrant payload. Retrieval relies on these names.
PAYLOAD_FIELDS = (
    "text",
    "source_id",
    "source_name",
    "content_hash",
    "page",
    "label",
    "section_path",
    "char_start",
    "char_end",
    "chunk_index",
    "chunker_version",
    "embedding_model",
    "sparse_language",
    "parser",
    "ingested_at",
    "metadata",
)


@dataclass(frozen=True)
class UploadedFile:
    name: str
    data: bytes


# ----- options ---------------------------------------------------------------------------


class ChunkOptions(BaseModel):
    strategy: ChunkStrategy = "recursive"
    chunk_size: int = Field(default=512, ge=16, le=8000, description="tokens")
    overlap: int = Field(default=64, ge=0, description="tokens")

    @model_validator(mode="after")
    def _overlap_smaller(self) -> "ChunkOptions":
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class IngestOptions(BaseModel):
    """Everything a caller may choose per run. Unset fields fall back to settings."""

    collection: str | None = Field(default=None, min_length=1, max_length=100)
    chunking: ChunkOptions | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # BM25 language for the sparse vectors: a bundled language, or "none".
    sparse_language: str | None = None

    @field_validator("sparse_language")
    @classmethod
    def _language(cls, value: str | None) -> str | None:
        return None if value is None else check_language(value)

    @field_validator("collection")
    @classmethod
    def _collection_name(cls, value: str | None) -> str | None:
        if value is not None and not all(c.isalnum() or c in "-_." for c in value):
            raise ValueError("collection may contain only letters, digits, '-', '_' and '.'")
        return value


# ----- parsing ---------------------------------------------------------------------------


class PageResult(BaseModel):
    page_no: int  # 1-based. For xlsx, one "page" per sheet.
    text: str
    parser: ParserName = "local"
    label: str = ""  # sheet name for xlsx
    cached: bool = False  # True when a paid parser result came from the cache


class ParseStats(BaseModel):
    pages_local: int = 0
    pages_llamaparse: int = 0  # sent to LlamaParser in this run
    pages_cached: int = 0  # served from the cache, not sent
    pages_skipped: int = 0  # scanned pages that could not be parsed


class Document(BaseModel):
    source_name: str
    content_hash: str  # sha256 of the file bytes
    pages: list[PageResult]
    warnings: list[str] = Field(default_factory=list)
    stats: ParseStats = Field(default_factory=ParseStats)


# ----- chunks and points -------------------------------------------------------------------


class TextChunk(BaseModel):
    text: str
    page_no: int
    label: str = ""
    section_path: str = ""
    char_start: int
    char_end: int
    tokens: int
    parser: ParserName = "local"


class SparseVec(BaseModel):
    indices: list[int]
    values: list[float]


class PointRecord(BaseModel):
    id: str
    dense: list[float]
    sparse: SparseVec
    payload: dict[str, Any]


class CollectionSpec(BaseModel):
    """What a collection was built with. Stored in the collection's own metadata."""

    embedding_model: str
    dense_dim: int
    sparse_model: str
    sparse_language: str
    doc_prefix: str = ""
    query_prefix: str = ""


class CollectionInfo(BaseModel):
    name: str
    points: int = 0
    spec: CollectionSpec | None = None


# ----- reports and API responses ---------------------------------------------------------------


class FileResult(BaseModel):
    source: str
    status: Literal["added", "updated", "skipped", "failed"]
    chunks: int = 0  # chunks this file has now
    reused: int = 0  # of those, chunks that were already stored and not embedded again
    error: str = ""


class IngestReport(BaseModel):
    run_id: str
    trace_id: str | None = None
    collection: str
    added: int = 0
    updated: int = 0
    skipped: int = 0
    deleted: int = 0
    chunks_written: int = 0  # chunks newly embedded and written
    chunks_reused: int = 0  # chunks already stored, identical text, not embedded again
    pages_local: int = 0
    pages_llamaparse: int = 0
    pages_cached: int = 0
    pages_skipped: int = 0
    warnings: list[str] = Field(default_factory=list)
    failed: list[dict[str, str]] = Field(default_factory=list)  # [{source, error}]
    files: list[FileResult] = Field(default_factory=list)
    duration_ms: float = 0.0


class SourceInfo(BaseModel):
    source_id: str
    source_name: str
    content_hash: str
    chunk_count: int
    pages: int
    parsers: dict[str, int] = Field(default_factory=dict)  # pages per parser
    last_ingested_at: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeleteReport(BaseModel):
    source_id: str
    chunks_deleted: int


class DeleteCollectionReport(BaseModel):
    collection: str
    chunks_deleted: int
    sources_forgotten: int


class ChunkPreview(BaseModel):
    index: int
    text: str
    tokens: int
    page_no: int
    label: str = ""
    section_path: str = ""
    char_start: int
    char_end: int
    parser: ParserName = "local"


class PreviewResponse(BaseModel):
    source_name: str
    trace_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    stats: ParseStats
    chunks: list[ChunkPreview]


JobState = Literal["queued", "running", "completed", "failed"]


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    created_at: float
    finished_at: float | None = None
    report: IngestReport | None = None
    error: str = ""


class JobAccepted(BaseModel):
    job_id: str
    state: JobState


class IngestStatus(BaseModel):
    pdf_parser: str
    llamaparse_configured: bool
    embed_model: str
    sparse_language: str
    languages: list[str]  # valid values for `sparse_language`
    qdrant_url: str
    manifest_path: str
    allowed_paths: list[str]
    supported_extensions: list[str]
    max_upload_mb: int
    # Defaults the UI pre-fills
    default_collection: str
    chunk_strategy: ChunkStrategy
    chunk_size: int
    chunk_overlap: int
