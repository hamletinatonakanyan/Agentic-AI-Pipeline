"""The contract: options, hits, results. The collection spec mirrors what ingest writes into the
collection's metadata (`ingest_spec`). It is read, never imported from ingest."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SearchMode = Literal["dense", "sparse", "hybrid"]
RewriteStrategy = Literal["none", "multi_query", "hyde", "step_back"]


class CollectionSpec(BaseModel):
    """How a collection was built. Tolerates fields added later."""

    model_config = ConfigDict(extra="ignore")

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


class ChunkItem(BaseModel):
    """One stored chunk, as it was ingested. For tools that need every row of a table."""

    chunk_id: str
    source_id: str = ""
    source_name: str = ""
    page: int | None = None
    chunk_index: int = 0
    text: str


class ChunkPage(BaseModel):
    chunks: list[ChunkItem]
    next: str | None = None  # pass back as `after` for the next page; None at the end


class SparseVec(BaseModel):
    indices: list[int]
    values: list[float]


class RawHit(BaseModel):
    """What the vector store returns."""

    id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class RetrievalOptions(BaseModel):
    """Unset fields fall back to settings."""

    collection: str | None = Field(default=None, min_length=1, max_length=100)
    mode: SearchMode = "hybrid"
    top_k: int | None = Field(default=None, ge=1, le=100)
    candidates: int | None = Field(default=None, ge=1, le=500)  # fetched before reranking
    # {"source_name": "a.pdf", "metadata.year": 2025, "page": [1, 2], "page": {"gte": 3}}
    filters: dict[str, Any] = Field(default_factory=dict)
    rewrite: RewriteStrategy = "none"
    rerank: bool = False
    rerank_model: str | None = None

    @model_validator(mode="after")
    def _candidates_cover_top_k(self) -> "RetrievalOptions":
        if self.top_k and self.candidates and self.candidates < self.top_k:
            raise ValueError("candidates must be at least top_k")
        return self


class RetrieveRequest(RetrievalOptions):
    query: str = Field(min_length=1)


class Hit(BaseModel):
    rank: int  # 1-based, after reranking when it ran
    chunk_id: str
    score: float  # the retrieval score (cosine, BM25, or RRF, depending on mode)
    rerank_score: float | None = None
    text: str
    payload: dict[str, Any]  # everything except `text`


class Timings(BaseModel):
    rewrite_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


class RetrievalResult(BaseModel):
    query: str
    collection: str
    rewritten_queries: list[str]  # what was actually searched (the original comes first)
    hits: list[Hit]
    timings: Timings
    # Echoed back so a result can be reproduced
    mode: SearchMode
    rewrite: RewriteStrategy
    rerank: bool
    rerank_model: str | None = None
    top_k: int
    candidates: int
    warnings: list[str] = Field(default_factory=list)
    trace_id: str | None = None


class RewriteRequest(BaseModel):
    query: str = Field(min_length=1)
    strategy: RewriteStrategy = "multi_query"


class RewriteResponse(BaseModel):
    query: str
    strategy: RewriteStrategy
    rewritten_queries: list[str]
    trace_id: str | None = None


class Candidate(BaseModel):
    text: str = Field(min_length=1)
    chunk_id: str | None = None


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    candidates: list[Candidate] = Field(min_length=1, max_length=500)
    top_n: int | None = Field(default=None, ge=1, le=500)
    model: str | None = None


class RerankedItem(BaseModel):
    rank: int
    chunk_id: str | None
    text: str
    rerank_score: float
    original_rank: int


class RerankResponse(BaseModel):
    query: str
    model: str
    items: list[RerankedItem]
    rerank_ms: float
    trace_id: str | None = None


class RerankModelInfo(BaseModel):
    name: str
    size_gb: float | None = None
    description: str = ""


class StrategiesResponse(BaseModel):
    modes: list[str]
    rewrite_strategies: list[str]
    rerank_models: list[RerankModelInfo]
    defaults: dict[str, Any]
    collections: list[CollectionInfo]


class RetrievalStatus(BaseModel):
    qdrant_url: str
    default_collection: str
    rerank_model: str
    top_k: int
    candidates: int
