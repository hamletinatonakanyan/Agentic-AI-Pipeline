from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import NoDecode, SettingsConfigDict

from common import BaseAppSettings

# Languages with a bundled stemmer and stop words in fastembed's BM25. Anything else must use
# "none", which turns stemming and stop words off.
BM25_LANGUAGES = [
    "arabic", "danish", "dutch", "english", "finnish", "french", "german", "greek",
    "hungarian", "italian", "norwegian", "portuguese", "romanian", "russian", "spanish",
    "swedish", "tamil", "turkish",
]  # fmt: skip
NO_LANGUAGE = "none"


class IngestSettings(BaseAppSettings):
    """Reads INGEST_* variables, plus the shared QDRANT_URL, QDRANT_API_KEY, LLAMA_CLOUD_API_KEY."""

    model_config = SettingsConfigDict(
        env_prefix="INGEST_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # --- PDFs -----------------------------------------------------------------------------
    pdf_parser: Literal["llamaparse", "local"] = "llamaparse"
    llamaparse_tier: Literal["fast", "cost_effective", "agentic", "agentic_plus"] = "fast"
    llamaparse_base_url: str = "https://api.cloud.llamaindex.ai"
    llamaparse_timeout_s: float = 300.0
    scanned_text_threshold: int = Field(default=50, ge=0)

    # --- Embeddings -----------------------------------------------------------------------
    embed_model: str = "google/embeddinggemma-300m"
    sparse_model: str = "Qdrant/bm25"
    sparse_language: str = "english"
    # Text added before each document / query for models that need it. Empty = use the built-in
    # value for known models (see embedding.py), or none for unknown models.
    doc_prefix: str | None = None
    query_prefix: str | None = None
    embed_batch_size: int = Field(default=16, ge=1)

    # --- Chunking -------------------------------------------------------------------------
    chunk_strategy: Literal["fixed", "recursive", "markdown"] = "recursive"
    chunk_size: int = Field(default=512, ge=16)
    chunk_overlap: int = Field(default=64, ge=0)

    # --- Storage and input ----------------------------------------------------------------
    default_collection: str = "docs"
    manifest_path: str = "data/ingest_manifest.sqlite"
    # Server-side folders that `ingest_path` may read. Comma-separated in .env.
    allowed_paths: Annotated[list[str], NoDecode] = Field(default_factory=list)
    max_upload_mb: int = Field(default=50, ge=1)

    # --- Shared names, no INGEST_ prefix ---------------------------------------------------
    qdrant_url: str = Field(
        default="http://localhost:6333",
        validation_alias=AliasChoices("QDRANT_URL", "INGEST_QDRANT_URL"),
    )
    qdrant_api_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("QDRANT_API_KEY", "INGEST_QDRANT_API_KEY"),
    )
    llama_cloud_api_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("LLAMA_CLOUD_API_KEY", "INGEST_LLAMA_CLOUD_API_KEY"),
    )

    @field_validator("allowed_paths", mode="before")
    @classmethod
    def _split_paths(cls, value: object) -> object:
        if isinstance(value, str):
            return [p.strip() for p in value.split(",") if p.strip()]
        return value

    @field_validator("sparse_language")
    @classmethod
    def _check_language(cls, value: str) -> str:
        return check_language(value)

    @model_validator(mode="after")
    def _check_overlap(self) -> "IngestSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("INGEST_CHUNK_OVERLAP must be smaller than INGEST_CHUNK_SIZE")
        return self


def check_language(value: str) -> str:
    language = value.strip().lower()
    if language != NO_LANGUAGE and language not in BM25_LANGUAGES:
        raise ValueError(
            f"unsupported BM25 language {value!r}. Use one of {BM25_LANGUAGES} or '{NO_LANGUAGE}'."
        )
    return language
