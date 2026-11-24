from pydantic import AliasChoices, Field
from pydantic_settings import SettingsConfigDict

from common import BaseAppSettings


class RetrievalSettings(BaseAppSettings):
    """Reads RETRIEVAL_* variables, plus the shared QDRANT_URL and QDRANT_API_KEY."""

    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_", env_file=".env", extra="ignore", populate_by_name=True
    )

    collection: str = "docs"
    top_k: int = Field(default=5, ge=1, le=100)
    candidates: int = Field(default=50, ge=1, le=500)
    # Multilingual cross-encoder, int8 quantised (0.57 GB). Downloads on first use.
    rerank_model: str = "BAAI/bge-reranker-v2-m3-int8"
    rrf_k: int = Field(default=60, ge=1)
    max_query_chars: int = Field(default=2000, ge=1)
    rewrite_queries: int = Field(default=3, ge=1, le=8)  # extra phrasings for multi_query

    qdrant_url: str = Field(
        default="http://localhost:6333",
        validation_alias=AliasChoices("QDRANT_URL", "RETRIEVAL_QDRANT_URL"),
    )
    qdrant_api_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("QDRANT_API_KEY", "RETRIEVAL_QDRANT_API_KEY"),
    )
