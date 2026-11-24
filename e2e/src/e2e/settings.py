from pydantic import AliasChoices, Field
from pydantic_settings import SettingsConfigDict

from common import BaseAppSettings


class E2ESettings(BaseAppSettings):
    """Reads E2E_* variables, plus the shared QDRANT_URL."""

    model_config = SettingsConfigDict(env_prefix="E2E_", env_file=".env", extra="ignore")

    # One flag per component. A disabled component is not mounted.
    enable_ingest: bool = True
    enable_retrieval: bool = True
    enable_evals: bool = True
    enable_mcp: bool = True
    enable_obs: bool = True
    enable_agent: bool = True

    # Shared name, no E2E_ prefix.
    qdrant_url: str = Field(
        default="http://localhost:6333",
        validation_alias=AliasChoices("QDRANT_URL", "E2E_QDRANT_URL"),
    )
    probe_timeout_s: float = 1.0
