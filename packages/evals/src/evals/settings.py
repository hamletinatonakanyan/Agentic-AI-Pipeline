from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import NoDecode, SettingsConfigDict

from common import BaseAppSettings


class EvalSettings(BaseAppSettings):
    """Reads EVAL_* variables, plus the shared QDRANT_URL and QDRANT_API_KEY."""

    model_config = SettingsConfigDict(
        env_prefix="EVAL_", env_file=".env", extra="ignore", populate_by_name=True
    )

    db_path: str = "data/evals.sqlite"
    # Judge and question drafting use the global model unless this is set.
    judge_model: str | None = None
    judge_prompt_version: str = "1"
    default_ks: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [1, 3, 5, 10])
    max_generate: int = Field(default=200, ge=1)
    target_timeout_s: float = Field(default=120.0, gt=0)

    # HTTP targets may only call these hosts, and may only send these environment variables as
    # headers. Otherwise a user of the UI could point a target at their own server and have it
    # receive the system's keys.
    allowed_target_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1"]
    )
    allowed_header_refs: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["APP_API_KEY"]
    )

    qdrant_url: str = Field(
        default="http://localhost:6333",
        validation_alias=AliasChoices("QDRANT_URL", "EVAL_QDRANT_URL"),
    )
    qdrant_api_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("QDRANT_API_KEY", "EVAL_QDRANT_API_KEY"),
    )

    @field_validator("default_ks", mode="before")
    @classmethod
    def _split_ks(cls, v: object) -> object:
        return [int(x) for x in v.split(",") if x.strip()] if isinstance(v, str) else v

    @field_validator("allowed_target_hosts", "allowed_header_refs", mode="before")
    @classmethod
    def _split_names(cls, v: object) -> object:
        return [x.strip() for x in v.split(",") if x.strip()] if isinstance(v, str) else v
