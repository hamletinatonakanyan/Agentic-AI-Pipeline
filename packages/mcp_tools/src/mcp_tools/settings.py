from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import NoDecode, SettingsConfigDict

from common import BaseAppSettings

OWN_SERVER_COMMAND = "python -m mcp_tools.server"


class McpSettings(BaseAppSettings):
    """Reads MCP_* variables."""

    model_config = SettingsConfigDict(env_prefix="MCP_", env_file=".env", extra="ignore")

    # ----- our own server --------------------------------------------------------------------
    server_name: str = "agentic-blocks"
    retrieval_url: str = "http://localhost:8000/api/retrieval"  # the stdio server calls this
    ingest_url: str = "http://localhost:8000/api/ingest"
    default_collection: str = "docs"
    max_query_chars: int = Field(default=2000, ge=1)
    passage_chars: int = Field(default=1200, ge=100)  # each search result is trimmed to this
    # Table tools (list_tables, query_table): how much they may read for one question.
    table_max_chunks: int = Field(default=5000, ge=1)  # chunks scanned to find tables
    table_max_rows: int = Field(default=200_000, ge=1)  # rows of one table that are loaded
    table_max_groups: int = Field(default=50, ge=1, le=500)  # groups or rows returned at most
    # Hosts the HTTP endpoint answers to (Host header), against DNS rebinding.
    http_allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["localhost:*", "127.0.0.1:*", "testserver"]
    )

    # ----- the client ----------------------------------------------------------------------------
    db_path: str = "data/mcp.sqlite"  # saved server registrations
    call_timeout_s: float = Field(default=30.0, gt=0)
    max_output_chars: int = Field(default=20_000, ge=100)  # longer tool output is cut and flagged
    max_description_chars: int = Field(default=1000, ge=50)  # tool descriptions are untrusted text
    tools_cache_ttl_s: float = Field(default=60.0, ge=0)  # how long a server's tool list is kept

    # What a registration may do. Without these, anyone with the API key could register a
    # server that runs any command, or sends your keys to any URL.
    # Full command lines, separated by `;`. A command must match one of them exactly.
    allowed_commands: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [OWN_SERVER_COMMAND]
    )
    allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1"]
    )
    # Environment variables a stdio server may receive, and headers an HTTP server may be sent,
    # by NAME. The value is read from the environment only when connecting.
    allowed_env_refs: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["APP_API_KEY", "MCP_RETRIEVAL_URL", "MCP_INGEST_URL"]
    )
    allowed_header_refs: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["APP_API_KEY"]
    )

    @field_validator("allowed_commands", mode="before")
    @classmethod
    def _split_commands(cls, v: object) -> object:
        return [x.strip() for x in v.split(";") if x.strip()] if isinstance(v, str) else v

    @field_validator(
        "allowed_hosts", "allowed_env_refs", "allowed_header_refs", "http_allowed_hosts",
        mode="before",
    )  # fmt: skip
    @classmethod
    def _split_names(cls, v: object) -> object:
        return [x.strip() for x in v.split(",") if x.strip()] if isinstance(v, str) else v
