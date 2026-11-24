from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from common import BaseAppSettings


class AgentSettings(BaseAppSettings):
    """Reads AGENT_* variables."""

    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    # Where the agent keeps its files: checkpoints (one save per step of every conversation),
    # long-term memory, the retrieval cache, and its own bookkeeping (threads, LLM cache).
    data_dir: str = "data/agent"
    # `sqlite` survives restarts. `memory` forgets everything when the process stops (tests).
    # (Postgres is not offered in v1.)
    checkpoint_backend: Literal["sqlite", "memory"] = "sqlite"

    # ----- behaviour -------------------------------------------------------------------------
    max_check_retries: int = Field(default=2, ge=0, le=5)  # regenerate after a failed check
    max_tool_rounds: int = Field(default=3, ge=1, le=10)  # ask-the-model-for-tools rounds
    # Which tool calls pause for a human: "sensitive" = every tool not marked read-only by its
    # server, "all" = every tool, "none" = none (not recommended for tools you do not control).
    approval: Literal["sensitive", "all", "none"] = "sensitive"
    history_turns: int = Field(default=8, ge=1, le=50)  # past messages shown to the model
    max_message_chars: int = Field(default=4000, ge=1)
    top_k: int = Field(default=5, ge=1, le=50)  # passages retrieved
    context_chars: int = Field(default=12_000, ge=500)  # passages given to the model, in total
    tool_output_chars: int = Field(default=6000, ge=100)  # per tool result given to the model
    run_timeout_s: float = Field(default=300.0, gt=0)

    # Standalone only: where retrieval and the MCP tools run. Inside e2e they are in-process.
    retrieval_url: str = "http://localhost:8002"
    mcp_url: str = "http://localhost:8004"
    http_timeout_s: float = Field(default=60.0, gt=0)

    # ----- tracing ---------------------------------------------------------------------------
    # Put the question, passages, tool arguments/results and the answer on the spans, so a trace
    # shows the whole run. Off keeps only names, counts and timings. The traces stay in the
    # local SQLite store.
    trace_content: bool = True
    trace_text_chars: int = Field(default=4000, ge=200)
    trace_passage_chars: int = Field(default=500, ge=50)

    # ----- caching ---------------------------------------------------------------------
    use_cache: bool = True  # a request can still turn it off
    cache_ttl_s: int = Field(default=600, ge=0)

    # ----- long-term memory ----------------------------------------------------------------------
    # Saving facts from chats is opt-in because it stores personal statements.
    auto_remember: bool = False
    memory_embed_model: str = "google/embeddinggemma-300m"
    memory_k: int = Field(default=5, ge=1, le=20)
    memory_min_score: float = Field(default=0.2, ge=0.0, le=1.0)

    def path(self, name: str) -> str:
        """A file inside `data_dir`, or `:memory:` when the backend is `memory`."""
        if self.checkpoint_backend == "memory":
            return ":memory:"
        return str(Path(self.data_dir) / name)
