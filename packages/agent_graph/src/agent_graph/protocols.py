"""What the agent needs from the outside. Everything is an interface, so the agent works with
fakes in tests, with HTTP adapters standalone, and with in-process services inside e2e."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from agent_graph.models import Passage, RetrievalChoice


class RetrievedContext(BaseModel):
    passages: list[Passage]
    rewritten_queries: list[str] = Field(default_factory=list)
    retrieval_ms: float | None = None
    trace_id: str | None = None


class Retriever(Protocol):
    def retrieve(self, query: str, choice: RetrievalChoice, top_k: int) -> RetrievedContext: ...

    # Optional: `def version(self, choice) -> str`, a value that changes whenever the searched
    # documents change. When present, it is part of the search cache key.


class AgentTool(BaseModel):
    """A tool as the model sees it."""

    name: str  # the name the model uses, for example "own__calculate"
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON Schema
    read_only: bool | None = None  # the server's `readOnlyHint`. None means "not stated".


class ToolOutcome(BaseModel):
    ok: bool
    content: str
    error_kind: str | None = None
    truncated: bool = False
    duration_ms: float = 0.0


class ToolProvider(Protocol):
    async def tools(self) -> list[AgentTool]:
        """The tools that may be offered to the model (already filtered by each server's
        allow-list)."""
        ...

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        """Run a tool by the name the model used. Never raises: failures are `ok=False`."""
        ...


class FactEmbedder(Protocol):
    """Embeds short texts for long-term memory. The prefixes some models need are applied by the
    memory layer, not here."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...
