from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from llm.tools import ToolTurn

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    """What every other package depends on. `OpenAILLM` is the real one, `FakeLLM` is for tests.

    There is no `temperature`, `top_p` or effort argument on purpose: the reasoning
    setting is global, and sampling parameters are never sent.
    """

    def complete(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int | None = None
    ) -> str: ...

    def complete_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        max_output_tokens: int | None = None,
    ) -> T:
        """The model's answer, validated against `schema`."""
        ...

    def complete_with_tools(
        self,
        items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        system: str | None = None,
        max_output_tokens: int | None = None,
    ) -> ToolTurn:
        """One turn of a tool conversation. See `llm.tools` for how to continue it."""
        ...

    def stream(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int | None = None
    ) -> "StreamHandle":
        """Iterate it for text pieces. `ttft_ms` and `total_ms` are set as it runs."""
        ...


class StreamHandle(Protocol):
    ttft_ms: float | None
    total_ms: float | None

    def __iter__(self): ...
