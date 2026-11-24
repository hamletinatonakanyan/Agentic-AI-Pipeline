"""A fake LLM for tests in any package: no network, no key. Records every call."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from llm.tools import ToolTurn

T = TypeVar("T", bound=BaseModel)


@dataclass
class Call:
    kind: str  # "complete" | "complete_json" | "complete_with_tools" | "stream"
    prompt: str
    system: str | None
    schema: type[BaseModel] | None = None
    items: list[dict[str, Any]] = field(default_factory=list)  # complete_with_tools
    tools: list[dict[str, Any]] = field(default_factory=list)  # complete_with_tools


class FakeStream:
    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces
        self.ttft_ms: float | None = None
        self.total_ms: float | None = None

    def __iter__(self):
        self.ttft_ms = 1.0
        yield from self._pieces
        self.total_ms = 2.0


class FakeLLM:
    """`script` decides the answers.

    - a list: answers in order (each a string for `complete`/`stream`, or a model instance or a
      JSON string for `complete_json`)
    - a function: called with the prompt, returns the answer
    - a string: always that answer
    """

    def __init__(
        self,
        script: list | Callable[[str], object] | str = "ok",
        tool_script: list | Callable[[list[dict]], object] | None = None,
    ) -> None:
        self._script = script
        self._tool_script = tool_script
        self.calls: list[Call] = []
        self._i = 0
        self._tool_i = 0

    def _next(self, prompt: str) -> object:
        if callable(self._script):
            return self._script(prompt)
        if isinstance(self._script, list):
            if self._i >= len(self._script):
                raise AssertionError(f"FakeLLM ran out of scripted answers at call {self._i + 1}")
            answer = self._script[self._i]
            self._i += 1
            return answer
        return self._script

    def complete(self, prompt: str, *, system: str | None = None, max_output_tokens=None) -> str:
        self.calls.append(Call("complete", prompt, system))
        return str(self._next(prompt))

    def complete_json(
        self, prompt: str, schema: type[T], *, system: str | None = None, max_output_tokens=None
    ) -> T:
        self.calls.append(Call("complete_json", prompt, system, schema))
        answer = self._next(prompt)
        if isinstance(answer, schema):
            return answer
        if isinstance(answer, BaseModel):
            return schema.model_validate(answer.model_dump())
        if isinstance(answer, dict):
            return schema.model_validate(answer)
        return schema.model_validate_json(str(answer))

    def complete_with_tools(
        self, items, tools, *, system: str | None = None, max_output_tokens=None
    ) -> ToolTurn:
        """`tool_script` is a list of turns (a ToolTurn, or a string for a text-only turn), or a
        function of the history."""
        last = items[-1] if items else {}
        self.calls.append(
            Call(
                "complete_with_tools",
                str(last.get("content", last)),
                system,
                None,
                list(items),
                list(tools),
            )
        )
        script = self._tool_script
        if script is None:
            raise AssertionError("FakeLLM was given no tool_script")
        if callable(script):
            answer = script(items)
        else:
            if self._tool_i >= len(script):
                raise AssertionError(f"FakeLLM ran out of tool turns at call {self._tool_i + 1}")
            answer = script[self._tool_i]
            self._tool_i += 1
        return answer if isinstance(answer, ToolTurn) else ToolTurn(text=str(answer))

    def stream(self, prompt: str, *, system: str | None = None, max_output_tokens=None):
        self.calls.append(Call("stream", prompt, system))
        text = str(self._next(prompt))
        return FakeStream(text.split(" ") if " " in text else [text])
