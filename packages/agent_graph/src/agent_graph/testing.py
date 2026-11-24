"""Fakes for tests in any package: no network, no model, no keys."""

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

from agent_graph.models import Passage, RetrievalChoice
from agent_graph.protocols import AgentTool, RetrievedContext, ToolOutcome


def default_passages() -> list[Passage]:
    return [
        Passage(
            chunk_id="c1", text="Cats purr when they are content.",
            source_name="cats.txt", page=1, score=0.9,
        ),
        Passage(
            chunk_id="c2", text="A cat sleeps most of the day.",
            source_name="cats.txt", page=2, score=0.8,
        ),
    ]  # fmt: skip


class FakeRetriever:
    def __init__(
        self, passages: list[Passage] | None = None, fail: Exception | None = None
    ) -> None:
        self.passages = passages if passages is not None else default_passages()
        self.fail = fail
        self.calls: list[tuple[str, RetrievalChoice, int]] = []
        self.data_version = "1"  # change it to say "the documents changed"
        self.version_error: Exception | None = None

    def version(self, choice: RetrievalChoice) -> str:
        if self.version_error:
            raise self.version_error
        return self.data_version

    def retrieve(self, query: str, choice: RetrievalChoice, top_k: int) -> RetrievedContext:
        self.calls.append((query, choice, top_k))
        if self.fail:
            raise self.fail
        return RetrievedContext(
            passages=self.passages[:top_k], rewritten_queries=[query], retrieval_ms=12.5
        )


class FakeTools:
    """A ToolProvider with scripted tools. `behaviour` maps a tool name to a function of the
    arguments (returning text or a ToolOutcome), or to an Exception to raise."""

    def __init__(
        self,
        tools: list[AgentTool] | None = None,
        behaviour: dict[str, Callable[[dict], Any]] | None = None,
    ) -> None:
        self._tools = tools if tools is not None else [
            AgentTool(
                name="own__calculate", description="Evaluate arithmetic.", read_only=True,
                parameters={"type": "object", "properties": {"expression": {"type": "string"}},
                            "required": ["expression"]},
            ),
        ]  # fmt: skip
        self.behaviour = behaviour or {}
        self.calls: list[tuple[str, dict]] = []

    async def tools(self) -> list[AgentTool]:
        return list(self._tools)

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        self.calls.append((name, arguments))
        fn = self.behaviour.get(name)
        if fn is None:
            return ToolOutcome(ok=True, content=f"result of {name}", duration_ms=1.0)
        result = fn(arguments)
        if isinstance(result, ToolOutcome):
            return result
        return ToolOutcome(ok=True, content=str(result), duration_ms=1.0)


class FakeEmbedder:
    """A bag-of-words embedding: texts sharing words are close. Deterministic, no model."""

    dim = 32

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            out.append(vec)
        return out


# ----- a scripted model for the agent --------------------------------------------------


class Brain:
    """An LLM that answers each KIND of call from its own list, so a test can say exactly what
    the model does at each step. When a list runs out, its last entry repeats.

    routes     the route chosen each time: "retrieve", "tool" or "direct"
    answers    the answer text each time (used for streaming and for plain completion)
    checks     for each self-check, the list of unsupported claims (an empty list = passes)
    facts      for each fact extraction, the facts found
    tool_turns for each tool turn: a ToolTurn, or a string for a text-only turn
    """

    def __init__(self, routes=None, answers=None, checks=None, facts=None, tool_turns=None,
                 standalone: str | None = None) -> None:  # fmt: skip
        from llm.testing import Call

        self._lists = {
            "route": list(routes or ["retrieve"]),
            "answer": list(answers or ["Cats purr when content [1]."]),
            "check": list(checks if checks is not None else [[]]),
            "facts": list(facts if facts is not None else [[]]),
            "tool": list(tool_turns or ["No tool needed."]),
        }
        self._i = {k: 0 for k in self._lists}
        self._standalone = standalone
        self.calls: list[Call] = []
        self.count = {k: 0 for k in self._lists}

    def _take(self, kind: str):
        items = self._lists[kind]
        item = items[min(self._i[kind], len(items) - 1)]
        self._i[kind] += 1
        self.count[kind] += 1
        return item

    def _record(self, kind, prompt, system, **extra):
        from llm.testing import Call

        self.calls.append(Call(kind, prompt, system, **extra))

    def complete_json(self, prompt, schema, *, system=None, max_output_tokens=None):
        from agent_graph.prompts import CheckOutput, FactsOutput, RouteDecision

        self._record("complete_json", prompt, system, schema=schema)
        if schema is RouteDecision:
            match = re.search(r"Latest message: (.*?)\n\nChoose a route", prompt, re.S)
            question = self._standalone or (match.group(1).strip() if match else "the question")
            return RouteDecision(
                route=self._take("route"), standalone_question=question, reason="because"
            )
        if schema is CheckOutput:
            return CheckOutput(unsupported_claims=list(self._take("check")))
        if schema is FactsOutput:
            return FactsOutput(facts=list(self._take("facts")))
        raise AssertionError(f"Brain has no script for {schema.__name__}")

    def complete(self, prompt, *, system=None, max_output_tokens=None):
        self._record("complete", prompt, system)
        return self._take("answer")

    def stream(self, prompt, *, system=None, max_output_tokens=None):
        from llm.testing import FakeStream

        self._record("stream", prompt, system)
        text = self._take("answer")
        words = text.split(" ")
        return FakeStream([w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)])

    def complete_with_tools(self, items, tools, *, system=None, max_output_tokens=None):
        from llm import ToolTurn

        self._record("complete_with_tools", "", system, items=list(items), tools=list(tools))
        turn = self._take("tool")
        return turn if isinstance(turn, ToolTurn) else ToolTurn(text=str(turn))

    # ----- conveniences for assertions -------------------------------------------------

    def prompts_of(self, kind: str) -> list[str]:
        return [c.prompt for c in self.calls if c.kind == kind]


def calls_turn(*calls: tuple[str, str, dict]):
    """A tool turn that asks for these calls: (call_id, tool name, arguments)."""
    from llm import ToolCall, ToolTurn

    return ToolTurn(
        calls=[ToolCall(call_id=i, name=n, arguments=a) for i, n, a in calls],
        output_items=[
            {"type": "function_call", "call_id": i, "name": n, "arguments": json.dumps(a)}
            for i, n, a in calls
        ],
    )
