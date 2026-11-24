"""What the agent writes onto its spans, so a trace shows the whole run: the question, what each
step saw and decided, and the answer.

Counts, names and timings are always recorded. Content (question, passages, tool arguments and
results, the answer) is recorded unless `AGENT_TRACE_CONTENT=false`. Text is clipped, and the
observability redactor still runs over everything afterwards.
"""

from typing import Any

from agent_graph.settings import AgentSettings

# Attribute names, in one place (the Observability page reads these).
SESSION = "session.id"  # the conversation (thread) id: OpenTelemetry's name for it
INPUT = "input.value"
OUTPUT = "output.value"


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class StepDetail:
    def __init__(self, settings: AgentSettings) -> None:
        self._on = settings.trace_content
        self._text = settings.trace_text_chars
        self._passage = settings.trace_passage_chars

    def text(self, sp: Any, key: str, value: str | None) -> None:
        if self._on and value:
            sp.set_attribute(key, clip(value, self._text))

    def value(self, sp: Any, key: str, value: Any) -> None:
        """A small structured value (a list of names, a dict of arguments), cut to size."""
        if self._on and value not in (None, "", [], {}):
            sp.set_attribute(key, _shrink(value, self._text))

    def passages(self, sp: Any, passages: list[dict[str, Any]]) -> None:
        if not self._on:
            return
        sp.set_attribute(
            "agent.passages",
            [
                {
                    "n": i,
                    "chunk_id": p.get("chunk_id"),
                    "source": p.get("source_name", ""),
                    "page": p.get("page"),
                    "score": round(float(p.get("score") or 0.0), 4),
                    "text": clip(p.get("text", ""), self._passage),
                }
                for i, p in enumerate(passages, start=1)
            ],
        )


def _shrink(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return clip(value, limit)
    if isinstance(value, dict):
        return {str(k): _shrink(v, limit) for k, v in list(value.items())[:50]}
    if isinstance(value, list | tuple):
        return [_shrink(v, limit) for v in list(value)[:50]]
    return value
