"""Tool calling, the model's own function-calling feature (Responses API).

How a tool turn works:
  1. we send the conversation so far (`items`) and the tool definitions
  2. the model answers with text, or with one or more `function_call` items
  3. we run the tools and send back everything the model said, in order, plus a
     `function_call_output` for each call, and ask again

The history is a list of plain dicts in the API's own item format, so it can be stored in a
checkpoint and replayed later. A reasoning model's own `reasoning` items must be sent back along
with its calls (they carry encrypted content in stateless mode), which is why `output_items`
keeps every item the model produced, untouched.
"""

import json
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    call_id: str
    name: str  # the name the model saw, e.g. "own__calculate"
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Set when the model's arguments were not a JSON object. The caller should send this back as
    # the tool's output so the model can correct itself, and must not run the tool.
    argument_error: str = ""


class ToolTurn(BaseModel):
    text: str = ""  # what the model said (may be empty when it only calls tools)
    calls: list[ToolCall] = Field(default_factory=list)
    # Everything the model produced, verbatim and in order. Append it to the history before
    # appending the tool outputs.
    output_items: list[dict[str, Any]] = Field(default_factory=list)


def user_item(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def assistant_item(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def tool_output_item(call_id: str, output: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def parse_arguments(raw: Any) -> tuple[dict[str, Any], str]:
    """(arguments, error). The model sends arguments as a JSON string, which can be malformed."""
    if isinstance(raw, dict):
        return raw, ""
    if raw in (None, ""):
        return {}, ""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}, "the arguments were not valid JSON"
    if not isinstance(value, dict):
        return {}, "the arguments must be a JSON object"
    return value, ""
