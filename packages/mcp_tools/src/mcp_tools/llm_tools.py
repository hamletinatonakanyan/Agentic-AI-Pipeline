"""Tools in the OpenAI function format, with names the model can use.

OpenAI function names must match `^[a-zA-Z0-9_-]{1,64}$`, and two servers may offer a tool with
the same name. So each tool is exposed as `<server>__<tool>`, cleaned up, and a lookup table maps
the model's choice back to the real server and tool.
"""

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from mcp_tools.models import ToolDescriptor

_BAD_NAME = re.compile(r"[^a-zA-Z0-9_-]")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
MAX_NAME = 64


@dataclass
class LlmToolSet:
    tools: list[dict[str, Any]] = field(default_factory=list)
    index: dict[str, tuple[str, str]] = field(default_factory=dict)  # llm name -> (server, tool)
    # llm name -> the server's readOnlyHint (None = the server did not say). Kept out of `tools`
    # because those are sent to the model as they are.
    read_only: dict[str, bool | None] = field(default_factory=dict)

    def resolve(self, llm_name: str) -> tuple[str, str] | None:
        return self.index.get(llm_name)


def clean_description(text: str, limit: int) -> str:
    """A tool's description is written by whoever runs the server, so it is untrusted text that
    ends up in the model's prompt. Strip control characters and keep it short."""
    text = _CONTROL.sub("", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def llm_name(server: str, tool: str, taken: set[str]) -> str:
    base = _BAD_NAME.sub("_", f"{server}__{tool}")[:MAX_NAME]
    name, n = base, 2
    while name in taken:
        suffix = f"_{n}"
        name = base[: MAX_NAME - len(suffix)] + suffix
        n += 1
    return name


def build_llm_tools(tools: list[ToolDescriptor], max_description: int = 1000) -> LlmToolSet:
    result = LlmToolSet()
    taken: set[str] = set()
    for tool in tools:
        name = llm_name(tool.server, tool.name, taken)
        taken.add(name)
        schema = copy.deepcopy(tool.input_schema) or {}
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        result.tools.append(
            {
                "type": "function",
                "name": name,
                "description": clean_description(tool.description, max_description),
                "parameters": schema,
                # Servers' schemas are not in OpenAI's strict form, so do not claim they are.
                "strict": False,
            }
        )
        result.index[name] = (tool.server, tool.name)
        result.read_only[name] = tool.read_only
    return result
