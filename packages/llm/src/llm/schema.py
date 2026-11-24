"""Pydantic model -> the JSON schema OpenAI's strict structured output accepts.

Strict mode demands: every object has `additionalProperties: false`, every property is listed in
`required` (optional means a union with null), and no `default`. Free-form objects (a dict with
arbitrary keys) are not allowed. Pydantic's own schema satisfies none of the first three, so this
rewrites it.
"""

from typing import Any

from pydantic import BaseModel


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    _visit(schema)
    return schema


def _visit(node: Any) -> None:
    if not isinstance(node, dict):
        return
    node.pop("default", None)

    extra = node.get("additionalProperties")
    if isinstance(extra, dict):
        raise ValueError(
            "free-form objects (dict fields with arbitrary keys) are not allowed in strict "
            "structured output. Use a list of {key, value} objects instead."
        )
    if node.get("type") == "object" or "properties" in node:
        properties = node.setdefault("properties", {})
        node["additionalProperties"] = False
        node["required"] = list(properties)
        for sub in properties.values():
            _visit(sub)

    for key in ("items", "prefixItems"):
        value = node.get(key)
        for item in value if isinstance(value, list) else [value]:
            _visit(item)
    for key in ("anyOf", "oneOf", "allOf"):
        for item in node.get(key, []):
            _visit(item)
    for sub in node.get("$defs", {}).values():
        _visit(sub)
