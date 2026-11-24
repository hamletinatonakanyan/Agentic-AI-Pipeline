"""Redaction: remove secrets and personal data from a span before it leaves the process."""

import re
from typing import Any

from observability.models import SpanData, SpanEvent

REDACTED = "[REDACTED]"

# An attribute whose KEY looks like this has its whole value replaced.
_SENSITIVE_KEY = re.compile(r"(api[_-]?key|secret|password|authorization)", re.IGNORECASE)


class Redactor:
    def __init__(self, patterns: list[str]) -> None:
        self._patterns = [re.compile(p) for p in patterns]

    def text(self, value: str) -> str:
        for pattern in self._patterns:
            value = pattern.sub(REDACTED, value)
        return value

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return self.mapping(value)
        if isinstance(value, list | tuple):
            return [self.value(v) for v in value]
        return value

    def mapping(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            k: REDACTED if _SENSITIVE_KEY.search(str(k)) else self.value(v) for k, v in data.items()
        }

    def span(self, data: SpanData) -> SpanData:
        return data.model_copy(
            update={
                "attributes": self.mapping(data.attributes),
                "status_message": self.text(data.status_message),
                "events": [
                    SpanEvent(name=e.name, ts=e.ts, attributes=self.mapping(e.attributes))
                    for e in data.events
                ],
            }
        )
