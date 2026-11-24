"""JSON-lines logging. One JSON object per line, easy to grep and to ship elsewhere.

When observability is on, it can attach `trace_id` and `span_id` to records
with `logger.info("msg", extra={"trace_id": ..., "span_id": ...})`.
"""

import json
import logging
import sys
from datetime import UTC, datetime

# Attributes every LogRecord has. Anything else was passed through `extra=`.
_STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Idempotent: calling it twice does not duplicate handlers."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    if any(getattr(h, "_common_json", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler._common_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)
