"""Turn a plain dict into a Qdrant filter.

    {"source_name": "a.pdf"}              -> must match
    {"page": [1, 2]}                      -> match any of
    {"page": {"gte": 3, "lt": 10}}        -> range
    {"metadata.year": 2025}               -> nested payload field
All conditions are ANDed.
"""

import re
from typing import Any

from qdrant_client import models

from common import AppError

_KEY = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,99}$")
_RANGE_KEYS = {"gt", "gte", "lt", "lte"}


def _bad(message: str) -> AppError:
    return AppError("invalid_filter", message, 422)


def build_filter(filters: dict[str, Any]) -> models.Filter | None:
    if not filters:
        return None
    conditions: list[models.Condition] = []
    for key, value in filters.items():
        if not _KEY.match(key):
            raise _bad(f"'{key}' is not a valid payload field name.")
        if isinstance(value, bool | str | int | float):
            match: Any = models.MatchValue(value=value)
            conditions.append(models.FieldCondition(key=key, match=match))
        elif isinstance(value, list):
            if not value or not all(
                isinstance(v, str | int) and not isinstance(v, bool) for v in value
            ):
                raise _bad(f"'{key}': a list must be non-empty and hold only strings or integers.")
            conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=value)))
        elif isinstance(value, dict):
            if not value or set(value) - _RANGE_KEYS:
                raise _bad(f"'{key}': a range may use only {sorted(_RANGE_KEYS)}.")
            if not all(
                isinstance(v, int | float) and not isinstance(v, bool) for v in value.values()
            ):
                raise _bad(f"'{key}': range bounds must be numbers.")
            conditions.append(models.FieldCondition(key=key, range=models.Range(**value)))
        else:
            raise _bad(f"'{key}': unsupported filter value (null is not supported).")
    return models.Filter(must=conditions)
