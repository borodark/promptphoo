"""Numeric guards for provider-controlled values. Ports src/util/numeric.ts."""

from __future__ import annotations

import math
from typing import Any

MAX_REASONABLE_TOKEN_COUNT = 1_000_000_000
_MAX_SAFE_INTEGER = 2**53 - 1


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def filter_finite_scores(scores: dict[str, Any]) -> dict[str, float]:
    return {k: v for k, v in scores.items() if _is_number(v) and math.isfinite(v)}


def is_safe_token_count(value: Any) -> bool:
    """A non-negative safe integer no larger than a billion (JS Number.isSafeInteger)."""
    if not _is_number(value) or not math.isfinite(value) or value != int(value):
        return False
    return abs(value) <= _MAX_SAFE_INTEGER and 0 <= value <= MAX_REASONABLE_TOKEN_COUNT


def is_safe_cost(value: Any) -> bool:
    return _is_number(value) and math.isfinite(value) and value >= 0
