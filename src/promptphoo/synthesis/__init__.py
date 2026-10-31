"""Shared helpers for `generate dataset` / `generate assertions`.

`dedent` ports the npm `dedent` 1.7.2 tag that upstream's synthesis prompts are written with:
it takes the template's *raw* segments (escape sequences not yet processed) and the
interpolated values, so the prompt texts stay byte-for-byte upstream's. `sample_array` and
`retry_with_deduplication` port src/util/generation.ts.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from promptphoo.jsvalues import js_string, json_stringify

logger = logging.getLogger("promptphoo")

_LEADING_INDENT = re.compile(r"^(\s+)\S+")


class InvariantError(RuntimeError):
    """src/util/invariant.ts: `Invariant failed: <message>`."""


def invariant(condition: Any, message: str | Callable[[], str] | None = None) -> None:
    if condition:
        return
    provided = message() if callable(message) else message
    raise InvariantError(f"Invariant failed: {provided}" if provided else "Invariant failed")


def _unescape(result: str) -> str:
    result = (
        result.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        .replace("\\v", "\v").replace("\\b", "\b").replace("\\f", "\f").replace("\\0", "\0")
    )  # fmt: skip
    result = re.sub(r"\\x([\da-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), result)
    result = re.sub(r"\\u\{([\da-fA-F]{1,6})\}", lambda m: chr(int(m.group(1), 16)), result)
    return re.sub(r"\\u([\da-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), result)


def dedent(raw: Sequence[str], *values: Any) -> str:
    """`dedent` used as a template tag, given `strings.raw` and the interpolated values."""
    result = ""
    for i, segment in enumerate(raw):
        segment = re.sub(r"\\\n[ \t]*", "", segment)
        segment = segment.replace("\\`", "`").replace("\\$", "$").replace("\\{", "{")
        result += segment
        if i < len(values):
            result += js_string(values[i])

    lines = result.split("\n")
    mindent: int | None = None
    for line in lines:
        m = _LEADING_INDENT.match(line)
        if m:
            indent = len(m.group(1))
            mindent = indent if not mindent else min(mindent, indent)
    if mindent is not None:
        result = "\n".join(line[mindent:] if line[:1] in (" ", "\t") else line for line in lines)
    return _unescape(result.strip())


def js_sort_key(value: str) -> bytes:
    """Array.prototype.sort's default order: UTF-16 code units."""
    return value.encode("utf-16-be", "surrogatepass")


def sample_array(array: list[Any], n: float) -> list[Any]:
    """Fisher-Yates shuffle, then the first `n` items (`slice(0, NaN)` is empty)."""
    logger.debug("Sampling %s items from array of length %d", js_string(n), len(array))
    shuffled = list(array)
    for i in range(len(shuffled) - 1, 0, -1):
        j = math.floor(random.random() * (i + 1))
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    if isinstance(n, float) and math.isnan(n):
        return []
    return shuffled[: max(0, int(min(n, len(array))))]


async def retry_with_deduplication(
    operation: Callable[[list[Any]], Awaitable[Any]],
    target_count: int,
    max_consecutive_retries: int = 2,
) -> list[Any]:
    """Call `operation` until `target_count` unique items (by JSON) are collected, giving up
    after `max_consecutive_retries` rounds that add nothing new."""
    all_items: list[Any] = []
    consecutive_retries = 0
    while len(all_items) < target_count and consecutive_retries <= max_consecutive_retries:
        new_items = await operation(all_items)
        if not isinstance(new_items, list):
            logger.warning("Operation returned non-iterable result. Skipping this iteration.")
            consecutive_retries += 1
            continue
        seen: dict[str, None] = {}
        for item in [*all_items, *new_items]:
            seen.setdefault(json_stringify(item), None)
        unique_new = [json.loads(s) for s in seen][len(all_items) :]
        all_items.extend(unique_new)
        logger.debug("Added %d unique items. Total: %d", len(unique_new), len(all_items))
        if not unique_new:
            consecutive_retries += 1
            logger.debug("No new unique items. Consecutive retries: %d", consecutive_retries)
        else:
            consecutive_retries = 0
    return all_items


def parse_int(value: Any) -> float:
    """Number.parseInt(value, 10): leading decimal digits, NaN when there are none."""
    m = re.match(r"^\s*([+-]?\d+)", js_string(value) if value is not None else "undefined")
    return int(m.group(1)) if m else math.nan
