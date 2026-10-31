"""Prompt and test filters for `eval`. Ports src/util/eval/filterPrompts.ts,
src/util/eval/filterTests.ts and src/util/filterRange.ts.

Option dicts use the camelCase FilterOptions keys; None means "not given".
"""

from __future__ import annotations

import logging
import math
import random as _random
import re
from collections.abc import Callable
from typing import Any

from promptphoo.jsregex import compile_js_regex
from promptphoo.jsvalues import js_string

logger = logging.getLogger(__name__)


def _imul(a: int, b: int) -> int:
    """JS Math.imul: 32-bit signed multiply."""
    r = (a * b) & 0xFFFFFFFF
    return r - 0x100000000 if r & 0x80000000 else r


def _i32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def _fnv1a(text: str) -> int:
    state = 2166136261
    data = text.encode("utf-16-le")  # charCodeAt iterates UTF-16 code units
    for k in range(0, len(data), 2):
        state = _imul(state ^ int.from_bytes(data[k : k + 2], "little"), 16777619)
    return state


def derive_scenario_sample_seed(seed: int, scenario_index: int) -> int:
    return _fnv1a(f"{js_string(seed)}:{scenario_index}") & 0xFFFFFFFF


def create_seeded_random(seed: int) -> Callable[[], float]:
    """mulberry32 seeded from FNV-1a of String(seed), bit-for-bit with Node."""
    state = _fnv1a(js_string(seed))

    def next_value() -> float:
        nonlocal state
        state = _i32(state + 0x6D2B79F5)
        v = state & 0xFFFFFFFF
        v = _imul(_i32(v ^ (v >> 15)), _i32(v | 1)) & 0xFFFFFFFF
        v = _i32(v ^ (v + _imul(_i32(v ^ (v >> 7)), _i32(v | 61)))) & 0xFFFFFFFF
        return ((v ^ (v >> 14)) & 0xFFFFFFFF) / 4294967296

    return next_value


def js_parse_int(value: Any) -> float:
    """Number.parseInt: leading integer of the string, NaN if none."""
    if isinstance(value, bool):
        return math.nan
    if isinstance(value, int | float):
        return math.trunc(value) if math.isfinite(value) else math.nan
    m = re.match(r"\s*([+-]?\d+)", str(value))
    return int(m.group(1)) if m else math.nan


def filter_prompts(prompts: list[dict[str, Any]], option: str | None) -> list[dict[str, Any]]:
    if not option:
        return prompts
    try:
        rx = compile_js_regex(option)
    except ValueError as e:
        raise ValueError(f'Invalid regex pattern for --filter-prompts: "{option}". {e}') from e
    return [
        p for p in prompts
        if (p.get("id") and rx.search(p["id"])) or (p.get("label") and rx.search(p["label"]))
    ]  # fmt: skip


# --- range --------------------------------------------------------------------------------------


def parse_filter_range(option: str) -> tuple[int, int | None]:
    m = re.fullmatch(r"(\d*)\s*:\s*(\d*)", option.strip())
    if not m or (m.group(1) == "" and m.group(2) == ""):
        raise ValueError(
            "--filter-range must be specified in start:end format using zero-based indices, "
            f"got: {option}"
        )
    for raw in m.groups():
        if raw and int(raw) > 2**53 - 1:
            raise ValueError(f"--filter-range bounds must be safe integers, got: {option}")
    start = int(m.group(1)) if m.group(1) else 0
    end = int(m.group(2)) if m.group(2) else None
    if end is not None and start > end:
        raise ValueError(f"--filter-range start must be less than or equal to end, got: {option}")
    return start, end


def filter_by_range(items: list[Any], option: str | None) -> list[Any]:
    if option is None:
        return items
    start, end = parse_filter_range(option)
    sliced = items[start:end]
    if items and not sliced:
        logger.warning(
            "--filter-range %s selected 0 tests (had %d before slicing). Check your range bounds.",
            option,
            len(items),
        )
    return sliced


# --- tests --------------------------------------------------------------------------------------


def _split_metadata_value(value: str) -> list[str]:
    alternatives = [""]
    i, n = 0, len(value)
    while i < n:
        end = i
        while end < n and value[end] == "\\":
            end += 1
        slashes = end - i
        if end < n and value[end] == ",":
            alternatives[-1] += "\\" * (slashes // 2)
            if slashes % 2:
                alternatives[-1] += ","
            else:
                alternatives.append("")
        else:
            alternatives[-1] += value[i : end + 1]
        i = end + 1
    return alternatives


def _metadata_filter(filters: str | list[str]) -> Callable[[dict[str, Any]], bool]:
    parsed: list[tuple[str, list[str]]] = []
    for f in filters if isinstance(filters, list) else [filters]:
        key, _, value = f.partition("=")
        if not key or value == "":
            raise ValueError("--filter-metadata must be specified in key=value format")
        values = _split_metadata_value(value)
        if "" in values:
            raise ValueError(f'--filter-metadata has an empty value in "{f}"')
        parsed.append((key, values))

    def matches(test: dict[str, Any]) -> bool:
        metadata = test.get("metadata")
        if not metadata:
            return False
        for key, values in parsed:
            actual = metadata.get(key)
            if isinstance(actual, list):
                ok = any(v in js_string(a) for v in values for a in actual)
            else:
                ok = actual is not None and any(v in js_string(actual) for v in values)
            if not ok:
                return False
        return True

    return matches


async def filter_tests(test_suite: dict[str, Any], options: dict[str, Any]) -> list[Any]:
    tests = list(test_suite.get("tests") or [])
    if not options:
        return tests

    if options.get("metadata"):
        tests = [t for t in tests if _metadata_filter(options["metadata"])(t)]

    if any(options.get(k) for k in ("failing", "failingOnly", "errorsOnly")):
        # Needs stored eval results; lands with the results DB / JSON reader.
        raise NotImplementedError(
            "--filter-failing / --filter-failing-only / --filter-errors-only are not supported yet"
        )

    if options.get("pattern"):
        try:
            rx = compile_js_regex(options["pattern"])
        except ValueError as e:
            raise ValueError(f'Invalid regex pattern "{options["pattern"]}": {e}') from e
        tests = [t for t in tests if t.get("description") and rx.search(t["description"])]

    if options.get("range") is not None:
        tests = filter_by_range(tests, options["range"])

    if options.get("firstN") is not None:
        count = js_parse_int(options["firstN"])
        if isinstance(count, float) and math.isnan(count):
            raise ValueError(f"firstN must be a number, got: {options['firstN']}")
        tests = tests[: int(count)]

    if options.get("sample") is not None:
        count = js_parse_int(options["sample"])
        if isinstance(count, float) and math.isnan(count):
            raise ValueError(f"sample must be a number, got: {options['sample']}")
        seed = options.get("sampleSeed")
        rand = _random.random if seed is None else create_seeded_random(seed)
        shuffled = list(tests)
        for i in range(len(shuffled) - 1, 0, -1):
            j = math.floor(rand() * (i + 1))
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        tests = shuffled[: int(count)]

    return tests
