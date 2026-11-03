"""promptphoo's JS value semantics against a Node golden of 10,974 cases.

The golden (`tests/fixtures/js_golden.json`) is a copy of the one Kung Phoo, the Elixir port,
recorded with plain Node v22.21.1; `tests/js_golden.py` reads it. Every category promptphoo
implements must match Node on every case, except the differences listed in EXPECTED_FAILURES
with their reason. `length` and `slice` are not driven: promptphoo has no general helper for
them (call sites slice UTF-16 inline, e.g. output/junit.py).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from js_golden import decode, js_same, load_golden

from promptphoo.fetch.errors import error_to_string
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import (
    UNDEFINED,
    is_truthy,
    js_string,
    json_stringify,
    number_to_string,
    to_fixed,
    to_precision,
)
from promptphoo.synthesis import js_sort_key

GOLDEN = load_golden()


def _js_sort(values: list[Any]) -> list[Any]:
    """Array.prototype.sort() with no comparator: undefined last, the rest by String()."""
    defined = sorted((v for v in values if v is not UNDEFINED), key=_sort_key)
    return defined + [v for v in values if v is UNDEFINED]


def _sort_key(value: Any) -> bytes:
    return js_sort_key(js_string(value))


# category -> (run one case, how to compare: "string" outputs compare exactly, "value" outputs
# by SameValue on the decoded value)
_OPS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "stringify": lambda c: json_stringify(decode(c["input"])),
    "stringify_indent": lambda c: json_stringify(decode(c["input"]), c["gap"]),
    "string": lambda c: js_string(decode(c["input"])),
    "number_to_string": lambda c: number_to_string(decode(c["input"])),
    "to_fixed": lambda c: to_fixed(decode(c["input"]), c["digits"]),
    "to_precision": lambda c: to_precision(decode(c["input"]), c["precision"]),
    "truthy": lambda c: is_truthy(decode(c["input"])),
    "sort": lambda c: _js_sort(decode(c["input"])),
    "parse": lambda c: js_json_parse(c["text"]),
}

# Cases that may differ from Node, by category and case index, each with the reason.
EXPECTED_FAILURES: dict[str, dict[int, str]] = {}

# JSON.parse SyntaxError messages are V8's own wording ("Unexpected token ']', "[1,]" is not
# valid JSON"); js_json_parse raises ValueError with Python's. No contract surface carries the
# message, so a parse error case passes when promptphoo also rejects the text.
_MESSAGE_FREE = {"parse"}


def _check(category: str, case: dict[str, Any]) -> str | None:
    """None if promptphoo matches Node on this case, else a description of the difference."""
    try:
        actual = _OPS[category](case)
    except Exception as error:
        if "error" not in case:
            return f"raised {error_to_string(error)!r}, Node gave {case['output']!r}"
        if category in _MESSAGE_FREE or error_to_string(error) == case["error"]:
            return None
        return f"raised {error_to_string(error)!r}, Node raised {case['error']!r}"
    if "error" in case:
        return f"returned {actual!r}, Node raised {case['error']!r}"
    expected = case["output"]
    if expected == {"$js": "undefined"} and category.startswith("stringify"):
        # json_stringify returns the text "undefined" where JSON.stringify returns undefined,
        # so it can be interpolated the way a template literal would (see its docstring).
        expected = "undefined"
    if isinstance(expected, str) and isinstance(actual, str):
        return None if actual == expected else f"{actual!r} != Node {expected!r}"
    return None if js_same(actual, decode(expected)) else f"{actual!r} != Node {expected!r}"


def _failures(category: str) -> dict[int, str]:
    return {
        i: problem
        for i, case in enumerate(GOLDEN[category])
        if (problem := _check(category, case)) is not None
    }


@pytest.mark.parametrize("category", sorted(_OPS))
def test_matches_node_golden(category: str):
    failures = _failures(category)
    expected = EXPECTED_FAILURES.get(category, {})
    unexpected = {i: p for i, p in failures.items() if i not in expected}
    fixed = sorted(set(expected) - set(failures))
    detail = "\n".join(
        f"  #{i} {GOLDEN[category][i].get('input', GOLDEN[category][i].get('text'))!r}: {p}"
        for i, p in list(unexpected.items())[:20]
    )
    assert not unexpected, f"{len(unexpected)}/{len(GOLDEN[category])} differ from Node:\n{detail}"
    assert not fixed, f"expected failures now pass, drop them: {fixed}"


def test_golden_covers_every_category_promptphoo_implements():
    recorded = {k for k, v in GOLDEN.items() if isinstance(v, list)}
    assert recorded - set(_OPS) == {"length", "slice"}
