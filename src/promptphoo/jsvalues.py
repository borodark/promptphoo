"""JavaScript value semantics needed for output parity with the Node implementation.

promptfoo hashes, stores and renders values the way V8 does; these helpers reproduce the parts
that observably differ from Python (JSON.stringify spacing and number formatting, String()).
"""

from __future__ import annotations

import json
import math
from typing import Any


class Undefined:
    """Stand-in for JS `undefined` where it must be distinguished from `null` (None)."""

    _instance: Undefined | None = None

    def __new__(cls) -> Undefined:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "undefined"


UNDEFINED = Undefined()


def number_to_string(n: float | int) -> str:
    """JS Number.prototype.toString for finite and special values."""
    if isinstance(n, bool):
        return "true" if n else "false"
    if isinstance(n, int):
        return str(n)
    if math.isnan(n):
        return "NaN"
    if math.isinf(n):
        return "Infinity" if n > 0 else "-Infinity"
    if n == 0:
        return "0"
    if n.is_integer() and abs(n) < 1e21:
        return str(int(n))
    r = repr(n)
    if "e" in r:
        mantissa, exp = r.split("e")
        sign = "-" if exp.startswith("-") else "+"
        exp_digits = exp.lstrip("+-").lstrip("0") or "0"
        e = int(exp)
        if -7 < e < 21:
            # V8 prints plain decimals in this range; repr may still use exponent form.
            return format(n, "f").rstrip("0").rstrip(".")
        return f"{mantissa}e{sign}{exp_digits}"
    return r


def js_string(value: Any) -> str:
    """JS String(value) / template-literal interpolation."""
    if value is None:
        return "null"
    if value is UNDEFINED:
        return "undefined"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return number_to_string(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return ",".join("" if v is None or v is UNDEFINED else js_string(v) for v in value)
    if isinstance(value, dict):
        return "[object Object]"
    return str(value)


def _prepare(value: Any) -> Any:
    """Convert to a structure json.dumps renders the way JSON.stringify does."""
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return _JsNumber(value) if value.is_integer() and abs(value) < 1e21 else value
    if isinstance(value, dict):
        return {str(k): _prepare(v) for k, v in value.items() if v is not UNDEFINED}
    if isinstance(value, list | tuple):
        return [None if v is UNDEFINED else _prepare(v) for v in value]
    if hasattr(value, "model_dump"):
        return _prepare(value.model_dump(by_alias=True, exclude_none=False))
    return str(value)


class _JsNumber(int):
    """An integral float that JSON.stringify prints without a fractional part."""

    def __new__(cls, value: float) -> _JsNumber:
        return super().__new__(cls, int(value))


def json_stringify(value: Any, indent: int | str | None = None) -> str:
    """JSON.stringify(value, null, indent). Returns 'undefined' for a bare undefined."""
    if value is UNDEFINED:
        return "undefined"
    if indent in (None, 0, ""):
        return json.dumps(_prepare(value), ensure_ascii=False, separators=(",", ":"))
    return json.dumps(_prepare(value), ensure_ascii=False, indent=indent)


def is_truthy(value: Any) -> bool:
    """JS truthiness: empty containers are truthy, 0/''/null/undefined/NaN are falsy."""
    if value is None or value is UNDEFINED or value is False:
        return False
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    if isinstance(value, str):
        return value != ""
    return True
