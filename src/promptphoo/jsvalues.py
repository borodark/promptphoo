"""JavaScript value semantics needed for output parity with the Node implementation.

promptfoo hashes, stores and renders values the way V8 does; these helpers reproduce the parts
that observably differ from Python (JSON.stringify spacing and number formatting, String()).
"""

from __future__ import annotations

import json
import math
import re
import secrets
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal
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
        return _integral_digits(n)
    r = repr(n)
    if "e" in r:
        mantissa, exp = r.split("e")
        sign = "-" if exp.startswith("-") else "+"
        exp_digits = exp.lstrip("+-").lstrip("0") or "0"
        e = int(exp)
        if -7 < e < 21:
            # V8 prints plain decimals in this range; repr may still use exponent form. Shift
            # repr's shortest round-trip digits rather than format(), which rounds to 6 places.
            neg = mantissa.startswith("-")
            digits = mantissa.lstrip("-").replace(".", "")
            point = e + 1  # digits before the decimal point
            if point <= 0:
                body = "0." + "0" * -point + digits
            elif point >= len(digits):
                body = digits + "0" * (point - len(digits))
            else:
                body = digits[:point] + "." + digits[point:]
            return ("-" if neg else "") + body
        return f"{mantissa}e{sign}{exp_digits}"
    return r


def _integral_digits(n: float) -> str:
    """An integral double below 1e21 as V8 prints it: the shortest round-trip digits, padded
    with zeros (2**60 prints as 1152921504606847000, not its exact 1152921504606846976)."""
    if abs(n) < 2**53:
        return str(int(n))
    mantissa, _, exp = repr(n).partition("e")
    sign = "-" if mantissa.startswith("-") else ""
    int_part, _, frac = mantissa.lstrip("-").partition(".")
    digits = (int_part + frac).rstrip("0") or "0"
    return sign + digits.ljust(int(exp or len(int_part) - 1) + 1, "0")


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


# json.dumps formats floats with float.__repr__ even for subclasses, so a float whose repr is
# not its JS form (1e-07 vs 1e-7, 1e-06 vs 0.000001) is swapped for a placeholder string that
# json_stringify replaces with number_to_string's text after encoding.
_NUM_MARK = f"\x00jsnum{secrets.token_hex(8)}_"


def _prepare(value: Any, numbers: list[str] | None = None) -> Any:
    """Convert to a structure json.dumps renders the way JSON.stringify does."""
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if value.is_integer() and abs(value) < 1e21:
            return _JsNumber(value)
        js = number_to_string(value)
        if numbers is None or js == repr(value):
            return value
        numbers.append(js)
        return f"{_NUM_MARK}{len(numbers) - 1}\x00"
    if isinstance(value, dict):
        # JSON.stringify drops undefined and function-valued properties.
        return {
            str(k): _prepare(v, numbers)
            for k, v in value.items()
            if v is not UNDEFINED and not callable(v)
        }
    if isinstance(value, list | tuple):
        return [None if v is UNDEFINED or callable(v) else _prepare(v, numbers) for v in value]
    if hasattr(value, "model_dump"):
        return _prepare(value.model_dump(by_alias=True, exclude_none=False), numbers)
    if hasattr(value, "to_json"):  # e.g. ApiProvider: JSON.stringify(provider) / toJSON()
        return _prepare(value.to_json(), numbers)
    return str(value)


class _JsNumber(int):
    """An integral float that JSON.stringify prints without a fractional part."""

    def __new__(cls, value: float) -> _JsNumber:
        return super().__new__(cls, int(_integral_digits(value)))


def json_stringify(value: Any, indent: int | str | None = None) -> str:
    """JSON.stringify(value, null, indent). Returns 'undefined' for a bare undefined."""
    if value is UNDEFINED:
        return "undefined"
    numbers: list[str] = []
    prepared = _prepare(value, numbers)
    if indent in (None, 0, ""):
        out = json.dumps(prepared, ensure_ascii=False, separators=(",", ":"))
    else:
        out = json.dumps(prepared, ensure_ascii=False, indent=indent)
    if numbers:
        out = re.sub(
            '"' + re.escape(json.dumps(_NUM_MARK)[1:-1]) + r'(\d+)\\u0000"',
            lambda m: numbers[int(m.group(1))],
            out,
        )
    return out


def js_sum(values: Iterable[float]) -> float:
    """`values.reduce((acc, v) => acc + v, 0)`: plain left-to-right addition.

    Python 3.12's built-in sum() adds floats with compensated summation, so its result can differ
    from JavaScript's in the last bit (sum([0.1] * 10) is 1.0; the JS reduce gives
    0.9999999999999999). Use this wherever a float total must match Node.
    """
    acc: float = 0
    for value in values:
        acc += value
    return acc


def is_truthy(value: Any) -> bool:
    """JS truthiness: empty containers are truthy, 0/''/null/undefined/NaN are falsy."""
    if value is None or value is UNDEFINED or value is False:
        return False
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    if isinstance(value, str):
        return value != ""
    return True


def to_fixed(value: float, digits: int = 0) -> str:
    """Number.prototype.toFixed: round half up on the exact binary value."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return number_to_string(value)
    if abs(value) >= 1e21:
        return number_to_string(value)
    quantum = Decimal(1).scaleb(-digits)
    d = Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP)
    text = f"{d:f}"
    return text[1:] if value == 0 and text.startswith("-") else text


def to_precision(value: float, precision: int) -> str:
    """Number.prototype.toPrecision (round half up on the exact value)."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return number_to_string(value)
    if value == 0:
        return "0" if precision == 1 else "0." + "0" * (precision - 1)
    d = Decimal(value)
    exponent = d.adjusted()
    rounded = d.quantize(Decimal(1).scaleb(exponent - precision + 1), rounding=ROUND_HALF_UP)
    exponent = rounded.adjusted()
    if exponent < -6 or exponent >= precision:
        mantissa = f"{rounded.scaleb(-exponent).quantize(Decimal(1).scaleb(1 - precision)):f}"
        return f"{mantissa}e{'+' if exponent >= 0 else '-'}{abs(exponent)}"
    places = max(precision - 1 - exponent, 0)
    return f"{rounded.quantize(Decimal(1).scaleb(-places)):f}"


def node_base64_decode(data: str) -> bytes:
    """Buffer.from(data, 'base64'): accepts url-safe letters, skips junk, stops at '='."""
    import base64
    import binascii
    import re

    cleaned = data.replace("-", "+").replace("_", "/")
    cleaned = cleaned.split("=", 1)[0]
    cleaned = re.sub(r"[^A-Za-z0-9+/]", "", cleaned)
    if len(cleaned) % 4 == 1:
        cleaned = cleaned[:-1]
    try:
        return base64.b64decode(cleaned + "=" * (-len(cleaned) % 4))
    except binascii.Error:
        return b""
