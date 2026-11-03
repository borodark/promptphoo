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
from decimal import ROUND_HALF_UP, Context, Decimal, localcontext
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

_MAX_SAFE_INTEGER = 2**53


def js_number(n: int) -> int | float:
    """The JS number an integer denotes: itself up to 2**53, else the nearest double (V8 has no
    exact integers past 2**53: JSON.parse("9007199254740993") is 9007199254740992)."""
    if -_MAX_SAFE_INTEGER <= n <= _MAX_SAFE_INTEGER:
        return n
    try:
        return float(n)
    except OverflowError:
        return math.inf if n > 0 else -math.inf


def js_json_int(text: str) -> int | float:
    """`parse_int` hook for json.loads: an integer literal as JSON.parse reads it. float() rounds
    the decimal text correctly, as V8 does, and also takes literals too long for int()."""
    if len(text) < 16:
        return -0.0 if text == "-0" else int(text)
    value = float(text)
    return int(value) if abs(value) <= _MAX_SAFE_INTEGER else value


def is_array_index(key: Any) -> bool:
    """A canonical array index ("0".."4294967294"): V8 enumerates such keys first."""
    return (
        isinstance(key, str)
        and key.isdecimal()
        and key.isascii()
        and (key[0] != "0" or key == "0")
        and int(key) < 4294967295
    )


def v8_ordered(obj: dict[str, Any]) -> dict[str, Any]:
    """`obj` in V8's property order: array-index keys first, ascending, then the other keys in
    insertion order. {b:1, 1:2, 0:3} enumerates (and stringifies) as {"0":3,"1":2,"b":1}.
    Returns `obj` itself when it has no index keys."""
    if not any(map(str.isdecimal, obj)):  # the common case, checked at C speed
        return obj
    indices = [key for key in obj if is_array_index(key)]
    if not indices:
        return obj
    ordered = {key: obj[key] for key in sorted(indices, key=int)}
    for key, value in obj.items():
        ordered.setdefault(key, value)
    return ordered


def js_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """`object_pairs_hook` for json.loads: the object JSON.parse builds (the last duplicate wins
    and keeps the first one's place, index keys come first)."""
    return v8_ordered(dict(pairs))


def number_to_string(n: float | int) -> str:
    """JS Number.prototype.toString for finite and special values."""
    if isinstance(n, bool):
        return "true" if n else "false"
    if isinstance(n, int):
        if -_MAX_SAFE_INTEGER <= n <= _MAX_SAFE_INTEGER:
            return str(n)
        return number_to_string(js_number(n))
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
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        if -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
            return value
        value = js_number(value)
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
        # JSON.stringify drops undefined and function-valued properties, and writes the rest in
        # V8's property order.
        return v8_ordered(
            {
                str(k): _prepare(v, numbers)
                for k, v in value.items()
                if v is not UNDEFINED and not callable(v)
            }
        )
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


# A surrogate pair split over two code points (JS sees one character), or a lone surrogate,
# which JSON.stringify writes as a \uXXXX escape.
_SURROGATES = re.compile(r"[\ud800-\udbff][\udc00-\udfff]|[\ud800-\udfff]")


def _surrogate(match: re.Match[str]) -> str:
    text = match.group()
    if len(text) == 2:
        return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le")
    return f"\\u{ord(text):04x}"


def _gap(indent: Any) -> str:
    """JSON.stringify's `space` argument as its gap: a number gives min(10, n) spaces, a string
    its first 10 UTF-16 code units, anything else (or less than 1) no gap."""
    if isinstance(indent, bool) or indent is None:
        return ""
    if isinstance(indent, int | float):
        count = 0 if math.isnan(indent) else min(10, indent)
        return " " * int(count) if count >= 1 else ""
    if isinstance(indent, str):
        if len(indent) <= 5 or indent.isascii():
            return indent[:10]
        units = indent.encode("utf-16-le", "surrogatepass")[:20]
        return units.decode("utf-16-le", "surrogatepass")
    return ""


def json_stringify(value: Any, indent: int | float | str | None = None) -> str:
    """JSON.stringify(value, null, indent). Returns 'undefined' for a bare undefined."""
    if value is UNDEFINED:
        return "undefined"
    numbers: list[str] = []
    prepared = _prepare(value, numbers)
    gap = _gap(indent)
    if not gap:
        out = json.dumps(prepared, ensure_ascii=False, separators=(",", ":"))
    else:
        out = json.dumps(prepared, ensure_ascii=False, indent=gap)
    if not out.isascii():
        try:
            out.encode("utf-8")  # fails only on a surrogate; far faster than scanning with re
        except UnicodeEncodeError:
            out = _SURROGATES.sub(_surrogate, out)
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


# toFixed and toPrecision print at most 21 + 100 significant digits; the default context's 28
# would make quantize raise InvalidOperation past that.
_DIGITS_CONTEXT = Context(prec=200, rounding=ROUND_HALF_UP)


def _range_error(message: str) -> Exception:
    """A JS RangeError, the way promptphoo raises JS throws (fetch.errors.JsError)."""
    from promptphoo.fetch.errors import JsError

    return JsError(message, name="RangeError")


def _to_integer(value: float) -> float:
    """ToIntegerOrInfinity: NaN is 0, finite values truncate toward zero."""
    if math.isnan(value):
        return 0
    return value if math.isinf(value) else math.trunc(value)


def to_fixed(value: float, digits: float = 0) -> str:
    """Number.prototype.toFixed: round half up on the exact binary value."""
    places = _to_integer(digits)
    if not 0 <= places <= 100:
        raise _range_error("toFixed() digits argument must be between 0 and 100")
    if isinstance(value, int):
        value = js_number(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return number_to_string(value)
    if abs(value) >= 1e21:
        return number_to_string(value)
    with localcontext(_DIGITS_CONTEXT):
        d = Decimal(value).quantize(Decimal(1).scaleb(-int(places)))
    text = f"{d:f}"
    return text[1:] if value == 0 and text.startswith("-") else text


def to_precision(value: float, precision: float) -> str:
    """Number.prototype.toPrecision (round half up on the exact value)."""
    p = _to_integer(precision)
    if isinstance(value, int):
        value = js_number(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return number_to_string(value)
    if not 1 <= p <= 100:
        raise _range_error("toPrecision() argument must be between 1 and 100")
    precision = int(p)
    if value == 0:
        return "0" if precision == 1 else "0." + "0" * (precision - 1)
    with localcontext(_DIGITS_CONTEXT):
        d = Decimal(value)
        exponent = d.adjusted()
        rounded = d.quantize(Decimal(1).scaleb(exponent - precision + 1))
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
