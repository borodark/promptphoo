"""Reader for `tests/fixtures/js_golden.json`, the Node golden of JS value semantics.

The file is a verbatim copy of `../kung_phoo/test/fixtures/js_golden.json` at kung_phoo commit
18a7b87, recorded with plain Node v22.21.1 by `../kung_phoo/tools/gen_js_golden.mjs`. Values
JSON cannot carry are tagged; the scheme, as documented in that recorder:

    {"$js":"undefined"}, {"$js":"NaN"}, {"$js":"Infinity"}, {"$js":"-Infinity"}, {"$js":"-0"}
    {"$js":"utf16","units":[...]}           a string with lone surrogates, as UTF-16 code units
    {"$js":"object","pairs":[[k, v], ...]}  an object; inputs define the pairs in this order,
                                            outputs list them in V8's enumeration order
    {"$js":"json","text":"..."}             (inputs) the value JSON.parse(text) gives
    {"$js":"date","ms":...}, {"$js":"bytes","base64":"..."}   (outputs only; unused here)

A `json` input is decoded with promptphoo's own `js_json_parse`, so those cases test the parse
boundary as well as the operation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import UNDEFINED

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "js_golden.json"

_SPECIAL = {
    "undefined": UNDEFINED,
    "NaN": math.nan,
    "Infinity": math.inf,
    "-Infinity": -math.inf,
    "-0": -0.0,
}


def load_golden() -> dict[str, Any]:
    # Plain json.load: integers beyond 2**53 stay exact Python ints, so the number cases also
    # check that promptphoo formats such ints as the doubles they round to.
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def from_units(units: list[int]) -> str:
    """A JS string from UTF-16 code units: pairs combine, lone surrogates stay code points."""
    out: list[str] = []
    i = 0
    while i < len(units):
        u = units[i]
        if 0xD800 <= u <= 0xDBFF and i + 1 < len(units) and 0xDC00 <= units[i + 1] <= 0xDFFF:
            out.append(chr(0x10000 + ((u - 0xD800) << 10) + (units[i + 1] - 0xDC00)))
            i += 2
            continue
        out.append(chr(u))
        i += 1
    return "".join(out)


def decode(tagged: Any) -> Any:
    """The Python value promptphoo uses for a tagged JS value."""
    if isinstance(tagged, list):
        return [decode(x) for x in tagged]
    if not isinstance(tagged, dict):
        return tagged
    tag = tagged["$js"]
    if tag in _SPECIAL:
        return _SPECIAL[tag]
    if tag == "utf16":
        return from_units(tagged["units"])
    if tag == "object":
        return {decode(k): decode(v) for k, v in tagged["pairs"]}
    if tag == "json":
        return js_json_parse(tagged["text"])
    raise ValueError(f"unsupported tag {tag}")


def js_same(a: Any, b: Any) -> bool:
    """SameValue over decoded values: -0 differs from 0, NaN equals NaN, ints equal floats, and
    objects must list the same keys in the same order."""
    if a is UNDEFINED or b is UNDEFINED:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        x, y = float(a), float(b)
        if math.isnan(x) or math.isnan(y):
            return math.isnan(x) and math.isnan(y)
        return x == y and math.copysign(1, x) == math.copysign(1, y)
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(js_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(js_same(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b
