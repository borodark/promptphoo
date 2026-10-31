"""JSON helpers with JavaScript semantics. Ports parts of src/util/json.ts."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from promptphoo.yamlload import load_yaml

_MAX_JSON_LENGTH = 100_000


def _reject_constant(name: str) -> Any:
    raise ValueError(f"Unexpected token {name} in JSON")


_DECODER = json.JSONDecoder(parse_constant=_reject_constant)


def js_json_parse(text: str) -> Any:
    """JSON.parse: like json.loads but rejects NaN/Infinity and only allows JSON whitespace."""
    stripped = text.strip(" \t\n\r")
    value, end = _DECODER.raw_decode(stripped)
    if end != len(stripped):
        raise ValueError("Unexpected non-whitespace character after JSON")
    return value


def is_valid_json(text: str) -> bool:
    try:
        js_json_parse(text)
        return True
    except ValueError:
        return False


def js_deep_equal(a: Any, b: Any) -> bool:
    """util.isDeepStrictEqual over JSON-like values: bool is not a number, 1 == 1.0."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(js_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(js_deep_equal(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


def _is_escaped_quote(line: str, index: int) -> bool:
    count = 0
    j = index - 1
    while j >= 0 and line[j] == "\\":
        count += 1
        j -= 1
    return count % 2 == 1


def _convert_line(line: str) -> str:
    state = "normal"
    result = ""
    i = 0
    while i < len(line):
        char = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        prev = line[i - 1] if i > 0 else ""
        if state == "normal":
            if char == "'" and not re.match(r"[a-zA-Z]", prev):
                state = "singleQuote"
                result += char
            elif char == '"':
                state = "doubleQuote"
                result += char
            elif char == "/" and nxt == "/":
                token_start = 0
                for j in range(i - 1, -1, -1):
                    if line[j].isspace():
                        token_start = j + 1
                        break
                if "://" in line[token_start : i + 2]:
                    result += char
                else:
                    count = 2
                    while i + count < len(line) and line[i + count] == "/":
                        count += 1
                    return result + "#" * (count // 2) + line[i + count :]
            else:
                result += char
        elif state == "singleQuote":
            result += char
            if char == "'" and not _is_escaped_quote(line, i) and not re.match(r"[a-zA-Z]", nxt):
                state = "normal"
        else:
            result += char
            if char == '"' and not _is_escaped_quote(line, i):
                state = "normal"
        i += 1
    return result


def convert_slash_comments_to_hash(text: str) -> str:
    return "\n".join(_convert_line(line) for line in text.split("\n"))


def _strict_object_at(text: str, start: int) -> tuple[Any, int] | None:
    try:
        value, end = _DECODER.raw_decode(text, start)
    except ValueError:
        return None
    return (value, end) if isinstance(value, dict) else None


def extract_json_objects(text: str) -> list[Any]:
    """Find JSON objects embedded in text: strict JSON first, then a tolerant pass that closes
    unbalanced braces and accepts YAML-flow syntax (unquoted keys, // comments)."""
    if len(text) <= _MAX_JSON_LENGTH:
        try:
            parsed = js_json_parse(text)
            if isinstance(parsed, dict):
                return [parsed]
        except ValueError:
            pass

    found: list[Any] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        strict = _strict_object_at(text, i)
        if strict and strict[1] - i <= _MAX_JSON_LENGTH:
            found.append(strict[0])
            i = strict[1]
            continue
        opens, closes = 1, 0
        j = i + 1
        limit = min(i + _MAX_JSON_LENGTH, n)
        advanced = False
        while j < limit and opens > closes:
            if text[j] == "{":
                opens += 1
            if text[j] == "}":
                closes += 1
            j += 1
            if opens == closes or j == n or j == i + _MAX_JSON_LENGTH:
                candidate = text[i:j] + "}" * max(0, opens - closes)
                try:
                    parsed = load_yaml(convert_slash_comments_to_hash(candidate), json=True)
                except Exception:
                    if opens == closes:
                        break
                    continue
                if isinstance(parsed, dict | list):
                    found.append(parsed)
                    i = j
                    advanced = True
                    break
        if not advanced:
            i += 1
    return found


def extract_first_json_object(text: str) -> Any:
    objects = extract_json_objects(text)
    if not objects:
        raise ValueError(f"Expected a JSON object, but got {json.dumps(text)}")
    return objects[0]
