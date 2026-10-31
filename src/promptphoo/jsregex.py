"""Compile JavaScript RegExp source with Python `re`.

Translates the syntax that differs: named groups `(?<n>...)` and `\\k<n>`, `[^]` (any char),
`[]` (never matches), `$` (end of input only, unless multiline), `\\/`, `\\d`/`\\w` ASCII
semantics, `\\u{...}` with the u flag, and the i/m/s flags.
"""

from __future__ import annotations

import re
from functools import lru_cache

# JS \\d and \\w are ASCII-only even with the u flag; \\s is Unicode in both.
_ASCII_CLASSES = {"d": "0-9", "w": "A-Za-z0-9_", "D": "", "W": ""}
_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def translate_js_regex(source: str, flags: str = "") -> str:
    multiline = "m" in flags
    out: list[str] = []
    i, n = 0, len(source)
    in_class = False
    while i < n:
        c = source[i]
        if c == "\\" and i + 1 < n:
            nxt = source[i + 1]
            if nxt == "/":
                out.append("/")
                i += 2
                continue
            if nxt == "k" and i + 2 < n and source[i + 2] == "<" and not in_class:
                end = source.index(">", i + 3)
                out.append(f"(?P={source[i + 3 : end]})")
                i = end + 1
                continue
            if nxt == "u" and i + 2 < n and source[i + 2] == "{":
                end = source.index("}", i + 3)
                out.append(f"\\U{int(source[i + 3 : end], 16):08x}")
                i = end + 1
                continue
            if nxt == "c" and i + 2 < n and source[i + 2].isalpha():
                out.append(f"\\x{ord(source[i + 2].upper()) - 64:02x}")
                i += 3
                continue
            if nxt in _ASCII_CLASSES:
                lower = _ASCII_CLASSES[nxt.lower()]
                if in_class:
                    # Negated shorthands inside a class have no ASCII spelling; keep them.
                    out.append(lower if nxt.islower() else source[i : i + 2])
                else:
                    out.append(f"[{lower}]" if nxt.islower() else f"[^{lower}]")
                i += 2
                continue
            out.append(source[i : i + 2])
            i += 2
            continue
        if in_class:
            if c == "]":
                in_class = False
            elif c == "[":
                out.append("\\[")
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        if c == "[":
            if source.startswith("[^]", i):
                out.append(r"[\s\S]")
                i += 3
                continue
            if source.startswith("[]", i):
                out.append("(?!)")
                i += 2
                continue
            in_class = True
            out.append(c)
            # A leading ']' (after optional '^') is literal in Python but closes in JS; JS
            # has no literal-leading-] so nothing to translate.
            i += 1
            continue
        if c == "(" and source.startswith("(?<", i) and not source.startswith(("(?<=", "(?<!"), i):
            out.append("(?P<")
            i += 3
            continue
        if c == "$" and not multiline:
            out.append(r"\Z")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


@lru_cache(maxsize=512)
def compile_js_regex(source: str, flags: str = "") -> re.Pattern[str]:
    py_flags = 0
    for f in flags:
        py_flags |= _FLAGS.get(f, 0)
    try:
        return re.compile(translate_js_regex(source, flags), py_flags)
    except re.error as e:
        raise ValueError(f"Invalid regular expression: /{source}/{flags}: {e}") from e
