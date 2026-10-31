"""Output/vars transforms. Ports src/util/transform.ts.

`file://x.py[:fn]` runs `fn(output, context)` (default `get_transform`). Inline strings are
JavaScript in Node; promptphoo evaluates single expressions in the Nunjucks-compatible Jinja
environment, so dotted paths, indexing and JS member idioms work unchanged
(`output.trim()`, `json.choices[0].message.content`, `JSON.parse(output).answer`, `a && b`).
Statements, arrow functions and `?:` raise a clear error pointing at a Python file instead.
"""

from __future__ import annotations

import inspect
import os
import re
from collections.abc import Callable
from typing import Any

import jinja2

from promptphoo import cli_state, python_hooks
from promptphoo.files import is_javascript_file
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import json_stringify
from promptphoo.templating import get_engine

INLINE_FUNCTION_LABEL = "[inline function]"
INLINE_STRING_LABEL = "[inline transform]"
FILE_TRANSFORM_LABEL = "[file transform]"

_UNSUPPORTED = re.compile(r"=>|\bfunction\b|\breturn\b|\bconst\b|\blet\b|\bvar\b|;|(?<!\?)\?(?!\?)")
_STRING = re.compile(r"""("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)""")


class TransformError(RuntimeError):
    pass


class _Json:
    @staticmethod
    def parse(text: str) -> Any:
        return js_json_parse(text)

    @staticmethod
    def stringify(value: Any, _replacer: Any = None, indent: Any = None) -> str:
        return json_stringify(value, indent)


def get_transform_label(t: Any) -> str:
    if t is None:
        return INLINE_STRING_LABEL
    if callable(t):
        name = getattr(t, "__name__", "")
        return f"{INLINE_FUNCTION_LABEL}: {name}" if name and name != "<lambda>" else (
            INLINE_FUNCTION_LABEL
        )
    if t.startswith("file://"):
        return FILE_TRANSFORM_LABEL
    single = re.sub(r"\s+", " ", t).strip()
    if len(single) > 80:
        single = single[:79] + "…"
    return f"{INLINE_STRING_LABEL}: {single}"


def _translate_js_expression(code: str) -> str:
    """Rewrite JS operators outside string literals into Jinja syntax."""
    parts = _STRING.split(code)
    out = []
    for i, part in enumerate(parts):
        if i % 2:  # string literal
            out.append(part if not part.startswith("`") else json_stringify(part[1:-1]))
            continue
        part = part.replace("?.", ".")  # undefined-chaining is the default here
        if _UNSUPPORTED.search(part):
            raise ValueError(
                "JavaScript transforms are not supported by promptphoo; use a single "
                "expression or a Python file (file://transform.py)"
            )
        part = re.sub(r"===?", "==", part)
        part = re.sub(r"!==?", "!=", part)
        part = part.replace("&&", " and ").replace("||", " or ").replace("??", " or ")  # approx.
        part = re.sub(r"!(?!=)", " not ", part)
        part = re.sub(r"\bnull\b|\bundefined\b", "none", part)
        out.append(part)
    return "".join(out)


def _inline_function(code: str, input_name: str) -> Callable[..., Any]:
    if cli_state.safe_mode():
        raise ValueError(
            "Inline transform execution is disabled in safe mode. Please use a file reference "
            'instead (e.g. "file://path/to/transform.py").'
        )
    if "\n" in code.strip():
        raise ValueError(
            "Multi-line JavaScript transforms are not supported by promptphoo; "
            "move the code into a Python file (file://transform.py)"
        )
    expression = get_engine().env.compile_expression(
        _translate_js_expression(code.strip()), undefined_to_none=True
    )

    def fn(value: Any, context: Any) -> Any:
        return expression(
            **{input_name: value, "context": context, "JSON": _Json,
               "process": {"env": dict(os.environ)}}
        )  # fmt: skip

    return fn


def _file_function(ref: str) -> Callable[..., Any]:
    target = ref[len("file://") :]
    idx = target.rfind(":")
    path, fn_name = (target[:idx], target[idx + 1 :]) if idx > 1 else (target, None)
    full = os.path.join(cli_state.base_path() or "", path)
    if is_javascript_file(full):
        raise ValueError(
            f"JavaScript transform files are not supported by promptphoo: {ref}. "
            "Use a Python file instead."
        )
    if not full.endswith(".py"):
        raise ValueError(f"Unsupported transform file format: file://{path}")

    async def fn(value: Any, context: Any) -> Any:
        return await python_hooks.run_python(full, fn_name or "get_transform", [value, context])

    return fn


async def transform(
    code_or_fn: str | Callable[..., Any],
    value: Any,
    context: dict[str, Any],
    validate_return: bool = True,
    input_type: str = "output",
) -> Any:
    label = get_transform_label(code_or_fn)
    if callable(code_or_fn):
        fn = code_or_fn
    elif code_or_fn.startswith("file://"):
        fn = _file_function(code_or_fn)
    else:
        try:
            fn = _inline_function(code_or_fn, input_type)
        except jinja2.TemplateSyntaxError as e:
            raise TransformError(f"Transform failed ({label}): {e}") from e
    try:
        ret = fn(value, context)
        if inspect.isawaitable(ret):
            ret = await ret
    except Exception as e:
        raise TransformError(f"Transform failed ({label}): {e}") from e
    if isinstance(ret, jinja2.Undefined):
        ret = None
    if validate_return and ret is None:
        raise TransformError(f"Transform function did not return a value\n\n{label}")
    return ret
