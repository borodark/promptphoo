"""HTTP provider transforms. Ports src/providers/httpTransforms.ts, src/providers/transformUtils.ts
and createSessionParser / createValidateStatus from src/providers/http.ts.

Node evaluates string transforms as JavaScript. promptphoo evaluates them as single expressions
with `transform.compile_inline_expression` (Jinja-backed, so dotted paths into dicts such as
`json.choices[0].message.content` work unchanged, as do `&&`, `||`, `===` and Python's
`x if c else y`). Arrow functions, `function` bodies and statements raise a "not supported"
error; move that code into a Python file instead.

`file://x.py[:fn]` references call a Python function (JSON round-tripped, as Node's Python
bridge does). The default function names are `transform_response(json, text, context)`,
`transform_request(prompt, vars, context)`, `session_parser(data)` and
`validate_status(status)`. JavaScript files are rejected.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

import jinja2

from promptphoo import cli_state, python_hooks
from promptphoo.files import is_javascript_file
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.sanitizer import safe_json_stringify, sanitize_object
from promptphoo.transform import compile_inline_expression

logger = logging.getLogger(__name__)

_NON_SERIALIZABLE_CONTEXT_KEYS = ("getCache", "logger", "filters", "originalProvider")


def js_typeof(value: Any) -> str:
    if value is None:
        return "object"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if callable(value):
        return "function"
    return "object"


def normalize_response_transform_result(value: Any) -> dict[str, Any]:
    """Ports src/providers/transformResult.ts."""
    if isinstance(value, dict) and ("output" in value or "error" in value):
        return value
    return {"output": value}


async def call_flexible(fn: Callable[..., Any], *args: Any) -> Any:
    """Call `fn` with as many positional args as it accepts (JS ignores extra arguments)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        params = None
    if params is not None and not any(p.kind in (p.VAR_POSITIONAL,) for p in params):
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        args = args[: len(positional)]
    result = fn(*args)
    if inspect.isawaitable(result):
        result = await result
    return result


def parse_file_transform_reference(reference: str) -> dict[str, str]:
    """Split `file://path[:fn]`; only script files (JS or .py) take a function suffix."""
    raw = reference[len("file://") :] if reference.startswith("file://") else reference
    idx = raw.rfind(":")
    if idx == -1:
        return {"filename": raw}
    candidate, fn = raw[:idx], raw[idx + 1 :]
    if candidate and fn and (is_javascript_file(candidate) or candidate.endswith(".py")):
        return {"filename": candidate, "functionName": fn}
    return {"filename": raw}


def _sanitize_hook_arg(value: Any) -> Any:
    if isinstance(value, dict) and any(k in value for k in _NON_SERIALIZABLE_CONTEXT_KEYS):
        return {k: v for k, v in value.items() if k not in _NON_SERIALIZABLE_CONTEXT_KEYS}
    return value


def load_python_reference(
    reference: str, default_function_name: str, malformed: Callable[[str], str]
) -> Callable[..., Awaitable[Any]]:
    """Resolve `file://x.py[:fn]` relative to basePath into an async callable."""
    ref = parse_file_transform_reference(reference)
    filename = ref["filename"]
    full = os.path.join(cli_state.base_path() or "", filename)
    full = os.path.abspath(full)
    if is_javascript_file(filename):
        raise ValueError(
            f"JavaScript transform files are not supported by promptphoo: {reference}. "
            "Use a Python file instead (file://transform.py)."
        )
    if not filename.endswith(".py"):
        raise ValueError(f"Unsupported transform file format: {reference}")
    fn_name = ref.get("functionName") or default_function_name
    try:
        python_hooks.load_function(full, fn_name)
    except (AttributeError, TypeError) as e:
        raise ValueError(malformed(filename)) from e

    async def call(*args: Any) -> Any:
        return await python_hooks.run_python(full, fn_name, [_sanitize_hook_arg(a) for a in args])

    return call


def load_transform_module(transform: Any, default_function_name: str) -> Any:
    """Load a file:// transform into a callable; other values pass through."""
    if not transform or callable(transform):
        return transform
    if isinstance(transform, str) and transform.startswith("file://"):
        return load_python_reference(
            transform,
            default_function_name,
            lambda filename: (
                f"Transform module malformed: {filename} must export a function or have a "
                "default export as a function"
            ),
        )
    return transform


def _compile(code: str) -> Callable[..., Any]:
    return compile_inline_expression(code.strip())


def _js_error(err: BaseException) -> str:
    """String(err) for an error thrown inside a JS transform."""
    if isinstance(err, jinja2.TemplateSyntaxError):
        return f"SyntaxError: {err}"
    return f"Error: {err}"


ResponseTransform = Callable[..., Awaitable[dict[str, Any]]]


def create_transform_response(parser: Any) -> ResponseTransform:
    """Build `(data, text, context) -> ProviderResponse` from a transformResponse config."""
    if not parser:

        async def default(data: Any, text: str, context: Any = None) -> dict[str, Any]:
            return {"output": data if is_truthy(data) else text}

        return default

    if callable(parser):

        async def from_function(data: Any, text: str, context: Any = None) -> dict[str, Any]:
            try:
                result = await call_flexible(parser, data, text, context)
                return normalize_response_transform_result(result)
            except Exception as err:
                logger.error(
                    "[Http Provider] Error in response transform function: Error: %s. Data: %s. "
                    "Text: %s. Context: %s.",
                    err,
                    safe_json_stringify(data),
                    text,
                    safe_json_stringify(context),
                )
                raise

        return from_function

    if isinstance(parser, str) and parser.startswith("file://"):
        raise ValueError(
            "Response transform with file:// reference should be pre-loaded before calling "
            "createTransformResponse. This is a bug in the HTTP provider implementation."
        )
    if isinstance(parser, str):

        async def from_expression(data: Any, text: str, context: Any = None) -> dict[str, Any]:
            try:
                fn = _compile(parser)
                json_value = data if is_truthy(data) else None
                try:
                    resp = fn(json=json_value, text=text, context=context)
                except Exception as e:
                    raise RuntimeError(
                        f"Transform failed: {e} : {text} : {json_stringify(json_value)} : "
                        f"{json_stringify(context)}"
                    ) from e
                return normalize_response_transform_result(resp)
            except Exception as err:
                logger.error(
                    "[Http Provider] Error in response transform: %s. Data: %s. Text: %s. "
                    "Context: %s.",
                    _js_error(err),
                    safe_json_stringify(data),
                    text,
                    safe_json_stringify(context),
                )
                raise ValueError(f"Failed to transform response: {_js_error(err)}") from err

        return from_expression

    raise ValueError(
        f"Unsupported response transform type: {js_typeof(parser)}. Expected a function, a string "
        "starting with 'file://' pointing to a JavaScript file, or a string containing a "
        "JavaScript expression."
    )


RequestTransform = Callable[..., Awaitable[Any]]


def create_transform_request(transform: Any) -> RequestTransform:
    """Build `(prompt, vars, context) -> transformed prompt` from a transformRequest config."""
    if not transform:

        async def identity(prompt: Any, vars: Any = None, context: Any = None) -> Any:
            return prompt

        return identity

    if callable(transform):

        async def from_function(prompt: Any, vars: Any = None, context: Any = None) -> Any:
            try:
                return await call_flexible(transform, prompt, vars, context)
            except Exception as err:
                message = f"Error in request transform function: {err}"
                logger.error(message)
                raise ValueError(message) from err

        return from_function

    if isinstance(transform, str) and transform.startswith("file://"):
        raise ValueError(
            "Request transform with file:// reference should be pre-loaded before calling "
            "createTransformRequest. This is a bug in the HTTP provider implementation."
        )
    if isinstance(transform, str):

        async def from_expression(prompt: Any, vars: Any = None, context: Any = None) -> Any:
            try:
                fn = _compile(transform)
                try:
                    return fn(prompt=prompt, vars=vars, context=context)
                except Exception as e:
                    raise RuntimeError(f"Transform failed: {e}") from e
            except Exception as err:
                logger.error(
                    "[Http Provider] Error in request transform: %s. Prompt: %s. Vars: %s. "
                    "Context: %s.",
                    _js_error(err),
                    prompt,
                    safe_json_stringify(vars),
                    safe_json_stringify(
                        sanitize_object(_sanitize_hook_arg(context), context="request transform")
                    ),
                )
                raise ValueError(f"Failed to transform request: {_js_error(err)}") from err

        return from_expression

    raise ValueError(
        f"Unsupported request transform type: {js_typeof(transform)}. Expected a function, a "
        "string starting with 'file://' pointing to a JavaScript file, or a string containing a "
        "JavaScript expression."
    )


SessionParser = Callable[[dict[str, Any]], Awaitable[Any]]


def create_session_parser(parser: Any) -> SessionParser:
    """Build `({headers, body}) -> session id` from a sessionParser / session.responseParser."""
    if not parser:

        async def empty(data: dict[str, Any]) -> Any:
            return ""

        return empty

    if callable(parser):

        async def from_function(data: dict[str, Any]) -> Any:
            return await call_flexible(parser, data)

        return from_function

    if isinstance(parser, str) and parser.startswith("file://"):
        fn = load_python_reference(
            parser,
            "session_parser",
            lambda filename: (
                f"Response transform malformed: {filename} must export a function or have a "
                "default export as a function"
            ),
        )

        async def from_file(data: dict[str, Any]) -> Any:
            return await fn(data)

        return from_file

    if isinstance(parser, str):
        expression = _compile(parser)

        async def from_expression(data: dict[str, Any]) -> Any:
            return expression(data=data)

        return from_expression

    raise ValueError(
        f"Unsupported response transform type: {js_typeof(parser)}. Expected a function, a string "
        "starting with 'file://' pointing to a JavaScript file, or a string containing a "
        "JavaScript expression."
    )


StatusValidator = Callable[[int], Awaitable[Any]]

_STATUS_LIST = re.compile(r"^\s*\d{3}(?:\s*-\s*\d{3})?(?:\s*,\s*\d{3}(?:\s*-\s*\d{3})?)*\s*\Z")


def _status_set_validator(spec: list[Any] | str) -> StatusValidator:
    """promptphoo extension: a list such as [200, 201, "400-499"] or the string "200-299,404"."""
    items: list[Any] = spec.split(",") if isinstance(spec, str) else list(spec)
    ranges: list[tuple[int, int]] = []
    for item in items:
        if isinstance(item, bool):
            raise ValueError(f"Invalid status validator entry: {item!r}")
        if isinstance(item, int):
            ranges.append((item, item))
            continue
        text = str(item).strip()
        m = re.fullmatch(r"(\d{3})\s*-\s*(\d{3})", text)
        if m:
            ranges.append((int(m.group(1)), int(m.group(2))))
        elif re.fullmatch(r"\d{3}", text):
            ranges.append((int(text), int(text)))
        else:
            raise ValueError(f"Invalid status validator entry: {item!r}")

    async def validate(status: int) -> bool:
        return any(lo <= status <= hi for lo, hi in ranges)

    return validate


def create_validate_status(validator: Any) -> StatusValidator:
    """Build `(status) -> bool` from validateStatus."""
    if not validator:

        async def accept_all(status: int) -> bool:
            return True

        return accept_all

    if callable(validator):

        async def from_function(status: int) -> Any:
            return await call_flexible(validator, status)

        return from_function

    if isinstance(validator, list):
        return _status_set_validator(validator)

    if isinstance(validator, str):
        if validator.startswith("file://"):
            filename = parse_file_transform_reference(validator)["filename"]
            try:
                fn = load_python_reference(
                    validator,
                    "validate_status",
                    lambda _f: "Exported value must be a function",
                )
            except Exception as err:
                raise ValueError(f"Status validator malformed: {filename} - {err}") from err

            async def from_file(status: int) -> Any:
                return await fn(status)

            return from_file
        if _STATUS_LIST.match(validator):
            return _status_set_validator(validator)
        try:
            expression = _compile(validator)
        except Exception as err:
            raise ValueError(f"Invalid status validator expression: {err}") from err

        async def from_expression(status: int) -> Any:
            return expression(status=status)

        return from_expression

    raise ValueError(
        f"Unsupported status validator type: {js_typeof(validator)}. Expected a function, a string "
        "starting with 'file://' pointing to a JavaScript file, or a string containing a "
        "JavaScript expression."
    )
