"""Tool-call assertions.

Ports:
- src/assertions/functionToolCall.ts (is-valid-function-call, is-valid-openai-function-call)
- src/assertions/openai.ts (is-valid-openai-tools-call)
- validateFunctionCall from src/providers/openai/util.ts and prepareOpenAiFunctions from
  src/providers/openai/prepare.ts (exported for the OpenAI provider's validate_function_tool_call)
- maybeLoadToolsFromExternalFile from src/util/file.ts
- src/assertions/toolCallF1.ts (tool-call-f1)
- src/assertions/skill.ts (skill-used)
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterator
from typing import Any

from promptphoo.assertions.base import (
    Assertion,
    AssertionParams,
    AssertionValueError,
    GradingResult,
    Handler,
    invariant,
)
from promptphoo.assertions.refusal import JS_WHITESPACE, js_trim
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import UNDEFINED, is_truthy, js_string, json_stringify, to_fixed

# --- errors ------------------------------------------------------------------------------------


class InvalidToolCallError(Exception):
    """contracts/providers.ts InvalidToolCallError: the output is not a valid call."""


class InvalidToolSchemaError(Exception):
    """providers/toolSchemaError.ts: tool definitions that cannot be loaded or compiled."""


def _error_message(err: BaseException) -> str:
    return str(err)


# --- functionToolCall.ts -----------------------------------------------------------------------


def tool_call_verdict(
    assertion: Assertion, inverse: bool, valid: bool, reason: str, label: str
) -> GradingResult:
    """Builds a validation verdict, inverted for `not-` assertions. Not for configuration errors."""
    passed = (not valid) if inverse else valid
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": ("Assertion passed" if passed else f"Expected output to not be a valid {label}")
        if inverse
        else reason,
        "assertion": assertion,
    }


def tool_call_error_verdict(
    assertion: Assertion, inverse: bool, err: BaseException, label: str
) -> GradingResult:
    """Only a known invalid-call verdict can satisfy a negated assertion."""
    reason = _error_message(err)
    # Node checks err.name, so any provider-defined class with this name counts.
    if type(err).__name__ != "InvalidToolCallError":
        return {"pass": False, "score": 0, "reason": reason, "assertion": assertion}
    return tool_call_verdict(assertion, inverse, False, reason, label)


def _function_tool_call_validator(provider: Any) -> Any:
    """hasFunctionToolCallValidator: Node's validateFunctionToolCall, snake_cased."""
    if provider is None:
        return None
    for name in ("validate_function_tool_call", "validate_function_call"):
        fn = getattr(provider, name, None)
        if callable(fn):
            return fn
    return None


def handle_is_valid_function_call(params: AssertionParams) -> GradingResult:
    assertion, inverse = params.assertion, params.inverse
    validator = _function_tool_call_validator(params.provider)
    # Without a validator there is no verdict to negate, so this fails under `not-` too.
    if validator is None:
        return {
            "pass": False,
            "score": 0,
            "reason": "Provider does not have functionality for checking function call.",
            "assertion": assertion,
        }
    try:
        validator(params.output, (params.test or {}).get("vars"))
        return tool_call_verdict(assertion, inverse, True, "Assertion passed", "function call")
    except Exception as err:
        return tool_call_error_verdict(assertion, inverse, err, "function call")


# --- providers/openai/prepare.ts: prepareOpenAiFunctions ---------------------------------------


def _render(value: Any, vars: dict[str, Any] | None) -> Any:
    from promptphoo.templating import render_vars_in_object

    return render_vars_in_object(value, vars)


def _is_file_reference(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("file://")


def _load_and_render(value: Any, vars: dict[str, Any] | None) -> Any:
    """Render authored values once. Loaded files are new authored input, not rendered output."""
    if _is_file_reference(value):
        from promptphoo.files import maybe_load_from_external_file

        reference = _render(value, vars)
        return _render(maybe_load_from_external_file(reference), vars)
    return _render(value, vars)


def _zod_type_name(value: Any) -> str:
    if value is UNDEFINED:
        return "undefined"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "NaN" if isinstance(value, float) and math.isnan(value) else "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _invalid_type(expected: str, path: list[Any], value: Any) -> dict[str, Any]:
    return {
        "expected": expected,
        "code": "invalid_type",
        "path": path,
        "message": f"Invalid input: expected {expected}, received {_zod_type_name(value)}",
    }


def _function_definition_issues(definitions: list[Any]) -> list[dict[str, Any]]:
    """z.array(FunctionDefinition).parse issues, in zod v4's shape."""
    issues: list[dict[str, Any]] = []
    for i, d in enumerate(definitions):
        if not isinstance(d, dict):
            issues.append(_invalid_type("object", [i], d))
            continue
        name = d.get("name", UNDEFINED)
        if not isinstance(name, str):
            issues.append(_invalid_type("string", [i, "name"], name))
        elif not js_trim(name):
            issues.append(
                {
                    "code": "custom",
                    "path": [i, "name"],
                    "message": "Expected a nonempty function name",
                }
            )
        if "description" in d and not isinstance(d["description"], str):
            issues.append(_invalid_type("string", [i, "description"], d["description"]))
        if "parameters" in d and not isinstance(d["parameters"], dict):
            issues.append(_invalid_type("record", [i, "parameters"], d["parameters"]))
        if "strict" in d and d["strict"] is not None and not isinstance(d["strict"], bool):
            issues.append(_invalid_type("boolean", [i, "strict"], d["strict"]))
    return issues


def prepare_openai_functions(value: Any, vars: dict[str, Any] | None = None) -> Any:
    loaded = _load_and_render(value, vars or {})
    if loaded is not None and not isinstance(loaded, list):
        raise ValueError("Expected the loaded functions to be an array")
    if loaded is None:
        return None
    issues = _function_definition_issues(loaded)
    if issues:
        raise ValueError(json.dumps(issues, indent=2, ensure_ascii=False))
    return loaded


# --- providers/openai/util.ts: validateFunctionCall --------------------------------------------


def validate_openai_function_call(
    output: Any, functions: Any = None, vars: dict[str, Any] | None = None
) -> None:
    """Raise InvalidToolCallError (bad call) or InvalidToolSchemaError (bad definitions)."""
    from promptphoo.ajv import compile_schema

    if isinstance(output, dict) and "function_call" in output:
        output = output["function_call"]
    function_call = output
    if (
        not isinstance(function_call, dict)
        or not isinstance(function_call.get("name"), str)
        or not isinstance(function_call.get("arguments"), str)
    ):
        raise InvalidToolCallError(
            f"OpenAI did not return a valid-looking function call: {json_stringify(function_call)}"
        )

    try:
        interpolated = prepare_openai_functions(functions, vars)
    except Exception as err:
        raise InvalidToolSchemaError(_error_message(err)) from err
    if not interpolated:
        raise InvalidToolSchemaError(
            f'Called "{function_call["name"]}", but there is no function with that name'
        )
    try:
        function_args = js_json_parse(function_call["arguments"])
    except ValueError as err:
        raise InvalidToolCallError(_error_message(err)) from err
    function_name = function_call["name"]
    definition = next((f for f in interpolated if f.get("name") == function_name), None)
    if definition is None:
        raise InvalidToolCallError(
            f'Called "{function_name}", but there is no function with that name'
        )
    parameters = definition.get("parameters")
    if parameters is None:
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    try:
        validate = compile_schema(parameters)
    except Exception as err:
        raise InvalidToolSchemaError(_error_message(err)) from err
    if not validate(function_args):
        raise InvalidToolCallError(
            f'Call to "{function_name}" does not match schema: {json_stringify(validate.errors)}'
        )


# --- util/file.ts: maybeLoadToolsFromExternalFile ----------------------------------------------


async def maybe_load_tools_from_external_file(
    tools: Any, vars: dict[str, Any] | None = None
) -> Any:
    from promptphoo import cli_state
    from promptphoo.files import is_javascript_file, maybe_load_from_external_file, parse_file_url

    rendered = _render(tools, vars)

    if isinstance(rendered, str) and rendered.startswith("file://"):
        file_path, function_name = parse_file_url(rendered)
        base_path = cli_state.base_path() or os.getcwd()
        is_py = file_path.endswith(".py")
        if function_name and (is_py or is_javascript_file(file_path)):
            try:
                if not is_py:
                    raise ValueError(
                        "JavaScript tool files are not supported by promptphoo; "
                        "use a Python file (file://tools.py:get_tools) instead"
                    )
                from promptphoo.python_hooks import run_python

                abs_path = os.path.abspath(os.path.join(base_path, file_path))
                definitions = await run_python(abs_path, function_name, [])
                if not is_truthy(definitions) or isinstance(definitions, str | bool | int | float):
                    returned = "null" if definitions is None else _zod_type_name(definitions)
                    raise ValueError(
                        f'Function "{function_name}" must return an array or object of tool '
                        f"definitions, but returned: {returned}"
                    )
                return definitions
            except Exception as err:
                raise ValueError(
                    f"Failed to load tools from {rendered}:\n{_error_message(err)}\n\n"
                    f'Make sure the function "{function_name}" exists and returns a valid tool '
                    f"definition array.\nResolved from: {base_path}"
                ) from err

        if is_py or is_javascript_file(file_path):
            ext = "Python" if is_py else "JavaScript"
            example = (
                '  def get_tools():\n      return [{"type": "function", "function": {...}}]'
                if is_py
                else '  module.exports.get_tools = () => [{ type: "function", function: {...} }];'
            )
            raise ValueError(
                f"Cannot load tools from {rendered}\n"
                f"{ext} files require a function name. Use this format:\n"
                f"  tools: file://{file_path}:get_tools\n\n"
                f"Your {ext} file should export a function that returns tool definitions:\n"
                f"{example}\n\nResolved from: {base_path}"
            )

    # Array entries are already rendered; only load their file references.
    if isinstance(rendered, list):
        results = [await maybe_load_tools_from_external_file(item) for item in rendered]
        if all(isinstance(r, list) for r in results):
            return [x for r in results for x in r]
        return results

    if not isinstance(rendered, str):
        return rendered

    loaded = maybe_load_from_external_file(rendered)
    if isinstance(loaded, str):
        if loaded.startswith("file://"):
            raise ValueError(
                f"Failed to load tools from {loaded}\n"
                "Ensure the file exists and contains valid JSON or YAML tool definitions."
            )
        if "def " in loaded or "import " in loaded:
            raise ValueError(
                "Invalid tools configuration: file appears to contain Python code.\n"
                "Python files require a function name. Use this format:\n"
                "  tools: file://tools.py:get_tools"
            )
        raise ValueError(
            "Invalid tools configuration: expected an array or object, but got a string.\n"
            "If using file://, ensure the file contains valid JSON or YAML tool definitions."
        )
    return loaded


# --- openai.ts ---------------------------------------------------------------------------------

_NOT_LINE_TERMINATOR = "[^\n\r\u2028\u2029]"
_MCP_ERROR = re.compile(rf"MCP Tool Error \(([^)]+)\): ({_NOT_LINE_TERMINATOR}+)")
_MCP_RESULT = re.compile(r"MCP Tool Result \(([^)]+)\):")


async def handle_is_valid_openai_tools_call(params: AssertionParams) -> GradingResult:
    assertion, inverse, output = params.assertion, params.inverse, params.output
    vars = (params.test or {}).get("vars")
    label = "OpenAI tools call"

    def verdict(valid: bool, reason: str) -> GradingResult:
        return tool_call_verdict(assertion, inverse, valid, reason, label)

    # Handle MCP tool outputs from Responses API
    output_str = output if isinstance(output, str) else json_stringify(output)
    if output_str == "undefined":
        output_str = ""

    if "MCP Tool Result" in output_str or "MCP Tool Error" in output_str:
        if "MCP Tool Error" in output_str:
            m = _MCP_ERROR.search(output_str)
            tool_name = m.group(1) if m else "unknown"
            error_msg = m.group(2) if m else "unknown error"
            return verdict(False, f"MCP tool call failed for {tool_name}: {error_msg}")
        m = _MCP_RESULT.search(output_str)
        tool_name = m.group(1) if m else "unknown"
        return verdict(True, f"MCP tool call succeeded for {tool_name}")

    # Handle traditional OpenAI function/tool calls
    if isinstance(output, dict) and "tool_calls" in output:
        output = output["tool_calls"]
    tools_output = output

    def looks_valid(tool_call: Any) -> bool:
        fn = tool_call.get("function") if isinstance(tool_call, dict) else None
        return (
            isinstance(fn, dict)
            and isinstance(fn.get("name"), str)
            and isinstance(fn.get("arguments"), str)
        )

    if (
        not isinstance(tools_output, list)
        or len(tools_output) == 0
        or not all(looks_valid(tc) for tc in tools_output)
    ):
        return verdict(
            False,
            f"OpenAI did not return a valid-looking tools response: {json_stringify(tools_output)}",
        )

    config = getattr(params.provider, "config", None)
    tools = config.get("tools") if isinstance(config, dict) else None
    try:
        if is_truthy(tools):
            loaded = await maybe_load_tools_from_external_file(tools, vars)
            if loaded is not UNDEFINED:
                tools = loaded
    except Exception as err:
        return tool_call_error_verdict(assertion, inverse, err, label)

    # Tools must be defined when validating tool calls. Missing tools is a
    # configuration error, so it fails under `not-` too.
    if not is_truthy(tools):
        return {
            "pass": False,
            "score": 0,
            "reason": "No tools configured in provider, but output contains tool calls",
            "assertion": assertion,
        }
    if not isinstance(tools, list) or any(
        not isinstance(tool, dict)
        or not isinstance(tool.get("type"), str)
        or (tool["type"] == "function" and "function" not in tool)
        for tool in tools
    ):
        return {
            "pass": False,
            "score": 0,
            "reason": "Expected tools to be an array of tool definitions",
            "assertion": assertion,
        }
    functions = [t["function"] for t in tools if t["type"] == "function" and "function" in t]
    try:
        for tool_output in tools_output:
            validate_openai_function_call(tool_output["function"], functions, vars)
        return verdict(True, "Assertion passed")
    except Exception as err:
        return tool_call_error_verdict(assertion, inverse, err, label)


# --- toolCallF1.ts -----------------------------------------------------------------------------

_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-+*]|[0-9]+[.)])[ \t]+")
_NON_WS = re.compile(f"[^{JS_WHITESPACE}]")
_OPENERS = {"{": True, "[": True}
_DELIMS = frozenset("{[}]")
MAX_UNMATCHED_DELIMITERS = 65_536


def _column_width(text: str) -> int:
    column = 0
    for char in text:
        column += 4 - (column % 4) if char == "\t" else 1
    return column


# A delimiter token is (char, index, starts_line, ends_line); None resets the candidate stack.
_Token = tuple[str, int, bool, bool] | None


def _json_delimiters(text: str) -> Iterator[_Token]:
    in_string = False
    escaped = False
    fence = ""
    fence_container = 0
    containers: list[int] = []
    line_start = 0
    n = len(text)

    while line_start < n:
        offset = line_start
        newline = text.find("\n", line_start)
        line_end = n if newline < 0 else newline
        content = text[line_start:line_end].rstrip(JS_WHITESPACE)
        line_start = line_end + 1
        m = _NON_WS.search(content)
        first_content = m.start() if m else -1
        indent = _column_width(content[: max(0, first_content)])
        if fence and content and indent < fence_container:
            fence = ""
            yield None
        if fence:
            marker = _FENCE.match(content)
            if (
                marker
                and indent <= fence_container + 3
                and marker.group(1)[0] == fence[0]
                and len(marker.group(1)) >= len(fence)
                and not js_trim(content[marker.end() :])
            ):
                fence = ""
            continue

        while content and containers and indent < containers[-1]:
            containers.pop()
        lst = _LIST_MARKER.match(content)
        if lst:
            containers.append(_column_width(lst.group(0)))
        container = containers[-1] if containers else 0
        fence_text = content[lst.end() :] if lst else content
        marker = _FENCE.match(fence_text)
        if (
            marker
            and (lst or indent <= container + 3)
            and (marker.group(1)[0] != "`" or "`" not in fence_text[marker.end() :])
        ):
            fence = marker.group(1)
            fence_container = container
            yield None
            continue

        last = len(content) - 1
        for column, char in enumerate(content):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in _DELIMS:
                yield (char, offset + column, column == first_content, column == last)

        # JSON strings cannot span literal newlines.
        if in_string:
            in_string = False
            escaped = False
            yield None


class ToolCallParseError(Exception):
    pass


def _json_blocks(text: str, skip_root: bool = False) -> Iterator[tuple[str, bool]]:
    # Pop complete pairs so independent blocks do not accumulate in memory.
    unmatched: list[int] = []
    stack_start = 0
    for token in _json_delimiters(text):
        if token is None:
            stack_start = len(unmatched)
        elif token[0] in _OPENERS:
            unmatched.append(token[1])
        elif len(unmatched) > stack_start and text[unmatched[-1]] == (
            "{" if token[0] == "}" else "["
        ):
            unmatched.pop()
        else:
            unmatched.append(token[1])
            stack_start = len(unmatched)
        if len(unmatched) > MAX_UNMATCHED_DELIMITERS:
            raise ToolCallParseError(
                f"Tool Call F1 exceeded its delimiter limit ({MAX_UNMATCHED_DELIMITERS})"
            )

    # Yield disjoint blocks; nested recovery skips the enclosing root.
    next_unmatched = 0
    depth = 0
    start = -1
    start_depth = 0
    starts_line = False
    root_depth = 1 if skip_root else 0
    for token in _json_delimiters(text):
        if token is None:
            continue
        char, index, token_starts_line, ends_line = token
        if next_unmatched < len(unmatched) and index == unmatched[next_unmatched]:
            next_unmatched += 1
            continue
        if char in _OPENERS:
            if start < 0 and depth == root_depth:
                start = index
                start_depth = depth
                starts_line = token_starts_line
            depth += 1
            continue
        depth -= 1
        if start >= 0 and depth == start_depth:
            if ends_line:
                yield text[start : index + 1], starts_line
            start = -1


def _extract_json_blocks(text: str) -> Iterator[Any]:
    pending = [_json_blocks(text)]
    remaining = 4 * len(text)
    while pending:
        nxt = next(pending[-1], None)
        if nxt is None:
            pending.pop()
            continue
        block, starts_line = nxt
        if len(block) > remaining:
            raise ToolCallParseError(
                "Tool Call F1 could not finish parsing malformed output within its work limit"
            )
        remaining -= len(block)
        try:
            parsed = js_json_parse(block)
        except (ValueError, RecursionError):
            # Each rescan is paid for by the failed parse, keeping total work linear.
            pending.append(_json_blocks(block, True))
            continue
        if starts_line:
            yield parsed


def _extract_tool_names(output: Any) -> dict[str, None]:
    """extractToolNames; an insertion-ordered dict stands in for a JS Set."""
    names: dict[str, None] = {}

    if output is None or output is UNDEFINED:
        return names

    if isinstance(output, str):
        try:
            parsed = js_json_parse(output)
        except (ValueError, RecursionError):
            # Providers join serialized calls at line boundaries.
            for block in _extract_json_blocks(output):
                names.update(_extract_tool_names(block))
            return names
        return _extract_tool_names(parsed)

    if isinstance(output, dict):
        obj = output
        # OpenAI format: { tool_calls: [{ function: { name: "..." } }] }
        if isinstance(obj.get("tool_calls"), list):
            for tc in obj["tool_calls"]:
                if isinstance(tc, dict):
                    fn = tc.get("function")
                    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                        names[fn["name"]] = None
                    if isinstance(tc.get("name"), str):
                        names[tc["name"]] = None
            return names
        # Anthropic single tool_use block / OpenAI Responses single function_call item
        if obj.get("type") in ("tool_use", "function_call") and isinstance(obj.get("name"), str):
            names[obj["name"]] = None
            return names
        # Google/Vertex single functionCall
        fc = obj.get("functionCall")
        if "functionCall" in obj and isinstance(fc, dict | list):
            if isinstance(fc, dict) and isinstance(fc.get("name"), str):
                names[fc["name"]] = None
            return names
        # Google Live format: { toolCall: { functionCalls: [...] } }
        tool_call = obj.get("toolCall")
        if "toolCall" in obj and isinstance(tool_call, dict | list):
            calls = tool_call.get("functionCalls") if isinstance(tool_call, dict) else None
            if isinstance(calls, list):
                for c in calls:
                    if isinstance(c, dict) and isinstance(c.get("name"), str):
                        names[c["name"]] = None
            return names
        return names

    # Arrays (Anthropic content blocks, Google arrays, OpenAI arrays)
    if isinstance(output, list):
        for block in output:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and isinstance(block.get("name"), str):
                names[block["name"]] = None
                continue
            fc = block.get("functionCall")
            if "functionCall" in block and isinstance(fc, dict | list):
                if isinstance(fc, dict) and isinstance(fc.get("name"), str):
                    names[fc["name"]] = None
                continue
            fn = block.get("function")
            if isinstance(fn, dict | list):
                if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                    names[fn["name"]] = None
                continue
            if isinstance(block.get("name"), str):
                names[block["name"]] = None

    return names


def handle_tool_call_f1(params: AssertionParams) -> GradingResult:
    assertion, rendered_value, inverse = params.assertion, params.rendered_value, params.inverse
    if isinstance(rendered_value, list):
        expected_tools = [t for t in (js_trim(js_string(v)) for v in rendered_value) if t]
    elif isinstance(rendered_value, str):
        expected_tools = [t for t in (js_trim(v) for v in rendered_value.split(",")) if t]
    else:
        invariant(
            False,
            '"tool-call-f1" assertion requires a value: array of tool names or '
            "comma-separated string",
        )
        expected_tools = []

    if not expected_tools:
        invariant(False, '"tool-call-f1" assertion requires at least one expected tool name')

    expected = dict.fromkeys(expected_tools)
    try:
        actual = _extract_tool_names(params.output)
    except ToolCallParseError as error:
        return {"pass": False, "score": 0, "reason": str(error), "assertion": assertion}

    intersection = sum(1 for t in expected if t in actual)
    precision = intersection / len(actual) if actual else 0
    recall = intersection / len(expected) if expected else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    threshold = assertion.get("threshold")
    if threshold is None:
        threshold = 1.0
    passed = (f1 >= threshold) != inverse

    expected_list = ", ".join(sorted(expected))
    actual_list = ", ".join(sorted(actual)) or "(none)"

    if passed:
        reason = (
            f"Tool Call F1: {to_fixed(f1, 3)} (precision={to_fixed(precision, 3)}, "
            f"recall={to_fixed(recall, 3)}). Expected: [{expected_list}], Called: [{actual_list}]"
        )
    else:
        reason = (
            f"Tool Call F1 score {to_fixed(f1, 3)} is {'above' if inverse else 'below'} threshold "
            f"{js_string(threshold)}. Expected: [{expected_list}], Called: [{actual_list}]. "
            f"Precision={to_fixed(precision, 3)}, Recall={to_fixed(recall, 3)}"
        )
    return {"pass": passed, "score": f1, "reason": reason, "assertion": assertion}


# --- skill.ts ----------------------------------------------------------------------------------


def _matches_pattern(name: str, pattern: str) -> bool:
    from promptphoo.assertions.trajectory import matches_pattern

    return matches_pattern(name, pattern)


def _get_skill_calls(params: AssertionParams) -> list[dict[str, Any]]:
    metadata = (params.provider_response or {}).get("metadata")
    raw = metadata.get("skillCalls") if isinstance(metadata, dict) else None
    if not isinstance(raw, list):
        return []
    return [
        e
        for e in raw
        if isinstance(e, dict) and isinstance(e.get("name"), str) and e.get("is_error") is not True
    ]


def _matches_skill(skill_call: dict[str, Any], matcher: dict[str, Any]) -> bool:
    if matcher.get("name") and skill_call["name"] != matcher["name"]:
        return False
    return not (
        matcher.get("pattern") and not _matches_pattern(skill_call["name"], matcher["pattern"])
    )


def _format_skill_call(skill_call: dict[str, Any]) -> str:
    details = ", ".join(
        js_string(v) for v in (skill_call.get("source"), skill_call.get("path")) if is_truthy(v)
    )
    return f"{skill_call['name']} ({details})" if details else skill_call["name"]


def _is_non_negative_integer(count: Any) -> bool:
    return (
        isinstance(count, int | float)
        and not isinstance(count, bool)
        and math.isfinite(count)
        and float(count).is_integer()
        and count >= 0
    )


def _resolve_skill_matchers(value: Any) -> tuple[str, Any]:
    def normalize_text(text: Any) -> str | None:
        return js_trim(text) if isinstance(text, str) else None

    def validate_count(field: str, count: Any) -> None:
        if not _is_non_negative_integer(count):
            raise AssertionValueError(
                f"skill-used assertion object {field} must be a finite non-negative integer"
            )

    if isinstance(value, str) and js_trim(value):
        return "list", [{"name": js_trim(value)}]

    if (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, str) and js_trim(item) for item in value)
    ):
        return "list", [{"name": js_trim(item)} for item in value]

    if isinstance(value, dict):
        name = normalize_text(value.get("name"))
        pattern = normalize_text(value.get("pattern"))
        if not name and not pattern:
            raise AssertionValueError(
                "skill-used assertion object must include a name or pattern property"
            )
        if "min" in value:
            validate_count("min", value["min"])
        if "max" in value:
            validate_count("max", value["max"])
        mn, mx = value.get("min"), value.get("max")
        mn = mn if isinstance(mn, int | float) and not isinstance(mn, bool) else None
        mx = mx if isinstance(mx, int | float) and not isinstance(mx, bool) else None
        if mn is not None and mx is not None and mx < mn:
            raise AssertionValueError(
                "skill-used assertion object max must be greater than or equal to min"
            )
        return "count", {
            "max": mx,
            "min": mn,
            "name": name,
            "pattern": pattern,
        }

    raise AssertionValueError(
        "skill-used assertion must have a string, string array, or object value"
    )


def _summary(actual_skills: list[str]) -> str:
    return ", ".join(actual_skills) if actual_skills else "(none)"


def _handle_list_skill(
    params: AssertionParams,
    skill_calls: list[dict[str, Any]],
    actual_skills: list[str],
    matchers: list[dict[str, Any]],
) -> GradingResult:
    def used(m: dict[str, Any]) -> bool:
        return any(_matches_skill(sc, m) for sc in skill_calls)

    missing = [m for m in matchers if not used(m)]
    matched = [m for m in matchers if used(m)]
    passed = len(matched) == 0 if params.inverse else len(missing) == 0
    expected_skills = ", ".join(m["name"] for m in matchers)
    actual_summary = _summary(actual_skills)

    if params.inverse:
        reason = (
            f"Forbidden skill(s) were not used: {expected_skills}"
            if passed
            else f"Forbidden skill(s) were used: {', '.join(m['name'] for m in matched)}. "
            f"Actual skills: {actual_summary}"
        )
    elif passed:
        reason = f"Observed required skill(s): {expected_skills}. Actual skills: {actual_summary}"
    else:
        reason = (
            f"Missing required skill(s): {', '.join(m['name'] for m in missing)}. "
            f"Actual skills: {actual_summary}"
        )
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": reason,
        "assertion": params.assertion,
    }


def _handle_count_skill(
    params: AssertionParams,
    skill_calls: list[dict[str, Any]],
    actual_skills: list[str],
    matcher: dict[str, Any],
) -> GradingResult:
    has_min = matcher["min"] is not None
    has_max = matcher["max"] is not None
    mn = matcher["min"] if has_min else (0 if has_max else 1)
    mx = matcher["max"]
    matching = [sc for sc in skill_calls if _matches_skill(sc, matcher)]
    count = len(matching)
    label = matcher["pattern"] or matcher["name"] or "*"

    if params.inverse:
        if has_min or (has_max and mx != 0):
            raise AssertionValueError(
                "not-skill-used object assertions only support name/pattern with no count "
                "bounds, or max: 0"
            )
        passed = count == 0
        return {
            "pass": passed,
            "score": 1 if passed else 0,
            "reason": f'Forbidden skill "{label}" was not used. Actual skills: '
            f"{_summary(actual_skills)}"
            if passed
            else f'Forbidden skill "{label}" was used {count} time(s). Matches: '
            f"{', '.join(_format_skill_call(sc) for sc in matching)}",
            "assertion": params.assertion,
        }

    passed = count >= mn and (mx is None or count <= mx)
    reason = f'Matched skill "{label}" {count} time(s)'
    reason += (
        f" (expected at least {js_string(mn)})"
        if mx is None
        else f" (expected {js_string(mn)}-{js_string(mx)})"
    )
    if matching:
        reason += f". Matches: {', '.join(_format_skill_call(sc) for sc in matching)}"
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": reason,
        "assertion": params.assertion,
    }


def handle_skill_used(params: AssertionParams) -> GradingResult:
    skill_calls = _get_skill_calls(params)
    actual_skills = [_format_skill_call(sc) for sc in skill_calls]
    value = params.rendered_value
    if value is None:
        value = params.assertion.get("value")
    kind, expected = _resolve_skill_matchers(value)
    if kind == "list":
        return _handle_list_skill(params, skill_calls, actual_skills, expected)
    return _handle_count_skill(params, skill_calls, actual_skills, expected)


HANDLERS: dict[str, Handler] = {
    "is-valid-function-call": handle_is_valid_function_call,
    "is-valid-openai-function-call": handle_is_valid_function_call,
    "is-valid-openai-tools-call": handle_is_valid_openai_tools_call,
    "tool-call-f1": handle_tool_call_f1,
    "skill-used": handle_skill_used,
}
