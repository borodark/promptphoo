"""Trajectory assertions over trace spans.

Ports src/assertions/trajectory.ts, src/assertions/trajectoryUtils.ts, matchesPattern from
src/assertions/traceUtils.ts and the attribute keys from src/tracing/toolAttributes.ts.

Traces are dicts shaped like Node's TraceData ({traceId, spans: [{name, spanId, startTime,
endTime, attributes, statusCode, statusMessage}], metadata}), read from
`params.assertion_value_context["trace"]`.
"""

from __future__ import annotations

import functools
import math
import re
from typing import Any

from promptphoo.assertions.base import AssertionParams, AssertionValueError, GradingResult, Handler
from promptphoo.assertions.refusal import js_trim
from promptphoo.jsonutil import js_deep_equal, js_json_parse
from promptphoo.jsvalues import UNDEFINED, is_truthy, js_string, json_stringify

_NOT_LT = "[^\n\r\u2028\u2029]"  # JS `.` without the s flag


@functools.lru_cache(maxsize=512)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    parts = [f"{_NOT_LT}*" if c == "*" else _NOT_LT if c == "?" else re.escape(c) for c in pattern]
    return re.compile("^" + "".join(parts) + r"\Z", re.IGNORECASE)


def matches_pattern(span_name: str, pattern: str) -> bool:
    """traceUtils.matchesPattern: case-insensitive glob (* and ?) over the whole name."""
    return _glob_regex(js_string(pattern)).search(js_string(span_name)) is not None


# --- tracing/toolAttributes.ts -----------------------------------------------------------------

COMMAND_ATTRIBUTE_KEYS = ("codex.command", "command", "command.name", "command_name")
SEARCH_ATTRIBUTE_KEYS = ("codex.search.query", "search.query", "search_query")

_TOOL_ATTRIBUTE_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "prefix": "tool",
        "nameSuffixes": [".name", "_name"],
        "argSuffixes": [".args", ".arguments", ".input", "_args", "_arguments", "_input"],
        "extraNameKeys": ["tool"],
    },
    {
        "prefix": "function",
        "nameSuffixes": [".name", "_name"],
        "argSuffixes": [".args", ".arguments", ".input", "_args", "_arguments"],
    },
    {
        "prefix": "ai.toolCall",
        "nameSuffixes": [".name"],
        "argSuffixes": [".args", ".arguments", ".input"],
    },  # noqa: E501
    {
        "prefix": "gen_ai.tool",
        "nameSuffixes": [".name"],
        "argSuffixes": [".args", ".arguments", ".input"],
        "extraArgKeys": ["gen_ai.tool.call.args", "gen_ai.tool.call.arguments"],
    },
    {
        "prefix": "agent.tool",
        "nameSuffixes": ["_name"],
        "argSuffixes": [".args", ".arguments", ".input"],
        "extraNameKeys": ["agent.tool", "agent.toolName"],
    },
    {
        "prefix": "codex.mcp",
        "argSuffixes": [".args", ".arguments", ".input"],
        "extraNameKeys": ["codex.mcp.tool"],
    },
)

TOOL_NAME_ATTRIBUTE_KEYS: tuple[str, ...] = tuple(
    key
    for f in _TOOL_ATTRIBUTE_FAMILIES
    for key in [f["prefix"] + s for s in f.get("nameSuffixes", [])] + f.get("extraNameKeys", [])
)
TOOL_ARGUMENT_ATTRIBUTE_KEYS: tuple[str, ...] = (
    *(
        key
        for f in _TOOL_ATTRIBUTE_FAMILIES
        for key in [f["prefix"] + s for s in f.get("argSuffixes", [])] + f.get("extraArgKeys", [])
    ),
    "args",
    "arguments",
    "input",
)


def get_first_string_attribute(attributes: Any, keys: tuple[str, ...]) -> str | None:
    if not isinstance(attributes, dict):
        return None
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, str) and js_trim(value):
            return js_trim(value)
    return None


def get_tool_name_from_attributes(attributes: Any) -> str | None:
    return get_first_string_attribute(attributes, TOOL_NAME_ATTRIBUTE_KEYS)


# --- trajectoryUtils.ts ------------------------------------------------------------------------

_GENERIC_QUERY_ATTRIBUTE_KEYS = ("query",)
_DEFAULT_COMMAND_TOOL_NAMES = ("exec_command", "local_shell", "shell")
_AI = re.ASCII | re.IGNORECASE
_SEARCH_SPAN_NAME = re.compile(
    r"(^|[\s._:/-])(search|find|lookup|retriev(?:e|al))(\Z|[\s._:/-])", re.IGNORECASE
)
_SEARCH_KEY = re.compile(r"(^|[._])(search|lookup|retriev(?:e|al))(\Z|[._])", _AI)
_TOOL_NAME_KEY = re.compile(rf"tool{_NOT_LT}?name|function{_NOT_LT}?name", _AI)
_TOOL_KEY = re.compile(r"(^|[._])tool(\Z|[._])", _AI)
_TOOL_KEY_EXCLUDE = re.compile(r"result|output|definition", _AI)
_ARGS_KEY_EXCLUDE = re.compile(r"result|output|error|status", _AI)
_ARGS_KEY = re.compile(r"(^|[._])(arguments|args|input)(\Z|[._])", _AI)
_COMMAND_KEY = re.compile(r"command", _AI)
_COMMAND_KEY_EXCLUDE = re.compile(r"output|result", _AI)
_REASONING_NAME = re.compile(r"^reasoning([_\s]|\Z)", re.IGNORECASE)
_WS_SPLIT = re.compile(r"\s+")


def _resolve_command_tool_names(extra: Any) -> set[str]:
    merged: set[str] = set(_DEFAULT_COMMAND_TOOL_NAMES)
    for name in extra or []:
        if isinstance(name, str) and js_trim(name):
            merged.add(js_trim(name).lower())
    return merged


def _attributes(span: dict[str, Any]) -> dict[str, Any]:
    attributes = span.get("attributes")
    return attributes if isinstance(attributes, dict) else {}


def _span_name(span: dict[str, Any]) -> str:
    return js_string(span.get("name", ""))


def _normalize_structured_attribute(value: Any) -> Any:
    if value is None or value is UNDEFINED:
        return UNDEFINED
    if isinstance(value, str):
        trimmed = js_trim(value)
        if not trimmed:
            return UNDEFINED
        try:
            return js_json_parse(trimmed)
        except (ValueError, RecursionError):
            return trimmed
    if isinstance(value, int | float | bool | dict | list):
        return value
    return UNDEFINED


def _is_search_like_span(span: dict[str, Any]) -> bool:
    name = _span_name(span)
    if _SEARCH_SPAN_NAME.search(name) or name.startswith("search "):
        return True
    return any(key != "query" and _SEARCH_KEY.search(key) for key in _attributes(span))


def _get_trajectory_step_status(step: dict[str, Any]) -> dict[str, Any] | None:
    code = step.get("statusCode")
    if code is None or code == 0:
        return None
    status: dict[str, Any] = {"code": code}
    if step.get("statusMessage"):
        status["message"] = step["statusMessage"]
    return status


def _get_command_executable(command: str) -> str | None:
    executable = _WS_SPLIT.split(js_trim(command))[0]
    return executable or None


def _get_trace_command_tool_names(trace: dict[str, Any]) -> set[str]:
    metadata = trace.get("metadata")
    names = metadata.get("commandToolNames") if isinstance(metadata, dict) else None
    configured = [n for n in names if isinstance(n, str)] if isinstance(names, list) else None
    return _resolve_command_tool_names(configured)


def _is_command_tool_name(tool_name: str | None, command_tool_names: set[str]) -> bool:
    return bool(tool_name) and js_trim(tool_name or "").lower() in command_tool_names


def _extract_tool_name(span: dict[str, Any]) -> str | None:
    attributes = _attributes(span)
    direct = get_tool_name_from_attributes(attributes)
    if direct:
        return direct

    for key, value in attributes.items():
        if not isinstance(value, str) or not js_trim(value):
            continue
        trimmed = js_trim(value)
        # A tool name is always a scalar identifier; skip JSON-serialized tool definitions.
        if trimmed.startswith(("[", "{")):
            try:
                parsed = js_json_parse(trimmed)
            except (ValueError, RecursionError):
                parsed = UNDEFINED
            if isinstance(parsed, dict | list):
                continue
        if _TOOL_NAME_KEY.search(key):
            return trimmed
        if _TOOL_KEY.search(key) and not _TOOL_KEY_EXCLUDE.search(key):
            return trimmed

    name = _span_name(span)
    if name.startswith("mcp "):
        slash = name.rfind("/")
        if slash != -1 and slash < len(name) - 1:
            return js_trim(name[slash + 1 :])
    return None


def _extract_tool_args(span: dict[str, Any]) -> Any:
    attributes = _attributes(span)
    for key in TOOL_ARGUMENT_ATTRIBUTE_KEYS:
        value = _normalize_structured_attribute(attributes.get(key, UNDEFINED))
        if value is not UNDEFINED:
            return value
    for key, raw in attributes.items():
        if _ARGS_KEY_EXCLUDE.search(key) or not _ARGS_KEY.search(key):
            continue
        value = _normalize_structured_attribute(raw)
        if value is not UNDEFINED:
            return value
    return UNDEFINED


def _extract_command(
    span: dict[str, Any], tool_name: str | None, get_tool_args: Any, command_tool_names: set[str]
) -> str | None:
    attributes = _attributes(span)
    direct = get_first_string_attribute(attributes, COMMAND_ATTRIBUTE_KEYS)
    if direct:
        return direct

    for key, value in attributes.items():
        if not isinstance(value, str) or not js_trim(value):
            continue
        if _COMMAND_KEY.search(key) and not _COMMAND_KEY_EXCLUDE.search(key):
            return js_trim(value)

    tool_args = get_tool_args()
    if _is_command_tool_name(tool_name, command_tool_names) and isinstance(tool_args, dict | list):
        args = tool_args if isinstance(tool_args, dict) else {}
        source: tuple[Any, str] | None = None
        for key, separator in (("cmd", " "), ("command", " "), ("commands", "; ")):
            if args.get(key, UNDEFINED) is not UNDEFINED:
                source = (args[key], separator)
                break
        command = source[0] if source else UNDEFINED
        if isinstance(command, str) and js_trim(command):
            return js_trim(command)
        if isinstance(command, list) and source:
            joined = source[1].join(p for p in (js_trim(js_string(x)) for x in command) if p)
            if joined:
                return joined

    name = _span_name(span)
    if name.startswith("exec "):
        return js_trim(name[len("exec ") :])
    return None


def _extract_search_query(span: dict[str, Any]) -> str | None:
    attributes = _attributes(span)
    direct = get_first_string_attribute(attributes, SEARCH_ATTRIBUTE_KEYS)
    if direct:
        return direct
    generic = get_first_string_attribute(attributes, _GENERIC_QUERY_ATTRIBUTE_KEYS)
    if generic and _is_search_like_span(span):
        return generic
    name = _span_name(span)
    if name.startswith("search "):
        return js_trim(re.sub(r'^"|"\Z', "", name[len("search ") :]))
    return None


def _is_reasoning_span(span: dict[str, Any]) -> bool:
    if _attributes(span).get("codex.item.type") == "reasoning":
        return True
    name = _span_name(span)
    return bool(_REASONING_NAME.search(name)) or name == "reasoning"


def _is_message_span(span: dict[str, Any]) -> bool:
    if _attributes(span).get("codex.item.type") == "agent_message":
        return True
    return _span_name(span) in ("agent response", "send input")


def _num(value: Any) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    return math.nan


def _compare_spans(left: tuple[int, dict[str, Any]], right: tuple[int, dict[str, Any]]) -> int:
    li, ls = left
    ri, rs = right

    def sign(diff: float) -> int:
        # V8's sort only tests `< 0`/`> 0`, so a NaN comparison behaves like "equal".
        return 0 if math.isnan(diff) else (-1 if diff < 0 else 1 if diff > 0 else 0)

    time_diff = _num(ls.get("startTime")) - _num(rs.get("startTime"))
    if time_diff != 0:
        return sign(time_diff)
    l_end = ls.get("endTime") if ls.get("endTime") is not None else ls.get("startTime")
    r_end = rs.get("endTime") if rs.get("endTime") is not None else rs.get("startTime")
    end_diff = _num(l_end) - _num(r_end)
    if end_diff != 0:
        return sign(end_diff)
    return li - ri


def extract_trajectory_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    command_tool_names = _get_trace_command_tool_names(trace)
    spans = [s for s in (trace.get("spans") or []) if isinstance(s, dict)]
    ordered = sorted(enumerate(spans), key=functools.cmp_to_key(_compare_spans))

    steps: list[dict[str, Any]] = []
    for _, span in ordered:
        tool_name = _extract_tool_name(span)
        cache: list[Any] = []

        def get_tool_args(span: dict[str, Any] = span, cache: list[Any] = cache) -> Any:
            if not cache:
                cache.append(_extract_tool_args(span))
            return cache[0]

        command = _extract_command(span, tool_name, get_tool_args, command_tool_names)
        search_query = _extract_search_query(span)

        span_name = _span_name(span)
        step_type = "span"
        name = span_name
        aliases: dict[str, None] = {span_name: None}
        args: Any = UNDEFINED

        if command and _is_command_tool_name(tool_name, command_tool_names):
            step_type, name = "command", command
            aliases[command] = None
            args = get_tool_args()
            if tool_name:
                aliases[tool_name] = None
            executable = _get_command_executable(command)
            if executable:
                aliases[executable] = None
        elif tool_name:
            step_type, name = "tool", tool_name
            aliases[tool_name] = None
            args = get_tool_args()
        elif command:
            step_type, name = "command", command
            aliases[command] = None
            executable = _get_command_executable(command)
            if executable:
                aliases[executable] = None
        elif search_query:
            step_type, name = "search", search_query
            aliases[search_query] = None
        elif _is_reasoning_span(span):
            step_type = "reasoning"
            aliases["reasoning"] = None
        elif _is_message_span(span):
            step_type = "message"
            aliases["message"] = None

        step: dict[str, Any] = {"aliases": list(aliases)}
        if args is not UNDEFINED:
            step["args"] = args
        step.update(
            attributes=_attributes(span),
            endTime=span.get("endTime"),
            name=name,
            spanId=span.get("spanId"),
            spanName=span_name,
            startTime=span.get("startTime"),
            statusCode=span.get("statusCode"),
            statusMessage=span.get("statusMessage"),
            type=step_type,
        )
        steps.append(step)
    return steps


def normalize_trajectory_matcher(matcher: Any, default_type: str | None = None) -> dict[str, Any]:
    if isinstance(matcher, str):
        out: dict[str, Any] = {"pattern": matcher}
        if default_type:
            out["type"] = default_type
        return out
    out = dict(matcher) if isinstance(matcher, dict) else {}
    if not is_truthy(out.get("type")) and default_type:
        out["type"] = default_type
    return out


def matches_trajectory_step(
    step: dict[str, Any], matcher: Any, default_type: str | None = None
) -> bool:
    m = normalize_trajectory_matcher(matcher, default_type)
    step_type = m.get("type")
    if is_truthy(step_type):
        allowed = step_type if isinstance(step_type, list) else [step_type]
        if step["type"] not in allowed:
            return False
    match_pattern: Any = m.get("pattern") or m.get("name")
    if not is_truthy(match_pattern):
        return True
    return any(matches_pattern(alias, match_pattern) for alias in step["aliases"])


def format_trajectory_step(step: dict[str, Any]) -> str:
    return f"{step['type']}:{step['name']}"


def format_trajectory_args(args: Any) -> str:
    if args is UNDEFINED:
        return "(none)"
    try:
        return json_stringify(args)
    except (TypeError, ValueError, RecursionError):
        return js_string(args)


MAX_JUDGE_SUMMARY_STEPS = 24
JUDGE_SUMMARY_HEAD_STEPS = 12
JUDGE_SUMMARY_TAIL_STEPS = 12


def _has_same_status(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    return (left or {}).get("code") == (right or {}).get("code") and (left or {}).get(
        "message"
    ) == (right or {}).get("message")


def _compact_judge_trajectory_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for step in steps:
        previous = compacted[-1] if compacted else None
        if (
            previous
            and previous["type"] == step["type"]
            and previous["name"] == step["name"]
            and previous.get("spanName") == step.get("spanName")
            and _has_same_status(previous.get("status"), step.get("status"))
        ):
            previous["collapsedCount"] = (previous.get("collapsedCount") or 1) + 1
            continue
        compacted.append(step)
    return compacted


def _truncate_judge_trajectory_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(steps) <= MAX_JUDGE_SUMMARY_STEPS:
        return steps
    return [
        *steps[:JUDGE_SUMMARY_HEAD_STEPS],
        {"omittedCount": len(steps) - MAX_JUDGE_SUMMARY_STEPS},
        *steps[-JUDGE_SUMMARY_TAIL_STEPS:],
    ]


def summarize_trajectory_for_judge(trace: dict[str, Any]) -> str:
    raw_steps: list[dict[str, Any]] = []
    for index, step in enumerate(extract_trajectory_steps(trace)):
        status = _get_trajectory_step_status(step)
        entry: dict[str, Any] = {"index": index + 1, "type": step["type"], "name": step["name"]}
        if step["spanName"] != step["name"]:
            entry["spanName"] = step["spanName"]
        if status:
            entry["status"] = status
        raw_steps.append(entry)
    compacted = _compact_judge_trajectory_steps(raw_steps)
    summary: dict[str, Any] = {}
    if "traceId" in trace:
        summary["traceId"] = trace["traceId"]
    summary.update(
        stepCount=len(raw_steps),
        compactedStepCount=len(compacted),
        steps=_truncate_judge_trajectory_steps(compacted),
    )
    return json_stringify(summary, 2)


# --- trajectory.ts -----------------------------------------------------------------------------


def _fail(message: str) -> AssertionValueError:
    return AssertionValueError(message)


def _get_trace_or_throw(params: AssertionParams) -> dict[str, Any]:
    trace = (params.assertion_value_context or {}).get("trace")
    if not isinstance(trace, dict) or not is_truthy(trace.get("spans")):
        raise _fail(f"No trace data available for {params.base_type} assertion")
    return trace


def _value(params: AssertionParams) -> Any:
    return (
        params.rendered_value
        if params.rendered_value is not None
        else params.assertion.get("value")
    )


def _format_step_list(labels: list[str]) -> str:
    return ", ".join(labels) if labels else "(none)"


def _label(matcher: dict[str, Any] | None) -> str:
    if not matcher:
        return "*"
    return js_string(matcher.get("pattern") or matcher.get("name") or "*")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _require_named_matcher(
    matcher: dict[str, Any], assertion_type: str, index: int | None = None
) -> None:
    if is_truthy(matcher.get("pattern")) or is_truthy(matcher.get("name")):
        return
    step_label = "object" if index is None else f"step {index + 1}"
    raise _fail(f"{assertion_type} assertion {step_label} must include a name or pattern property")


def _tool_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in extract_trajectory_steps(trace) if s["type"] == "tool"]


def _result(params: AssertionParams, passed: bool, reason: str) -> GradingResult:
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": reason,
        "assertion": params.assertion,
    }


def _resolve_tool_matchers(value: Any) -> tuple[str, Any]:
    if isinstance(value, str):
        return "list", [normalize_trajectory_matcher(value, "tool")]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "list", [normalize_trajectory_matcher(item, "tool") for item in value]
    if isinstance(value, dict):
        matcher = normalize_trajectory_matcher(value, "tool")
        matcher["max"] = value.get("max") if _is_number(value.get("max")) else None
        matcher["min"] = value.get("min") if _is_number(value.get("min")) else None
        return "count", matcher
    raise _fail("trajectory:tool-used assertion must have a string, string array, or object value")


def handle_trajectory_tool_used(params: AssertionParams) -> GradingResult:
    trace = _get_trace_or_throw(params)
    steps = _tool_steps(trace)
    kind, expected = _resolve_tool_matchers(_value(params))

    if kind == "list":
        if not expected:
            raise _fail("trajectory:tool-used assertion requires at least one expected tool")
        missing = [m for m in expected if not any(matches_trajectory_step(s, m) for s in steps)]
        matched = [m for m in expected if any(matches_trajectory_step(s, m) for s in steps)]
        passed = len(matched) == 0 if params.inverse else len(missing) == 0
        actual_tools = _format_step_list([format_trajectory_step(s) for s in steps])
        expected_tools = ", ".join(_label(m) for m in expected)

        if params.inverse:
            reason = (
                f"Forbidden tool(s) were not used: {expected_tools}"
                if passed
                else f"Forbidden tool(s) were used: {', '.join(_label(m) for m in matched)}. "
                f"Actual tools: {actual_tools}"
            )
        elif passed:
            reason = f"Observed required tool(s): {expected_tools}. Actual tools: {actual_tools}"
        else:
            reason = (
                f"Missing required tool(s): {', '.join(_label(m) for m in missing)}. "
                f"Actual tools: {actual_tools}"
            )
        return _result(params, passed, reason)

    matcher = expected
    mn = matcher["min"] if matcher["min"] is not None else 1
    mx = matcher["max"]
    if not is_truthy(matcher.get("pattern")) and not is_truthy(matcher.get("name")):
        raise _fail("trajectory:tool-used assertion object must include a name or pattern property")
    matching = [s for s in steps if matches_trajectory_step(s, matcher)]
    count = len(matching)
    base_pass = count >= mn and (mx is None or count <= mx)
    passed = (not base_pass) if params.inverse else base_pass
    label = _label(matcher)

    reason = f'Matched tool "{label}" {count} time(s)'
    if mx is None:
        reason += f" (expected at least {js_string(mn)})"
    else:
        reason += f" (expected {js_string(mn)}-{js_string(mx)})"
    if matching:
        reason += f". Matches: {', '.join(format_trajectory_step(s) for s in matching)}"
    if params.inverse:
        reason = (
            f'Tool "{label}" matched {count} time(s), which violates the inverse assertion'
            if base_pass
            else f'Tool "{label}" did not satisfy the forbidden match condition'
        )
    return _result(params, passed, reason)


def _resolve_sequence_value(value: Any) -> tuple[str, list[Any]]:
    if isinstance(value, list):
        return "in_order", value
    if isinstance(value, dict):
        return (value.get("mode") or "in_order"), (value.get("steps") or [])
    raise _fail("trajectory:tool-sequence assertion must have an array or object value")


def handle_trajectory_tool_sequence(params: AssertionParams) -> GradingResult:
    trace = _get_trace_or_throw(params)
    tool_steps = _tool_steps(trace)
    mode, raw_steps = _resolve_sequence_value(_value(params))
    expected: list[dict[str, Any]] = []
    for index, step in enumerate(raw_steps if isinstance(raw_steps, list) else []):
        matcher = normalize_trajectory_matcher(step, "tool")
        _require_named_matcher(matcher, "trajectory:tool-sequence", index)
        expected.append(matcher)
    if not expected:
        raise _fail("trajectory:tool-sequence assertion requires at least one expected step")

    actual_tools = _format_step_list([format_trajectory_step(s) for s in tool_steps])
    if mode == "exact":
        base_pass = len(tool_steps) == len(expected) and all(
            matches_trajectory_step(tool_steps[i], m) for i, m in enumerate(expected)
        )
        reason = (
            f"Observed exact tool sequence: {actual_tools}"
            if base_pass
            else f"Expected exact tool sequence of {', '.join(_label(m) for m in expected)}, "
            f"but actual tools were {actual_tools}"
        )
    else:
        expected_index = 0
        matched_steps: list[str] = []
        for step in tool_steps:
            if expected_index >= len(expected):
                break
            if matches_trajectory_step(step, expected[expected_index]):
                matched_steps.append(format_trajectory_step(step))
                expected_index += 1
        base_pass = expected_index == len(expected)
        if base_pass:
            reason = (
                f"Observed tool sequence in order: {', '.join(matched_steps)}. "
                f"Actual tools: {actual_tools}"
            )
        else:
            next_expected = _label(
                expected[expected_index] if expected_index < len(expected) else None
            )
            reason = (
                f'Expected tool "{next_expected}" was not observed in order. '
                f"Actual tools: {actual_tools}"
            )

    passed = (not base_pass) if params.inverse else base_pass
    if params.inverse:
        reason = (
            f"Forbidden tool sequence was observed. Actual tools: {actual_tools}"
            if base_pass
            else "Forbidden tool sequence was not observed"
        )
    return _result(params, passed, reason)


def _matches_expected_args_partial(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_matches_expected_args_partial(actual[i], e) for i, e in enumerate(expected))
        )
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _matches_expected_args_partial(actual[key], value)
            for key, value in expected.items()
        )
    return js_deep_equal(actual, expected)


def _strip_defaults(actual: Any, defaults: dict[str, Any] | None) -> tuple[Any, list[str]]:
    if defaults is None or not isinstance(actual, dict):
        return actual, []
    cleaned: dict[str, Any] = {}
    stripped: list[str] = []
    for key, value in actual.items():
        if key in defaults and js_deep_equal(value, defaults[key]):
            stripped.append(key)
            continue
        cleaned[key] = value
    return cleaned, stripped


def _matches_ignore_key(entry: str, key: str) -> bool:
    if entry == key:
        return True
    if "*" in entry or "?" in entry:
        return matches_pattern(key, entry)
    return False


def _strip_ignored_args(value: Any, ignore: list[str]) -> tuple[Any, list[str]]:
    if not ignore or not isinstance(value, dict):
        return value, []
    cleaned: dict[str, Any] = {}
    stripped: list[str] = []
    for key, entry_value in value.items():
        if any(_matches_ignore_key(entry, key) for entry in ignore):
            stripped.append(key)
            continue
        cleaned[key] = entry_value
    return cleaned, stripped


def _matches_tool_args(
    actual: Any, expected: Any, mode: str, defaults: dict[str, Any] | None, ignore: list[str]
) -> bool:
    # `ignore` applies to both payloads; `defaults` only to the observed one.
    cleaned_actual = _strip_defaults(_strip_ignored_args(actual, ignore)[0], defaults)[0]
    cleaned_expected = _strip_ignored_args(expected, ignore)[0]
    if mode == "exact":
        return js_deep_equal(cleaned_actual, cleaned_expected)
    return _matches_expected_args_partial(cleaned_actual, cleaned_expected)


def _resolve_tool_args_match_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail("trajectory:tool-args-match assertion must have an object value")
    matcher = normalize_trajectory_matcher(value, "tool")
    _require_named_matcher(matcher, "trajectory:tool-args-match")

    expected_args = value["args"] if "args" in value else value.get("arguments", UNDEFINED)
    if expected_args is UNDEFINED:
        raise _fail(
            "trajectory:tool-args-match assertion must include an args or arguments property"
        )

    mode = value.get("mode", UNDEFINED)
    if mode is UNDEFINED:
        mode = "partial"
    elif mode not in ("partial", "exact"):
        raise _fail('trajectory:tool-args-match assertion mode must be "partial" or "exact"')

    defaults = value.get("defaults", UNDEFINED)
    if defaults is UNDEFINED:
        defaults = None
    elif not isinstance(defaults, dict):
        raise _fail(
            "trajectory:tool-args-match assertion defaults must be an object mapping argument "
            "names to default values"
        )

    raw_ignore = value.get("ignore", UNDEFINED)
    ignore: list[str] = []
    if raw_ignore is not UNDEFINED:
        entries = raw_ignore if isinstance(raw_ignore, list) else [raw_ignore]
        for entry in entries:
            if not isinstance(entry, str) or len(js_trim(entry)) == 0:
                raise _fail(
                    "trajectory:tool-args-match assertion ignore must be a non-empty string or "
                    "an array of non-empty strings"
                )
        ignore = entries

    return {
        "matcher": matcher,
        "expectedArgs": expected_args,
        "mode": mode,
        "defaults": defaults,
        "ignore": ignore,
    }


def handle_trajectory_tool_args_match(params: AssertionParams) -> GradingResult:
    trace = _get_trace_or_throw(params)
    tool_steps = _tool_steps(trace)
    v = _resolve_tool_args_match_value(_value(params))
    matcher, expected_args, mode = v["matcher"], v["expectedArgs"], v["mode"]
    defaults, ignore = v["defaults"], v["ignore"]
    label = _label(matcher)
    actual_tools = _format_step_list([format_trajectory_step(s) for s in tool_steps])
    matching = [s for s in tool_steps if matches_trajectory_step(s, matcher)]
    with_args = [s for s in matching if "args" in s]
    matched = next(
        (
            s
            for s in with_args
            if _matches_tool_args(s["args"], expected_args, mode, defaults, ignore)
        ),
        None,
    )
    base_pass = matched is not None
    passed = (not base_pass) if params.inverse else base_pass
    ignored_args = _strip_ignored_args(matched["args"], ignore)[1] if matched else []
    ignored_defaults = (
        _strip_defaults(_strip_ignored_args(matched["args"], ignore)[0], defaults)[1]
        if matched
        else []
    )
    ignored_args_suffix = (
        f". Ignored argument(s): {', '.join(ignored_args)}" if ignored_args else ""
    )
    ignored_defaults_suffix = (
        f". Ignored default argument(s): {', '.join(ignored_defaults)}" if ignored_defaults else ""
    )
    expected_label = format_trajectory_args(expected_args)
    observed_label = (
        ", ".join(format_trajectory_args(s["args"]) for s in with_args) if with_args else "(none)"
    )

    if params.inverse:
        if matched is not None:
            reason = (
                f'Forbidden argument match for tool "{label}" was observed on '
                f"{format_trajectory_step(matched)}. "
                f"Args: {format_trajectory_args(matched['args'])}"
            )
        elif not matching:
            reason = (
                f'Forbidden argument match for tool "{label}" was not observed because no tool '
                "call matched it"
            )
        else:
            reason = (
                f'Forbidden argument match for tool "{label}" was not observed. '
                f"Observed args: {observed_label}"
            )
    elif matched is not None:
        reason = (
            f'Tool "{label}" matched expected arguments ({mode}) on '
            f"{format_trajectory_step(matched)}. Args: {format_trajectory_args(matched['args'])}"
            f"{ignored_args_suffix}{ignored_defaults_suffix}"
        )
    elif not matching:
        reason = f'No tool call matched "{label}". Actual tools: {actual_tools}'
    elif not with_args:
        reason = (
            f'Tool "{label}" was observed but no arguments were captured. '
            f"Actual tools: {actual_tools}"
        )
    else:
        reason = (
            f'No call to tool "{label}" matched expected arguments ({mode}): {expected_label}. '
            f"Observed args: {observed_label}"
        )
    return _result(params, passed, reason)


def handle_trajectory_step_count(params: AssertionParams) -> GradingResult:
    trace = _get_trace_or_throw(params)
    steps = extract_trajectory_steps(trace)
    value = _value(params)
    if not isinstance(value, dict):
        raise _fail("trajectory:step-count assertion must have an object value")
    matcher = normalize_trajectory_matcher(value)
    mn = value.get("min") if _is_number(value.get("min")) else None
    mx = value.get("max") if _is_number(value.get("max")) else None
    matcher["min"], matcher["max"] = mn, mx
    if mn is None and mx is None:
        raise _fail("trajectory:step-count assertion must include a min or max property")

    matching = [s for s in steps if matches_trajectory_step(s, matcher)]
    count = len(matching)
    base_pass = (mn is None or count >= mn) and (mx is None or count <= mx)
    passed = (not base_pass) if params.inverse else base_pass

    filter_parts: list[str] = []
    if is_truthy(matcher.get("type")):
        types = matcher["type"] if isinstance(matcher["type"], list) else [matcher["type"]]
        filter_parts.append(f"type={'|'.join(js_string(t) for t in types)}")
    pattern = matcher.get("pattern") or matcher.get("name")
    if is_truthy(pattern):
        filter_parts.append(f"pattern={js_string(pattern)}")

    reason = f"Matched {count} trajectory step(s)"
    if filter_parts:
        reason += f" for {', '.join(filter_parts)}"
    if mn is not None and mx is not None:
        reason += f" (expected {js_string(mn)}-{js_string(mx)})"
    elif mn is not None:
        reason += f" (expected at least {js_string(mn)})"
    elif mx is not None:
        reason += f" (expected at most {js_string(mx)})"
    if matching:
        reason += f". Matches: {', '.join(format_trajectory_step(s) for s in matching)}"
    if params.inverse:
        reason = (
            "Trajectory step count satisfied the forbidden range"
            if base_pass
            else "Trajectory step count did not satisfy the forbidden range"
        )
    return _result(params, passed, reason)


def _resolve_goal_success_value(value: Any) -> dict[str, str]:
    if isinstance(value, str) and js_trim(value):
        return {"goal": js_trim(value)}
    if isinstance(value, dict) and isinstance(value.get("goal"), str) and js_trim(value["goal"]):
        return {"goal": js_trim(value["goal"])}
    raise _fail(
        "trajectory:goal-success assertion must have a string value or an object with a goal "
        "property"
    )


async def handle_trajectory_goal_success(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.llm_grading import is_grader_failure, matches_trajectory_goal_success

    trace = _get_trace_or_throw(params)
    goal = _resolve_goal_success_value(_value(params))["goal"]
    trajectory = summarize_trajectory_for_judge(trace)
    result = await matches_trajectory_goal_success(
        goal,
        trajectory,
        params.output_string,
        (params.test or {}).get("options"),
        (params.assertion_value_context or {}).get("vars"),
        params.assertion,
        params.provider_call_context,
    )
    if not params.inverse:
        return result
    if is_grader_failure(result):
        return {**result, "assertion": params.assertion}
    return {
        **result,
        "assertion": params.assertion,
        "pass": not result["pass"],
        "score": 0 if result["pass"] else 1,
        "reason": f"Agent unexpectedly achieved the goal: {goal}"
        if result["pass"]
        else f"Agent did not achieve the forbidden goal: {goal}",
    }


HANDLERS: dict[str, Handler] = {
    "trajectory:goal-success": handle_trajectory_goal_success,
    "trajectory:tool-args-match": handle_trajectory_tool_args_match,
    "trajectory:step-count": handle_trajectory_step_count,
    "trajectory:tool-sequence": handle_trajectory_tool_sequence,
    "trajectory:tool-used": handle_trajectory_tool_used,
}
