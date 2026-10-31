"""Trace assertions over the spans of the test case's trace.

Ports src/assertions/traceUtils.ts, src/assertions/traceSpanCount.ts,
src/assertions/traceSpanDuration.ts and src/assertions/traceErrorSpans.ts.

The trace is `params.assertion_value_context["trace"]` ({traceId, evaluationId, testCaseId,
metadata, spans}), which `run_assertion` loads from the trace store. As upstream, the handlers
read the raw `assertion.value` and ignore a `not-` prefix.
"""

from __future__ import annotations

import functools
import math
import re
from typing import Any

from promptphoo.assertions.base import AssertionParams, AssertionValueError, GradingResult, Handler
from promptphoo.jsvalues import is_truthy, js_string, json_stringify, to_fixed

_NOT_LT = "[^\n\r\u2028\u2029]"  # JS `.` without the s flag
_ERROR_STATUS_MESSAGE = re.compile(r"error|failed|failure|exception|timeout|abort", re.IGNORECASE)


# --- traceUtils.ts ----------------------------------------------------------------------------


@functools.lru_cache(maxsize=512)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    parts = [f"{_NOT_LT}*" if c == "*" else _NOT_LT if c == "?" else re.escape(c) for c in pattern]
    return re.compile("^" + "".join(parts) + r"\Z", re.IGNORECASE)


def matches_pattern(span_name: str, pattern: str) -> bool:
    """Match a span name against a glob-like pattern: `*` is any characters, `?` is one
    character, case-insensitive over the whole name."""
    return _glob_regex(js_string(pattern)).search(js_string(span_name)) is not None


def with_trace_attribute_filter_context(reason: str, attributes: Any = None) -> str:
    keys = list(attributes) if isinstance(attributes, dict) else []
    if keys:
        return f"{reason}. Attribute filters applied to keys: {json_stringify(keys)}."
    return reason


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _strict_equal(a: Any, b: Any) -> bool:
    """JS `===` for attribute values (strings, booleans, numbers)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        return a == b
    return type(a) is type(b) and a == b


def filter_trace_spans(
    spans: list[dict[str, Any]], pattern: str, attributes: Any = None
) -> list[dict[str, Any]]:
    if attributes is not None:
        if not isinstance(attributes, dict) or any(
            not isinstance(v, str | bool) and not (_is_number(v) and math.isfinite(v))
            for v in attributes.values()
        ):
            raise AssertionValueError(
                "Trace assertion attributes must be an object with string, boolean, or finite "
                "number values"
            )
    entries = list((attributes or {}).items())
    out = []
    for span in spans:
        if not matches_pattern(span.get("name", ""), pattern):
            continue
        span_attributes = span.get("attributes")
        if all(
            isinstance(span_attributes, dict)
            and key in span_attributes
            and _strict_equal(span_attributes[key], value)
            for key, value in entries
        ):
            out.append(span)
    return out


def _require_spans(params: AssertionParams) -> list[dict[str, Any]]:
    trace = (params.assertion_value_context or {}).get("trace")
    if not is_truthy(trace) or not isinstance(trace, dict) or not is_truthy(trace.get("spans")):
        raise AssertionValueError(f"No trace data available for {params.base_type} assertion")
    return trace["spans"]


def _num(value: Any) -> str:
    return js_string(value)


# --- traceSpanCount.ts ------------------------------------------------------------------------


def handle_trace_span_count(params: AssertionParams) -> GradingResult:
    assertion = params.assertion
    spans = _require_spans(params)
    value = assertion.get("value")
    if not isinstance(value, dict) or not is_truthy(value.get("pattern")):
        raise AssertionValueError(
            "trace-span-count assertion must have a value object with pattern property"
        )
    pattern = value["pattern"]
    min_count = value.get("min")
    max_count = value.get("max")
    attributes = value.get("attributes")

    matching = filter_trace_spans(spans, pattern, attributes)
    count = len(matching)
    passed = True
    if min_count is not None and count < min_count:
        passed = False
        reason = (
            f'Found {count} spans matching pattern "{js_string(pattern)}", expected at least '
            f"{_num(min_count)}"
        )
    elif max_count is not None and count > max_count:
        passed = False
        reason = (
            f'Found {count} spans matching pattern "{js_string(pattern)}", expected at most '
            f"{_num(max_count)}"
        )
    else:
        reason = f'Found {count} spans matching pattern "{js_string(pattern)}"'
        if min_count is not None and max_count is not None:
            reason += f" (expected {_num(min_count)}-{_num(max_count)})"
        elif min_count is not None:
            reason += f" (expected at least {_num(min_count)})"
        elif max_count is not None:
            reason += f" (expected at most {_num(max_count)})"
    if matching:
        names = list(dict.fromkeys(js_string(s.get("name")) for s in matching))
        reason += f". Matched spans: {', '.join(names)}"
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": with_trace_attribute_filter_context(reason, attributes),
        "assertion": assertion,
    }


# --- traceSpanDuration.ts ---------------------------------------------------------------------


def _calculate_percentile(durations: list[float], percentile: float) -> float:
    if not durations:
        return 0
    ordered = sorted(durations)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    return ordered[max(0, index)]


def handle_trace_span_duration(params: AssertionParams) -> GradingResult:
    assertion = params.assertion
    spans = _require_spans(params)
    value = assertion.get("value")
    if not isinstance(value, dict) or not _is_number(value.get("max")):
        raise AssertionValueError(
            "trace-span-duration assertion must have a value object with max property"
        )
    pattern = value.get("pattern")
    if pattern is None:
        pattern = "*"
    max_ms = value["max"]
    percentile = value.get("percentile")
    attributes = value.get("attributes")
    if percentile is not None and (
        not _is_number(percentile)
        or not math.isfinite(percentile)
        or percentile < 0
        or percentile > 100
    ):
        raise AssertionValueError(
            "trace-span-duration assertion percentile must be a number between 0 and 100"
        )

    matching = [
        s
        for s in filter_trace_spans(spans, pattern, attributes)
        if s.get("startTime") is not None and s.get("endTime") is not None
    ]
    if not matching:
        return {
            "pass": True,
            "score": 1,
            "reason": with_trace_attribute_filter_context(
                f'No spans found matching pattern "{js_string(pattern)}" with complete timing data',
                attributes,
            ),
            "assertion": assertion,
        }
    span_durations = [
        {"name": js_string(s.get("name")), "duration": s["endTime"] - s["startTime"]}
        for s in matching
    ]

    def describe(entries: list[dict[str, Any]]) -> str:
        return ", ".join(f"{s['name']} ({_num(s['duration'])}ms)" for s in entries)

    passed = True
    if percentile is None:
        slow = [s for s in span_durations if s["duration"] > max_ms]
        if slow:
            passed = False
            top3 = sorted(slow, key=lambda s: s["duration"], reverse=True)[:3]
            reason = f"{len(slow)} span(s) exceed duration threshold {_num(max_ms)}ms. "
            reason += f"Slowest: {describe(top3)}"
        else:
            longest = max(s["duration"] for s in span_durations)
            reason = (
                f'All {len(matching)} spans matching pattern "{js_string(pattern)}" completed '
                f"within {_num(max_ms)}ms (max: {_num(longest)}ms)"
            )
    else:
        percentile_value = _calculate_percentile(
            [s["duration"] for s in span_durations], percentile
        )
        if percentile_value > max_ms:
            passed = False
            slowest = sorted(
                (s for s in span_durations if s["duration"] >= percentile_value),
                key=lambda s: s["duration"],
                reverse=True,
            )[:3]
            reason = (
                f"{_num(percentile)}th percentile duration ({_num(percentile_value)}ms) exceeds "
                f"threshold {_num(max_ms)}ms. "
            )
            reason += f"Slowest spans: {describe(slowest)}"
        else:
            reason = (
                f"{_num(percentile)}th percentile duration ({_num(percentile_value)}ms) is "
                f"within threshold {_num(max_ms)}ms"
            )
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": with_trace_attribute_filter_context(reason, attributes),
        "assertion": assertion,
    }


# --- traceErrorSpans.ts -----------------------------------------------------------------------


def _js_number(value: Any) -> float:
    from promptphoo.matchers.rubric import js_number

    return js_number(value)


def is_error_span(span: dict[str, Any]) -> bool:
    status_code = span.get("statusCode")
    if status_code == 2 and not isinstance(status_code, bool):
        return True
    if is_truthy(status_code) and _js_number(status_code) >= 400:
        return True
    attributes = span.get("attributes")
    if is_truthy(attributes) and isinstance(attributes, dict):
        for key in ("error", "exception", "failed", "failure"):
            attr = attributes.get(key)
            if attr is True or attr == "true" or isinstance(attr, dict | list):
                return True
        http_status_code = attributes.get("http.status_code")
        if http_status_code is not None:
            status_number = _js_number(http_status_code)
            if not math.isnan(status_number) and status_number >= 400:
                return True
        if (
            attributes.get("otel.status_code") == "ERROR"
            or attributes.get("status.code") == "ERROR"
        ):
            return True
    status_message = span.get("statusMessage")
    return bool(
        is_truthy(status_message) and _ERROR_STATUS_MESSAGE.search(js_string(status_message))
    )


def handle_trace_error_spans(params: AssertionParams) -> GradingResult:
    assertion = params.assertion
    spans = _require_spans(params)
    value = assertion.get("value")
    max_count: Any = None
    max_percentage: Any = None
    pattern: Any = "*"
    attributes: Any = None
    if _is_number(value):
        max_count = value
    elif isinstance(value, dict):
        max_count = value.get("max_count")
        max_percentage = value.get("max_percentage")
        pattern = value.get("pattern") if is_truthy(value.get("pattern")) else "*"
        attributes = value.get("attributes")
    if max_count is None and max_percentage is None:
        max_count = 0  # Default to no errors allowed

    matching = filter_trace_spans(spans, pattern, attributes)
    if not matching:
        return {
            "pass": True,
            "score": 1,
            "reason": with_trace_attribute_filter_context(
                f'No spans found matching pattern "{js_string(pattern)}"', attributes
            ),
            "assertion": assertion,
        }
    error_spans = [s for s in matching if is_error_span(s)]
    error_count = len(error_spans)
    error_percentage = (error_count / len(matching)) * 100

    passed = True
    if max_count is not None and error_count > max_count:
        passed = False
        details = []
        for span in error_spans[:3]:
            detail = js_string(span.get("name"))
            if is_truthy(span.get("statusMessage")):
                detail += f" ({js_string(span['statusMessage'])})"
            elif is_truthy(span.get("statusCode")):
                detail += f" (status: {js_string(span['statusCode'])})"
            details.append(detail)
        reason = f"Found {error_count} error spans, expected at most {_num(max_count)}. "
        reason += f"Errors: {', '.join(details)}"
        if len(error_spans) > 3:
            reason += f" and {len(error_spans) - 3} more"
    elif max_percentage is not None and error_percentage > max_percentage:
        passed = False
        reason = f"Error rate {to_fixed(error_percentage, 1)}% exceeds threshold "
        reason += f"{_num(max_percentage)}% ({error_count} errors out of {len(matching)} spans)"
    elif error_count == 0:
        reason = f'No errors found in {len(matching)} spans matching pattern "{js_string(pattern)}"'
    else:
        reason = (
            f"Found {error_count} error(s) in {len(matching)} spans "
            f"({to_fixed(error_percentage, 1)}%)"
        )
        if max_count is not None:
            reason += f", within threshold of {_num(max_count)}"
        if max_percentage is not None:
            reason += f", within threshold of {_num(max_percentage)}%"
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": with_trace_attribute_filter_context(reason, attributes),
        "assertion": assertion,
    }


HANDLERS: dict[str, Handler] = {
    "trace-error-spans": handle_trace_error_spans,
    "trace-span-count": handle_trace_span_count,
    "trace-span-duration": handle_trace_span_duration,
}
