"""JUnit XML output (`-o results.junit.xml`).

Ports src/util/junit.ts: one `<testsuite>` per (provider, prompt) pair, one `<testcase>` per
result. Model outputs, vars and grader reasons are never written, only assertion types.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from promptphoo.jsvalues import js_sum, json_stringify, number_to_string, to_fixed
from promptphoo.output.xmlbuilder import XMLBuilder

MAX_JUNIT_NAME_LENGTH = 512
MAX_JUNIT_DETAIL_LENGTH = 8192
JUNIT_ASSERTION_FAILURE_MESSAGE = "Assertion failed"
JUNIT_EVALUATION_ERROR_MESSAGE = "Evaluation error"
INVALID_XML_CHARACTERS = re.compile("[^\t\n\r\x20-퟿-�\U00010000-\U0010ffff]")
_JS_WS = re.compile("[\t\n\x0b\x0c\r \xa0  -     　﻿]+")

_ASSERT = 1


def _json_key(value: Any) -> str:
    """JSON.stringify output; lone surrogates are escaped as in well-formed JSON.stringify."""
    return re.sub("[\ud800-\udfff]", lambda m: f"\\u{ord(m.group(0)):04x}", json_stringify(value))


def _utf16_len(value: str) -> int:
    return len(value.encode("utf-16-le", "surrogatepass")) // 2


def _utf16_slice(value: str, end: int) -> str:
    return value.encode("utf-16-le", "surrogatepass")[: end * 2].decode("utf-16-le", "replace")


def truncate_text(value: str, max_length: int) -> str:
    if _utf16_len(value) <= max_length:
        return value
    cutoff = max_length - 3
    units = value.encode("utf-16-le", "surrogatepass")
    unit = int.from_bytes(units[(cutoff - 1) * 2 : cutoff * 2], "little") if cutoff > 0 else 0
    end = cutoff + 1 if 0xD800 <= unit <= 0xDBFF else cutoff
    return f"{_utf16_slice(value, end)}..."


def normalize_inline_text(
    value: str | None, fallback: str, max_length: int = MAX_JUNIT_NAME_LENGTH
) -> str:
    sanitized = _JS_WS.sub(" ", value or "")
    sanitized = INVALID_XML_CHARACTERS.sub("", sanitized)
    sanitized = _JS_WS.sub(" ", sanitized).strip(" ")
    return truncate_text(sanitized or fallback, max_length)


def format_duration_seconds(duration_ms: Any) -> str:
    safe = (
        duration_ms
        if isinstance(duration_ms, int | float)
        and not isinstance(duration_ms, bool)
        and duration_ms == duration_ms
        and duration_ms not in (float("inf"), float("-inf"))
        and duration_ms > 0
        else 0
    )
    return number_to_string(float(to_fixed(safe / 1000, 3)))


def _iso(ms: Any) -> str | None:
    try:
        dt = datetime.fromtimestamp(float(ms) / 1000, UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _test_case_name(result: dict[str, Any]) -> str:
    prefix = f"test {result['testIdx'] + 1}"
    description = normalize_inline_text(
        result.get("description") or (result.get("testCase") or {}).get("description"), ""
    )
    return f"{prefix}: {description}" if description else prefix


def _failed_assertion_labels(grading: dict[str, Any] | None) -> list[str]:
    failed = [c for c in ((grading or {}).get("componentResults") or []) if not c.get("pass")]
    if failed:
        return [(c.get("assertion") or {}).get("type") or "assertion" for c in failed]
    return [(grading.get("assertion") or {}).get("type") or "assertion"] if grading else []


def _failure_details(result: dict[str, Any]) -> str:
    from promptphoo.jsvalues import js_string

    lines = [
        f"Score: {js_string(result.get('score'))}",
        f"Reason: {JUNIT_ASSERTION_FAILURE_MESSAGE}",
    ]
    labels = _failed_assertion_labels(result.get("gradingResult"))
    if labels:
        lines.append("Failed assertions:")
        lines.extend(f"- {label}" for label in labels)
    return truncate_text(INVALID_XML_CHARACTERS.sub("", "\n".join(lines)), MAX_JUNIT_DETAIL_LENGTH)


def _project(result: Any) -> dict[str, Any]:
    return result.to_evaluate_result() if hasattr(result, "to_evaluate_result") else result


def _iterate_results(eval_record: Any) -> list[dict[str, Any]]:
    version = eval_record.version() if hasattr(eval_record, "version") else 4
    if version < 4:
        old = getattr(eval_record, "old_results", None) or {}
        return [_project(r) for r in old.get("results") or []]
    if not getattr(eval_record, "persisted", False):
        return [_project(r) for r in eval_record.results]
    return [_project(r) for r in eval_record.load_results()]


def _build_test_case(result: dict[str, Any]) -> dict[str, Any]:
    testcase: dict[str, Any] = {
        "@_name": _test_case_name(result),
        "@_time": format_duration_seconds(result.get("latencyMs")),
    }
    if not result.get("success"):
        if result.get("failureReason") == _ASSERT:
            testcase["failure"] = {
                "#text": _failure_details(result),
                "@_message": JUNIT_ASSERTION_FAILURE_MESSAGE,
                "@_type": "assertion",
            }
        else:
            testcase["error"] = {
                "#text": f"Reason: {JUNIT_EVALUATION_ERROR_MESSAGE}",
                "@_message": JUNIT_EVALUATION_ERROR_MESSAGE,
                "@_type": "error",
            }
    return testcase


def _locale_key(value: str) -> tuple[str, str]:
    return (value.casefold(), value.swapcase())


def _build_suites(eval_record: Any) -> list[dict[str, Any]]:
    suites: dict[str, dict[str, Any]] = {}
    providers_by_display: dict[str, set[str]] = {}
    for result in _iterate_results(eval_record):
        provider = result.get("provider") or {}
        provider_key = _json_key([provider.get("id") or "", provider.get("label") or ""])
        prompt_key = result.get("promptId") or f"prompt-index:{result['promptIdx']}"
        key = _json_key([provider_key, prompt_key])
        suite = suites.get(key)
        if suite is None:
            raw_name = provider.get("label") or provider.get("id") or ""
            display = normalize_inline_text(raw_name, "unknown provider")
            providers_by_display.setdefault(display, set()).add(provider_key)
            suite = {
                "providerKey": provider_key,
                "rawName": raw_name,
                "promptKey": prompt_key,
                "promptIdx": result["promptIdx"],
                "displayName": "",
                "errors": 0,
                "failures": 0,
                "testcases": [],
                "timeMs": 0,
            }
            suites[key] = suite
        suite["promptIdx"] = min(suite["promptIdx"], result["promptIdx"])
        suite["testcases"].append((result["testIdx"], _build_test_case(result)))
        suite["timeMs"] += result.get("latencyMs") or 0
        if not result.get("success"):
            if result.get("failureReason") == _ASSERT:
                suite["failures"] += 1
            else:
                suite["errors"] += 1

    prompt_counts: dict[str, int] = {}
    ordered = sorted(suites.values(), key=lambda s: (s["promptIdx"], _locale_key(s["promptKey"])))
    for suite in ordered:
        base = normalize_inline_text(suite["rawName"], "unknown provider")
        collision = len(providers_by_display.get(base, set())) > 1
        suffix = (
            f" ({hashlib.sha256(suite['providerKey'].encode()).hexdigest()[:16]})"
            if collision or INVALID_XML_CHARACTERS.search(suite["rawName"])
            else ""
        )
        name = normalize_inline_text(
            suite["rawName"], "unknown provider", MAX_JUNIT_NAME_LENGTH - len(suffix)
        )
        ordinal = prompt_counts.get(suite["providerKey"], 0) + 1
        prompt_counts[suite["providerKey"]] = ordinal
        suite["displayName"] = f"[{name}] prompt {ordinal}{suffix}"
        for _, testcase in suite["testcases"]:
            testcase["@_classname"] = suite["displayName"]
    return list(suites.values())


def create_junit_xml(eval_record: Any) -> str:
    suites = _build_suites(eval_record)
    tests = sum(len(s["testcases"]) for s in suites)
    failures = sum(s["failures"] for s in suites)
    errors = sum(s["errors"] for s in suites)
    total_ms = js_sum(s["timeMs"] for s in suites)
    timestamp = _iso(getattr(eval_record, "created_at", None))
    builder = XMLBuilder(format=True, ignore_attributes=False, indent_by="  ")
    testsuites = []
    for suite in suites:
        entry: dict[str, Any] = {
            "@_errors": suite["errors"],
            "@_failures": suite["failures"],
            "@_name": suite["displayName"],
            "@_skipped": 0,
            "@_tests": len(suite["testcases"]),
            "@_time": format_duration_seconds(suite["timeMs"]),
        }
        if timestamp:
            entry["@_timestamp"] = timestamp
        entry["testcase"] = [tc for _, tc in sorted(suite["testcases"], key=lambda t: t[0])]
        testsuites.append(entry)
    xml = builder.build(
        {
            "?xml": {"@_version": "1.0", "@_encoding": "UTF-8"},
            "testsuites": {
                "@_errors": errors,
                "@_failures": failures,
                "@_name": "promptfoo",
                "@_skipped": 0,
                "@_tests": tests,
                "@_time": format_duration_seconds(total_ms),
                "testsuite": testsuites,
            },
        }
    )
    return INVALID_XML_CHARACTERS.sub("", xml)


def write_junit_xml_output(output_path: str, eval_record: Any) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(create_junit_xml(eval_record))
