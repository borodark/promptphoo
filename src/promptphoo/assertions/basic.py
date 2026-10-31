"""Deterministic string/number assertions. Ports src/assertions/{equals,contains,startsWith,
regex,json,wordCount,cost,latency,finishReason,perplexity,normalize}.ts."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

from promptphoo.ajv import compile_schema, errors_text
from promptphoo.assertions.base import AssertionParams, GradingResult, invariant, result
from promptphoo.csvtests import parse_comma_separated_values
from promptphoo.jsonutil import extract_json_objects, js_deep_equal, js_json_parse
from promptphoo.jsregex import compile_js_regex
from promptphoo.jsvalues import js_string, json_stringify, number_to_string, to_fixed, to_precision
from promptphoo.yamlload import load_yaml

n = number_to_string


def normalize_for_comparison(text: str, mode: Any) -> str:
    if not mode:
        return text
    return unicodedata.normalize("NFC" if mode is True else mode, text)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def handle_equals(p: AssertionParams) -> GradingResult:
    value = p.rendered_value
    if isinstance(value, dict | list) or value is None:
        try:
            passed = js_deep_equal(value, js_json_parse(p.output_string)) != p.inverse
        except ValueError:
            passed = p.inverse
        shown = json_stringify(value)
    else:
        mode = p.assertion.get("normalizeUnicode")
        passed = (
            normalize_for_comparison(js_string(value), mode)
            == normalize_for_comparison(p.output_string, mode)
        ) != p.inverse
        shown = js_string(value)
    return result(
        p,
        passed,
        f'Expected output "{p.output_string}" to {"not " if p.inverse else ""}equal "{shown}"',
    )


def _contains(p: AssertionParams, mode: str, ignore_case: bool = False) -> GradingResult:
    kind = f"{'i' if ignore_case else ''}contains{'' if mode == 'single' else '-' + mode}"
    value = p.value_from_script if p.value_from_script is not None else p.rendered_value
    norm = p.assertion.get("normalizeUnicode") if mode == "single" and not ignore_case else None
    out = normalize_for_comparison(p.output_string, norm)

    def includes(item: Any) -> bool:
        if ignore_case:
            return js_string(item).lower() in p.output_string.lower()
        return normalize_for_comparison(js_string(item), norm) in out

    if mode == "single":
        invariant(
            (isinstance(value, str) and value != "")
            or (_is_number(value) and not (isinstance(value, float) and math.isnan(value))),
            f'"{kind}" assertion type must have a string or number value',
        )
        matches = includes(value)
        expectation = f'"{js_string(value)}"'
    else:
        invariant(value, f'"{kind}" assertion type must have a value')
        if isinstance(value, str):
            value = parse_comma_separated_values(value)
        invariant(isinstance(value, list), f'"{kind}" assertion type must have an array value')
        joined = ", ".join(js_string(v) for v in value)
        if mode == "any":
            matches = any(includes(v) for v in value)
            expectation = f'one of "{joined}"'
        else:
            missing = [v for v in value if not includes(v)]
            matches = not missing
            expectation = (
                f"all of [{joined}]. Missing: [{', '.join(js_string(v) for v in missing)}]"
            )
    passed = matches != p.inverse
    return result(
        p, passed, f"Expected output to {'not ' if p.inverse else ''}contain {expectation}"
    )


def handle_starts_with(p: AssertionParams) -> GradingResult:
    value = p.rendered_value
    invariant(value, '"starts-with" assertion type must have a string value')
    invariant(isinstance(value, str), '"starts-with" assertion type must have a string value')
    passed = p.output_string.startswith(value) != p.inverse
    return result(
        p, passed, f'Expected output to {"not " if p.inverse else ""}start with "{value}"'
    )


def handle_regex(p: AssertionParams) -> GradingResult:
    value = p.rendered_value
    invariant(value, '"regex" assertion type must have a string value')
    invariant(isinstance(value, str), '"regex" assertion type must have a string value')
    try:
        rx = compile_js_regex(value)
    except ValueError as e:
        return {"pass": False, "score": 0, "reason": f"Invalid regex pattern: {e}",
                "assertion": p.assertion}  # fmt: skip
    passed = (rx.search(p.output_string) is not None) != p.inverse
    return result(
        p, passed, f'Expected output to {"not " if p.inverse else ""}match regex "{value}"'
    )


def _schema_validator(p: AssertionParams, kind: str):
    value = p.rendered_value
    if isinstance(value, str):
        if value.startswith("file://"):
            invariant(
                p.value_from_script,
                f"{kind} references a file that does not export a JSON schema",
            )
            return compile_schema(p.value_from_script)
        return compile_schema(load_yaml(value))
    if isinstance(value, dict | list):
        return compile_schema(value)
    raise ValueError(f"{kind} assertion must have a string or object value")


def handle_is_json(p: AssertionParams) -> GradingResult:
    parsed: Any = None
    has_parsed = False
    try:
        parsed = js_json_parse(p.output_string)
        has_parsed = True
        passed = not p.inverse
    except ValueError:
        passed = p.inverse
    if has_parsed and p.rendered_value:
        validate = _schema_validator(p, "is-json")
        valid = validate(parsed)
        passed = not valid if p.inverse else valid
        if not passed:
            return {
                "pass": False,
                "score": 0,
                "reason": "Output is JSON that conforms to the provided schema"
                if p.inverse
                else f"JSON does not conform to the provided schema. Errors: "
                f"{errors_text(validate.errors)}",
                "assertion": p.assertion,
            }
    return result(p, passed, f"Expected output to {'not ' if p.inverse else ''}be valid JSON")


def handle_contains_json(p: AssertionParams) -> GradingResult:
    message = f"Expected output to {'not ' if p.inverse else ''}contain valid JSON"
    objects = extract_json_objects(p.output_string)
    passed = not objects if p.inverse else bool(objects)
    for obj in objects:
        if p.rendered_value:
            validate = _schema_validator(p, "contains-json")
            valid = validate(obj)
            passed = not valid if p.inverse else valid
            if valid:
                if p.inverse:
                    message = "Output contains JSON conforming to the provided schema"
                break
            message = (
                f"JSON does not conform to the provided schema. Errors: "
                f"{errors_text(validate.errors)}"
            )
    reason = "Assertion passed" if passed else message
    return {"pass": passed, "score": 1 if passed else 0, "reason": reason, "assertion": p.assertion}


def _js_number(value: Any) -> float:
    """Number(value) for the string/number inputs assertions accept."""
    if _is_number(value):
        return value
    text = js_string(value).strip()
    if text == "":
        return 0
    try:
        return float(int(text, 0)) if re.fullmatch(r"0[xXoObB][0-9a-fA-F]+", text) else float(text)
    except ValueError:
        return math.nan


def handle_word_count(p: AssertionParams) -> GradingResult:
    value = p.value_from_script if p.value_from_script is not None else p.rendered_value
    invariant(value is not None, '"word-count" assertion must have a value')
    count = len(re.findall(r"\S+", p.output_string.strip()))
    inverse = p.inverse
    reason = "Assertion passed"
    if isinstance(value, dict):
        lo, hi = value.get("min"), value.get("max")
        invariant(
            lo is not None or hi is not None,
            '"word-count" assertion object must have "min" and/or "max" properties',
        )
        if lo is not None and hi is not None:
            invariant(
                lo <= hi,
                f'"word-count" assertion: min ({number_to_string(lo)}) must be less than or '
                f"equal to max ({number_to_string(hi)})",
            )
            base = lo <= count <= hi
            passed = not base if inverse else base
            if not passed:
                reason = (
                    f"Expected word count to not be between {n(lo)} and {n(hi)}, but got {count}"
                    if inverse
                    else f"Word count {count} is not between {n(lo)} and {n(hi)}"
                )
        elif lo is None:
            assert hi is not None
            base = count <= hi
            passed = not base if inverse else base
            if not passed:
                reason = (
                    f"Expected word count to be greater than {n(hi)}, but got {count}"
                    if inverse
                    else f"Word count {count} is greater than maximum {n(hi)}"
                )
        else:
            base = count >= lo
            passed = not base if inverse else base
            if not passed:
                reason = (
                    f"Expected word count to be less than {n(lo)}, but got {count}"
                    if inverse
                    else f"Word count {count} is less than minimum {n(lo)}"
                )
    else:
        invariant(
            _is_number(value) or (isinstance(value, str) and not math.isnan(_js_number(value))),
            '"word-count" assertion value must be a number or an object with min/max properties',
        )
        expected = _js_number(value)
        base = count == expected
        passed = not base if inverse else base
        if not passed:
            shown = number_to_string(expected)
            reason = (
                f"Expected word count to not equal {shown}, but got {count}"
                if inverse
                else f"Word count {count} does not equal expected {shown}"
            )
    return {"pass": passed, "score": 1 if passed else 0, "reason": reason, "assertion": p.assertion}


def handle_cost(p: AssertionParams) -> GradingResult:
    a, cost = p.assertion, p.cost
    threshold = a.get("threshold")
    if threshold is None:
        if a.get("metric") and a.get("weight") == 0 and not p.inverse:
            if cost is None or not math.isfinite(cost) or cost < 0:
                raise ValueError("Cost metric requires a finite, non-negative provider cost")
            return {"pass": True, "score": cost, "reason": f"Cost: ${number_to_string(cost)}",
                    "assertion": a}  # fmt: skip
        raise ValueError("Cost assertion must have a threshold")
    if cost is None:
        raise ValueError("Cost assertion does not support providers that do not return cost")
    passed = (cost <= threshold) != p.inverse
    cmp = "less than or equal to" if p.inverse else "greater than"
    return result(
        p, passed, f"Cost {to_precision(cost, 2)} is {cmp} threshold {number_to_string(threshold)}"
    )


def handle_latency(p: AssertionParams) -> GradingResult:
    threshold = p.assertion.get("threshold")
    if threshold is None:
        raise ValueError("Latency assertion must have a threshold in milliseconds")
    if p.latency_ms is None:
        raise ValueError(
            "Latency assertion does not support cached results. Rerun the eval with --no-cache"
        )
    passed = (p.latency_ms <= threshold) != p.inverse
    cmp = "less than or equal to" if p.inverse else "greater than"
    return result(
        p,
        passed,
        f"Latency {number_to_string(p.latency_ms)}ms is {cmp} threshold "
        f"{number_to_string(threshold)}ms",
    )


def handle_finish_reason(p: AssertionParams) -> GradingResult:
    value = p.rendered_value if p.rendered_value is not None else p.assertion.get("value")
    invariant(isinstance(value, str), '"finish-reason" assertion type must have a string value')
    actual = p.provider_response.get("finishReason")
    if not actual:
        passed = p.inverse
        return result(p, passed, "Provider did not supply stop/finish reason")
    assert isinstance(value, str)
    passed = (value.lower() == actual.lower()) != p.inverse
    return result(
        p,
        passed,
        f'Expected finish reason {"not " if p.inverse else ""}"{value}" but got "{actual}"',
    )


def _perplexity(log_probs: list[float] | None, kind: str) -> float:
    if not log_probs:
        raise ValueError(f"{kind} assertion does not support providers that do not return logProbs")
    return math.exp(-(sum(log_probs) / len(log_probs)))


def handle_perplexity(p: AssertionParams) -> GradingResult:
    perplexity = _perplexity(p.log_probs, "Perplexity")
    threshold = p.assertion.get("threshold")
    passed = True if threshold is None else (perplexity <= threshold) != p.inverse
    cmp = "less than or equal to" if p.inverse else "greater than"
    return result(
        p,
        passed,
        f"Perplexity {to_fixed(perplexity, 2)} is {cmp} threshold {js_string(threshold)}",
    )


def handle_perplexity_score(p: AssertionParams) -> GradingResult:
    norm = 1 / (1 + _perplexity(p.log_probs, "perplexity-score"))
    threshold = p.assertion.get("threshold")
    passed = True if threshold is None else (norm >= threshold) != p.inverse
    cmp = "greater than or equal to" if p.inverse else "less than"
    return {
        "pass": passed,
        "score": 1 - norm if p.inverse else norm,
        "reason": "Assertion passed"
        if passed
        else f"Perplexity score {to_fixed(norm, 2)} is {cmp} threshold {js_string(threshold)}",
        "assertion": p.assertion,
    }


HANDLERS = {
    "equals": handle_equals,
    "contains": lambda p: _contains(p, "single"),
    "icontains": lambda p: _contains(p, "single", True),
    "contains-any": lambda p: _contains(p, "any"),
    "icontains-any": lambda p: _contains(p, "any", True),
    "contains-all": lambda p: _contains(p, "all"),
    "icontains-all": lambda p: _contains(p, "all", True),
    "starts-with": handle_starts_with,
    "regex": handle_regex,
    "is-json": handle_is_json,
    "contains-json": handle_contains_json,
    "word-count": handle_word_count,
    "cost": handle_cost,
    "latency": handle_latency,
    "finish-reason": handle_finish_reason,
    "perplexity": handle_perplexity,
    "perplexity-score": handle_perplexity_score,
}
