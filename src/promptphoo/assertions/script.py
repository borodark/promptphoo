"""Script-backed assertions (python, webhook) and script result normalization. Ports
src/assertions/{python,webhook,scriptBody,scriptResultNormalization}.ts and isGradingResult."""

from __future__ import annotations

import math
import re
from typing import Any

import httpx

from promptphoo import envars, python_hooks
from promptphoo.assertions.base import AssertionParams, GradingResult, invariant
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import js_string, json_stringify, number_to_string

_FLOAT_PREFIX = re.compile(r"\s*([+-]?(?:Infinity|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))")


def js_parse_float(value: str) -> float:
    m = _FLOAT_PREFIX.match(value)
    if not m:
        return math.nan
    text = m.group(1)
    return float(text.replace("Infinity", "inf"))


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _finite_record(value: Any) -> bool:
    return isinstance(value, dict) and all(_finite(v) for v in value.values())


def _valid_fields(r: Any) -> bool:
    return (
        isinstance(r, dict)
        and isinstance(r.get("pass"), bool)
        and _finite(r.get("score"))
        and isinstance(r.get("reason"), str)
        and (r.get("namedScores") is None or _finite_record(r["namedScores"]))
        and (r.get("namedScoreWeights") is None or _finite_record(r["namedScoreWeights"]))
        and ("tokensUsed" not in r or isinstance(r["tokensUsed"], dict | type(None)))
        and (r.get("componentResults") is None or isinstance(r["componentResults"], list))
        and (r.get("assertion") is None or isinstance(r["assertion"], dict))
        and ("comment" not in r or isinstance(r["comment"], str))
    )


def is_grading_result(result: Any) -> bool:
    if not _valid_fields(result):
        return False
    return all(is_grading_result(c) for c in result.get("componentResults") or [])


def _recorded_score(value: Any) -> Any:
    if isinstance(value, bool):
        return 1 if value else 0
    if value is None:
        return 0
    if isinstance(value, str) and value.strip() != "":
        try:
            score = float(value.strip())
        except ValueError:
            return value
        return score if math.isfinite(score) else value
    return value


def _legacy_shapes(result: Any, is_component: bool) -> Any:
    if not isinstance(result, dict):
        return result
    changes: dict[str, Any] = {}
    if is_component and isinstance(result.get("pass"), bool):
        if result.get("reason") is None:
            changes["reason"] = ""
        if result.get("score") is None:
            changes["score"] = 1 if result["pass"] else 0
    named = result.get("namedScores")
    if isinstance(named, dict):
        recorded = {k: _recorded_score(v) for k, v in named.items()}
        if recorded != named:
            changes["namedScores"] = recorded
    components = result.get("componentResults")
    if isinstance(components, list):
        converted = [_legacy_shapes(c, True) for c in components]
        if converted != components:
            changes["componentResults"] = converted
    return {**result, **changes} if changes else result


def as_grading_result(result: Any) -> GradingResult | None:
    if is_grading_result(result):
        return result
    converted = _legacy_shapes(result, False)
    return converted if converted is not result and is_grading_result(converted) else None


def map_snake_case_to_camel_case(obj: dict[str, Any]) -> dict[str, Any]:
    result = dict(obj)
    for snake, camel in (
        ("pass_", "pass"),
        ("named_scores", "namedScores"),
        ("named_score_weights", "namedScoreWeights"),
        ("component_results", "componentResults"),
        ("tokens_used", "tokensUsed"),
    ):
        if snake in result and camel not in result:
            result[camel] = result[snake]
    if isinstance(result.get("componentResults"), list):
        result["componentResults"] = [
            map_snake_case_to_camel_case(c) if isinstance(c, dict) else c
            for c in result["componentResults"]
        ]
    return result


def _append_reason(reason: str, suffix: Any) -> str:
    return f"{reason}\n{suffix}" if isinstance(suffix, str) and suffix else reason


def normalize_script_assertion_result(
    assertion: dict, result: Any, inverse: bool, labels: tuple[str, str], suffix: Any = None
) -> GradingResult:
    code, _language = labels

    def failure(raw: bool) -> str:
        return _append_reason(f"{code} returned {'true' if raw else 'false'}", suffix)

    if isinstance(result, bool):
        passed = result != inverse
        return {"pass": passed, "score": 1 if passed else 0,
                "reason": "Assertion passed" if passed else failure(result),
                "assertion": assertion}  # fmt: skip
    if isinstance(result, int | float):
        threshold = assertion.get("threshold")
        raw = result > 0 if threshold is None else result >= threshold
        passed = raw != inverse
        return {"pass": passed, "score": result,
                "reason": "Assertion passed" if passed else failure(raw),
                "assertion": assertion}  # fmt: skip
    passed = result["pass"] != inverse
    if inverse:
        reason = "Assertion passed" if passed else (result.get("reason") or f"{code} returned true")
    else:
        reason = result.get("reason")
    return {
        **result,
        "pass": passed,
        "reason": reason,
        "assertion": result.get("assertion") or assertion,
    }


def normalize_script_object_result(
    assertion: dict, result: Any, inverse: bool, labels: tuple[str, str], suffix: Any = None
) -> GradingResult:
    _code, language = labels
    grading = as_grading_result(map_snake_case_to_camel_case(result))
    if not grading:
        raise ValueError(
            f"{language} assertion must return a boolean, number, or {{pass, score, reason}} "
            f"object with finite scores and weights. Got type object."
        )
    grading = dict(grading)
    threshold = assertion.get("threshold")
    if threshold is not None and grading["score"] < threshold:
        grading["pass"] = False
        message = (
            f"{language} score {number_to_string(grading['score'])} is less than threshold "
            f"{number_to_string(threshold)}"
        )
        grading["reason"] = f"{message}: {grading['reason']}" if grading.get("reason") else message
    return normalize_script_assertion_result(
        assertion, {**grading, "assertion": assertion}, inverse, labels, suffix
    )


def normalize_script_result(
    assertion: dict, result: Any, inverse: bool, labels: tuple[str, str], suffix: Any = None
) -> GradingResult:
    _code, language = labels
    lower = result.lower() if isinstance(result, str) else None
    if result is True or lower == "true":
        return normalize_script_assertion_result(assertion, True, inverse, labels, suffix)
    if isinstance(result, bool) or lower == "false":
        return normalize_script_assertion_result(assertion, False, inverse, labels, suffix)
    if isinstance(result, str) and result.startswith("{"):
        try:
            parsed = js_json_parse(result)
        except ValueError as e:
            raise ValueError(f"Invalid JSON: SyntaxError: {e} when parsing result: {result}") from e
        if not isinstance(parsed, dict | list):
            raise ValueError(
                f"{language} assertion must return a boolean, number, or {{pass, score, reason}} "
                f"object. Got instead: {result}"
            )
        return normalize_script_object_result(assertion, parsed, inverse, labels, suffix)
    if isinstance(result, dict | list):
        return normalize_script_object_result(assertion, result, inverse, labels, suffix)
    score = js_parse_float(js_string(result))
    if math.isnan(score):
        raise ValueError(
            f"{language} assertion must return a boolean, number, or {{pass, score, reason}} "
            f"object. Instead got:\n{js_string(result)}"
        )
    return normalize_script_assertion_result(assertion, score, inverse, labels, suffix)


def build_script_body(rendered: str, default_indent: str) -> str:
    if "\n" not in rendered:
        return f"{default_indent}return {rendered}"
    match = re.search(r"^(?!\s*$)\s+", rendered, re.MULTILINE)
    indent = match.group(0) if match else default_indent
    return "\n".join(f"{indent}{line}" for line in rendered.split("\n"))


async def handle_python(params: AssertionParams) -> GradingResult:
    invariant(isinstance(params.rendered_value, str), "python assertion must have a string value")
    assertion = params.assertion
    try:
        if params.value_from_script is None:
            code = (
                "import json\n\ndef main(output, context):\n"
                f"{build_script_body(params.rendered_value, '    ')}\n"
            )
            result = await python_hooks.run_python_code(
                code, "main", [params.output, params.assertion_value_context]
            )
        else:
            result = params.value_from_script
        return normalize_script_result(
            assertion, result, params.inverse, ("Python code", "Python"), assertion.get("value")
        )
    except Exception as e:
        return {
            "pass": False,
            "score": 0,
            "reason": f"Python code execution failed: {e}",
            "assertion": assertion,
        }


async def handle_webhook(params: AssertionParams) -> GradingResult:
    url = params.rendered_value
    invariant(url, '"webhook" assertion type must have a URL value')
    invariant(isinstance(url, str), '"webhook" assertion type must have a URL value')
    assertion, inverse = params.assertion, params.inverse
    try:
        body = json_stringify(
            {"output": params.output,
             "context": {"prompt": params.prompt, "vars": params.test.get("vars") or {}}}
        )  # fmt: skip
        timeout = envars.get_env_int("WEBHOOK_TIMEOUT", 5000) / 1000
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url, content=body, headers={"Content-Type": "application/json"}
            )
        if not response.is_success:
            raise RuntimeError(f"Webhook response status: {response.status_code}")
        data = response.json()
        invariant(
            isinstance(data, dict) and isinstance(data.get("pass"), bool),
            'Webhook response must be a JSON object with a boolean "pass" property',
        )
        score = data.get("score")
        invariant(
            score is None or (_finite(score) and 0 <= score <= 1),
            'Webhook response "score" must be a finite number between 0 and 1',
        )
        passed = data["pass"] != inverse
        score = score if score is not None else (1 if data["pass"] else 0)
        reason = data.get("reason") or (
            "Assertion passed" if passed else f"Webhook returned {'true' if inverse else 'false'}"
        )
        return {"pass": passed, "score": 1 - score if inverse else score, "reason": reason,
                "assertion": assertion}  # fmt: skip
    except Exception as e:
        return {"pass": False, "score": 0, "reason": f"Webhook error: {e}", "assertion": assertion}


def _unsupported(language: str):
    def handler(params: AssertionParams) -> GradingResult:
        raise ValueError(
            f"{language} assertions are not supported by promptphoo. "
            "Use a `python` assertion or file://assert.py instead."
        )

    return handler


HANDLERS = {
    "python": handle_python,
    "webhook": handle_webhook,
    "javascript": _unsupported("JavaScript"),
    "ruby": _unsupported("Ruby"),
}
