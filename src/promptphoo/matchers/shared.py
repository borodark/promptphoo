# SHIM: replaced by the grading-infrastructure branch at merge
"""Shared matcher helpers. Ports src/matchers/shared.ts (only what comparison.py needs)."""

from __future__ import annotations

from typing import Any

from promptphoo.jsonutil import js_json_parse


def _or0(value: Any) -> Any:
    return 0 if value is None else value


def normalize_matcher_token_usage(token_usage: dict[str, Any] | None) -> dict[str, Any]:
    tu = token_usage or {}
    prompt = _or0(tu.get("prompt"))
    completion = _or0(tu.get("completion"))
    cached = _or0(tu.get("cached"))
    component_total = prompt + completion
    cached_response = tu.get("numRequests") == 0 and cached > 0 and component_total <= cached
    total = tu.get("total")
    if total is None:
        total = 0 if cached_response else component_total
    out: dict[str, Any] = {
        "total": total,
        "prompt": prompt,
        "completion": completion,
        "cached": cached,
        "numRequests": _or0(tu.get("numRequests")),
        "completionDetails": tu.get("completionDetails")
        or {"reasoning": 0, "acceptedPrediction": 0, "rejectedPrediction": 0},
    }
    if tu.get("incurredTokenUsage"):
        out["incurredTokenUsage"] = tu["incurredTokenUsage"]
    return out


def fail(reason: str, tokens_used: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "pass": False,
        "reason": reason,
        "score": 0,
        "tokensUsed": normalize_matcher_token_usage(tokens_used),
    }


def grader_fail(reason: str, tokens_used: dict[str, Any] | None = None) -> dict[str, Any]:
    return {**fail(reason, tokens_used), "metadata": {"graderError": True}}


def try_parse(content: Any) -> Any:
    if not isinstance(content, str):
        return content  # JSON.parse(String(x)) round-trips numbers/booleans; objects throw
    try:
        return js_json_parse(content)
    except (ValueError, TypeError):
        pass
    return content
