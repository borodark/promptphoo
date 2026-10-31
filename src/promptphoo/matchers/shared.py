# SHIM: replaced by the grading-infrastructure branch at merge
"""Shared matcher helpers. Ports src/matchers/shared.ts."""

from __future__ import annotations

import math
import re
from typing import Any

from promptphoo.jsonutil import js_json_parse

# JS String.prototype.trim() / regex \s whitespace.
_JS_WS = "\t\n\v\f\r                  　﻿"  # noqa: E501


def _nullish(value: Any, default: Any) -> Any:
    return default if value is None else value


def normalize_matcher_token_usage(token_usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = token_usage or {}
    prompt = _nullish(usage.get("prompt"), 0)
    completion = _nullish(usage.get("completion"), 0)
    cached = _nullish(usage.get("cached"), 0)
    component_total = prompt + completion
    cached_response = usage.get("numRequests") == 0 and cached > 0 and component_total <= cached
    out: dict[str, Any] = {
        "total": _nullish(usage.get("total"), 0 if cached_response else component_total),
        "prompt": prompt,
        "completion": completion,
        "cached": cached,
        "numRequests": _nullish(usage.get("numRequests"), 0),
        "completionDetails": usage.get("completionDetails")
        or {"reasoning": 0, "acceptedPrediction": 0, "rejectedPrediction": 0},
    }
    if usage.get("incurredTokenUsage"):
        out["incurredTokenUsage"] = usage["incurredTokenUsage"]
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


def invert_score(score: Any) -> float:
    finite = isinstance(score, int | float) and not isinstance(score, bool) and math.isfinite(score)
    return min(1, max(0, 1 - (score if finite else 0)))


def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of equal length")
    acc: float = 0
    for a, b in zip(vec_a, vec_b, strict=True):
        acc += a * b
    return acc


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    product = dot_product(vec_a, vec_b)
    mag_a = math.sqrt(sum(v * v for v in vec_a))
    mag_b = math.sqrt(sum(v * v for v in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0
    return product / (mag_a * mag_b)


def euclidean_distance(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of equal length")
    acc: float = 0
    for a, b in zip(vec_a, vec_b, strict=True):
        diff = a - b
        acc += diff * diff
    return math.sqrt(acc)


def try_parse(content: Any) -> Any:
    try:
        return js_json_parse(content)
    except Exception:
        return content


def split_into_sentences(text: str) -> list[str]:
    return [s for s in text.split("\n") if s.strip(_JS_WS) != ""]


_ENUMERATION_MARKER_ONLY = re.compile(r"^\d+[.)]$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[" + _JS_WS + r"]+")


def split_text_into_sentences(text: str) -> list[str]:
    lines = [line for line in text.split("\n") if line.strip(_JS_WS) != ""]
    segments = lines if len(lines) > 1 else _SENTENCE_BOUNDARY.split(text)
    out = []
    for sentence in segments:
        s = sentence.strip(_JS_WS)
        if s and not _ENUMERATION_MARKER_ONLY.match(s):
            out.append(s)
    return out
