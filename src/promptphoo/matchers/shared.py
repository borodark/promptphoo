# SHIM: replaced by the grading-infrastructure branch at merge
"""Matcher helpers. Ports src/matchers/shared.ts."""

from __future__ import annotations

import math
import re
from typing import Any

from promptphoo.jsonutil import js_json_parse

TokenUsage = dict[str, Any]


def _or0(value: Any) -> Any:
    return value if value is not None else 0


def normalize_matcher_token_usage(token_usage: TokenUsage | None) -> TokenUsage:
    usage = token_usage or {}
    prompt = _or0(usage.get("prompt"))
    completion = _or0(usage.get("completion"))
    cached = _or0(usage.get("cached"))
    component_total = prompt + completion
    cached_response = usage.get("numRequests") == 0 and cached > 0 and component_total <= cached
    total = usage.get("total")
    out: TokenUsage = {
        "total": total if total is not None else (0 if cached_response else component_total),
        "prompt": prompt,
        "completion": completion,
        "cached": cached,
        "numRequests": _or0(usage.get("numRequests")),
        "completionDetails": usage.get("completionDetails")
        or {"reasoning": 0, "acceptedPrediction": 0, "rejectedPrediction": 0},
    }
    if usage.get("incurredTokenUsage"):
        out["incurredTokenUsage"] = usage["incurredTokenUsage"]
    return out


def fail(reason: str, tokens_used: TokenUsage | None = None) -> dict[str, Any]:
    return {
        "pass": False,
        "reason": reason,
        "score": 0,
        "tokensUsed": normalize_matcher_token_usage(tokens_used),
    }


def grader_fail(reason: str, tokens_used: TokenUsage | None = None) -> dict[str, Any]:
    return {**fail(reason, tokens_used), "metadata": {"graderError": True}}


def invert_score(score: Any) -> float:
    finite = isinstance(score, int | float) and not isinstance(score, bool) and math.isfinite(score)
    return min(1, max(0, 1 - (score if finite else 0)))


def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of equal length")
    return sum(a * b for a, b in zip(vec_a, vec_b, strict=True))


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
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b, strict=True)))


def try_parse(content: Any) -> Any:
    try:
        return js_json_parse(content)
    except Exception:
        pass
    return content


def split_into_sentences(text: str) -> list[str]:
    return [s for s in text.split("\n") if s.strip() != ""]


_ENUMERATION_MARKER_ONLY = re.compile(r"^\d+[.)]$")


def split_text_into_sentences(text: str) -> list[str]:
    lines = [line for line in text.split("\n") if line.strip() != ""]
    segments = lines if len(lines) > 1 else re.split(r"(?<=[.!?])\s+", text)
    stripped = (s.strip() for s in segments)
    return [s for s in stripped if s and not _ENUMERATION_MARKER_ONLY.match(s)]
