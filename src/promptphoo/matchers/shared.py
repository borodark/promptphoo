"""Shared matcher helpers. Ports src/matchers/shared.ts."""

from __future__ import annotations

import math
import re
from typing import Any

from promptphoo.assertions.refusal import JS_WHITESPACE, js_trim
from promptphoo.jsonutil import js_json_parse


def normalize_matcher_token_usage(token_usage: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize token usage for matcher results.

    Unlike the evaluator-level normalizeTokenUsage, this excludes the `assertions` field and
    preserves the existing completionDetails shape and any incurred usage reported by the
    provider.
    """
    usage = token_usage or {}

    def get(key: str) -> Any:
        value = usage.get(key)
        return 0 if value is None else value

    prompt = get("prompt")
    completion = get("completion")
    cached = get("cached")
    component_total = prompt + completion
    cached_response = (
        token_usage is not None
        and usage.get("numRequests") == 0
        and cached > 0
        and component_total <= cached
    )
    total = usage.get("total")
    out: dict[str, Any] = {
        "total": total if total is not None else (0 if cached_response else component_total),
        "prompt": prompt,
        "completion": completion,
        "cached": cached,
        "numRequests": get("numRequests"),
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
    """Same shape as `fail`, but tagged with `metadata.graderError` so inverse-aware callers
    (e.g. `not-llm-rubric`, `not-g-eval`) propagate the failure verbatim rather than flipping a
    transport/parse error into a spurious pass."""
    return {**fail(reason, tokens_used), "metadata": {"graderError": True}}


def _is_finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def invert_score(score: Any) -> float:
    """Invert a grader score for a negated assertion, clamped to [0, 1] so a NaN or
    out-of-range score cannot become a misleading negative or inflated value."""
    return min(1, max(0, 1 - (score if _is_finite(score) else 0)))


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
        pass
    return content


def split_into_sentences(text: str) -> list[str]:
    return [sentence for sentence in text.split("\n") if js_trim(sentence) != ""]


_ENUMERATION_MARKER_ONLY = re.compile(r"^[0-9]+[.)]$")
_SENTENCE_BOUNDARY = re.compile(f"(?<=[.!?])[{JS_WHITESPACE}]+")


def split_text_into_sentences(text: str) -> list[str]:
    """Segment text into units for sentence-level metrics (e.g. the RAGAS context relevance
    denominator).

    Text spanning two or more non-empty lines is already segmented (one unit per line).
    Otherwise it is one prose block, split on `.`, `!` or `?` followed by whitespace. Bare
    enumeration markers ("1.", "2)") are dropped.
    """
    lines = [line for line in text.split("\n") if js_trim(line) != ""]
    segments = lines if len(lines) > 1 else _SENTENCE_BOUNDARY.split(text)
    out: list[str] = []
    for segment in segments:
        sentence = js_trim(segment)
        if sentence and not _ENUMERATION_MARKER_ONLY.match(sentence):
            out.append(sentence)
    return out
