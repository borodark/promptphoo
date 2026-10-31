"""Classifier matcher. Ports src/matchers/classification.ts (matchesClassification).

Works with any provider that has `call_classification_api(text)`, such as a Python provider
or a fake in tests; the HuggingFace classification provider is not required.
"""

from __future__ import annotations

import sys
from typing import Any

from promptphoo.jsvalues import is_truthy, js_string, to_fixed
from promptphoo.matchers.providers import call_grading_provider, get_and_check_provider
from promptphoo.matchers.shared import grader_fail, normalize_matcher_token_usage

EPSILON = sys.float_info.epsilon


async def matches_classification(
    expected: str | None,
    output: str,
    threshold: float,
    grading: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pass if `output` is classified as `expected` (or as anything, when `expected` is None)
    with a score of at least `threshold`."""
    provider = await get_and_check_provider(
        "classification", (grading or {}).get("provider"), None, "classification check"
    )
    resp = await call_grading_provider(
        provider, "classification", lambda _context: provider.call_classification_api(output)
    )
    classification = resp.get("classification")
    scores = list((classification if classification is not None else {}).values())
    if not is_truthy(classification) or not scores:
        failure = grader_fail(
            "No classification scores returned"
            if is_truthy(classification)
            else resp.get("error") or "Unknown error fetching classification",
            resp.get("tokenUsage"),
        )
        if resp.get("cached"):
            failure["metadata"] = {**failure["metadata"], "cachedResponse": True}
        return failure

    assert classification is not None
    score = max(scores) if expected is None else (classification.get(expected) or 0)
    passed = score >= threshold - EPSILON
    subject = (
        "Maximum classification score"
        if expected is None
        else f"Classification {expected} has score"
    )
    op = ">=" if passed else "<"
    out: dict[str, Any] = {
        "pass": passed,
        "score": score,
        "reason": f"{subject} {to_fixed(score, 2)} {op} {js_string(threshold)}",
    }
    if resp.get("tokenUsage"):
        out["tokensUsed"] = normalize_matcher_token_usage(resp["tokenUsage"])
    if resp.get("cached"):
        out["metadata"] = {"cachedResponse": True}
    return out
