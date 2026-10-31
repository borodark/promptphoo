"""Shared threshold and negation helpers for model-graded assertions.

Ports src/assertions/ragDefaults.ts.
"""

from __future__ import annotations

from typing import Any

from promptphoo.matchers.llm_grading import is_grader_failure
from promptphoo.matchers.shared import invert_score

DEFAULT_RAG_ASSERTION_THRESHOLD = 0.5


def apply_rag_inverse(result: dict[str, Any], inverse: bool) -> dict[str, Any]:
    if not inverse or is_grader_failure(result):
        return result
    return {**result, "pass": not result["pass"], "score": invert_score(result.get("score"))}


def finalize_graded_assertion(
    resp: dict[str, Any],
    assertion: dict[str, Any],
    inverse: bool,
    attach_assertion: bool = False,
) -> dict[str, Any]:
    # Grader failures must remain failures under negation.
    if is_grader_failure(resp):
        return {**resp, "assertion": assertion}
    score = invert_score(resp.get("score")) if inverse else resp.get("score")
    out = {**resp, "pass": resp["pass"] != inverse, "score": score}
    if attach_assertion:
        out["assertion"] = assertion
    return out
