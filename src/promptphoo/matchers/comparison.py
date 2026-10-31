"""Comparison matchers over all outputs of one test. Ports src/matchers/comparison.ts
(matchesSelectBest, selectMaxScore)."""

from __future__ import annotations

import math
import re
from typing import Any

from promptphoo.jsvalues import js_string, to_fixed

_FIRST_INTEGER = re.compile(r"[0-9]+")  # JS \d is ASCII-only


def _js_number(value: Any) -> str:
    return js_string(value)


async def matches_select_best(
    criteria: str,
    outputs: list[Any],
    grading: dict[str, Any] | None = None,
    vars: dict[str, Any] | None = None,  # noqa: A002 - Node's parameter name
    provider_call_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from promptphoo.matchers.providers import call_provider_with_context, get_and_check_provider
    from promptphoo.matchers.rubric import load_rubric_prompt, render_llm_rubric_prompt
    from promptphoo.matchers.shared import fail, normalize_matcher_token_usage, try_parse
    from promptphoo.prompts.grading import SELECT_BEST_PROMPT
    from promptphoo.providers.defaults import get_default_providers

    if len(outputs) < 2:
        raise ValueError("select-best assertion must have at least two outputs to compare between")
    text_provider = await get_and_check_provider(
        "text",
        (grading or {}).get("provider"),
        (await get_default_providers())["gradingProvider"],
        "select-best check",
    )

    rubric_prompt = await load_rubric_prompt((grading or {}).get("rubricPrompt"),
                                             SELECT_BEST_PROMPT)  # fmt: skip
    template_vars = {
        **(vars or {}),
        "criteria": criteria,
        "outputs": [try_parse(o) for o in outputs],
    }
    prompt_text = await render_llm_rubric_prompt(rubric_prompt, template_vars)

    resp = await call_provider_with_context(
        text_provider, prompt_text, "select-best", template_vars, provider_call_context
    )
    token_usage = resp.get("tokenUsage")
    if resp.get("cached"):
        tu = token_usage or {}
        total = tu.get("total")
        if total is None:
            total = (tu.get("prompt") or 0) + (tu.get("completion") or 0)
        cached = tu.get("cached")
        tokens_used = normalize_matcher_token_usage(
            {**tu, "cached": max(cached if cached is not None else 0, total)}
        )
    else:
        tokens_used = normalize_matcher_token_usage(token_usage)
    cache_metadata: dict[str, Any] = (
        {"metadata": {"cachedResponse": True}} if resp.get("cached") else {}
    )

    def failure_results(reason: str) -> list[dict[str, Any]]:
        return [{**fail(reason, tokens_used), **cache_metadata} for _ in outputs]

    output = resp.get("output")
    if resp.get("error") or not output:
        return failure_results(resp.get("error") or "No output")

    if not isinstance(output, str):
        raise ValueError("select-best produced malformed response")

    match = _FIRST_INTEGER.search(output.strip())
    verdict: float = int(match.group(0)) if match else math.nan

    if math.isnan(verdict) or verdict < 0 or verdict >= len(outputs):
        shown = "NaN" if math.isnan(verdict) else str(int(verdict))
        return failure_results(f"Invalid select-best verdict: {shown}")

    results: list[dict[str, Any]] = []
    for index in range(len(outputs)):
        selected = index == verdict
        results.append({
            "pass": selected,
            "score": 1 if selected else 0,
            "reason": f"Output selected as the best: {criteria}" if selected
            else f"Output not selected: {criteria}",
            "tokensUsed": tokens_used,
            **cache_metadata,
        })  # fmt: skip
    return results


async def select_max_score(
    outputs: list[Any],
    results_with_grading_results: list[dict[str, Any]],
    assertion: dict[str, Any],
) -> list[dict[str, Any]]:
    """Each entry of `results_with_grading_results` is `{gradingResult?, unavailable?}`."""
    if len(outputs) < 2:
        raise ValueError("max-score assertion must have at least two outputs to compare between")

    # Parse options from assertion value
    value = assertion.get("value") or {}
    is_obj = isinstance(value, dict)
    method = value["method"] if is_obj and "method" in value else "average"
    weights = value["weights"] if is_obj and "weights" in value else {}
    threshold = value["threshold"] if is_obj and "threshold" in value else None

    # Calculate aggregate score for each output
    scores: list[dict[str, Any]] = []
    for index, result in enumerate(results_with_grading_results):
        if result.get("unavailable"):
            scores.append({"index": index, "score": 0, "componentCount": 0, "totalWeight": 0,
                           "unavailable": True})  # fmt: skip
            continue
        # Get component results from gradingResult if available
        component_results = (result.get("gradingResult") or {}).get("componentResults") or []

        # Filter out max-score and select-best assertions
        relevant = [
            r
            for r in component_results
            if r.get("assertion")
            and r["assertion"].get("type") != "max-score"
            and r["assertion"].get("type") != "select-best"
        ]
        if not relevant:
            raise ValueError(
                "max-score requires at least one other assertion (besides max-score or "
                "select-best) to aggregate scores from"
            )

        # Calculate weighted scores for each assertion
        total_weighted_score: float = 0
        total_weight: float = 0
        for component in relevant:
            assertion_type = (component.get("assertion") or {}).get("type") or "unknown"
            weight = weights.get(assertion_type) if isinstance(weights, dict) else None
            weight = 1.0 if weight is None else weight  # Default weight is 1
            score = component.get("score") or 0
            total_weighted_score += score * weight
            total_weight += weight

        # Calculate aggregate score based on method
        if method == "sum":
            aggregate = total_weighted_score
        else:
            # Average method (default)
            aggregate = total_weighted_score / total_weight if total_weight > 0 else 0
        scores.append({"index": index, "score": aggregate, "componentCount": len(relevant),
                       "totalWeight": total_weight})  # fmt: skip

    # Find max score (with deterministic tie-breaking by index)
    max_score = -math.inf
    winner_index = 0
    for i, s in enumerate(scores):
        if not s.get("unavailable") and s["score"] > max_score:
            max_score = s["score"]
            winner_index = i

    # Apply threshold if specified
    meets_threshold = threshold is None or max_score >= threshold

    out: list[dict[str, Any]] = []
    for s in scores:
        if s.get("unavailable"):
            out.append({"pass": False, "score": 0,
                        "reason": "Target response unavailable for comparison"})  # fmt: skip
            continue
        score = s["score"]
        is_winner = s["index"] == winner_index and meets_threshold
        if is_winner:
            reason = f"Selected as highest scoring output (score: {to_fixed(score, 3)})"
        elif score == max_score and not meets_threshold:
            reason = (
                f"Not selected - score {to_fixed(score, 3)} below threshold {_js_number(threshold)}"
            )
        else:
            reason = f"Not selected (score: {to_fixed(score, 3)}, max: {to_fixed(max_score, 3)})"
        out.append({
            "pass": is_winner,
            "score": 1 if is_winner else 0,
            "reason": reason,
            "namedScores": {
                "maxScore": score,
                "assertionCount": s["componentCount"],
                "totalWeight": s["totalWeight"],
            },
        })  # fmt: skip
    return out
