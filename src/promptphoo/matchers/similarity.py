"""Embedding similarity matcher. Ports src/matchers/similarity.ts (matchesSimilarity).

Upstream's remote-grading branch (cosine similarity under a `redteam` config) is not ported:
redteam is out of scope, so the grading provider is always called locally.
"""

from __future__ import annotations

import asyncio
import math
import sys
from typing import Any

from promptphoo.jsvalues import is_truthy, js_string, to_fixed
from promptphoo.matchers.providers import (
    call_embedding_provider,
    call_grading_provider,
    get_and_check_provider,
)
from promptphoo.matchers.shared import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    fail,
    normalize_matcher_token_usage,
)
from promptphoo.tokens import accumulate_token_usage

EPSILON = sys.float_info.epsilon


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _calculate_similarity_score(
    expected: list[float], output: list[float], metric: str, tokens_used: dict[str, Any]
) -> float | dict[str, Any]:
    if metric == "cosine":
        return cosine_similarity(expected, output)
    if metric == "dot_product":
        return dot_product(expected, output)
    if metric == "euclidean":
        # The distance is stored as the similarity; distance semantics apply when grading.
        return euclidean_distance(expected, output)
    return fail(f"Unsupported metric: {metric}", tokens_used)


def _build_similarity_result(
    similarity: float, threshold: float, inverse: bool, metric: str, tokens_used: dict[str, Any]
) -> dict[str, Any]:
    if not _is_finite_number(similarity):
        return fail(f"Invalid similarity score: {js_string(similarity)}", tokens_used)
    t = js_string(threshold)
    if metric == "euclidean":
        # For distance metrics lower is better, and the threshold is the maximum distance.
        distance = similarity
        passed = distance >= threshold - EPSILON if inverse else distance <= threshold + EPSILON
        normalized = 1 / (1 + distance)
        score = 1 - normalized if inverse else normalized
        below = f"Distance {to_fixed(distance, 2)} is less than or equal to threshold {t}"
        above = f"Distance {to_fixed(distance, 2)} is greater than threshold {t}"
        reason = (above if inverse else below) if passed else (below if inverse else above)
        return {"pass": passed, "score": score, "reason": reason, "tokensUsed": tokens_used}

    # For similarity metrics higher is better, and the threshold is the minimum similarity.
    passed = similarity <= threshold + EPSILON if inverse else similarity >= threshold - EPSILON
    score = 1 - similarity if inverse else similarity
    greater = f"Similarity {to_fixed(similarity, 2)} is greater than or equal to threshold {t}"
    less = f"Similarity {to_fixed(similarity, 2)} is less than threshold {t}"
    reason = (less if inverse else greater) if passed else (greater if inverse else less)
    return {"pass": passed, "score": score, "reason": reason, "tokensUsed": tokens_used}


async def _calculate_provider_similarity(
    provider: Any,
    expected: str,
    output: str,
    metric: str,
    tokens_used: dict[str, Any],
    call_context: dict[str, Any] | None = None,
) -> float | dict[str, Any]:
    has_similarity_api = hasattr(provider, "call_similarity_api")
    if metric == "cosine" and has_similarity_api:
        resp = await call_grading_provider(
            provider,
            "similarity",
            lambda _context: provider.call_similarity_api(expected, output),
            call_context=call_context,
        )
        accumulate_token_usage(tokens_used, resp.get("tokenUsage"))
        if resp.get("error"):
            return fail(resp["error"], tokens_used)
        if resp.get("similarity") is None:
            return fail("Unknown error fetching similarity", tokens_used)
        if not _is_finite_number(resp["similarity"]):
            return fail(f"Invalid similarity score: {js_string(resp['similarity'])}", tokens_used)
        return resp["similarity"]

    if not callable(getattr(provider, "call_embedding_api", None)):
        if has_similarity_api:
            return fail(
                f"Provider {provider.id()} only supports cosine similarity via callSimilarityApi",
                tokens_used,
            )
        raise ValueError("Provider must implement callSimilarityApi or callEmbeddingApi")

    def embed(text: str) -> Any:
        return call_grading_provider(
            provider,
            "similarity.embedding",
            lambda context: call_embedding_provider(provider, text, context),
            operation_name="embeddings",
            call_context=call_context,
        )

    expected_embedding, output_embedding = await asyncio.gather(embed(expected), embed(output))

    merged = normalize_matcher_token_usage(None)
    accumulate_token_usage(merged, expected_embedding.get("tokenUsage"))
    accumulate_token_usage(merged, output_embedding.get("tokenUsage"))
    accumulate_token_usage(tokens_used, merged)

    if expected_embedding.get("error") or output_embedding.get("error"):
        return fail(
            expected_embedding.get("error")
            or output_embedding.get("error")
            or "Unknown error fetching embeddings",
            tokens_used,
        )
    if not is_truthy(expected_embedding.get("embedding")) or not is_truthy(
        output_embedding.get("embedding")
    ):
        return fail("Embedding not found", tokens_used)
    return _calculate_similarity_score(
        expected_embedding["embedding"], output_embedding["embedding"], metric, tokens_used
    )


async def matches_similarity(
    expected: str,
    output: str,
    threshold: float,
    inverse: bool = False,
    grading: dict[str, Any] | None = None,
    metric: str = "cosine",
    call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from promptphoo.providers.defaults import get_default_providers

    defaults = await get_default_providers()
    provider = await get_and_check_provider(
        "embedding",
        (grading or {}).get("provider"),
        defaults.get("embeddingProvider"),
        "similarity check",
    )
    tokens_used = normalize_matcher_token_usage(None)
    similarity = await _calculate_provider_similarity(
        provider, expected, output, metric, tokens_used, call_context
    )
    if isinstance(similarity, dict):
        return similarity
    return _build_similarity_result(similarity, threshold, inverse, metric, tokens_used)
