"""matchesSimilarity. Ports the deterministic cases of upstream test/matchers/similarity.test.ts."""

from __future__ import annotations

import math
from typing import Any

import pytest
from rag_fakes import FakeEmbedding, embedding_result, use_defaults

from promptphoo.matchers.similarity import matches_similarity
from promptphoo.providers.base import ApiProvider


def default_embed(text: str) -> Any:
    if text in ("Expected output", "Sample output"):
        return embedding_result(1, 0)
    if text == "Different output":
        return embedding_result(0, 1)
    return ValueError("Unexpected input")


@pytest.fixture
def embedder(monkeypatch):
    provider = FakeEmbedding(default_embed)
    use_defaults(monkeypatch, embeddingProvider=provider)
    return provider


class NativeSimilarity(ApiProvider):
    def __init__(self, response: Any, embed: Any = None) -> None:
        super().__init__({})
        self.response = response
        self.similarity_calls: list[tuple[str, str]] = []
        if embed is not None:
            self.embed_calls: list[str] = []

            async def call_embedding_api(text: str) -> Any:
                self.embed_calls.append(text)
                return embed(text)

            self.call_embedding_api = call_embedding_api

    def default_id(self) -> str:
        return "native-similarity"

    async def call_api(self, prompt, context=None, options=None):
        raise NotImplementedError

    async def call_similarity_api(self, expected: str, output: str) -> Any:
        self.similarity_calls.append((expected, output))
        return self.response


USAGE = {"total": 10, "prompt": 4, "completion": 6, "cached": 0, "numRequests": 0}


def without_details(result: dict[str, Any]) -> dict[str, Any]:
    """Drop tokensUsed.completionDetails (accumulation adds the cache-token counters)."""
    usage = {k: v for k, v in result["tokensUsed"].items() if k != "completionDetails"}
    return {**result, "tokensUsed": usage}


async def test_passes_above_threshold(embedder):
    assert without_details(await matches_similarity("Expected output", "Sample output", 0.5)) == {
        "pass": True,
        "score": 1,
        "reason": "Similarity 1.00 is greater than or equal to threshold 0.5",
        "tokensUsed": USAGE,
    }


async def test_fails_below_threshold(embedder):
    assert without_details(
        await matches_similarity("Expected output", "Different output", 0.9)
    ) == {
        "pass": False,
        "score": 0,
        "reason": "Similarity 0.00 is less than threshold 0.9",
        "tokensUsed": USAGE,
    }


async def test_zero_magnitude_embeddings(embedder):
    embedder.fn = lambda t: {
        "embedding": [0, 0, 0],
        "tokenUsage": {"total": 5, "prompt": 2, "completion": 3},
    }
    result = await matches_similarity("Expected output", "Sample output", 0.5)
    assert (result["pass"], result["score"]) == (False, 0)
    assert result["reason"] == "Similarity 0.00 is less than threshold 0.5"


async def test_inverse(embedder):
    fail = await matches_similarity("Expected output", "Sample output", 0.5, True)
    assert (fail["pass"], fail["score"]) == (False, 0)
    assert fail["reason"] == "Similarity 1.00 is greater than or equal to threshold 0.5"
    ok = await matches_similarity("Expected output", "Different output", 0.9, True)
    assert (ok["pass"], ok["score"]) == (True, 1)
    assert ok["reason"] == "Similarity 0.00 is less than threshold 0.9"


async def test_uses_grading_provider_override(monkeypatch, embedder):
    override = FakeEmbedding(lambda t: embedding_result(1, 0), id="override")
    result = await matches_similarity("a", "b", 0.5, False, {"provider": override})
    assert result["pass"] is True
    assert override.inputs_seen == ["a", "b"]
    assert embedder.inputs_seen == []


async def test_provider_exceptions_propagate(embedder):
    embedder.fn = lambda t: RuntimeError("API call failed")
    with pytest.raises(RuntimeError, match="API call failed"):
        await matches_similarity("Expected output", "Sample output", 0.5)


async def test_templates_are_not_rendered(monkeypatch, embedder):
    monkeypatch.setenv("PROMPTFOO_DISABLE_TEMPLATING", "true")
    embedder.fn = lambda t: {"embedding": [1, 2, 3], "tokenUsage": {"total": 10}}
    await matches_similarity("Expected {{ var }}", "Output {{ var }}", 0.8)
    assert sorted(embedder.inputs_seen) == ["Expected {{ var }}", "Output {{ var }}"]


@pytest.mark.parametrize(
    ("output", "threshold", "inverse", "passed", "score"),
    [
        ("Sample output", 0.5, False, True, 1),
        ("Different output", 0.9, False, False, 0),
        ("Sample output", 0.5, True, False, 0),
    ],
)
async def test_dot_product(embedder, output, threshold, inverse, passed, score):
    result = await matches_similarity(
        "Expected output", output, threshold, inverse, None, "dot_product"
    )
    assert (result["pass"], result["score"]) == (passed, score)


VECTORS = {
    "small": [3, 4],
    "large": [6, 8],
    "opposite": [-3, -4],
    "huge": [1e200, 1e200],
    "negativeHuge": [-1e200, -1e200],
}


def vector_embed(text: str) -> Any:
    v = VECTORS[text]
    return embedding_result(v[0], v[1])


@pytest.mark.parametrize(
    ("metric", "output", "inverse", "score", "passed"),
    [
        ("cosine", "opposite", False, -1, False),
        ("cosine", "opposite", True, 2, True),
        ("dot_product", "large", False, 50, True),
        ("dot_product", "large", True, -49, False),
        ("dot_product", "opposite", False, -25, False),
        ("dot_product", "opposite", True, 26, True),
    ],
)
async def test_native_score_and_inverse_complement(
    embedder, metric, output, inverse, score, passed
):
    embedder.fn = vector_embed
    result = await matches_similarity("small", output, 0.75, inverse, None, metric)
    assert result["pass"] is passed
    assert result["score"] == pytest.approx(score)


@pytest.mark.parametrize(
    ("metric", "output", "invalid"),
    [
        ("cosine", "huge", "NaN"),
        ("dot_product", "huge", "Infinity"),
        ("dot_product", "negativeHuge", "-Infinity"),
        ("euclidean", "small", "Infinity"),
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
async def test_rejects_non_finite_scores(embedder, metric, output, invalid, inverse):
    embedder.fn = vector_embed
    result = await matches_similarity("huge", output, 0.75, inverse, None, metric)
    assert without_details(result) == {
        "pass": False,
        "score": 0,
        "reason": f"Invalid similarity score: {invalid}",
        "tokensUsed": USAGE,
    }


async def test_euclidean(embedder):
    near = await matches_similarity(
        "Expected output", "Sample output", 0.1, False, None, "euclidean"
    )
    assert near["pass"] is True
    assert near["reason"] == "Distance 0.00 is less than or equal to threshold 0.1"
    assert near["score"] == 1

    far = await matches_similarity(
        "Expected output", "Different output", 0.5, False, None, "euclidean"
    )
    assert far["pass"] is False
    assert far["score"] == 1 / (1 + math.sqrt(2))
    assert far["reason"] == "Distance 1.41 is greater than threshold 0.5"

    inverse = await matches_similarity(
        "Expected output", "Different output", 0.5, True, None, "euclidean"
    )
    assert inverse["pass"] is True
    assert inverse["score"] == 1 - 1 / (1 + math.sqrt(2))
    assert inverse["reason"] == "Distance 1.41 is greater than threshold 0.5"


@pytest.mark.parametrize(
    ("similarity", "shown"),
    [
        (math.nan, "NaN"),
        (math.inf, "Infinity"),
        ("0.9", "0.9"),
        ({"pass": True}, "[object Object]"),
    ],
)
async def test_rejects_malformed_native_similarity(monkeypatch, similarity, shown):
    use_defaults(monkeypatch)
    provider = NativeSimilarity({"similarity": similarity})
    result = await matches_similarity("expected", "output", 0.8, False, {"provider": provider})
    assert (result["pass"], result["score"]) == (False, 0)
    assert result["reason"] == f"Invalid similarity score: {shown}"


async def test_native_similarity_normalizes_token_usage(monkeypatch):
    use_defaults(monkeypatch)
    provider = NativeSimilarity(
        {"similarity": 0.9, "tokenUsage": {"total": 5, "prompt": 2, "completion": 3}}
    )
    result = await matches_similarity("expected", "output", 0.8, False, {"provider": provider})
    assert result["pass"] is True
    assert result["reason"] == "Similarity 0.90 is greater than or equal to threshold 0.8"
    assert result["tokensUsed"]["completionDetails"] == {
        "reasoning": 0,
        "acceptedPrediction": 0,
        "rejectedPrediction": 0,
    }


async def test_native_similarity_errors(monkeypatch):
    use_defaults(monkeypatch)
    errored = NativeSimilarity({"error": "boom"})
    assert (await matches_similarity("e", "o", 0.8, False, {"provider": errored}))[
        "reason"
    ] == "boom"
    missing = NativeSimilarity({})
    result = await matches_similarity("e", "o", 0.8, False, {"provider": missing})
    assert result["reason"] == "Unknown error fetching similarity"


async def test_native_similarity_rejects_other_metrics(monkeypatch):
    use_defaults(monkeypatch)
    provider = NativeSimilarity({"similarity": 0.9})
    result = await matches_similarity(
        "expected", "output", 0.8, False, {"provider": provider}, "dot_product"
    )
    assert result["pass"] is False
    assert result["reason"] == (
        "Provider native-similarity only supports cosine similarity via callSimilarityApi"
    )


async def test_hybrid_provider_uses_embeddings_for_other_metrics(monkeypatch):
    use_defaults(monkeypatch)
    provider = NativeSimilarity(
        {"similarity": 0.1},
        embed=lambda t: {"embedding": [1, 0] if t == "expected" else [0.5, 0], "tokenUsage": {}},
    )
    result = await matches_similarity(
        "expected", "output", 0.4, False, {"provider": provider}, "dot_product"
    )
    assert (result["pass"], result["score"]) == (True, 0.5)
    assert provider.similarity_calls == []
    assert sorted(provider.embed_calls) == ["expected", "output"]


async def test_embedding_errors(embedder):
    embedder.fn = lambda t: {"error": "embedding down"} if t == "b" else embedding_result(1, 0)
    assert (await matches_similarity("a", "b", 0.5))["reason"] == "embedding down"
    embedder.fn = lambda t: {"embedding": []}
    # An empty array is truthy in JS, so it reaches the score (cosine of zero-length vectors).
    assert (await matches_similarity("a", "b", 0.5))["score"] == 0
    embedder.fn = lambda t: {}
    assert (await matches_similarity("a", "b", 0.5))["reason"] == "Embedding not found"


async def test_rejects_providers_without_embedding_support(monkeypatch):
    use_defaults(monkeypatch)

    class Plain(ApiProvider):
        def default_id(self) -> str:
            return "plain"

    with pytest.raises(ValueError, match="Provider plain is not a valid embedding provider"):
        await matches_similarity("a", "b", 0.5, False, {"provider": Plain()})
