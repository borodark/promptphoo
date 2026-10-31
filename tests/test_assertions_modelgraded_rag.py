# ruff: noqa: E501
"""answer-relevance, context-*, similar, classifier, moderation and search-rubric handlers.

Ports the deterministic cases of upstream test/assertions/{answerRelevance,contextFaithfulness,
contextRecall,contextRelevance,contextUtils,ragInverse,similar,classifier,moderation,
searchRubric}.test.ts. Matchers are replaced by recording fakes where upstream mocks them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from rag_fakes import FakeTextAndEmbedding

from promptphoo.assertions import context_utils
from promptphoo.assertions.base import AssertionParams, coerce_string
from promptphoo.assertions.context_utils import resolve_context
from promptphoo.assertions.index import get_handlers, run_assertion
from promptphoo.assertions.modelgraded_rag import (
    handle_answer_relevance,
    handle_classifier,
    handle_context_faithfulness,
    handle_context_recall,
    handle_context_relevance,
    handle_moderation,
    handle_search_rubric,
    handle_similar,
)
from promptphoo.matchers import classification, moderation, rag, search, similarity
from promptphoo.transform import INLINE_FUNCTION_LABEL, INLINE_STRING_LABEL

CONTEXT_REQUIRED = (
    'Context is required for context-based assertions. Provide either a "context" variable '
    '(string or array of strings) in your test case or use "contextTransform" to extract '
    "context from the provider response."
)


def params(
    assertion: dict[str, Any],
    output: Any = "output",
    *,
    test: dict[str, Any] | None = None,
    prompt: str | None = "prompt",
    rendered: Any = None,
    provider_response: dict[str, Any] | None = None,
) -> AssertionParams:
    return AssertionParams(
        assertion=assertion,
        base_type=assertion["type"].removeprefix("not-"),
        inverse=assertion["type"].startswith("not-"),
        output=output,
        output_string=coerce_string(output),
        rendered_value=rendered if rendered is not None else assertion.get("value"),
        prompt=prompt,
        provider_response=provider_response
        if provider_response is not None
        else {"output": output},
        test=test if test is not None else {"vars": {}, "options": {}},
    )


class Recorder:
    """Replaces a matcher: records its positional arguments and returns queued results."""

    def __init__(self, *results: dict[str, Any]) -> None:
        self.results = list(results)
        self.calls: list[tuple[Any, ...]] = []

    async def __call__(self, *args: Any) -> dict[str, Any]:
        self.calls.append(args)
        result = self.results[min(len(self.calls) - 1, len(self.results) - 1)]
        return dict(result)


def mock(monkeypatch, module: Any, name: str, *results: dict[str, Any]) -> Recorder:
    recorder = Recorder(*results)
    monkeypatch.setattr(module, name, recorder)
    return recorder


def test_handlers_are_registered():
    handlers = get_handlers()
    for name in (
        "answer-relevance", "classifier", "context-faithfulness", "context-recall",
        "context-relevance", "moderation", "search-rubric", "similar", "similar:cosine",
        "similar:dot", "similar:euclidean",
    ):  # fmt: skip
        assert name in handlers


# ---- resolveContext ----------------------------------------------------------------------------


async def test_resolve_context_from_vars():
    test = {"vars": {"context": "test context"}, "options": {}}
    assert await resolve_context({"type": "context-faithfulness"}, test, "output", "prompt") == (
        "test context"
    )
    chunks = ["chunk 1", "chunk 2", "chunk 3"]
    test = {"vars": {"context": chunks}}
    assert await resolve_context({"type": "context-faithfulness"}, test, "o", "p") == chunks


async def test_resolve_context_rejects_non_string_chunks():
    test = {"vars": {"context": ["valid", 123, "invalid"]}}
    with pytest.raises(
        ValueError,
        match="Invariant failed: Invalid context: expected an array of strings, but found "
        "number at index 1",
    ):
        await resolve_context({"type": "context-faithfulness"}, test, "o", "p")


async def test_resolve_context_fallback():
    result = await resolve_context({"type": "context-recall"}, {"vars": {}}, "o", "p", "fallback")
    assert result == "fallback"


async def test_resolve_context_transform_uses_output_and_context(monkeypatch):
    seen: list[tuple[Any, ...]] = []

    async def fake_transform(code, value, context):
        seen.append((code, value, context))
        return "transformed context"

    monkeypatch.setattr(context_utils, "transform", fake_transform)
    assertion = {"type": "context-faithfulness", "contextTransform": "output.context"}
    test = {"vars": {"context": "original context"}}
    result = await resolve_context(assertion, test, {"context": "data"}, "prompt")
    assert result == "transformed context"
    assert seen == [
        (
            "output.context",
            {"context": "data"},
            {"vars": {"context": "original context"}, "prompt": {"label": "prompt"}},
        )
    ]
    seen.clear()
    response = {"providerTransformedOutput": {"context": "x"}, "metadata": {"m": 1}}
    await resolve_context(assertion, {"vars": {}}, "raw", "prompt", None, response)
    assert seen[0][1] == {"context": "x"}
    assert seen[0][2]["metadata"] == {"m": 1}


async def test_resolve_context_real_transforms():
    assertion = {"type": "context-faithfulness", "contextTransform": "output.chunks"}
    result = await resolve_context(assertion, {"vars": {}}, {"chunks": ["a", "b"]}, "p")
    assert result == ["a", "b"]
    fn_assertion = {"type": "context-faithfulness", "contextTransform": lambda o, c: o["ctx"]}
    assert await resolve_context(fn_assertion, {"vars": {}}, {"ctx": "c"}, "p") == "c"


@pytest.mark.parametrize(("value", "kind"), [(123, "number"), (["valid", 123], "object")])
async def test_resolve_context_transform_type_errors(monkeypatch, value, kind):
    async def fake_transform(code, v, context):
        return value

    monkeypatch.setattr(context_utils, "transform", fake_transform)
    assertion = {"type": "context-faithfulness", "contextTransform": "output.invalid"}
    with pytest.raises(ValueError) as e:
        await resolve_context(assertion, {"vars": {}}, "output", "prompt")
    label = f"{INLINE_STRING_LABEL}: output.invalid"
    assert str(e.value) == (
        f"Failed to transform context using expression '{label}': Invariant failed: "
        f"contextTransform must return a string or array of strings. Got {kind}. Check your "
        f"transform expression: {label}"
    )


async def test_resolve_context_transform_failure_unwraps_label():
    def boom(output, context):
        raise RuntimeError("Transform failed")

    assertion = {"type": "context-faithfulness", "contextTransform": boom}
    with pytest.raises(ValueError) as e:
        await resolve_context(assertion, {"vars": {}}, "output", "prompt")
    assert str(e.value) == (
        f"Failed to transform context using expression '{INLINE_FUNCTION_LABEL}: boom': "
        "Transform failed"
    )


@pytest.mark.parametrize("context", [None, "", [], ["valid chunk", "", "another chunk"]])
async def test_resolve_context_requires_context(context):
    test = {"vars": {} if context is None else {"context": context}}
    with pytest.raises(ValueError, match="Invariant failed: Context is required"):
        await resolve_context({"type": "context-faithfulness"}, test, "o", "p")


@pytest.mark.parametrize("value", ["", []])
async def test_resolve_context_transform_must_be_nonempty(monkeypatch, value):
    async def fake_transform(code, v, context):
        return value

    monkeypatch.setattr(context_utils, "transform", fake_transform)
    assertion = {"type": "context-faithfulness", "contextTransform": "output.empty"}
    with pytest.raises(ValueError) as e:
        await resolve_context(assertion, {"vars": {}}, "o", "p")
    assert str(e.value) == f"Invariant failed: {CONTEXT_REQUIRED}"


# ---- answer-relevance --------------------------------------------------------------------------

RESULT = {"pass": True, "score": 0.8, "reason": "test reason"}


async def test_answer_relevance_handler(monkeypatch):
    rec = mock(monkeypatch, rag, "matches_answer_relevance", RESULT)
    p = params({"type": "answer-relevance", "threshold": 0.7}, "test output", prompt="test prompt")
    assert await handle_answer_relevance(p) == {"assertion": p.assertion, **RESULT}
    assert rec.calls == [("test prompt", "test output", 0.7, {}, None)]

    p = params(
        {"type": "answer-relevance"},
        "o",
        test={"vars": {"query": "vars query"}, "options": {"provider": "x"}},
    )
    await handle_answer_relevance(p)
    assert rec.calls[1] == ("vars query", "o", 0.5, {"provider": "x"}, None)


async def test_answer_relevance_handler_invariants(monkeypatch):
    mock(monkeypatch, rag, "matches_answer_relevance", RESULT)
    with pytest.raises(
        ValueError, match="answer-relevance assertion type must evaluate a string output"
    ):
        await handle_answer_relevance(params({"type": "answer-relevance"}, {"not": "string"}))
    with pytest.raises(ValueError, match="answer-relevance assertion type must have a prompt"):
        await handle_answer_relevance(params({"type": "answer-relevance"}, prompt=None))


async def test_answer_relevance_handler_inverse(monkeypatch):
    low = {"pass": False, "score": 0.4, "reason": "Relevance 0.40 is less than threshold 0.5"}
    mock(monkeypatch, rag, "matches_answer_relevance", low)
    p = params({"type": "not-answer-relevance"})
    result = await handle_answer_relevance(p)
    assert result == {**low, "assertion": p.assertion, "pass": True, "score": 0.6}


def _inverse_provider(vector: list[float], response: dict[str, Any] | None = None):
    usage = {"total": 1, "prompt": 1, "completion": 0}
    return FakeTextAndEmbedding(
        {**(response or {"output": "Generated question"}), "tokenUsage": usage},
        lambda text: {
            "embedding": [1, 0] if text == "Original question" else vector,
            "tokenUsage": usage,
        },
    )


async def grade_relevance(inverse: bool, vector: list[float], response: Any = None):
    provider = _inverse_provider(vector, response)
    return await run_assertion(
        assertion={"type": "not-answer-relevance" if inverse else "answer-relevance"},
        prompt="Original question",
        provider_response={"output": "Unrelated answer"},
        test={"options": {"provider": provider}},
    )


@pytest.mark.parametrize(
    ("vector", "positive_score", "inverse_score", "positive_pass"),
    [
        ([-1, 0], -1, 1, False),
        ([0, 1], 0, 1, False),
        ([1, 0], 1, 0, True),
        ([3, 4], 0.6, 0.4, True),
    ],
)
async def test_answer_relevance_inverse_scoring(
    vector, positive_score, inverse_score, positive_pass
):
    """ragInverse.test.ts: the inverted score is bounded to [0, 1]."""
    positive = await grade_relevance(False, vector)
    inverse = await grade_relevance(True, vector)
    assert positive["pass"] is positive_pass
    assert positive["score"] == pytest.approx(positive_score)
    assert inverse["pass"] is (not positive_pass)
    assert inverse["score"] == pytest.approx(inverse_score)
    assert inverse["assertion"] == {"type": "not-answer-relevance"}
    assert inverse["reason"] == positive["reason"]
    assert inverse["metadata"] == positive["metadata"]
    usage = inverse["tokensUsed"]
    assert (usage["total"], usage["prompt"], usage["completion"]) == (7, 7, 0)
    assert inverse["metadata"]["averageSimilarity"] == pytest.approx(positive_score)
    assert inverse["metadata"]["threshold"] == 0.5
    assert len(inverse["metadata"]["generatedQuestions"]) == 3


async def test_answer_relevance_inverse_keeps_grader_errors():
    result = await grade_relevance(True, [-1, 0], {"error": "Grader unavailable"})
    assert (result["pass"], result["score"], result["reason"]) == (False, 0, "Grader unavailable")
    assert result["metadata"] == {"graderError": True}
    usage = result["tokensUsed"]
    assert (usage["total"], usage["prompt"], usage["completion"]) == (1, 1, 0)


# ---- context-faithfulness ----------------------------------------------------------------------

FAITH_TEST = {"vars": {"query": "test query", "context": "test context"}, "options": {}}


async def test_context_faithfulness_handler(monkeypatch):
    mock_result = {"pass": True, "score": 0.9, "reason": "Content is faithful to context"}
    rec = mock(monkeypatch, rag, "matches_context_faithfulness", mock_result)
    p = params({"type": "context-faithfulness", "threshold": 0.7}, "test output", test=FAITH_TEST)
    assert await handle_context_faithfulness(p) == {
        "assertion": p.assertion,
        **mock_result,
        "metadata": {"context": "test context"},
    }
    assert rec.calls == [
        ("test query", "test output", "test context", 0.7, {}, FAITH_TEST["vars"], None)
    ]


async def test_context_faithfulness_handler_invariants(monkeypatch):
    mock(monkeypatch, rag, "matches_context_faithfulness", RESULT)
    a = {"type": "context-faithfulness"}
    with pytest.raises(ValueError, match="requires a test with variables"):
        await handle_context_faithfulness(params(a, test={"options": {}}))
    with pytest.raises(ValueError, match='requires a "query" variable with the user question'):
        await handle_context_faithfulness(params(a, test={"vars": {"query": 1, "context": "c"}}))
    with pytest.raises(ValueError, match="requires string output from the provider"):
        await handle_context_faithfulness(params(a, {"x": 1}, test=FAITH_TEST))


async def test_context_faithfulness_uses_context_transform(monkeypatch):
    rec = mock(
        monkeypatch, rag, "matches_context_faithfulness", {"pass": True, "score": 1, "reason": "ok"}
    )
    a = {"type": "context-faithfulness", "contextTransform": "output.split('|')[1]"}
    p = params(
        a,
        "answer|retrieved context",
        test={"vars": {"query": "q"}, "options": {}},
        provider_response={"output": "answer|retrieved context"},
    )
    result = await handle_context_faithfulness(p)
    assert rec.calls[0][2] == "retrieved context"
    assert result["metadata"] == {"context": "retrieved context"}


async def test_context_faithfulness_inverse(monkeypatch):
    low = {"pass": False, "score": 0.4, "reason": "Faithfulness 0.40 is < 0.5"}
    mock(monkeypatch, rag, "matches_context_faithfulness", low)
    result = await handle_context_faithfulness(
        params({"type": "not-context-faithfulness"}, test=FAITH_TEST)
    )
    assert (result["pass"], result["score"]) == (True, 0.6)
    assert result["metadata"] == {"context": "test context"}

    high = {"pass": True, "score": 0.9, "reason": "Faithfulness 0.90 is >= 0.5"}
    mock(monkeypatch, rag, "matches_context_faithfulness", high)
    result = await handle_context_faithfulness(
        params({"type": "not-context-faithfulness"}, test=FAITH_TEST)
    )
    assert result["pass"] is False
    assert result["score"] == pytest.approx(0.1)

    err = {
        "pass": False,
        "score": 0,
        "reason": "grading provider failed",
        "metadata": {"graderError": True},
    }
    mock(monkeypatch, rag, "matches_context_faithfulness", err)
    result = await handle_context_faithfulness(
        params({"type": "not-context-faithfulness"}, test=FAITH_TEST)
    )
    assert (result["pass"], result["score"], result["reason"]) == (
        False,
        0,
        "grading provider failed",
    )


# ---- context-recall ----------------------------------------------------------------------------


async def test_context_recall_handler(monkeypatch):
    matcher = {
        "pass": True,
        "score": 0.9,
        "reason": "Recall 0.90 is >= 0.8",
        "metadata": {"sentenceAttributions": [{"sentence": "s", "attributed": True}]},
    }
    rec = mock(monkeypatch, rag, "matches_context_recall", matcher)
    test = {"vars": {"context": "test context"}, "options": {}}
    p = params({"type": "context-recall", "threshold": 0.8}, rendered="expected", test=test)
    result = await handle_context_recall(p)
    assert result["metadata"] == {
        "context": "test context",
        "sentenceAttributions": [{"sentence": "s", "attributed": True}],
    }
    assert rec.calls == [("test context", "expected", 0.8, {}, test["vars"], None)]


async def test_context_recall_handler_falls_back_to_prompt(monkeypatch):
    rec = mock(monkeypatch, rag, "matches_context_recall", RESULT)
    p = params(
        {"type": "context-recall"},
        rendered="val",
        prompt="the prompt",
        test={"vars": {}, "options": {}},
    )
    await handle_context_recall(p)
    assert rec.calls == [("the prompt", "val", 0.5, {}, {}, None)]


async def test_context_recall_handler_invariants(monkeypatch):
    mock(monkeypatch, rag, "matches_context_recall", RESULT)
    with pytest.raises(
        ValueError,
        match=r"context-recall assertion requires a string value \(expected answer or fact to verify\)",
    ):
        await handle_context_recall(params({"type": "context-recall", "value": 5}))
    with pytest.raises(ValueError, match="context-recall assertion requires a prompt"):
        await handle_context_recall(params({"type": "context-recall", "value": "v"}, prompt=None))


async def test_context_recall_handler_inverse(monkeypatch):
    test = {"vars": {"context": "test context"}, "options": {}}
    mock(monkeypatch, rag, "matches_context_recall", {"pass": True, "score": 0.7, "reason": "r"})
    result = await handle_context_recall(
        params({"type": "not-context-recall", "value": "v"}, test=test)
    )
    assert result["pass"] is False
    assert result["score"] == pytest.approx(0.3)
    err = {"pass": False, "score": 0, "reason": "boom", "metadata": {"graderError": True}}
    mock(monkeypatch, rag, "matches_context_recall", err)
    result = await handle_context_recall(
        params({"type": "not-context-recall", "value": "v"}, test=test)
    )
    assert (result["pass"], result["score"]) == (False, 0)
    assert result["metadata"] == {"graderError": True, "context": "test context"}


# ---- context-relevance -------------------------------------------------------------------------


async def test_context_relevance_handler(monkeypatch):
    matcher = {"pass": True, "score": 0.9, "reason": "r", "metadata": {"extractedSentences": ["s"]}}
    rec = mock(monkeypatch, rag, "matches_context_relevance", matcher)
    test = {"vars": {"query": "q", "context": "test context"}, "options": {}}
    result = await handle_context_relevance(params({"type": "context-relevance"}, test=test))
    assert result["metadata"] == {"context": "test context", "extractedSentences": ["s"]}
    assert rec.calls == [("q", "test context", 0.5, {}, None)]


async def test_context_relevance_handler_invariants(monkeypatch):
    mock(monkeypatch, rag, "matches_context_relevance", RESULT)
    a = {"type": "context-relevance"}
    with pytest.raises(
        ValueError, match="context-relevance assertion requires a test with variables"
    ):
        await handle_context_relevance(params(a, test={"options": {}}))
    with pytest.raises(
        ValueError,
        match='context-relevance assertion requires a "query" variable with the user question',
    ):
        await handle_context_relevance(params(a, test={"vars": {"context": "c"}}))


async def test_context_relevance_handler_inverse(monkeypatch):
    test = {"vars": {"query": "q", "context": "test context"}, "options": {}}
    mock(monkeypatch, rag, "matches_context_relevance", {"pass": True, "score": 0.5, "reason": "r"})
    result = await handle_context_relevance(params({"type": "not-context-relevance"}, test=test))
    assert (result["pass"], result["score"]) == (False, 0.5)
    err = {"pass": False, "score": 0, "reason": "x", "metadata": {"graderError": True}}
    mock(monkeypatch, rag, "matches_context_relevance", err)
    result = await handle_context_relevance(params({"type": "not-context-relevance"}, test=test))
    assert result["pass"] is False
    assert result["metadata"] == {"graderError": True, "context": "test context"}


# ---- similar -----------------------------------------------------------------------------------


async def fake_similarity(expected, output, threshold, inverse, grading, metric, ctx):
    same = expected == output
    if inverse:
        return {"pass": not same, "score": 0 if same else 1, "reason": "inverse"}
    return {"pass": same, "score": 1 if same else 0, "reason": "match" if same else "no match"}


@pytest.fixture
def similar_calls(monkeypatch):
    calls: list[tuple[Any, ...]] = []

    async def recording(*args):
        calls.append(args)
        return await fake_similarity(*args)

    monkeypatch.setattr(similarity, "matches_similarity", recording)
    return calls


async def test_similar_string_and_array(similar_calls):
    result = await handle_similar(
        params({"type": "similar", "value": "hello world"}, "hello world")
    )
    assert (result["pass"], result["score"]) == (True, 1)
    result = await handle_similar(
        params({"type": "similar", "value": ["hello world", "hi world"]}, "hi world")
    )
    assert (result["pass"], result["score"]) == (True, 1)
    assert similar_calls[0] == ("hello world", "hello world", 0.75, False, {}, "cosine", None)


async def test_similar_custom_threshold_and_metric(similar_calls):
    await handle_similar(params({"type": "similar", "value": "a", "threshold": 0.5}, "a"))
    await handle_similar(params({"type": "similar:dot", "value": "a"}, "a"))
    await handle_similar(params({"type": "similar:euclidean", "value": "a"}, "a"))
    await handle_similar(params({"type": "similar:cosine", "value": "a"}, "a"))
    assert [(c[2], c[5]) for c in similar_calls] == [
        (0.5, "cosine"),
        (0.75, "dot_product"),
        (0.75, "euclidean"),
        (0.75, "cosine"),
    ]


async def test_similar_unknown_metric(similar_calls):
    with pytest.raises(ValueError, match="Unknown similarity metric: manhattan"):
        await handle_similar(params({"type": "similar:manhattan", "value": "a"}, "a"))


async def test_similar_inverse(similar_calls):
    result = await handle_similar(params({"type": "not-similar", "value": "hello"}, "goodbye"))
    assert (result["pass"], result["score"]) == (True, 1)


async def test_similar_array_none_pass(monkeypatch):
    mock(
        monkeypatch,
        similarity,
        "matches_similarity",
        {"pass": False, "score": 0.2},
        {"pass": False, "score": 0.6},
    )
    p = params({"type": "similar", "value": ["far value", "closer value"], "threshold": 0.9}, "x")
    assert await handle_similar(p) == {
        "assertion": p.assertion,
        "pass": False,
        "score": 0.6,
        "reason": "None of the provided values met the similarity threshold",
    }


async def test_not_similar_array_fails_on_any_match(monkeypatch):
    rec = mock(
        monkeypatch,
        similarity,
        "matches_similarity",
        {"pass": False, "score": 0.05, "reason": "too similar"},
        {"pass": True, "score": 0.9, "reason": "dissimilar"},
    )
    p = params(
        {"type": "not-similar", "value": ["forbidden A", "unrelated B"], "threshold": 0.75}, "x"
    )
    result = await handle_similar(p)
    assert (result["pass"], result["score"], result["reason"]) == (False, 0.05, "too similar")
    assert len(rec.calls) == 1


async def test_not_similar_array_reports_closest(monkeypatch):
    mock(
        monkeypatch,
        similarity,
        "matches_similarity",
        {"pass": True, "score": 0.9, "reason": "dissimilar to A"},
        {"pass": True, "score": 0.78, "reason": "closest to B"},
        {"pass": True, "score": 0.95, "reason": "dissimilar to C"},
    )
    p = params({"type": "not-similar", "value": ["A", "B", "C"], "threshold": 0.75}, "x")
    result = await handle_similar(p)
    assert (result["pass"], result["score"], result["reason"]) == (True, 0.78, "closest to B")


async def test_similar_invariants(similar_calls):
    with pytest.raises(
        ValueError, match="Similarity assertion type must have a string or array of strings value"
    ):
        await handle_similar(params({"type": "similar", "value": "test"}, rendered=5))
    with pytest.raises(
        ValueError, match="Similarity assertion must have at least one value to compare against"
    ):
        await handle_similar(params({"type": "similar", "value": []}, rendered=[]))


# ---- classifier --------------------------------------------------------------------------------


async def test_classifier_handler(monkeypatch):
    rec = mock(
        monkeypatch,
        classification,
        "matches_classification",
        {"pass": True, "score": 0.8, "reason": "ok"},
    )
    test = {"options": {"provider": "huggingface:text-classification:x"}}
    p = params({"type": "classifier", "value": "positive", "threshold": 0.6}, "great", test=test)
    assert await handle_classifier(p) == {
        "assertion": p.assertion,
        "pass": True,
        "score": 0.8,
        "reason": "ok",
    }
    assert rec.calls == [("positive", "great", 0.6, test["options"])]


async def test_classifier_default_threshold_and_inverse(monkeypatch):
    rec = mock(
        monkeypatch,
        classification,
        "matches_classification",
        {"pass": True, "score": 0.75, "reason": "r"},
    )
    p = params({"type": "not-classifier"}, "text")
    result = await handle_classifier(p)
    assert (result["pass"], result["score"]) == (False, 0.25)
    assert rec.calls == [(None, "text", 1, {})]


async def test_classifier_rejects_non_string_values(monkeypatch):
    mock(monkeypatch, classification, "matches_classification", RESULT)
    with pytest.raises(
        ValueError, match='"classifier" assertion type must have a string value or be undefined'
    ):
        await handle_classifier(params({"type": "classifier", "value": 5}))


async def test_classifier_grader_failure_is_not_inverted(monkeypatch):
    err = {"pass": False, "score": 0, "reason": "Unknown error", "metadata": {"graderError": True}}
    mock(monkeypatch, classification, "matches_classification", err)
    p = params({"type": "not-classifier"})
    assert await handle_classifier(p) == {"assertion": p.assertion, **err}


async def test_classifier_end_to_end_with_any_classification_provider():
    from promptphoo.providers.base import ApiProvider

    class Sentiment(ApiProvider):
        def default_id(self) -> str:
            return "sentiment"

        async def call_classification_api(self, text):
            return {"classification": {"positive": 0.9, "negative": 0.1}}

    result = await run_assertion(
        assertion={
            "type": "classifier",
            "value": "positive",
            "threshold": 0.5,
            "provider": Sentiment(),
        },
        test={},
        provider_response={"output": "I love it"},
    )
    assert (result["pass"], result["score"]) == (True, 0.9)
    assert result["reason"] == "Classification positive has score 0.90 >= 0.5"


# ---- moderation --------------------------------------------------------------------------------

FLAGGED = {"pass": False, "score": 0, "reason": "Moderation flags detected: harassment"}
CLEAN = {"pass": True, "score": 1, "reason": "No moderation flags detected"}
USAGE = {"total": 5, "prompt": 2, "completion": 3, "cached": 0, "numRequests": 0}


def moderation_params(assertion_type: str = "moderation", **kw: Any) -> AssertionParams:
    return params(
        {"type": assertion_type, "value": ["harassment"]}, kw.pop("output", "output"), **kw
    )


@pytest.mark.parametrize(
    ("assertion_type", "matcher", "passed", "score"),
    [
        ("moderation", CLEAN, True, 1),
        ("moderation", FLAGGED, False, 0),
        ("not-moderation", FLAGGED, True, 1),
        ("not-moderation", CLEAN, False, 0),
    ],
)
async def test_moderation_handler_verdicts(monkeypatch, assertion_type, matcher, passed, score):
    mock(monkeypatch, moderation, "matches_moderation", {**matcher, "tokensUsed": USAGE})
    p = moderation_params(assertion_type)
    result = await handle_moderation(p)
    assert result == {
        **matcher,
        "tokensUsed": USAGE,
        "pass": passed,
        "score": score,
        "assertion": p.assertion,
    }


async def test_moderation_handler_keeps_api_errors(monkeypatch):
    err = {
        "pass": False,
        "score": 0,
        "reason": "Moderation API error: provider unavailable",
        "metadata": {"graderError": True},
    }
    mock(monkeypatch, moderation, "matches_moderation", err)
    p = moderation_params("not-moderation")
    assert await handle_moderation(p) == {**err, "assertion": p.assertion}


@pytest.mark.parametrize(
    ("response", "prompt", "expected"),
    [
        ({"output": "o", "metadata": {"redteamFinalPrompt": "modified prompt"}}, "prompt", "modified prompt"),
        ({"output": "o", "prompt": "provider-generated prompt", "metadata": {"redteamFinalPrompt": "r"}}, "original prompt", "provider-generated prompt"),
        ({"output": "o", "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "Hello world"}]}, "original prompt", "Hello world"),
        ({"output": "o", "metadata": {"redteamFinalPrompt": "redteam prompt"}}, "original prompt", "redteam prompt"),
        ({"output": "o"}, "original prompt", "original prompt"),
        ({"output": "o", "prompt": ""}, "original prompt", "original prompt"),
        ({"output": "o", "prompt": []}, "original prompt", "original prompt"),
        ({"output": "o"}, json.dumps([{"role": "system", "content": "Ignore this system message"}, {"role": "user", "content": "Moderate this user request"}, {"role": "assistant", "content": "Ignore this assistant reply"}]), "Moderate this user request"),
        ({"output": "o"}, json.dumps([{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "input_text", "text": "Moderate this multimodal user request"}, {"type": "input_image", "image_url": "https://example.test/image.png"}]}, {"role": "assistant", "content": "a"}]), "Moderate this multimodal user request"),
        ({"output": "o"}, "- role: system\n  content: Ignore this system message\n- role: user\n  content: Moderate this YAML user request\n- role: assistant\n  content: Ignore this assistant reply", "Moderate this YAML user request"),
    ],
)  # fmt: skip
async def test_moderation_prompt_selection(monkeypatch, response, prompt, expected):
    rec = mock(monkeypatch, moderation, "matches_moderation", CLEAN)
    await handle_moderation(moderation_params(prompt=prompt, provider_response=response))
    assert rec.calls == [(expected, "output", ["harassment"], {})]


async def test_moderation_value_must_be_string_array(monkeypatch):
    mock(monkeypatch, moderation, "matches_moderation", CLEAN)
    with pytest.raises(
        ValueError, match="moderation assertion value must be a string array if set"
    ):
        await handle_moderation(params({"type": "moderation", "value": "harassment"}))
    with pytest.raises(ValueError, match="moderation assertion type must have a prompt"):
        await handle_moderation(params({"type": "moderation"}, prompt=None))


# ---- search-rubric -----------------------------------------------------------------------------


async def test_search_rubric_handler(monkeypatch):
    expected = {
        "pass": True,
        "score": 1,
        "reason": "The output correctly states the weather in Tokyo",
    }
    rec = mock(monkeypatch, search, "matches_search_rubric", expected)
    test = {
        "vars": {"city": "Tokyo"},
        "options": {"provider": "anthropic:messages:claude-opus-4-6"},
    }
    p = params(
        {"type": "search-rubric", "value": "Weather in {{city}}"},
        "It is sunny in Tokyo",
        test=test,
        rendered="Weather in Tokyo",
        provider_response={"output": "It is sunny in Tokyo"},
    )
    assert await handle_search_rubric(p) == expected
    assert rec.calls == [
        (
            "Weather in Tokyo",
            "It is sunny in Tokyo",
            test["options"],
            test["vars"],
            p.assertion,
            None,
            None,
        )
    ]


async def test_search_rubric_handler_requires_string(monkeypatch):
    with pytest.raises(ValueError, match="search-rubric assertion type must have a string value"):
        await handle_search_rubric(params({"type": "search-rubric"}, rendered=None))


@pytest.mark.parametrize(
    ("matcher", "passed", "score", "reason"),
    [
        ({"pass": True, "score": 0.8, "reason": "Information is current"}, False, 0.2, "Output requires web search verification: Information is current"),
        ({"pass": False, "score": 0.1, "reason": "Information is outdated"}, True, 0.9, "Output does not require web search verification: Information is outdated"),
        ({"pass": True, "score": 5, "reason": "matched"}, False, 0, "Output requires web search verification: matched"),
        ({"pass": False, "score": float("nan"), "reason": "malformed"}, True, 1, "Output does not require web search verification: malformed"),
    ],
)  # fmt: skip
async def test_search_rubric_handler_inverse(monkeypatch, matcher, passed, score, reason):
    mock(monkeypatch, search, "matches_search_rubric", matcher)
    result = await handle_search_rubric(params({"type": "not-search-rubric", "value": "v"}))
    assert result["pass"] is passed
    assert result["score"] == pytest.approx(score)
    assert result["reason"] == reason


async def test_search_rubric_handler_keeps_grader_failures(monkeypatch):
    err = {
        "pass": False,
        "score": 0,
        "reason": "Search rubric evaluation failed: search unavailable",
        "metadata": {"graderError": True},
    }
    mock(monkeypatch, search, "matches_search_rubric", err)
    assert await handle_search_rubric(params({"type": "not-search-rubric", "value": "v"})) == err
