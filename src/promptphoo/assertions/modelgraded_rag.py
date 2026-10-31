"""RAG, similarity, classifier, moderation and search-rubric assertions.

Ports src/assertions/{answerRelevance,contextFaithfulness,contextRecall,contextRelevance,similar,
classifier,moderation,searchRubric}.ts. Context resolution is in context_utils.py and the
negation helpers in rag_defaults.py.
"""

from __future__ import annotations

from typing import Any

from promptphoo.assertions.base import AssertionParams, GradingResult, Handler
from promptphoo.assertions.context_utils import invariant, resolve_context
from promptphoo.assertions.rag_defaults import (
    DEFAULT_RAG_ASSERTION_THRESHOLD,
    apply_rag_inverse,
    is_grader_failure,
)
from promptphoo.matchers.shared import invert_score


def _threshold(assertion: dict[str, Any], default: float) -> Any:
    threshold = assertion.get("threshold")
    return default if threshold is None else threshold


def _vars(params: AssertionParams) -> dict[str, Any]:
    return params.test.get("vars") or {}


async def handle_answer_relevance(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.rag import matches_answer_relevance

    invariant(
        isinstance(params.output, str),
        "answer-relevance assertion type must evaluate a string output",
    )
    invariant(params.prompt, "answer-relevance assertion type must have a prompt")
    query = _vars(params).get("query")
    input_ = query if isinstance(query, str) else (params.prompt or "")
    result = await matches_answer_relevance(
        input_,
        params.output,
        _threshold(params.assertion, DEFAULT_RAG_ASSERTION_THRESHOLD),
        params.test.get("options"),
        params.provider_call_context,
    )
    return {"assertion": params.assertion, **apply_rag_inverse(result, params.inverse)}


async def handle_context_faithfulness(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.rag import matches_context_faithfulness

    test = params.test
    invariant(
        test.get("vars") is not None,
        "context-faithfulness assertion requires a test with variables",
    )
    invariant(
        isinstance(test["vars"].get("query"), str),
        'context-faithfulness assertion requires a "query" variable with the user question',
    )
    invariant(
        isinstance(params.output, str),
        "context-faithfulness assertion requires string output from the provider",
    )
    context = await resolve_context(
        params.assertion, test, params.output, params.prompt, None, params.provider_response
    )
    result = await matches_context_faithfulness(
        test["vars"]["query"],
        params.output,
        context,
        _threshold(params.assertion, DEFAULT_RAG_ASSERTION_THRESHOLD),
        test.get("options"),
        test.get("vars"),
        params.provider_call_context,
    )
    return {
        "assertion": params.assertion,
        **apply_rag_inverse(result, params.inverse),
        "metadata": {"context": context},
    }


async def handle_context_recall(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.rag import matches_context_recall

    invariant(
        isinstance(params.rendered_value, str),
        "context-recall assertion requires a string value (expected answer or fact to verify)",
    )
    invariant(params.prompt, "context-recall assertion requires a prompt")
    test = params.test
    context = await resolve_context(
        params.assertion,
        test,
        params.output,
        params.prompt,
        params.prompt,
        params.provider_response,
    )
    # RAGAS context-recall checks whether the ground truth can be attributed to the context.
    result = await matches_context_recall(
        context,
        params.rendered_value,
        _threshold(params.assertion, DEFAULT_RAG_ASSERTION_THRESHOLD),
        test.get("options"),
        test.get("vars"),
        params.provider_call_context,
    )
    return {
        "assertion": params.assertion,
        **apply_rag_inverse(result, params.inverse),
        "metadata": {"context": context, **(result.get("metadata") or {})},
    }


async def handle_context_relevance(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.rag import matches_context_relevance

    test = params.test
    invariant(
        test.get("vars") is not None, "context-relevance assertion requires a test with variables"
    )
    invariant(
        isinstance(test["vars"].get("query"), str),
        'context-relevance assertion requires a "query" variable with the user question',
    )
    context = await resolve_context(
        params.assertion, test, params.output, params.prompt, None, params.provider_response
    )
    result = await matches_context_relevance(
        test["vars"]["query"],
        context,
        _threshold(params.assertion, DEFAULT_RAG_ASSERTION_THRESHOLD),
        test.get("options"),
        params.provider_call_context,
    )
    return {
        "assertion": params.assertion,
        **apply_rag_inverse(result, params.inverse),
        "metadata": {"context": context, **(result.get("metadata") or {})},
    }


_SIMILARITY_METRICS = {"cosine": "cosine", "dot": "dot_product", "euclidean": "euclidean"}


async def handle_similar(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.similarity import matches_similarity

    rendered = params.rendered_value
    assertion = params.assertion
    inverse = params.inverse
    invariant(
        isinstance(rendered, str | list),
        "Similarity assertion type must have a string or array of strings value",
    )
    invariant(
        not isinstance(rendered, list) or len(rendered) > 0,
        "Similarity assertion must have at least one value to compare against",
    )
    threshold = _threshold(assertion, 0.75)

    # Parse the metric from the assertion type (e.g. 'similar:dot' -> 'dot_product').
    metric = "cosine"
    if ":" in assertion["type"]:
        suffix = assertion["type"].split(":")[1]
        if suffix not in _SIMILARITY_METRICS:
            raise ValueError(f"Unknown similarity metric: {suffix}")
        metric = _SIMILARITY_METRICS[suffix]

    async def match(value: Any) -> dict[str, Any]:
        return await matches_similarity(
            value,
            params.output_string,
            threshold,
            inverse,
            params.test.get("options"),
            metric,
            params.provider_call_context,
        )

    if not isinstance(rendered, list):
        return {"assertion": assertion, **(await match(rendered))}

    # `not-similar` passes only when the output is dissimilar to every value: fail on the first
    # value it is too similar to; otherwise report the closest (lowest-scoring) value. For the
    # positive case, pass on the first match; otherwise report the best score.
    closest: dict[str, Any] | None = None
    max_score: float = 0 if inverse else float("-inf")
    for value in rendered:
        result = await match(value)
        if (not result["pass"]) if inverse else result["pass"]:
            return {"assertion": assertion, **result}
        if inverse and (closest is None or result["score"] < closest["score"]):
            closest = result
        elif not inverse and result["score"] > max_score:
            max_score = result["score"]
    if inverse:
        invariant(closest, "Similarity result missing")
        assert closest is not None
        return {"assertion": assertion, **closest}
    return {
        "assertion": assertion,
        "pass": False,
        "score": max_score,
        "reason": "None of the provided values met the similarity threshold",
    }


async def handle_classifier(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.classification import matches_classification

    invariant(
        isinstance(params.rendered_value, str) or params.rendered_value is None,
        '"classifier" assertion type must have a string value or be undefined',
    )
    result = await matches_classification(
        params.rendered_value,
        params.output_string,
        _threshold(params.assertion, 1),
        params.test.get("options"),
    )
    if is_grader_failure(result):
        # A broken grader is not evidence either way; never invert it into a pass.
        return {"assertion": params.assertion, **result}
    if params.inverse:
        result["pass"] = not result["pass"]
        result["score"] = 1 - result["score"]
    return {"assertion": params.assertion, **result}


def _get_moderation_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p for p in (_get_moderation_text(part) for part in content) if p is not None]
        return "\n".join(parts) if parts else None
    if isinstance(content, dict):
        text = content.get("text")
        return _get_moderation_text(text if text is not None else content.get("content"))
    return None


def _get_last_moderation_prompt(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            prompt = _get_moderation_text(message.get("content"))
            if prompt is not None:
                return prompt
    for message in reversed(messages):
        content = message.get("content") if isinstance(message, dict) else None
        prompt = _get_moderation_text(content)
        if prompt is not None:
            return prompt
    return None


async def handle_moderation(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.moderation import matches_moderation
    from promptphoo.output.results_table import get_actual_prompt
    from promptphoo.providers.shared import parse_chat_prompt

    assertion = params.assertion
    # Priority: the provider-reported prompt, then redteamFinalPrompt, then the original prompt.
    prompt_to_moderate = get_actual_prompt(params.provider_response) or (params.prompt or "")
    invariant(prompt_to_moderate, "moderation assertion type must have a prompt")
    value = assertion.get("value")
    invariant(
        not value or (isinstance(value, list) and value and isinstance(value[0], str)),
        "moderation assertion value must be a string array if set",
    )
    # Use the last user message of a serialized chat prompt (JSON or YAML).
    try:
        parsed = parse_chat_prompt(prompt_to_moderate, None)
        if isinstance(parsed, list) and parsed:
            last = _get_last_moderation_prompt(parsed)
            prompt_to_moderate = last if last is not None else prompt_to_moderate
    except Exception:
        pass

    result = await matches_moderation(
        prompt_to_moderate,
        params.output_string,
        value if isinstance(value, list) else [],
        params.test.get("options"),
    )
    if is_grader_failure(result):
        return {**result, "assertion": assertion}
    inverse = params.inverse
    return {
        **result,
        "pass": (not result["pass"]) if inverse else result["pass"],
        "score": 1 - result["score"] if inverse else result["score"],
        "assertion": assertion,
    }


async def handle_search_rubric(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.search import matches_search_rubric

    if not isinstance(params.rendered_value, str):
        raise ValueError("search-rubric assertion type must have a string value")
    result = await matches_search_rubric(
        params.rendered_value,
        params.provider_response.get("output"),
        params.test.get("options"),
        params.test.get("vars"),
        params.assertion,
        params.provider,
        params.provider_call_context,
    )
    if is_grader_failure(result):
        return result
    if params.inverse:
        passed = not result["pass"]
        prefix = (
            "Output does not require web search verification"
            if passed
            else "Output requires web search verification"
        )
        return {
            **result,
            "pass": passed,
            "score": invert_score(result["score"]),
            "reason": f"{prefix}: {result['reason']}",
        }
    return result


HANDLERS: dict[str, Handler] = {
    "answer-relevance": handle_answer_relevance,
    "classifier": handle_classifier,
    "context-faithfulness": handle_context_faithfulness,
    "context-recall": handle_context_recall,
    "context-relevance": handle_context_relevance,
    "moderation": handle_moderation,
    "search-rubric": handle_search_rubric,
    "similar": handle_similar,
    "similar:cosine": handle_similar,
    "similar:dot": handle_similar,
    "similar:euclidean": handle_similar,
}
