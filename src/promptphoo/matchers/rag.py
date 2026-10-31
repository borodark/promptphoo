"""RAG matchers (RAGAS-style; the prompts are RAGAS's, Apache-2.0, see NOTICE).

Ports src/matchers/rag.ts: matchesAnswerRelevance, matchesContextRecall,
matchesContextRelevance and matchesContextFaithfulness."""

from __future__ import annotations

import re
import sys
from typing import Any

from promptphoo.assertions.context_utils import invariant, serialize_context
from promptphoo.cache import with_cache_namespace
from promptphoo.jsvalues import is_truthy, js_string, to_fixed
from promptphoo.matchers.providers import (
    call_embedding_provider,
    call_grading_provider,
    call_provider_with_context,
    get_and_check_provider,
)
from promptphoo.matchers.rubric import load_rubric_prompt, render_llm_rubric_prompt
from promptphoo.matchers.shared import (
    cosine_similarity,
    grader_fail,
    normalize_matcher_token_usage,
    split_into_sentences,
    split_text_into_sentences,
    try_parse,
)
from promptphoo.prompts.grading import (
    ANSWER_RELEVANCY_GENERATE,
    CONTEXT_FAITHFULNESS_LONGFORM,
    CONTEXT_FAITHFULNESS_NLI_STATEMENTS,
    CONTEXT_RECALL,
    CONTEXT_RECALL_ATTRIBUTED_TOKEN,
    CONTEXT_RECALL_NOT_ATTRIBUTED_TOKEN,
    CONTEXT_RELEVANCE,
    CONTEXT_RELEVANCE_BAD,
)
from promptphoo.tokens import accumulate_token_usage

EPSILON = sys.float_info.epsilon  # Number.EPSILON
_JS_WS = "\t\n\v\f\r                  　﻿"  # noqa: E501
_WS_RUN = re.compile("[" + _JS_WS + "]+")
_LIST_MARKER = re.compile(r"^[" + _JS_WS + r"]*(?:[0-9]+[.)]|[-*])[" + _JS_WS + r"]+")
_NUMBERED_SENTENCE = re.compile(r"^[0-9]+\.[" + _JS_WS + r"]*([^.]+\.)")
_YES_NO = re.compile(r"\b(?:yes|no)\b", re.ASCII)


def _js_trim(text: str) -> str:
    return text.strip(_JS_WS)


async def _default_providers() -> dict[str, Any]:
    from promptphoo.providers.defaults import get_default_providers

    return await get_default_providers()


def _grading(grading: dict[str, Any] | None, key: str) -> Any:
    return (grading or {}).get(key)


async def matches_answer_relevance(
    input: str,
    output: str,
    threshold: float,
    grading: dict[str, Any] | None = None,
    provider_call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = await _default_providers()
    embedding_provider = await get_and_check_provider(
        "embedding",
        _grading(grading, "provider"),
        defaults.get("embeddingProvider"),
        "answer relevancy check",
    )
    text_provider = await get_and_check_provider(
        "text",
        _grading(grading, "provider"),
        defaults.get("gradingProvider"),
        "answer relevancy check",
    )

    tokens_used = normalize_matcher_token_usage(None)

    rubric_prompt = await load_rubric_prompt(
        _grading(grading, "rubricPrompt"), ANSWER_RELEVANCY_GENERATE
    )
    parsed_output = try_parse(output)
    prompt_text = await render_llm_rubric_prompt(rubric_prompt, {"answer": parsed_output})

    candidate_questions: list[str] = []
    for i in range(3):
        # Each sample uses its own cache namespace. Otherwise the identical prompt hits the
        # cache and every question after the first is a copy of it.
        with with_cache_namespace(None if i == 0 else f"answer-relevance:sample:{i}"):
            resp = await call_provider_with_context(
                text_provider,
                prompt_text,
                "answer-relevance",
                {"answer": parsed_output},
                provider_call_context,
            )
        accumulate_token_usage(tokens_used, resp.get("tokenUsage"))
        if resp.get("error") or not is_truthy(resp.get("output")):
            return grader_fail(resp.get("error") or "No output", tokens_used)
        invariant(
            isinstance(resp["output"], str), "answer relevancy check produced malformed response"
        )
        candidate_questions.append(resp["output"])

    invariant(
        callable(getattr(embedding_provider, "call_embedding_api", None)),
        f"Provider {embedding_provider.id()} must implement callEmbeddingApi for similarity check",
    )

    async def embed(text: str) -> dict[str, Any]:
        return await call_grading_provider(
            embedding_provider,
            "answer-relevance.embedding",
            lambda context, _headers=None: call_embedding_provider(
                embedding_provider, text, context
            ),
            call_context=provider_call_context,
            operation_name="embeddings",
        )

    input_embedding_resp = await embed(input)
    accumulate_token_usage(tokens_used, input_embedding_resp.get("tokenUsage"))
    if input_embedding_resp.get("error") or not is_truthy(input_embedding_resp.get("embedding")):
        return grader_fail(input_embedding_resp.get("error") or "No embedding", tokens_used)
    input_embedding = input_embedding_resp["embedding"]

    similarities: list[float] = []
    questions_with_scores: list[dict[str, Any]] = []
    for question in candidate_questions:
        resp = await embed(question)
        accumulate_token_usage(tokens_used, resp.get("tokenUsage"))
        if resp.get("error") or not is_truthy(resp.get("embedding")):
            return grader_fail(resp.get("error") or "No embedding", tokens_used)
        question_similarity = cosine_similarity(input_embedding, resp["embedding"])
        similarities.append(question_similarity)
        questions_with_scores.append({"question": question, "similarity": question_similarity})

    total: float = 0
    for s in similarities:
        total += s
    similarity = total / len(similarities)
    passed = similarity >= threshold - EPSILON
    metadata = {
        "generatedQuestions": questions_with_scores,
        "averageSimilarity": similarity,
        "threshold": threshold,
    }
    if passed:
        reason = (
            f"Relevance {to_fixed(similarity, 2)} is greater than threshold {js_string(threshold)}"
        )
    else:
        reason = (
            f"Relevance {to_fixed(similarity, 2)} is less than threshold {js_string(threshold)}"
        )
    return {
        "pass": passed,
        "score": similarity,
        "reason": reason,
        "tokensUsed": tokens_used,
        "metadata": metadata,
    }


async def matches_context_recall(
    context: str | list[str],
    ground_truth: str,
    threshold: float,
    grading: dict[str, Any] | None = None,
    vars: dict[str, Any] | None = None,
    provider_call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text_provider = await get_and_check_provider(
        "text",
        _grading(grading, "provider"),
        (await _default_providers()).get("gradingProvider"),
        "context recall check",
    )
    context_string = serialize_context(context)
    rubric_prompt = await load_rubric_prompt(_grading(grading, "rubricPrompt"), CONTEXT_RECALL)
    template_vars = {**(vars or {}), "context": context_string, "groundTruth": ground_truth}
    prompt_text = await render_llm_rubric_prompt(rubric_prompt, template_vars)

    resp = await call_provider_with_context(
        text_provider,
        prompt_text,
        "context-recall",
        {**(vars or {}), "context": context_string, "groundTruth": ground_truth},
        provider_call_context,
    )
    if resp.get("error") or not is_truthy(resp.get("output")):
        return grader_fail(resp.get("error") or "No output", resp.get("tokenUsage"))
    invariant(isinstance(resp["output"], str), "context-recall produced malformed response")

    # Only lines with attribution markers count; LLMs may add preamble text before the list.
    attributed = CONTEXT_RECALL_ATTRIBUTED_TOKEN.lower()
    not_attributed = CONTEXT_RECALL_NOT_ATTRIBUTED_TOKEN.lower()
    sentences = [
        line
        for line in split_into_sentences(resp["output"])
        if attributed in line.lower() or not_attributed in line.lower()
    ]

    sentence_attributions: list[dict[str, Any]] = []
    numerator = 0
    for sentence in sentences:
        lower = sentence.lower()
        is_attributed = not_attributed not in lower and attributed in lower
        if is_attributed:
            numerator += 1
        match = _NUMBERED_SENTENCE.match(sentence)
        clean = _js_trim(match.group(1)) if match else _js_trim(sentence.split(".")[0])
        sentence_attributions.append({"sentence": clean, "attributed": is_attributed})

    # Nonempty output with no classification lines is a malformed grading attempt, not a
    # genuine 0.00 recall verdict.
    if not sentences:
        return {
            "pass": False,
            "score": 0,
            "reason": "Grader returned no [Attributed]/[Not Attributed] classifications",
            "tokensUsed": normalize_matcher_token_usage(resp.get("tokenUsage")),
            "metadata": {
                "sentenceAttributions": [],
                "totalSentences": 0,
                "attributedSentences": 0,
                "score": 0,
                "graderError": True,
            },
        }

    score = numerator / len(sentences)
    passed = score >= threshold - EPSILON
    op = ">=" if passed else "<"
    return {
        "pass": passed,
        "score": score,
        "reason": f"Recall {to_fixed(score, 2)} is {op} {js_string(threshold)}",
        "tokensUsed": normalize_matcher_token_usage(resp.get("tokenUsage")),
        "metadata": {
            "sentenceAttributions": sentence_attributions,
            "totalSentences": len(sentences),
            "attributedSentences": numerator,
            "score": score,
        },
    }


def _normalize_for_comparison(value: str) -> str:
    return _js_trim(_WS_RUN.sub(" ", value.lower()))


async def matches_context_relevance(
    question: str,
    context: str | list[str],
    threshold: float,
    grading: dict[str, Any] | None = None,
    provider_call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text_provider = await get_and_check_provider(
        "text",
        _grading(grading, "provider"),
        (await _default_providers()).get("gradingProvider"),
        "context relevance check",
    )
    context_string = serialize_context(context)
    rubric_prompt = await load_rubric_prompt(_grading(grading, "rubricPrompt"), CONTEXT_RELEVANCE)
    prompt_text = await render_llm_rubric_prompt(
        rubric_prompt, {"context": context_string, "query": question}
    )
    resp = await call_provider_with_context(
        text_provider,
        prompt_text,
        "context-relevance",
        {"context": context_string, "query": question},
        provider_call_context,
    )
    if resp.get("error") or not is_truthy(resp.get("output")):
        return grader_fail(resp.get("error") or "No output", resp.get("tokenUsage"))
    invariant(isinstance(resp["output"], str), "context-relevance produced malformed response")
    output: str = resp["output"]

    # Segment the context into units: chunks if provided, otherwise by line when it is already
    # segmented (multiple non-empty lines) or by sentence when it is a single prose block.
    is_list = isinstance(context, list)
    pre_segmented = is_list or (
        len([line for line in context.split("\n") if _js_trim(line) != ""]) > 1
    )
    if isinstance(context, list):
        context_units = [chunk for chunk in context if len(_js_trim(chunk)) > 0]
    else:
        context_units = split_text_into_sentences(context)
    total_context_units = len(context_units)

    # Segment the grader output into the same kind of units as the context.
    insufficient_information = CONTEXT_RELEVANCE_BAD in output
    segment_relevant = split_into_sentences if pre_segmented else split_text_into_sentences
    segments = [] if insufficient_information else segment_relevant(output)

    # Only count segments whose sentences are quoted from the context; dedupe normalized text.
    context_normalized = _normalize_for_comparison(context_string)
    quoted_segments: dict[str, str] = {}
    for segment in segments:
        # A list marker needs following whitespace; otherwise preserve numbers like 1.2 and -5.
        key = _normalize_for_comparison(_LIST_MARKER.sub("", segment, count=1))
        sentences = split_text_into_sentences(key)
        if (
            sentences
            and all(sentence in context_normalized for sentence in sentences)
            and key not in quoted_segments
        ):
            quoted_segments[key] = segment
    relevant_sentences = list(quoted_segments.values())

    # Nonempty output that quotes nothing from the context is a malformed grading attempt.
    if segments and not relevant_sentences:
        return {
            "pass": False,
            "score": 0,
            "reason": "Grader output does not quote the provided context",
            "tokensUsed": normalize_matcher_token_usage(resp.get("tokenUsage")),
            "metadata": {"graderError": True, "graderOutputs": {"final": output}},
        }

    numerator = min(len(relevant_sentences), total_context_units)
    score = numerator / total_context_units if total_context_units > 0 else 0
    passed = score >= threshold - EPSILON
    op = ">=" if passed else "<"
    return {
        "pass": passed,
        "score": score,
        "reason": f"Context relevance {to_fixed(score, 2)} is {op} {js_string(threshold)}",
        "tokensUsed": normalize_matcher_token_usage(resp.get("tokenUsage")),
        "metadata": {
            "graderOutputs": {"final": output},
            "extractedSentences": relevant_sentences,
            "totalContextUnits": total_context_units,
            "totalContextSentences": total_context_units,  # Backward compatibility
            "contextUnits": context_units,
            "relevantSentenceCount": numerator,
            "insufficientInformation": insufficient_information,
            "score": score,
        },
    }


def _rubric_part(rubric_prompt: Any, index: int) -> Any:
    item = (
        rubric_prompt[index]
        if isinstance(rubric_prompt, list) and index < len(rubric_prompt)
        else None
    )
    if isinstance(item, str):
        return item
    return item.get("content") if isinstance(item, dict) else None


async def matches_context_faithfulness(
    query: str,
    output: str,
    context: str | list[str],
    threshold: float,
    grading: dict[str, Any] | None = None,
    vars: dict[str, Any] | None = None,
    provider_call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text_provider = await get_and_check_provider(
        "text",
        _grading(grading, "provider"),
        (await _default_providers()).get("gradingProvider"),
        "faithfulness check",
    )
    tokens_used = normalize_matcher_token_usage(None)

    rubric = _grading(grading, "rubricPrompt")
    if rubric:
        invariant(isinstance(rubric, list), "rubricPrompt must be an array")
    longform_prompt = await load_rubric_prompt(
        _rubric_part(rubric, 0), CONTEXT_FAITHFULNESS_LONGFORM
    )
    nli_prompt = await load_rubric_prompt(
        _rubric_part(rubric, 1), CONTEXT_FAITHFULNESS_NLI_STATEMENTS
    )

    longform_vars = {**(vars or {}), "question": query, "answer": try_parse(output)}
    prompt_text = await render_llm_rubric_prompt(longform_prompt, longform_vars)
    resp = await call_provider_with_context(
        text_provider,
        prompt_text,
        "context-faithfulness-longform",
        {**(vars or {}), "question": query, "answer": try_parse(output)},
        provider_call_context,
    )
    accumulate_token_usage(tokens_used, resp.get("tokenUsage"))
    if resp.get("error") or not is_truthy(resp.get("output")):
        return grader_fail(resp.get("error") or "No output", tokens_used)
    invariant(isinstance(resp["output"], str), "context-faithfulness produced malformed response")

    context_string = serialize_context(context)
    statements = split_into_sentences(resp["output"])
    if not statements:
        return grader_fail("Could not extract context-faithfulness statements", tokens_used)
    nli_vars = {**(vars or {}), "context": context_string, "statements": statements}
    prompt_text = await render_llm_rubric_prompt(nli_prompt, nli_vars)
    resp = await call_provider_with_context(
        text_provider,
        prompt_text,
        "context-faithfulness-nli",
        {**(vars or {}), "context": context_string, "statements": statements},
        provider_call_context,
    )
    accumulate_token_usage(tokens_used, resp.get("tokenUsage"))
    if resp.get("error") or not is_truthy(resp.get("output")):
        return grader_fail(resp.get("error") or "No output", tokens_used)
    invariant(isinstance(resp["output"], str), "context-faithfulness produced malformed response")

    final_answer = "Final verdict for each statement in order:".lower()
    verdicts = _js_trim(resp["output"].lower())
    score: float = 0
    parsed_verdict = False
    if final_answer in verdicts:
        verdicts = verdicts[verdicts.index(final_answer) + len(final_answer) :]
        parsed_verdicts = [a for a in verdicts.split(".") if _js_trim(a) != ""]
        if parsed_verdicts and all(_YES_NO.search(a) for a in parsed_verdicts):
            parsed_verdict = True
            unsupported = len([a for a in parsed_verdicts if "yes" not in a])
            missing = max(0, len(statements) - len(parsed_verdicts))
            score = 1 - (unsupported + missing) / len(statements)
    else:
        no_count = verdicts.count("verdict: no")
        yes_count = verdicts.count("verdict: yes")
        if no_count + yes_count > 0:
            parsed_verdict = True
            missing = max(0, len(statements) - no_count - yes_count)
            score = 1 - (no_count + missing) / len(statements)
    if not parsed_verdict:
        return grader_fail("Could not parse context-faithfulness verdicts", tokens_used)
    score = min(1, max(0, score))
    passed = score >= threshold - EPSILON
    op = ">=" if passed else "<"
    return {
        "pass": passed,
        "score": score,
        "reason": f"Faithfulness {to_fixed(score, 2)} is {op} {js_string(threshold)}",
        "tokensUsed": tokens_used,
    }
