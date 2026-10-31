"""`conversation-relevance` assertion, ported from DeepEval. Ports
src/external/assertions/deepeval.ts (handleConversationRelevance)."""

from __future__ import annotations

import sys
from typing import Any

from promptphoo.assertions.base import AssertionParams, GradingResult, Handler
from promptphoo.assertions.context_utils import invariant
from promptphoo.jsonutil import extract_json_objects
from promptphoo.jsvalues import is_truthy, js_string
from promptphoo.tokens import accumulate_token_usage, create_empty_token_usage

EPSILON = sys.float_info.epsilon  # Number.EPSILON
DEFAULT_WINDOW_SIZE = 5
# DeepEval's default pass threshold for the conversation relevancy metric.
DEFAULT_THRESHOLD = 0.5


async def handle_conversation_relevance(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.deepeval import (
        ConversationRelevancyTemplate,
        matches_conversation_relevance,
    )
    from promptphoo.matchers.providers import (
        call_provider_with_context,
        get_and_check_provider,
    )
    from promptphoo.providers.defaults import get_default_providers

    assertion, inverse, test = params.assertion, params.inverse, params.test
    test_vars = test.get("vars")
    conversation = (test_vars or {}).get("_conversation")
    messages: list[dict[str, Any]]
    if isinstance(conversation, list) and len(conversation) > 0:
        messages = conversation
    else:
        invariant(
            isinstance(params.output_string, str),
            "conversational-relevance assertion type must have a string value",
        )
        invariant(params.prompt, "conversational-relevance assertion type must have a prompt")
        messages = [{"input": params.prompt, "output": params.output_string}]

    config = assertion.get("config") or {}
    window_size: Any = config.get("windowSize")
    window_size = int(window_size) if is_truthy(window_size) else DEFAULT_WINDOW_SIZE
    threshold = assertion.get("threshold")
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    relevant_count = 0
    total_windows = 0
    irrelevancies: list[str] = []
    tokens_used = create_empty_token_usage()

    def _tokens() -> dict[str, Any]:
        return {"tokensUsed": tokens_used} if tokens_used["total"] > 0 else {}

    # DeepEval creates a window for each message position, with varying sizes.
    for i in range(len(messages)):
        window_messages = messages[max(0, i - window_size + 1) : i + 1]
        result = await matches_conversation_relevance(
            window_messages,
            1.0,  # Use 1.0 threshold for individual windows
            test_vars,
            test.get("options"),
            params.provider_call_context,
        )

        if result.get("tokensUsed"):
            accumulate_token_usage(tokens_used, result["tokensUsed"])

        if (result.get("metadata") or {}).get("graderError") is True:
            out = {k: v for k, v in result.items() if k != "tokensUsed"}
            return {**out, "assertion": assertion, **_tokens()}

        if result.get("pass"):
            relevant_count += 1
        elif (
            is_truthy(result.get("reason"))
            and result["reason"] != "Response is not relevant to the conversation context"
        ):
            irrelevancies.append(result["reason"])

        total_windows += 1

    score = relevant_count / total_windows if total_windows > 0 else 0
    passed = (score >= threshold - EPSILON) != inverse

    fallback = f"{relevant_count} out of {total_windows} conversation windows were relevant"
    reason = fallback
    if irrelevancies and score < 1.0:
        options = test.get("options") or {}
        text_provider = await get_and_check_provider(
            "text",
            options.get("provider"),
            (await get_default_providers()).get("gradingProvider"),
            "conversation relevancy reason generation",
        )
        reason_prompt = ConversationRelevancyTemplate.generate_reason(score, irrelevancies)
        resp = await call_provider_with_context(
            text_provider,
            reason_prompt,
            "conversation-relevance-reason",
            {"score": js_string(score), "irrelevancies": irrelevancies},
            params.provider_call_context,
        )
        output = resp.get("output")
        if is_truthy(output) and isinstance(output, str):
            try:
                json_objects = extract_json_objects(output)
                if json_objects:
                    parsed = json_objects[0]
                    generated = parsed.get("reason") if isinstance(parsed, dict) else None
                    reason = generated if is_truthy(generated) else fallback
            except Exception:
                reason = fallback
            if resp.get("tokenUsage"):
                accumulate_token_usage(tokens_used, resp["tokenUsage"])

    return {
        "assertion": assertion,
        "pass": passed,
        "score": 1 - score if inverse else score,
        "reason": reason,
        **_tokens(),
    }


HANDLERS: dict[str, Handler] = {
    "conversation-relevance": handle_conversation_relevance,
}
