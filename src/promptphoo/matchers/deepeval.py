"""Conversation relevancy matcher, ported from DeepEval. Ports
src/external/matchers/deepeval.ts (matchesConversationRelevance) and
src/external/matchers/conversationRelevancyTemplate.ts (ConversationRelevancyTemplate).

The template texts are upstream's JS template literals, byte for byte (including the trailing
spaces on some lines); tests/fixtures/conversation_relevance_golden.json holds the output of
the upstream module under Node (tools/conversation_relevance_golden.sh).
"""

from __future__ import annotations

import sys
from typing import Any

from promptphoo.assertions.context_utils import invariant
from promptphoo.jsonutil import extract_json_objects
from promptphoo.jsvalues import UNDEFINED, is_truthy, js_string, json_stringify
from promptphoo.matchers.providers import call_provider_with_context, get_and_check_provider
from promptphoo.matchers.rubric import load_rubric_prompt
from promptphoo.matchers.shared import grader_fail
from promptphoo.templating import get_engine

EPSILON = sys.float_info.epsilon  # Number.EPSILON

_VERDICTS_HEAD = """Based on the given list of message exchanges between a user and an LLM, generate a JSON object to indicate whether the LAST `assistant` message is relevant to context in messages. The JSON will have 2 fields: 'verdict' and 'reason'.
The 'verdict' key should STRICTLY be either 'yes' or 'no', which states whether the last `assistant` message is relevant according to the context in messages \nProvide a 'reason' ONLY if the answer is 'no'. \nYou MUST USE the previous messages (if any) provided in the list of messages to make an informed judgement on relevancy.

**
IMPORTANT: Please make sure to only return in JSON format.
Example Messages:
[
    {
        "role": "user",
        "content": "Hi! I have something I want to tell you"
    },
    {
        "role": "assistant",
        "content": "Sure, what is it?"
    },
    {
        "role": "user",
        "content": "I've a sore throat, what meds should I take?"
    },
    {
        "role": "assistant",
        "content": "Not sure, but isn't it a nice day today?"
    }
]

Example JSON:
{
    "verdict": "no",
    "reason": "The LLM responded 'isn't it a nice day today' to a message that asked about how to treat a sore throat, which is completely irrelevant."
}
===== END OF EXAMPLE ======
You MUST ONLY provide a verdict for the LAST message on the list but MUST USE context from the previous messages.
You DON'T have to provide a reason if the answer is 'yes'.
ONLY provide a 'no' answer if the LLM response is COMPLETELY irrelevant to the message input.
Vague LLM responses to vague inputs, such as greetings DOES NOT count as irrelevancies!
**

Messages:
"""  # noqa: E501

_REASON_HEAD = """Below is a list of irrelevancies drawn from some messages in a conversation, which you have minimal knowledge of. It is a list of strings explaining why the 'assistant' messages are irrelevant to the 'user' messages.
Given the relevancy score, which is a 0-1 score indicating how irrelevant the OVERALL AI messages are in a conversation (higher the better), CONCISELY summarize the irrelevancies to justify the score. \n
** \nIMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{
    "reason": "The score is <relevancy_score> because <your_reason>."
}

Always quote WHICH MESSAGE and the INFORMATION in the reason in your final reason.
Be sure in your reason, as if you know what the `assistant` messages from messages in a conversation is from the irrelevancies.
**

Relevancy Score:
"""  # noqa: E501


class ConversationRelevancyTemplate:
    @staticmethod
    def generate_verdicts(sliding_window: list[dict[str, Any]]) -> str:
        return f"{_VERDICTS_HEAD}{json_stringify(sliding_window, 2)}\n\nJSON:"

    @staticmethod
    def generate_reason(score: float, irrelevancies: list[str]) -> str:
        return (
            f"{_REASON_HEAD}{js_string(score)}\n\nIrrelevancies:\n"
            f"{json_stringify(irrelevancies, 2)}\n\nJSON:"
        )


def _content(value: Any) -> Any:
    """`typeof v === 'string' ? v : JSON.stringify(v)`; a missing key stays undefined."""
    return value if isinstance(value, str) else json_stringify(value)


def _message_role(role: str, msg: dict[str, Any], key: str) -> dict[str, Any]:
    # JSON.stringify(undefined) is undefined, so the window serialization drops `content`.
    if key not in msg or msg[key] is UNDEFINED:
        return {"role": role}
    return {"role": role, "content": _content(msg[key])}


async def matches_conversation_relevance(
    messages: list[dict[str, Any]],
    threshold: float,
    vars: dict[str, Any] | None = None,  # noqa: A002 - Node's parameter name
    grading: dict[str, Any] | None = None,
    provider_call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from promptphoo.providers.defaults import get_default_providers

    text_provider = await get_and_check_provider(
        "text",
        (grading or {}).get("provider"),
        (await get_default_providers()).get("gradingProvider"),
        "conversation relevancy check",
    )

    # Messages contain runtime data (model output in msg.output, conversation prompts in
    # msg.input) and must NOT be rendered as Nunjucks templates (SSTI). Pass them as data only.
    message_roles: list[dict[str, Any]] = []
    for msg in messages:
        message_roles.append(_message_role("user", msg, "input"))
        message_roles.append(_message_role("assistant", msg, "output"))

    rubric_prompt = (grading or {}).get("rubricPrompt")
    loaded_rubric_prompt = (
        await load_rubric_prompt(rubric_prompt, "") if is_truthy(rubric_prompt) else ""
    )

    if loaded_rubric_prompt:
        # Custom rubric prompt (user config) is rendered as a template.
        prompt_text = get_engine(None, False, True).render_string(
            loaded_rubric_prompt, {**(vars or {}), "messages": messages}
        )
    else:
        prompt_text = ConversationRelevancyTemplate.generate_verdicts(message_roles)

    resp = await call_provider_with_context(
        text_provider,
        prompt_text,
        "conversation-relevance",
        {**(vars or {}), "messages": messages},
        provider_call_context,
    )
    if resp.get("error") or not is_truthy(resp.get("output")):
        return grader_fail(resp.get("error") or "No output", resp.get("tokenUsage"))

    invariant(
        isinstance(resp["output"], str),
        "conversation relevancy check produced malformed response",
    )

    try:
        json_objects = extract_json_objects(resp["output"])
        if not json_objects:
            raise ValueError("No JSON object found in response")
        result = json_objects[0]
        verdict = result.get("verdict") if isinstance(result, dict) else None
        if verdict not in ("yes", "no"):
            raise ValueError("Conversation relevance grader returned an invalid verdict")
        passed = verdict == "yes"
        score = 1 if passed else 0
        reason = result.get("reason")
        out: dict[str, Any] = {
            "pass": score >= threshold - EPSILON,
            "score": score,
            "reason": reason
            if is_truthy(reason)
            else f"Response {'is' if passed else 'is not'} relevant to the conversation context",
        }
        if resp.get("tokenUsage") is not None:
            out["tokensUsed"] = resp["tokenUsage"]
        return out
    except Exception as err:
        return grader_fail(f"Error parsing output: {err}", resp.get("tokenUsage"))
