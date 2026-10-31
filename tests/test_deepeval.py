"""conversation-relevance matcher and handler.

Ports test/external/{conversationRelevancy,assertions,matchers}.test.ts. Grader prompts are
compared byte for byte with tests/fixtures/conversation_relevance_golden.json, written by
tools/conversation_relevance_golden.mjs from upstream's own template module and real Nunjucks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rag_fakes import FakeText

from promptphoo.assertions.base import AssertionParams, AssertionValueError
from promptphoo.assertions.deepeval import handle_conversation_relevance
from promptphoo.assertions.index import get_handlers, run_assertion
from promptphoo.jsvalues import json_stringify
from promptphoo.matchers.deepeval import (
    ConversationRelevancyTemplate,
    matches_conversation_relevance,
)
from promptphoo.providers import defaults

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "conversation_relevance_golden.json").read_text(
        encoding="utf-8"
    )
)
TOKENS = {"total": 10, "prompt": 5, "completion": 5, "cached": 0}


def verdict(v: str, reason: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"verdict": v}
    if reason is not None:
        body["reason"] = reason
    return {"output": json.dumps(body), "tokenUsage": dict(TOKENS)}


@pytest.fixture
def default_grader(monkeypatch: pytest.MonkeyPatch):
    """Install `provider` as every default provider (upstream mocks getDefaultProviders)."""

    def install(provider: Any) -> Any:
        async def fake_defaults(env=None):
            return {
                k: provider
                for k in (
                    "embeddingProvider",
                    "gradingJsonProvider",
                    "gradingProvider",
                    "llmRubricProvider",
                    "moderationProvider",
                    "suggestionsProvider",
                    "synthesizeProvider",
                )
            }

        monkeypatch.setattr(defaults, "get_default_providers", fake_defaults)
        return provider

    return install


def params(
    assertion: dict[str, Any],
    *,
    test: dict[str, Any],
    prompt: str | None = "test prompt",
    output_string: Any = "hello world",
) -> AssertionParams:
    return AssertionParams(
        assertion=assertion,
        base_type="conversation-relevance",
        inverse=assertion["type"].startswith("not-"),
        output=output_string,
        output_string=output_string,
        rendered_value=None,
        prompt=prompt,
        provider_response={"output": output_string},
        test=test,
    )


# --- ConversationRelevancyTemplate (golden) ---------------------------------------------------


def _roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        for role, key in (("user", "input"), ("assistant", "output")):
            if key not in m:
                out.append({"role": role})
            else:
                v = m[key]
                out.append(
                    {"role": role, "content": v if isinstance(v, str) else json_stringify(v)}
                )
    return out


@pytest.mark.parametrize("name", sorted(GOLDEN["reasons"]))
def test_generate_reason_matches_upstream(name):
    case = GOLDEN["reasons"][name]
    assert (
        ConversationRelevancyTemplate.generate_reason(case["score"], case["irrelevancies"])
        == case["prompt"]
    )


def test_generate_verdicts_contains_window():
    messages = [
        {"role": "user", "content": "What is the weather?"},
        {"role": "assistant", "content": "It is sunny today."},
    ]
    prompt = ConversationRelevancyTemplate.generate_verdicts(messages)
    assert (
        "generate a JSON object to indicate whether the LAST `assistant` message is relevant"
        in prompt
    )
    assert json.dumps(messages, indent=2) in prompt


def test_generate_reason_contains_score():
    irrelevancies = [
        "Response about weather was not related to math question",
        "Assistant talked about food when asked about programming",
    ]
    prompt = ConversationRelevancyTemplate.generate_reason(0.6, irrelevancies)
    assert "Relevancy Score:\n0.6" in prompt
    assert json.dumps(irrelevancies, indent=2) in prompt


@pytest.mark.parametrize("name", sorted(GOLDEN["verdicts"]))
async def test_matcher_sends_golden_verdict_prompt(default_grader, name):
    case = GOLDEN["verdicts"][name]
    grader = default_grader(FakeText(verdict("yes")))
    await matches_conversation_relevance(case["messages"], 0.5, {"variable": "test"})
    assert grader.sent == [case["prompt"]]
    assert (
        ConversationRelevancyTemplate.generate_verdicts(_roles(case["messages"]))
        == (case["prompt"])
    )


@pytest.mark.parametrize("name", sorted(GOLDEN["rubrics"]))
async def test_custom_rubric_prompt_matches_nunjucks(default_grader, name):
    case = GOLDEN["rubrics"][name]
    grader = default_grader(FakeText(verdict("yes")))
    await matches_conversation_relevance(
        case["messages"], 0.5, case["vars"], {"rubricPrompt": case["rubricPrompt"]}
    )
    assert grader.sent == [case["prompt"]]


# --- matchesConversationRelevance (test/external/matchers.test.ts) ----------------------------


async def test_relevant_response_passes(default_grader):
    default_grader(FakeText(verdict("yes")))
    result = await matches_conversation_relevance(
        [{"input": "What is the capital of France?", "output": "The capital of France is Paris."}],
        0.5,
    )
    assert result == {
        "pass": True,
        "score": 1,
        "reason": "Response is relevant to the conversation context",
        "tokensUsed": TOKENS,
    }


async def test_irrelevant_response_fails(default_grader):
    default_grader(FakeText(verdict("no", "Response is completely unrelated to the question")))
    result = await matches_conversation_relevance(
        [{"input": "What is the capital of France?", "output": "The weather is nice today."}],
        0.5,
    )
    assert result["pass"] is False
    assert result["score"] == 0
    assert result["reason"] == "Response is completely unrelated to the question"


async def test_irrelevant_without_reason_uses_fallback(default_grader):
    default_grader(FakeText(verdict("no")))
    result = await matches_conversation_relevance([{"input": "q", "output": "a"}], 0.5)
    assert result["reason"] == "Response is not relevant to the conversation context"


async def test_conversation_context_in_prompt(default_grader):
    grader = default_grader(FakeText(verdict("yes", "answers with context")))
    messages = [
        {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
        {
            "input": "What is its population?",
            "output": "Paris has a population of about 2.2 million people.",
        },
    ]
    result = await matches_conversation_relevance(messages, 0.5)
    assert result["pass"] is True
    assert "What is the capital of France?" in grader.sent[0]
    assert "What is its population?" in grader.sent[0]


async def test_provider_error(default_grader):
    default_grader(
        FakeText({"error": "API error", "tokenUsage": {"total": 5, "prompt": 5, "completion": 0}})
    )
    result = await matches_conversation_relevance([{"input": "q", "output": "Paris"}], 0.5)
    assert result["pass"] is False
    assert result["reason"] == "API error"
    assert result["metadata"] == {"graderError": True}
    assert result["tokensUsed"]["total"] == 5
    assert result["tokensUsed"]["prompt"] == 5
    assert result["tokensUsed"]["completion"] == 0
    assert result["tokensUsed"]["cached"] == 0


async def test_no_output(default_grader):
    default_grader(FakeText({"output": ""}))
    result = await matches_conversation_relevance([{"input": "q", "output": "a"}], 0.5)
    assert result["reason"] == "No output"
    assert result["metadata"] == {"graderError": True}


async def test_malformed_response(default_grader):
    default_grader(FakeText({"output": "not json", "tokenUsage": TOKENS}))
    result = await matches_conversation_relevance([{"input": "q", "output": "Paris"}], 0.5)
    assert result["pass"] is False
    assert result["reason"] == "Error parsing output: No JSON object found in response"
    assert result["metadata"] == {"graderError": True}


async def test_invalid_verdict(default_grader):
    default_grader(FakeText(verdict("maybe")))
    result = await matches_conversation_relevance([{"input": "q", "output": "a"}], 0.5)
    assert result["reason"] == (
        "Error parsing output: Conversation relevance grader returned an invalid verdict"
    )


async def test_non_string_output_is_invariant_error(default_grader):
    default_grader(FakeText({"output": {"verdict": "yes"}}))
    with pytest.raises(
        AssertionValueError,
        match="^Invariant failed: conversation relevancy check produced malformed response$",
    ):
        await matches_conversation_relevance([{"input": "q", "output": "a"}], 0.5)


async def test_markdown_json_response(default_grader):
    default_grader(FakeText({"output": '```json\n{"verdict": "yes"}\n```', "tokenUsage": TOKENS}))
    result = await matches_conversation_relevance([{"input": "q", "output": "Paris"}], 0.5)
    assert result["pass"] is True
    assert result["score"] == 1


async def test_threshold_epsilon(default_grader):
    default_grader(FakeText(verdict("yes")))
    result = await matches_conversation_relevance(
        [{"input": "q", "output": "a"}], 1 + 2.220446049250313e-16
    )
    assert result["pass"] is True


async def test_custom_rubric_dump(default_grader):
    grader = default_grader(FakeText(verdict("yes")))
    messages = [{"input": "What is the capital of France?", "output": "Paris"}]
    await matches_conversation_relevance(
        messages, 0.5, {}, {"rubricPrompt": "Custom rubric template with {{ messages | dump(2) }}"}
    )
    assert grader.sent[0] == "Custom rubric template with " + json.dumps(messages, indent=2)


async def test_env_access_in_output_not_rendered(default_grader, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "SHOULD_NOT_RENDER")
    grader = default_grader(FakeText(verdict("yes")))
    messages = [
        {
            "input": "Tell me a secret",
            "output": "The answer is {{env.OPENAI_API_KEY}} and {{env.AWS_SECRET_ACCESS_KEY}}",
        }
    ]
    await matches_conversation_relevance(messages, 0.5, {})
    assert "{{env.OPENAI_API_KEY}}" in grader.sent[0]
    assert "{{env.AWS_SECRET_ACCESS_KEY}}" in grader.sent[0]
    assert "SHOULD_NOT_RENDER" not in grader.sent[0]


async def test_template_syntax_in_conversation_not_rendered(default_grader):
    grader = default_grader(FakeText(verdict("yes")))
    messages = GOLDEN["verdicts"]["ssti"]["messages"]
    await matches_conversation_relevance(messages, 0.5, {"variable": "test", "SECRET": "val"})
    prompt = grader.sent[0]
    assert "What is {{variable}}?" in prompt
    assert "range.constructor(" in prompt
    assert "execSync('id')" in prompt
    assert "Follow up with {{env.SECRET}}" in prompt
    assert "{% for x in range(10) %}{{x}}{% endfor %}" in prompt


async def test_reserved_messages_var_wins_over_user_vars(default_grader):
    grader = default_grader(FakeText(verdict("yes")))
    messages = [{"input": "What is 2+2?", "output": "4"}]
    await matches_conversation_relevance(
        messages,
        0.5,
        {"messages": "vars messages sentinel", "extra": "kept user var"},
        {"rubricPrompt": "messages={{ messages | dump }}\nextra={{ extra }}"},
    )
    prompt, context = grader.calls[0]
    assert "What is 2+2?" in prompt
    assert "extra=kept user var" in prompt
    assert "vars messages sentinel" not in prompt
    assert context["vars"]["messages"] == messages
    assert context["vars"]["extra"] == "kept user var"
    assert context["prompt"]["label"] == "conversation-relevance"


async def test_grading_provider_from_options(default_grader):
    default_grader(FakeText(verdict("no", "default should not be used")))
    configured = FakeText(verdict("yes"))
    result = await matches_conversation_relevance(
        [{"input": "q", "output": "a"}], 0.5, None, {"provider": configured}
    )
    assert result["pass"] is True
    assert len(configured.calls) == 1


# --- handleConversationRelevance (test/external/assertions.test.ts) ---------------------------


def window(prompt: str) -> list[dict[str, Any]]:
    """The message window serialized into a verdict prompt."""
    return json.loads(prompt.split("\n\nMessages:\n", 1)[1].rsplit("\n\nJSON:", 1)[0])


def by_marker(marker: str):
    def respond(prompt: str) -> dict[str, Any]:
        if marker in prompt:
            return verdict("no", "The response is unrelated to the question asked")
        return verdict("yes")

    return respond


async def test_single_message_pair():
    result = await handle_conversation_relevance(
        params(
            {"type": "conversation-relevance", "threshold": 0.8},
            test={"vars": {}, "options": {"provider": FakeText(verdict("yes"))}},
            prompt="What is the weather like?",
            output_string="The weather is sunny today.",
        )
    )
    assert result["pass"] is True
    assert result["score"] == 1
    assert result["reason"] == "1 out of 1 conversation windows were relevant"
    assert result["tokensUsed"]["total"] == 10


async def test_conversation_overrides_output_string():
    grader = FakeText(verdict("yes"))
    result = await handle_conversation_relevance(
        params(
            {"type": "conversation-relevance", "threshold": 0.8},
            test={
                "vars": {"_conversation": [{"input": "Hi", "output": "Hello! How can I help?"}]},
                "options": {"provider": grader},
            },
            prompt="This should be ignored",
            output_string="This should be ignored",
        )
    )
    assert result["pass"] is True
    assert result["score"] == 1
    assert "This should be ignored" not in grader.sent[0]


SUNNY = [
    {"input": "Hi", "output": "Hello! How can I help you?"},
    {"input": "What is the weather like?", "output": "The weather is sunny today."},
    {
        "input": "What should I wear?",
        "output": "Given the sunny weather, you might want to wear light clothes.",
    },
    {"input": "Thanks!", "output": "You're welcome! Enjoy the sunshine!"},
]


async def test_sliding_windows():
    conversation = [*SUNNY, {"input": "What is 2+2?", "output": "The capital of France is Paris."}]

    def respond(prompt: str) -> dict[str, Any]:
        if "Below is a list of irrelevancies" in prompt:
            return {"output": "not json"}
        if "2+2" in prompt and "Paris" in prompt:
            return verdict(
                "no",
                "The response about Paris is completely unrelated to the mathematical question "
                "about 2+2",
            )
        return verdict("yes")

    grader = FakeText(respond)
    result = await handle_conversation_relevance(
        params(
            {"type": "conversation-relevance", "threshold": 0.85, "config": {"windowSize": 3}},
            test={"vars": {"_conversation": conversation}, "options": {"provider": grader}},
            output_string="This should be ignored",
        )
    )
    assert result["pass"] is False
    assert result["score"] == 0.8
    # The reason grader returned no JSON, so the count fallback is used.
    assert result["reason"] == "4 out of 5 conversation windows were relevant"
    # Window i covers messages max(0, i-2)..i.
    windows = [window(p) for p in grader.sent[:5]]
    assert [len(w) // 2 for w in windows] == [1, 2, 3, 3, 3]
    assert windows[4][0]["content"] == "What should I wear?"
    reason_prompt = grader.sent[5]
    assert "Relevancy Score:\n0.8\n" in reason_prompt
    assert "The response about Paris is completely unrelated" in reason_prompt
    assert grader.calls[5][1]["vars"] == {
        "score": "0.8",
        "irrelevancies": [
            "The response about Paris is completely unrelated to the mathematical question "
            "about 2+2"
        ],
    }
    assert grader.calls[5][1]["prompt"]["label"] == "conversation-relevance-reason"


async def test_reason_generation():
    count = 0

    def respond(prompt: str) -> dict[str, Any]:
        nonlocal count
        count += 1
        if "Below is a list of irrelevancies" in prompt:
            return {
                "output": json.dumps(
                    {
                        "reason": "The score is 0.6 because 2 out of 5 responses were irrelevant "
                        "to the conversation context."
                    }
                ),
                "tokenUsage": {"total": 20, "prompt": 10, "completion": 10, "cached": 0},
            }
        if count in (3, 5):
            return verdict("no", f"Response {count} was irrelevant")
        return verdict("yes")

    conversation = [{"input": f"Question {i}", "output": f"Answer {i}"} for i in range(1, 6)]
    result = await handle_conversation_relevance(
        params(
            {"type": "conversation-relevance", "threshold": 0.8, "config": {"windowSize": 3}},
            test={
                "vars": {"_conversation": conversation},
                "options": {"provider": FakeText(respond)},
            },
            output_string="test",
        )
    )
    assert "The score is 0.6 because 2 out of 5 responses were irrelevant" in result["reason"]
    assert result["pass"] is False
    assert result["score"] == 0.6
    assert result["tokensUsed"]["total"] == 70
    assert result["tokensUsed"]["prompt"] == 35


async def test_reason_without_reason_key_falls_back():
    def respond(prompt: str) -> dict[str, Any]:
        if "Below is a list of irrelevancies" in prompt:
            return {"output": json.dumps({"summary": "x"})}
        return verdict("no", "off topic")

    result = await handle_conversation_relevance(
        params(
            {"type": "conversation-relevance"},
            test={"vars": {}, "options": {"provider": FakeText(respond)}},
        )
    )
    assert result["reason"] == "0 out of 1 conversation windows were relevant"


async def test_reason_grader_error_falls_back():
    def respond(prompt: str) -> dict[str, Any]:
        if "Below is a list of irrelevancies" in prompt:
            return {"error": "down", "tokenUsage": {"total": 3, "prompt": 3, "completion": 0}}
        return verdict("no", "off topic")

    result = await handle_conversation_relevance(
        params(
            {"type": "conversation-relevance"},
            test={"vars": {}, "options": {"provider": FakeText(respond)}},
        )
    )
    assert result["reason"] == "0 out of 1 conversation windows were relevant"
    # Usage from a reason call without output is not added (upstream only adds it on output).
    assert result["tokensUsed"]["total"] == 10


async def test_reason_uses_default_grader(default_grader):
    default = default_grader(
        FakeText(
            lambda p: (
                {"output": '{"reason": "summarized"}'}
                if "Below is a list of irrelevancies" in p
                else verdict("no", "off topic")
            )
        )
    )
    result = await handle_conversation_relevance(
        params({"type": "conversation-relevance"}, test={"vars": {}})
    )
    assert result["reason"] == "summarized"
    assert len(default.calls) == 2


async def test_shorter_than_window():
    result = await handle_conversation_relevance(
        params(
            {"type": "conversation-relevance", "threshold": 0.8, "config": {"windowSize": 5}},
            test={
                "vars": {"_conversation": SUNNY[:2]},
                "options": {"provider": FakeText(verdict("yes"))},
            },
        )
    )
    assert result["pass"] is True
    assert result["score"] == 1


async def test_default_window_size():
    conversation = [{"input": f"q{i}", "output": f"a{i}"} for i in range(7)]
    grader = FakeText(verdict("yes"))
    result = await handle_conversation_relevance(
        params(
            {"type": "conversation-relevance", "threshold": 0.8, "config": {"windowSize": 0}},
            test={"vars": {"_conversation": conversation}, "options": {"provider": grader}},
        )
    )
    assert result["score"] == 1
    assert [m["content"] for m in window(grader.sent[6])][::2] == ["q2", "q3", "q4", "q5", "q6"]


@pytest.mark.parametrize("conversation", [[], None])
async def test_empty_or_missing_conversation_uses_output(conversation):
    grader = FakeText(verdict("yes"))
    test_vars = {} if conversation is None else {"_conversation": conversation}
    result = await handle_conversation_relevance(
        params(
            {"type": "conversation-relevance", "threshold": 0.8},
            test={"vars": test_vars, "options": {"provider": grader}},
            prompt="What is the weather like?",
            output_string="The weather is sunny today.",
        )
    )
    assert result["pass"] is True
    assert result["score"] == 1
    assert "The weather is sunny today." in grader.sent[0]


async def test_missing_prompt_is_invariant_error():
    with pytest.raises(
        AssertionValueError,
        match="^Invariant failed: conversational-relevance assertion type must have a prompt$",
    ):
        await handle_conversation_relevance(
            params(
                {"type": "conversation-relevance"},
                test={"vars": {}, "options": {"provider": FakeText(verdict("yes"))}},
                prompt=None,
            )
        )


@pytest.mark.parametrize(
    ("assertion", "score", "passed"),
    [
        ({"type": "conversation-relevance"}, 0, False),  # issue #10141: default 0.5
        ({"type": "conversation-relevance", "threshold": 0}, 0, True),
        ({"type": "not-conversation-relevance"}, 1, True),
    ],
)
async def test_thresholds(assertion, score, passed):
    result = await handle_conversation_relevance(
        params(
            assertion,
            test={"vars": {}, "options": {"provider": FakeText(verdict("no", "irrelevant"))}},
            prompt="What is the weather like?",
            output_string="The capital of France is Paris.",
        )
    )
    assert result["score"] == score
    assert result["pass"] is passed


async def test_inverse_grader_error_does_not_pass():
    result = await handle_conversation_relevance(
        params(
            {"type": "not-conversation-relevance"},
            test={"vars": {}, "options": {"provider": FakeText({"error": "grader unavailable"})}},
            prompt="What is the weather like?",
            output_string="The capital of France is Paris.",
        )
    )
    assert result["pass"] is False
    assert result["score"] == 0
    assert "grader unavailable" in result["reason"]
    assert result["metadata"] == {"graderError": True}
    assert "tokensUsed" not in result


async def test_inverse_invalid_verdict_does_not_pass():
    result = await handle_conversation_relevance(
        params(
            {"type": "not-conversation-relevance"},
            test={"vars": {}, "options": {"provider": FakeText(verdict("maybe"))}},
            prompt="What is the weather like?",
            output_string="The capital of France is Paris.",
        )
    )
    assert result["pass"] is False
    assert result["score"] == 0
    assert "invalid verdict" in result["reason"]
    assert result["metadata"] == {"graderError": True}


async def test_prior_usage_kept_when_later_window_fails():
    grader = FakeText(
        [
            {
                "output": json.dumps({"verdict": "yes"}),
                "tokenUsage": {"total": 10, "prompt": 5, "completion": 5},
            },
            {
                "error": "grader unavailable",
                "tokenUsage": {"total": 12, "prompt": 7, "completion": 5},
            },
        ]
    )
    result = await handle_conversation_relevance(
        params(
            {"type": "not-conversation-relevance"},
            test={
                "vars": {
                    "_conversation": [
                        {"input": "First question", "output": "First answer"},
                        {"input": "Second question", "output": "Second answer"},
                    ]
                },
                "options": {"provider": grader},
            },
            prompt="Second question",
            output_string="Second answer",
        )
    )
    assert result["pass"] is False
    assert result["score"] == 0
    assert result["assertion"] == {"type": "not-conversation-relevance"}
    usage = result["tokensUsed"]
    assert (usage["total"], usage["prompt"], usage["completion"]) == (22, 12, 10)


async def test_score_exactly_on_default_threshold():
    conversation = [
        SUNNY[0],
        {"input": "What is 2+2?", "output": "The capital of France is Paris."},
    ]
    result = await handle_conversation_relevance(
        params(
            {"type": "conversation-relevance"},
            test={
                "vars": {"_conversation": conversation},
                "options": {"provider": FakeText(by_marker("Paris"))},
            },
        )
    )
    assert result["score"] == 0.5
    assert result["pass"] is True


async def test_score_just_below_default_threshold():
    conversation = [
        *SUNNY[:2],
        {"input": "What is 2+2?", "output": "The capital of France is Paris."},
        SUNNY[3],
        {"input": "Anything else?", "output": "Let me know if you need anything."},
    ]
    result = await handle_conversation_relevance(
        params(
            {"type": "conversation-relevance"},
            test={
                "vars": {"_conversation": conversation},
                "options": {"provider": FakeText(by_marker("Paris"))},
            },
        )
    )
    assert result["score"] == 0.4
    assert result["pass"] is False


# --- registry / run_assertion -----------------------------------------------------------------


def test_handler_registered():
    assert get_handlers()["conversation-relevance"] is handle_conversation_relevance


async def test_run_assertion_not_prefix():
    result = await run_assertion(
        assertion={"type": "not-conversation-relevance"},
        prompt="What is the weather like?",
        provider_response={"output": "The capital of France is Paris."},
        test={"options": {"provider": FakeText(verdict("no", "irrelevant"))}},
    )
    assert result["pass"] is True
    assert result["score"] == 1
