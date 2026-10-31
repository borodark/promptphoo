"""Ports test/matchers/{factuality,closed-qa,agent-rubric,trajectory-goal-success}.test.ts,
test/assertions/{modelGradedClosedQa,agentRubric,trajectoryGoalSuccess}.test.ts and the
summarizeTrajectoryForJudge case of test/assertions/trajectory.test.ts.

Rendered default grading prompts are compared with real Nunjucks output from
tools/grading_render_golden.js (tests/fixtures/grading_render_golden.json).

Not ported: the Codex default agent grader and copy_working_dir workspaces (D26).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from promptphoo import providers as providers_pkg
from promptphoo.assertions import modelgraded
from promptphoo.assertions.base import AssertionParams, AssertionValueError
from promptphoo.assertions.index import get_handlers, run_assertion
from promptphoo.assertions.trajectory import (
    handle_trajectory_goal_success,
    summarize_trajectory_for_judge,
)
from promptphoo.matchers import agent as agent_matchers
from promptphoo.matchers import llm_grading
from promptphoo.matchers.agent import matches_agent_rubric
from promptphoo.matchers.llm_grading import (
    matches_closed_qa,
    matches_factuality,
    matches_trajectory_goal_success,
)
from promptphoo.matchers.rubric import render_llm_rubric_prompt
from promptphoo.providers import defaults
from promptphoo.providers.base import ApiProvider

FIXTURES = Path(__file__).parent / "fixtures"
TOKENS = {"total": 10, "prompt": 5, "completion": 5}
NORMALIZED_TOKENS = {
    "total": 10, "prompt": 5, "completion": 5, "cached": 0, "numRequests": 0,
    "completionDetails": {"reasoning": 0, "acceptedPrediction": 0, "rejectedPrediction": 0},
}  # fmt: skip


class FakeProvider(ApiProvider):
    def __init__(self, response: Any = None, provider_id: str = "fake-grader") -> None:
        super().__init__({})
        self._pid = provider_id
        self.responses = response if isinstance(response, list) else [response]
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def default_id(self) -> str:
        return self._pid

    async def call_api(self, prompt, context=None, options=None):
        self.calls.append((prompt, context))
        resp = self.responses[min(len(self.calls), len(self.responses)) - 1]
        if isinstance(resp, Exception):
            raise resp
        return resp if resp is not None else {}


@pytest.fixture
def grader(monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    """The default grading provider (upstream DefaultGradingProvider)."""
    provider = FakeProvider(
        {"output": "(A) The submitted answer is a subset.", "tokenUsage": TOKENS}
    )

    async def fake_defaults(env=None):
        return {"gradingProvider": provider, "gradingJsonProvider": provider}

    monkeypatch.setattr(defaults, "get_default_providers", fake_defaults)
    return provider


# --- rendering parity with Nunjucks ----------------------------------------------------------

GOLDEN = json.loads((FIXTURES / "grading_render_golden.json").read_text("utf-8"))
CASES = {c["name"]: c for c in json.loads((FIXTURES / "grading_render_cases.json").read_text())}


@pytest.mark.parametrize("golden", GOLDEN, ids=[g["name"] for g in GOLDEN])
async def test_render_llm_rubric_prompt_matches_nunjucks(golden: dict[str, Any]):
    case = CASES[golden["name"]]
    assert await render_llm_rubric_prompt(case["template"], case["context"]) == golden["rendered"]


# --- matchesFactuality --------------------------------------------------------------------------


def _factuality_result(passed: bool, reason: str, score: float, **extra: Any) -> dict[str, Any]:
    return {"pass": passed, "reason": reason, "score": score, "tokensUsed": NORMALIZED_TOKENS,
            **extra}  # fmt: skip


SUBSET = "The submitted answer is a subset of the expert answer and is fully consistent with it."
DISAGREE = "There is a disagreement between the submitted answer and the expert answer."


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (f"(A) {SUBSET}", _factuality_result(True, SUBSET, 1)),
        (json.dumps({"category": "A", "reason": SUBSET}), _factuality_result(True, SUBSET, 1)),
        ("(A) This is a custom reason for category A.",
         _factuality_result(True, "This is a custom reason for category A.", 1)),
        (f"(D) {DISAGREE}", _factuality_result(False, DISAGREE, 0)),
        (json.dumps({"category": "D", "reason": DISAGREE}), _factuality_result(False, DISAGREE, 0)),
        ('{"category": "A"}', _factuality_result(True, f"Category A: {SUBSET}", 1)),
        ('{"category": "Z", "reason": "Invalid category"}',
         _factuality_result(False, "Invalid category value: Z", 0,
                            metadata={"graderError": True})),
        ("I am not able to grade this submission.",
         _factuality_result(False, "Factuality checker output did not match expected format: "
                            "I am not able to grade this submission.", 0,
                            metadata={"graderError": True})),
    ],
)  # fmt: skip
async def test_factuality_verdicts(grader: FakeProvider, output: str, expected: dict[str, Any]):
    grader.responses = [{"output": output, "tokenUsage": TOKENS}]
    assert await matches_factuality("Input text", "Expected output", "Sample output", {}) == (
        expected
    )


async def test_factuality_score_overrides(grader: FakeProvider):
    grader.responses = [{"output": json.dumps({"category": "A", "reason": SUBSET}),
                         "tokenUsage": TOKENS}]  # fmt: skip
    grading = {"factuality": {"subset": 0.8, "superset": 0.9, "agree": 1, "disagree": 0,
                              "differButFactual": 0.7}}  # fmt: skip
    assert await matches_factuality("i", "e", "o", grading) == _factuality_result(True, SUBSET, 0.8)


async def test_factuality_provider_error_is_grader_failure(grader: FakeProvider):
    grader.responses = [{"error": "Grader provider unavailable"}]
    result = await matches_factuality("Input text", "Expected output", "Sample output", {})
    assert result["reason"] == "Grader provider unavailable"
    assert result["metadata"] == {"graderError": True}
    assert (result["pass"], result["score"]) == (False, 0)


async def test_factuality_exception_propagates(grader: FakeProvider):
    grader.responses = [RuntimeError("An error occurred")]
    with pytest.raises(RuntimeError, match="An error occurred"):
        await matches_factuality("i", "e", "o", {})


async def test_factuality_requires_grading_config():
    with pytest.raises(ValueError, match="Cannot grade output without grading config"):
        await matches_factuality("i", "e", "o", None)


async def test_factuality_default_prompt_render(grader: FakeProvider):
    await matches_factuality(
        "What is the capital of France?", "Paris", "The capital of France is Paris.", {}
    )
    golden = next(g for g in GOLDEN if g["name"] == "factuality default")
    prompt, context = grader.calls[0]
    assert prompt == golden["rendered"]
    assert context is not None
    assert context["isGrading"] is True
    assert context["prompt"] == {"raw": golden["rendered"], "label": "factuality"}


async def test_factuality_custom_prompt_and_provider():
    custom = FakeProvider({"output": "(B) The submitted answer is a superset of the expert answer.",
                           "tokenUsage": TOKENS})  # fmt: skip
    rubric = (
        "Compare these answers:\nQuestion: {{input}}\nReference: {{ideal}}\n"
        "Submitted: {{completion}}\n\nDetermine if submitted answer is factually correct."
    )
    result = await matches_factuality(
        "What is the capital of France?",
        "Paris",
        "The capital of France is Paris.",
        {"rubricPrompt": rubric, "provider": custom},
    )
    assert result == _factuality_result(
        True, "The submitted answer is a superset of the expert answer.", 1
    )
    assert custom.calls[0][0] == (
        "Compare these answers:\nQuestion: What is the capital of France?\nReference: Paris\n"
        "Submitted: The capital of France is Paris.\n\nDetermine if submitted answer is "
        "factually correct."
    )


async def test_factuality_disable_templating_still_renders_grader(
    grader: FakeProvider, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PROMPTFOO_DISABLE_TEMPLATING", "true")
    grader.responses = [{"output": '{"category": "A", "reason": "ok"}', "tokenUsage": TOKENS}]
    await matches_factuality(
        "Input {{ var }}", "Expected {{ var }}", "Output {{ var }}", {"provider": grader}
    )
    prompt, context = grader.calls[0]
    assert context is not None
    assert context["vars"] == {"input": "Input {{ var }}", "ideal": "Expected {{ var }}",
                               "completion": "Output {{ var }}"}  # fmt: skip
    # Grader rendering ignores PROMPTFOO_DISABLE_TEMPLATING; the values are inserted verbatim.
    assert "Input {{ var }}" in prompt


async def test_factuality_reserved_vars_win(grader: FakeProvider):
    await matches_factuality(
        "input from prompt",
        "ideal from assertion",
        "completion from provider",
        {"rubricPrompt": "input={{ input }}\nideal={{ ideal }}\ncompletion={{ completion }}\n"
                         "extra={{ extra }}"},
        {"input": "vars input sentinel", "ideal": "vars ideal sentinel",
         "completion": "vars completion sentinel", "extra": "kept user var"},
    )  # fmt: skip
    prompt, context = grader.calls[0]
    assert prompt == (
        "input=input from prompt\nideal=ideal from assertion\ncompletion=completion from provider"
        "\nextra=kept user var"
    )
    assert context is not None
    assert context["vars"] == {
        "input": "input from prompt",
        "ideal": "ideal from assertion",
        "completion": "completion from provider",
        "extra": "kept user var",
    }


# --- matchesClosedQa ----------------------------------------------------------------------------


async def test_closed_qa_pass(grader: FakeProvider):
    grader.responses = [{"output": "foo \n \n bar\n Y Y \n", "tokenUsage": TOKENS}]
    assert await matches_closed_qa("Input text", "Expected output", "Sample output", {}) == {
        "pass": True,
        "score": 1,
        "reason": "The submission meets the criterion:\nfoo \n \n bar\n Y Y \n",
        "tokensUsed": NORMALIZED_TOKENS,
    }


async def test_closed_qa_fail(grader: FakeProvider):
    grader.responses = [{"output": "foo bar N \n", "tokenUsage": TOKENS}]
    result = await matches_closed_qa("Input text", "Expected output", "Sample output", {})
    assert result == {
        "pass": False,
        "score": 0,
        "reason": "The submission does not meet the criterion:\nfoo bar N \n",
        "tokensUsed": NORMALIZED_TOKENS,
    }


async def test_closed_qa_malformed_is_grader_failure(grader: FakeProvider):
    grader.responses = [{"output": "maybe", "tokenUsage": TOKENS}]
    result = await matches_closed_qa("i", "e", "o", {})
    assert result["reason"] == "Model grader produced a malformed response:\nmaybe"
    assert result["metadata"] == {"graderError": True}


@pytest.mark.parametrize(
    ("response", "reason"),
    [({"error": "Grader provider unavailable"}, "Grader provider unavailable"),
     ({"output": None}, "No output")],
)  # fmt: skip
async def test_closed_qa_provider_failures(grader: FakeProvider, response, reason):
    grader.responses = [response]
    result = await matches_closed_qa("Input text", "Expected output", "Sample output", {})
    assert (result["pass"], result["score"], result["reason"]) == (False, 0, reason)
    assert result["metadata"] == {"graderError": True}
    assert result["tokensUsed"]["numRequests"] == 0


async def test_closed_qa_escaping_keeps_json_prompt(grader: FakeProvider):
    grader.responses = [{"output": "foo \n \n bar\n Y Y", "tokenUsage": TOKENS}]
    await matches_closed_qa(
        'Input "text" with \\ escape characters and \\"nested\\" escapes',
        'Expected "output" with \\\\ escape characters and \\"nested\\" escapes',
        'Sample "output" with \\\\ escape characters and \\"nested\\" escapes',
        {},
    )
    messages = json.loads(grader.calls[0][0])
    assert 'Input "text" with \\ escape characters' in messages[0]["content"]


async def test_closed_qa_default_prompt_render(grader: FakeProvider):
    grader.responses = [{"output": "Y", "tokenUsage": TOKENS}]
    await matches_closed_qa("Tell me a joke", "is funny", "[1, 2]", {})
    golden = next(g for g in GOLDEN if g["name"] == "closedqa default")
    assert grader.calls[0][0] == golden["rendered"]
    context = grader.calls[0][1]
    assert context is not None
    assert context["prompt"]["label"] == "model-graded-closedqa"
    assert context["vars"]["completion"] == [1, 2]


async def test_closed_qa_reserved_vars_win(grader: FakeProvider):
    grader.responses = [{"output": "Y", "tokenUsage": TOKENS}]
    await matches_closed_qa(
        "input from prompt",
        "criteria from assertion",
        "completion from provider",
        {"rubricPrompt": "input={{ input }}\ncriteria={{ criteria }}\n"
                         "completion={{ completion }}\nextra={{ extra }}"},
        {"input": "x", "criteria": "y", "completion": "z", "extra": "kept user var"},
    )  # fmt: skip
    assert grader.calls[0][0] == (
        "input=input from prompt\ncriteria=criteria from assertion\n"
        "completion=completion from provider\nextra=kept user var"
    )


# --- matchesAgentRubric -------------------------------------------------------------------------


def _agent_response() -> dict[str, Any]:
    return {
        "output": json.dumps({"pass": True, "score": 1, "reason": "agent verified evidence"}),
        "metadata": {"toolCalls": [{"name": "read_file"}]},
        "tokenUsage": {"total": 5, "prompt": 3, "completion": 2},
    }


@pytest.fixture
def load_provider(monkeypatch: pytest.MonkeyPatch):
    claude = FakeProvider(_agent_response(), "anthropic:claude-agent-sdk")
    loaded: dict[str, Any] = {"provider": claude, "calls": []}

    async def fake_load(provider_id, **kw):
        loaded["calls"].append((provider_id, kw))
        return loaded["provider"]

    monkeypatch.setattr(providers_pkg, "load_api_provider", fake_load)
    return loaded


async def test_agent_rubric_explicit_agent_runtime(load_provider):
    result = await matches_agent_rubric(
        "Inspect the artifact", "done", {"provider": "anthropic:claude-agent-sdk"}
    )
    assert load_provider["calls"][0][0] == "anthropic:claude-agent-sdk"
    claude = load_provider["provider"]
    prompt, context = claude.calls[0]
    assert "untrusted evidence" in prompt
    assert context["prompt"]["label"] == "agent-rubric"
    assert "config" not in context["prompt"]
    assert result["pass"] is True
    assert result["metadata"]["agentProvider"] == "anthropic:claude-agent-sdk"
    assert result["metadata"]["toolCalls"] == [{"name": "read_file"}]


async def test_agent_rubric_default_prompt_render():
    codex = FakeProvider(_agent_response(), "openai:codex-sdk")
    result = await matches_agent_rubric(
        "Creates foo.py", "Created file foo.py", {"provider": codex}
    )
    golden = next(g for g in GOLDEN if g["name"] == "agent-rubric default")
    assert codex.calls[0][0] == golden["rendered"]
    assert result["metadata"]["renderedGradingPrompt"] == golden["rendered"]


async def test_agent_rubric_reserved_vars_win():
    codex = FakeProvider(_agent_response(), "openai:codex-sdk")
    await matches_agent_rubric(
        "rubric from assertion",
        "output from provider",
        {"provider": codex,
         "rubricPrompt": "output={{ output }}\nrubric={{ rubric }}\nextra={{ extra }}"},
        {"output": "vars output sentinel", "rubric": "vars rubric sentinel",
         "extra": "kept user var"},
    )  # fmt: skip
    prompt, context = codex.calls[0]
    assert (
        prompt == "output=output from provider\nrubric=rubric from assertion\nextra=kept user var"
    )
    assert context["vars"] == {"output": "output from provider", "rubric": "rubric from assertion",
                               "extra": "kept user var"}  # fmt: skip


@pytest.mark.parametrize(
    "provider_id",
    ["openai:responses:gpt-5.5", "openai:codex-security", "openai:codex-security:gpt-5.6-sol"],
)
async def test_agent_rubric_rejects_non_agentic_graders(load_provider, provider_id: str):
    provider = FakeProvider({"output": json.dumps({"findings": []})}, provider_id)
    load_provider["provider"] = provider
    with pytest.raises(ValueError, match="requires an agentic grading provider"):
        await matches_agent_rubric("Inspect the artifact", "done", {"provider": provider_id})
    assert provider.calls == []


async def test_agent_rubric_without_provider_has_no_codex_default():
    with pytest.raises(ValueError, match="requires an agentic grading provider"):
        await matches_agent_rubric("Verify", "It exists", {})


async def test_agent_rubric_applies_threshold():
    codex = FakeProvider(
        {"output": json.dumps({"pass": True, "score": 0.4, "reason": "partial evidence"})},
        "openai:codex-sdk",
    )
    result = await matches_agent_rubric(
        "Verify evidence",
        "done",
        {"provider": codex},
        {},
        {"type": "agent-rubric", "value": "Verify evidence", "threshold": 0.5},
    )
    assert (result["pass"], result["score"], result["reason"]) == (False, 0.4, "partial evidence")


@pytest.mark.parametrize("working_dir", ["/tmp", {"dir": "/tmp"}])
async def test_agent_rubric_never_trusts_reported_working_dir(working_dir):
    codex = FakeProvider(_agent_response(), "openai:codex-sdk")
    await matches_agent_rubric("Check", "Done", {"provider": codex}, {}, None, None, working_dir)
    assert "config" not in codex.calls[0][1]["prompt"]


def test_is_agentic_provider():
    assert agent_matchers.is_agentic_provider(FakeProvider(None, "opencode:sdk"))
    assert agent_matchers.is_agentic_provider(FakeProvider(None, "openai:codex-sdk:gpt-5"))
    assert not agent_matchers.is_agentic_provider(FakeProvider(None, "openai:codex-sdkx"))
    assert not agent_matchers.is_agentic_provider(None)


# --- matchesTrajectoryGoalSuccess -------------------------------------------------------------


async def test_trajectory_goal_success_renders_prompt():
    provider = FakeProvider(
        {"output": json.dumps({"pass": True, "score": 0.85, "reason": "Goal achieved"}),
         "tokenUsage": {"total": 9, "prompt": 5, "completion": 4}}
    )  # fmt: skip
    grading = {
        "provider": provider,
        "rubricPrompt": "Goal={{ goal }}\nTrajectory={{ trajectory }}\nOutput={{ output }}",
    }
    result = await matches_trajectory_goal_success(
        "Resolve the order lookup task",
        '{"stepCount":2}',
        "The order shipped yesterday.",
        grading,
        {"orderId": "123"},
    )
    prompt, context = provider.calls[0]
    rendered = (
        'Goal=Resolve the order lookup task\nTrajectory={"stepCount":2}\n'
        "Output=The order shipped yesterday."
    )
    assert prompt == rendered
    assert context["prompt"] == {"raw": rendered, "label": "trajectory:goal-success"}
    assert context["vars"] == {"orderId": "123", "goal": "Resolve the order lookup task",
                               "output": "The order shipped yesterday.",
                               "trajectory": '{"stepCount":2}'}  # fmt: skip
    assert result == {
        "pass": True,
        "score": 0.85,
        "reason": "Goal achieved",
        "tokensUsed": {"total": 9, "prompt": 5, "completion": 4, "cached": 0, "numRequests": 0,
                       "completionDetails": NORMALIZED_TOKENS["completionDetails"]},
        "metadata": {"renderedGradingPrompt": rendered},
    }  # fmt: skip


async def test_trajectory_goal_success_requires_grading():
    with pytest.raises(ValueError, match="Cannot grade output without grading config"):
        await matches_trajectory_goal_success("Resolve", '{"stepCount":1}', "output", None)


async def test_trajectory_goal_success_threshold():
    provider = FakeProvider(
        {"output": json.dumps({"pass": True, "score": 0.6, "reason": "Partial progress"})}
    )
    result = await matches_trajectory_goal_success(
        "Resolve the task",
        '{"stepCount":1}',
        "I found part of the answer.",
        {"provider": provider, "rubricPrompt": "Goal={{ goal }}"},
        {},
        {"type": "trajectory:goal-success", "threshold": 0.8, "value": "Resolve the task"},
    )
    assert (result["pass"], result["score"], result["reason"]) == (False, 0.6, "Partial progress")


async def test_trajectory_goal_success_default_prompt_render():
    provider = FakeProvider({"output": '{"pass": true, "score": 1, "reason": "ok"}'})
    case = CASES["trajectory goal success default"]["context"]
    await matches_trajectory_goal_success(
        case["goal"], case["trajectory"], case["output"], {"provider": provider}
    )
    golden = next(g for g in GOLDEN if g["name"] == "trajectory goal success default")
    assert provider.calls[0][0] == golden["rendered"]


# --- summarizeTrajectoryForJudge and the trajectory:goal-success handler ---------------------

TRACE: dict[str, Any] = {
    "traceId": "test-trace-id",
    "evaluationId": "test-evaluation-id",
    "testCaseId": "test-test-case-id",
    "metadata": {"test": "value"},
    "spans": [
        {"spanId": "span-1", "name": "chat gpt-5", "startTime": 1000, "endTime": 1800,
         "attributes": {"promptfoo.provider.id": "openai:gpt-5"}},
        {"spanId": "span-2", "name": "tool.call", "startTime": 1100, "endTime": 1200,
         "attributes": {"tool.name": "search_orders"}},
        {"spanId": "span-3", "name": "tool.call", "startTime": 1400, "endTime": 1500,
         "attributes": {"tool.name": "compose_reply"}},
    ],
}  # fmt: skip


def test_summarize_trajectory_compacts_and_truncates():
    spans = [
        {"spanId": f"search-{i}", "name": "document_search", "startTime": 1000 + i,
         "endTime": 1100 + i, "attributes": {"query": "refund policy"}}
        for i in range(4)
    ] + [
        {"spanId": f"tool-{i}", "name": "tool.call", "startTime": 2000 + i, "endTime": 2100 + i,
         "attributes": {"tool.name": f"tool_{i + 1}"}}
        for i in range(25)
    ]  # fmt: skip
    summary = json.loads(summarize_trajectory_for_judge({**TRACE, "spans": spans}))
    assert (summary["stepCount"], summary["compactedStepCount"]) == (29, 26)
    assert len(summary["steps"]) == 25
    assert summary["steps"][0] == {"index": 1, "type": "search", "name": "refund policy",
                                   "spanName": "document_search", "collapsedCount": 4}  # fmt: skip
    assert summary["steps"][12] == {"omittedCount": 2}
    assert summary["steps"][-1] == {"index": 29, "type": "tool", "name": "tool_25",
                                    "spanName": "tool.call"}  # fmt: skip


def test_summarize_trajectory_is_pretty_json():
    text = summarize_trajectory_for_judge(TRACE)
    assert text.startswith('{\n  "traceId": "test-trace-id",\n  "stepCount": 3,')


def _goal_params(**overrides: Any) -> AssertionParams:
    base: dict[str, Any] = {
        "assertion": {"type": "trajectory:goal-success", "value": "Resolve the order lookup task"},
        "base_type": "trajectory:goal-success",
        "inverse": False,
        "output": "test output",
        "output_string": "test output",
        "provider_response": {"output": "test output"},
        "test": {},
        "assertion_value_context": {"vars": {}, "trace": TRACE},
    }
    base.update(overrides)
    return AssertionParams(**base)


@pytest.fixture
def goal_matcher(monkeypatch: pytest.MonkeyPatch):
    recorded: dict[str, Any] = {"calls": [], "result": {"pass": True, "score": 1, "reason": "ok"}}

    async def fake(*args):
        recorded["calls"].append(args)
        return recorded["result"]

    monkeypatch.setattr(llm_grading, "matches_trajectory_goal_success", fake)
    return recorded


async def test_goal_success_handler_passes_context(goal_matcher):
    goal_matcher["result"] = {"pass": True, "score": 0.9, "reason": "Goal achieved"}
    call_ctx = {"originalProvider": None, "prompt": {"raw": "p", "label": "p"},
                "vars": {"orderId": "123"}}  # fmt: skip
    params = _goal_params(
        assertion_value_context={"vars": {"orderId": "123"}, "trace": TRACE},
        provider_call_context=call_ctx,
        test={"vars": {"orderId": "{{ order_id }}"}},
    )
    assert await handle_trajectory_goal_success(params) == goal_matcher["result"]
    goal, trajectory, output, grading, vars_, assertion, ctx = goal_matcher["calls"][0]
    assert (goal, output, grading, vars_, ctx) == (
        "Resolve the order lookup task",
        "test output",
        None,
        {"orderId": "123"},
        call_ctx,
    )
    assert assertion is params.assertion
    assert '"stepCount": 3' in trajectory
    assert "search_orders" in trajectory and "compose_reply" in trajectory


async def test_goal_success_handler_object_value(goal_matcher):
    value = {"goal": "Send the user a confirmed shipping status"}
    params = _goal_params(assertion={"type": "trajectory:goal-success", "value": value},
                          rendered_value=value)  # fmt: skip
    await handle_trajectory_goal_success(params)
    assert goal_matcher["calls"][0][0] == "Send the user a confirmed shipping status"


async def test_goal_success_handler_inverse(goal_matcher):
    assertion = {"type": "not-trajectory:goal-success", "value": "Resolve the order lookup task"}
    params = _goal_params(inverse=True, assertion=assertion)
    assert await handle_trajectory_goal_success(params) == {
        "pass": False,
        "score": 0,
        "reason": "Agent unexpectedly achieved the goal: Resolve the order lookup task",
        "assertion": assertion,
    }
    goal_matcher["result"] = {"pass": False, "score": 0, "reason": "no"}
    result = await handle_trajectory_goal_success(params)
    assert (result["pass"], result["score"]) == (True, 1)
    assert (
        result["reason"]
        == "Agent did not achieve the forbidden goal: Resolve the order lookup task"
    )


async def test_goal_success_handler_inverse_keeps_grader_failure(goal_matcher):
    failure = {"pass": False, "score": 0, "reason": "Could not extract JSON",
               "metadata": {"graderError": True}}  # fmt: skip
    goal_matcher["result"] = failure
    assertion = {"type": "not-trajectory:goal-success", "value": "x"}
    result = await handle_trajectory_goal_success(_goal_params(inverse=True, assertion=assertion))
    assert result == {**failure, "assertion": assertion}


async def test_goal_success_handler_requires_goal(goal_matcher):
    params = _goal_params(assertion={"type": "trajectory:goal-success", "value": {}},
                          rendered_value={})  # fmt: skip
    with pytest.raises(AssertionValueError, match="must have a string value or an object"):
        await handle_trajectory_goal_success(params)


# --- assertion handlers -------------------------------------------------------------------------


def _params(base_type: str, **overrides: Any) -> AssertionParams:
    base: dict[str, Any] = {
        "assertion": {"type": base_type},
        "base_type": base_type,
        "inverse": False,
        "output": "test output",
        "output_string": "test output",
        "prompt": "test prompt",
        "provider_response": {},
        "rendered_value": "test value",
        "test": {"options": {}, "vars": {}},
    }
    base.update(overrides)
    return AssertionParams(**base)


@pytest.fixture
def recorded_matchers(monkeypatch: pytest.MonkeyPatch):
    recorded: dict[str, Any] = {"calls": [], "result": {"pass": True, "score": 1,
                                                        "reason": "test reason"}}  # fmt: skip

    async def fake(*args):
        recorded["calls"].append(args)
        return recorded["result"]

    for name in ("matches_closed_qa", "matches_factuality"):
        monkeypatch.setattr(llm_grading, name, fake)
    monkeypatch.setattr(agent_matchers, "matches_agent_rubric", fake)
    return recorded


@pytest.mark.parametrize(
    ("base_type", "message"),
    [("model-graded-closedqa", "model-graded-closedqa assertion type must have a string value"),
     ("factuality", "factuality assertion type must have a string value")],
)  # fmt: skip
async def test_closed_qa_and_factuality_require_string(recorded_matchers, base_type, message):
    handler = get_handlers()[base_type]
    with pytest.raises(AssertionValueError, match=message):
        await handler(_params(base_type, rendered_value={}))


@pytest.mark.parametrize("base_type", ["model-graded-closedqa", "factuality"])
async def test_closed_qa_and_factuality_require_prompt(recorded_matchers, base_type):
    with pytest.raises(AssertionValueError, match=f"{base_type} assertion type must have a prompt"):
        await get_handlers()[base_type](_params(base_type, prompt=None))


@pytest.mark.parametrize("base_type", ["model-graded-closedqa", "factuality",
                                       "model-graded-factuality"])  # fmt: skip
async def test_closed_qa_and_factuality_call_matcher(recorded_matchers, base_type):
    params = _params(base_type, test={"options": {"rubricPrompt": "test rubric"},
                                      "vars": {"var": "value"}})  # fmt: skip
    result = await get_handlers()[base_type](params)
    assert recorded_matchers["calls"][0] == (
        "test prompt", "test value", "test output", {"rubricPrompt": "test rubric"},
        {"var": "value"}, None,
    )  # fmt: skip
    assert result == {"pass": True, "score": 1, "reason": "test reason",
                      "assertion": {"type": base_type}}  # fmt: skip


async def test_closed_qa_inverse_and_grader_failure(recorded_matchers):
    recorded_matchers["result"] = {"pass": True, "score": 0.75, "reason": "r"}
    result = await modelgraded.handle_model_graded_closed_qa(
        _params("model-graded-closedqa", inverse=True)
    )
    assert (result["pass"], result["score"]) == (False, 0.25)
    failure = {"pass": False, "score": 0, "reason": "No output", "metadata": {"graderError": True}}
    recorded_matchers["result"] = failure
    result = await modelgraded.handle_model_graded_closed_qa(
        _params("model-graded-closedqa", inverse=True)
    )
    assert result == {**failure, "assertion": {"type": "model-graded-closedqa"}}


def _agent_params(**overrides: Any) -> AssertionParams:
    fields: dict[str, Any] = {
        "assertion": {"type": "agent-rubric", "value": "Verify the claimed change"},
        "output": "Implemented",
        "output_string": "Implemented",
        "rendered_value": "Verify the claimed change",
        "provider_response": {"output": "Implemented"},
    }
    return _params("agent-rubric", **{**fields, **overrides})


async def test_agent_rubric_handler_passes_inputs(recorded_matchers):
    params = _agent_params()
    assert await modelgraded.handle_agent_rubric(params) == recorded_matchers["result"]
    assert recorded_matchers["calls"][0] == (
        "Verify the claimed change", "Implemented", {}, {}, params.assertion, None, None,
    )  # fmt: skip


async def test_agent_rubric_handler_passes_reported_working_dir(recorded_matchers):
    params = _agent_params(
        provider_response={"output": "Implemented", "metadata": {"workingDir": "/tmp/workspace"}}
    )
    await modelgraded.handle_agent_rubric(params)
    assert recorded_matchers["calls"][0][-1] == "/tmp/workspace"


async def test_agent_rubric_handler_structured_rubric_prompt(recorded_matchers):
    rubric_prompt = [{"role": "system", "content": "Inspect the artifact"}]
    params = _agent_params(
        assertion={"type": "agent-rubric"},
        rendered_value=None,
        test={"vars": {}, "options": {"rubricPrompt": rubric_prompt}},
    )
    await modelgraded.handle_agent_rubric(params)
    serialized = json.dumps(rubric_prompt, separators=(",", ":"))
    assert params.test["options"]["rubricPrompt"] == serialized
    assert params.assertion["value"] == serialized
    assert recorded_matchers["calls"][0][:3] == ("", "Implemented", {"rubricPrompt": serialized})


async def test_agent_rubric_handler_inverse(recorded_matchers):
    recorded_matchers["result"] = {"pass": True, "score": 0.8, "reason": "verified"}
    result = await modelgraded.handle_agent_rubric(_agent_params(inverse=True))
    assert result["pass"] is False
    assert result["score"] == pytest.approx(0.2)
    failure = {"pass": False, "score": 0, "reason": "No output", "metadata": {"graderError": True}}
    recorded_matchers["result"] = failure
    params = _agent_params(inverse=True)
    expected = {**failure, "assertion": params.assertion}
    assert await modelgraded.handle_agent_rubric(params) == expected


async def test_pi_is_not_supported():
    with pytest.raises(NotImplementedError, match="pi assertion is not supported"):
        await modelgraded.handle_pi_scorer(_params("pi"))


# --- through run_assertion --------------------------------------------------------------------


async def test_run_assertion_not_factuality_with_provider_override(grader: FakeProvider):
    override = FakeProvider({"output": '{"category": "D", "reason": "disagrees"}'}, "override")
    result = await run_assertion(
        assertion={"type": "not-factuality", "value": "Paris is in {{ country }}",
                   "provider": override},
        test={"vars": {"country": "France"}},
        provider_response={"output": "Paris is in Spain"},
        prompt="Where is Paris?",
    )  # fmt: skip
    assert grader.calls == []
    prompt, context = override.calls[0]
    assert context["vars"]["ideal"] == "Paris is in France"
    assert "Paris is in Spain" in prompt
    assert (result["pass"], result["score"], result["reason"]) == (True, 1, "disagrees")
    assert result["metadata"] == {"renderedAssertionValue": "Paris is in France"}


async def test_run_assertion_closed_qa_rubric_prompt_override(grader: FakeProvider):
    grader.responses = [{"output": "Y"}]
    result = await run_assertion(
        assertion={"type": "model-graded-closedqa", "value": "is polite",
                   "rubricPrompt": "Q={{input}} C={{criteria}} A={{completion}}"},
        test={},
        provider_response={"output": "Hello there"},
        prompt="Greet me",
    )  # fmt: skip
    assert grader.calls[0][0] == "Q=Greet me C=is polite A=Hello there"
    assert result["pass"] is True
    assert result["reason"] == "The submission meets the criterion:\nY"


async def test_run_assertion_closed_qa_transform(grader: FakeProvider):
    grader.responses = [{"output": "N"}]
    await run_assertion(
        assertion={"type": "model-graded-closedqa", "value": "c", "transform": "output.upper()",
                   "rubricPrompt": "A={{completion}}"},
        test={},
        provider_response={"output": "hello"},
        prompt="p",
    )  # fmt: skip
    assert grader.calls[0][0] == "A=HELLO"
