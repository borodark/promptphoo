"""Ports test/matchers/g-eval.test.ts and test/assertions/geval.test.ts.

The grader is a fake provider; the default grading provider is replaced by monkeypatching
`promptphoo.providers.defaults.get_default_providers`. Handler tests replace
`matches_g_eval` the way upstream's vi.mock does.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from promptphoo.assertions.base import AssertionParams, AssertionValueError
from promptphoo.assertions.index import run_assertion
from promptphoo.assertions.modelgraded import handle_g_eval
from promptphoo.matchers import llm_grading
from promptphoo.matchers.llm_grading import (
    GEVAL_PROMPT_EVALUATE,
    GEVAL_PROMPT_STEPS,
    matches_g_eval,
)
from promptphoo.providers import defaults
from promptphoo.providers.base import ApiProvider

UPSTREAM_PROMPTS = next(
    (
        p / "promptfoo/src/prompts/index.ts"
        for p in Path(__file__).resolve().parents
        if (p / "promptfoo/src/prompts/index.ts").exists()
    ),
    Path("/nonexistent"),
)


class FakeGrader(ApiProvider):
    """Records every call; answers from a queue of responses or a handler function."""

    def __init__(self, responses: list[Any] | None = None, handler: Any = None, pid: str = "fake"):
        super().__init__({})
        self.responses = list(responses or [])
        self.handler = handler
        self.pid = pid
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def default_id(self) -> str:
        return self.pid

    async def call_api(self, prompt: str, context: Any = None, options: Any = None) -> Any:
        self.calls.append((prompt, context))
        if self.handler is not None:
            return self.handler(prompt)
        return self.responses.pop(0)


def clarity_steps() -> dict[str, Any]:
    return {
        "output": '{"steps": ["Check clarity"]}',
        "tokenUsage": {"total": 3, "prompt": 1, "completion": 2},
    }


def writing_steps() -> dict[str, Any]:
    return {
        "output": '{"steps": ["Check clarity", "Evaluate coherence", "Assess grammar"]}',
        "tokenUsage": {"total": 10, "prompt": 5, "completion": 5},
    }


def default_handler(prompt: str) -> dict[str, Any]:
    if "generate 3-4 concise evaluation steps" in prompt:
        return writing_steps()
    return {
        "output": '{"score": 8, "reason": "The response is well-structured and clear"}',
        "tokenUsage": {"total": 15, "prompt": 8, "completion": 7},
    }


@pytest.fixture
def default_grader(monkeypatch: pytest.MonkeyPatch) -> FakeGrader:
    grader = FakeGrader(handler=default_handler, pid="openai:gpt-6-sol")

    async def fake_defaults(env: Any = None) -> dict[str, Any]:
        return {"gradingProvider": grader, "gradingJsonProvider": grader}

    monkeypatch.setattr(defaults, "get_default_providers", fake_defaults)
    return grader


def queue(grader: FakeGrader, *responses: dict[str, Any]) -> None:
    grader.handler = None
    grader.responses = list(responses)


def contains(total: int, prompt: int | None = None, completion: int | None = None):
    expected = {"total": total, "prompt": prompt, "completion": completion}
    return {k: v for k, v in expected.items() if v is not None}


def assert_usage(result: dict[str, Any], **expected: int) -> None:
    usage = result["tokensUsed"]
    assert {k: usage[k] for k in expected} == expected


def render_steps(criteria: str) -> str:
    return GEVAL_PROMPT_STEPS.replace("{{criteria}}", criteria)


def render_evaluate(criteria: str, steps: list[str], input_: str, output: str) -> str:
    return (
        GEVAL_PROMPT_EVALUATE.replace("{{criteria}}", criteria)
        .replace("{{steps}}", "\n- ".join(steps))
        .replace("{{maxScore}}", "10")
        .replace("{{input}}", input_)
        .replace("{{output}}", output)
    )


# --- vendored prompt text ------------------------------------------------------------------------


@pytest.mark.skipif(not UPSTREAM_PROMPTS.exists(), reason="upstream promptfoo checkout not found")
@pytest.mark.parametrize("name", ["GEVAL_PROMPT_STEPS", "GEVAL_PROMPT_EVALUATE"])
def test_geval_prompts_are_byte_identical_to_upstream(name: str):
    source = UPSTREAM_PROMPTS.read_text(encoding="utf-8")
    match = re.search(rf"export const {name} = `(.*?)`;", source, re.S)
    assert match
    assert getattr(llm_grading, name) == match.group(1)


def test_geval_prompt_key_phrases():
    assert GEVAL_PROMPT_STEPS.startswith("\nGiven evaluation criteria that outline how")
    assert GEVAL_PROMPT_STEPS.endswith("formatted as required in a minified JSON:\nJSON:\n")
    # Upstream keeps trailing spaces after "two keys:" and "IMPORTANT:".
    assert "return a JSON with two keys: \n" in GEVAL_PROMPT_EVALUATE
    assert "IMPORTANT: \n- Return output ONLY" in GEVAL_PROMPT_EVALUATE


# --- matchesGEval --------------------------------------------------------------------------------


async def test_evaluates_with_default_prompts(default_grader: FakeGrader):
    criteria = "Evaluate coherence and clarity"
    result = await matches_g_eval(criteria, "Test input", "Test output", 0.7)
    assert {k: result[k] for k in ("pass", "score", "reason")} == {
        "pass": True,
        "score": 0.8,
        "reason": "The response is well-structured and clear",
    }
    assert set(result) == {"pass", "score", "reason", "tokensUsed"}
    assert len(default_grader.calls) == 2

    steps_prompt, steps_context = default_grader.calls[0]
    assert steps_prompt == render_steps(criteria)
    assert steps_context is not None
    assert steps_context["prompt"] == {"raw": steps_prompt, "label": "g-eval-steps"}
    assert steps_context["vars"] == {"criteria": criteria}
    assert steps_context["isGrading"] is True

    eval_prompt, eval_context = default_grader.calls[1]
    assert eval_prompt == render_evaluate(
        criteria,
        ["Check clarity", "Evaluate coherence", "Assess grammar"],
        "Test input",
        "Test output",
    )
    assert "**Evaluation Steps**\n- Check clarity\n- Evaluate coherence\n- Assess grammar\n" in (
        eval_prompt
    )
    assert eval_context is not None
    assert eval_context["prompt"]["label"] == "g-eval"


@pytest.mark.parametrize(
    "criteria",
    [
        "the answer should contain harmful or inappropriate content",
        "the answer should not contain harmful or inappropriate content",
    ],
    ids=["positive", "negated"],
)
async def test_renders_observed_condition_scoring(default_grader: FakeGrader, criteria: str):
    await matches_g_eval(criteria, "Test input", "Test output", 0.7)
    assert len(default_grader.calls) == 2
    first, second = default_grader.calls[0][0], default_grader.calls[1][0]
    assert criteria in first
    assert criteria in second
    assert (
        "the condition described by the Evaluation Criteria is fully and clearly observed in "
        "the Reply"
    ) in second
    assert "and 0 indicates that it is not observed at all" in second


async def test_custom_rubric_prompts(default_grader: FakeGrader):
    queue(
        default_grader,
        {
            "output": '{"steps": ["Custom step 1", "Custom step 2"]}',
            "tokenUsage": {"total": 10, "prompt": 5, "completion": 5},
        },
        {
            "output": '{"score": 8, "reason": "Custom evaluation complete", "pass": true}',
            "tokenUsage": {"total": 15, "prompt": 8, "completion": 7},
        },
    )
    grading = {
        "rubricPrompt": {
            "steps": "Custom steps template with {{criteria}}",
            "evaluate": "Custom evaluation template with {{criteria}} and {{steps}}",
        }
    }
    result = await matches_g_eval(
        "Evaluate coherence and clarity", "Test input", "Test output", 0.7, grading
    )
    assert result["score"] == 0.8
    assert [p for p, _ in default_grader.calls] == [
        "Custom steps template with Evaluate coherence and clarity",
        "Custom evaluation template with Evaluate coherence and clarity and Custom step 1\n- "
        "Custom step 2",
    ]


async def test_configured_provider_receives_context_for_both_phases(default_grader: FakeGrader):
    criteria = "Reward factual grounding and completeness"
    provider = FakeGrader(
        [
            {
                "output": '{"steps": ["Check factual alignment", "Check completeness"]}',
                "tokenUsage": {"total": 11, "prompt": 6, "completion": 5},
            },
            {
                "output": '{"score": 8, "reason": "The answer is grounded and complete"}',
                "tokenUsage": {"total": 13, "prompt": 7, "completion": 6},
            },
        ],
        pid="litellm:gemini-pro",
    )
    result = await matches_g_eval(
        criteria,
        "Question: what is the capital of France?",
        "Paris is the capital of France.",
        0.7,
        {"provider": provider},
    )
    assert {k: result[k] for k in ("pass", "score", "reason")} == {
        "pass": True,
        "score": 0.8,
        "reason": "The answer is grounded and complete",
    }
    assert_usage(result, total=24, prompt=13, completion=11)
    assert default_grader.calls == []
    assert len(provider.calls) == 2
    first_prompt, first_ctx = provider.calls[0]
    assert "generate 3-4 concise evaluation steps" in first_prompt
    assert first_ctx is not None
    assert first_ctx["prompt"]["label"] == "g-eval-steps"
    assert first_ctx["vars"] == {"criteria": criteria}
    second_prompt, second_ctx = provider.calls[1]
    assert "Check factual alignment" in second_prompt
    assert second_ctx is not None
    assert second_ctx["prompt"]["label"] == "g-eval"
    v = second_ctx["vars"]
    assert v["criteria"] == criteria
    assert v["input"] == "Question: what is the capital of France?"
    assert v["maxScore"] == "10"
    assert v["output"] == "Paris is the capital of France."
    assert "Check completeness" in v["steps"]


async def test_fails_when_score_below_threshold(default_grader: FakeGrader):
    queue(
        default_grader,
        writing_steps(),
        {
            "output": '{"score": 3, "reason": "The response lacks coherence"}',
            "tokenUsage": {"total": 15, "prompt": 8, "completion": 7},
        },
    )
    result = await matches_g_eval("Evaluate coherence and clarity", "Test input", "Out", 0.7)
    assert {k: result[k] for k in ("pass", "score", "reason")} == {
        "pass": False,
        "score": 0.3,
        "reason": "The response lacks coherence",
    }


async def test_step_generation_provider_error(default_grader: FakeGrader):
    queue(
        default_grader,
        {"error": "steps provider unavailable", "tokenUsage": contains(3, 1, 2)},
    )
    result = await matches_g_eval("Evaluate coherence", "Test input", "Test output", 0.7)
    assert result["pass"] is False
    assert result["score"] == 0
    assert result["reason"] == "steps provider unavailable"
    assert result["metadata"] == {"graderError": True}
    assert_usage(result, total=3, prompt=1, completion=2)
    assert len(default_grader.calls) == 1


async def test_step_generation_no_output_names_grader(default_grader: FakeGrader):
    provider = FakeGrader(
        [{"output": "", "tokenUsage": contains(3, 1, 2)}], pid="litellm:gemini-pro"
    )
    result = await matches_g_eval(
        "Evaluate coherence", "Test input", "Test output", 0.7, {"provider": provider}
    )
    assert (
        result["reason"] == "G-Eval step-generation call to litellm:gemini-pro returned no output"
    )
    assert result["metadata"] == {"graderError": True}
    assert (result["pass"], result["score"]) == (False, 0)
    assert_usage(result, total=3, prompt=1, completion=2)
    assert default_grader.calls == []
    assert len(provider.calls) == 1


async def test_invalid_steps_shape(default_grader: FakeGrader):
    queue(default_grader, {"output": '{"steps":"Check clarity"}', "tokenUsage": contains(3, 1, 2)})
    result = await matches_g_eval("Evaluate coherence", "Test input", "Test output", 0.7)
    assert result["reason"] == (
        'G-Eval steps response has invalid or missing steps: "Check clarity"'
    )
    assert result["metadata"] == {"graderError": True}
    assert_usage(result, total=3, prompt=1, completion=2)
    assert len(default_grader.calls) == 1


async def test_evaluation_provider_error(default_grader: FakeGrader):
    queue(
        default_grader,
        clarity_steps(),
        {"error": "evaluation provider unavailable", "tokenUsage": contains(4, 2, 2)},
    )
    result = await matches_g_eval("Evaluate coherence", "Test input", "Test output", 0.7)
    assert result["reason"] == "evaluation provider unavailable"
    assert result["metadata"] == {"graderError": True}
    assert_usage(result, total=7, prompt=3, completion=4)


async def test_evaluation_no_output_names_grader(default_grader: FakeGrader):
    provider = FakeGrader(
        [clarity_steps(), {"tokenUsage": contains(4, 2, 2)}], pid="litellm:gemini-pro"
    )
    result = await matches_g_eval(
        "Evaluate coherence", "Test input", "Test output", 0.7, {"provider": provider}
    )
    assert result["reason"] == "G-Eval evaluation call to litellm:gemini-pro returned no output"
    assert result["metadata"] == {"graderError": True}
    assert_usage(result, total=7, prompt=3, completion=4)
    assert default_grader.calls == []
    assert len(provider.calls) == 2


async def test_evaluation_result_not_a_string(default_grader: FakeGrader):
    queue(
        default_grader,
        clarity_steps(),
        {"output": {"score": 8, "reason": "object output"}, "tokenUsage": contains(4, 2, 2)},
    )
    result = await matches_g_eval("Evaluate coherence", "Test input", "Test output", 0.7)
    assert result["reason"] == "LLM-proposed evaluation result response is not a string"
    assert result["metadata"] == {"graderError": True}
    assert_usage(result, total=7)


async def _evaluate_with(grader: FakeGrader, output: str) -> dict[str, Any]:
    queue(grader, clarity_steps(), {"output": output, "tokenUsage": contains(4, 2, 2)})
    return await matches_g_eval("Evaluate coherence", "Test input", "Test output", 0.7)


@pytest.mark.parametrize(
    ("output", "label"),
    [
        ('{"score": null, "reason": "null score"}', "null"),
        ('{"score": "", "reason": "blank score"}', '""'),
    ],
    ids=["null score", "blank string score"],
)
async def test_invalid_score(default_grader: FakeGrader, output: str, label: str):
    result = await _evaluate_with(default_grader, output)
    assert result["reason"] == f"G-Eval result has invalid or missing score: {label}"
    assert result["metadata"] == {"graderError": True}
    assert (result["pass"], result["score"]) == (False, 0)
    assert_usage(result, total=7)


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ('{"score": 11, "reason": "too high"}', "G-Eval result score 11 is outside the"),
        ('{"score": -1, "reason": "too low"}', "G-Eval result score -1 is outside the"),
    ],
)
async def test_score_out_of_range(default_grader: FakeGrader, output: str, reason: str):
    result = await _evaluate_with(default_grader, output)
    assert result["reason"] == f"{reason} expected 0-10 range"
    assert result["metadata"] == {"graderError": True}
    assert_usage(result, total=7, prompt=3, completion=4)


@pytest.mark.parametrize(
    ("output", "label"),
    [
        ('{"score": 8}', "undefined"),
        ('{"score": 8, "reason": null}', "null"),
        ('{"score": 8, "reason": 42}', "42"),
        ('{"score": 8, "reason": ""}', '""'),
        ('{"score": 8, "reason": "   "}', '"   "'),
    ],
    ids=["missing", "null", "numeric", "empty", "whitespace"],
)
async def test_invalid_reason(default_grader: FakeGrader, output: str, label: str):
    result = await _evaluate_with(default_grader, output)
    assert result["reason"] == f"G-Eval result has invalid or missing reason: {label}"
    assert result["metadata"] == {"graderError": True}
    assert (result["pass"], result["score"]) == (False, 0)
    assert_usage(result, total=7, prompt=3, completion=4)


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ('{"steps": ["ok", 42]}', 'G-Eval steps response contains invalid steps: ["ok",42]'),
        ('{"steps": ["ok", "   "]}', 'G-Eval steps response contains invalid steps: ["ok","   "]'),
        ('{"steps": []}', "LLM does not propose any evaluation step"),
    ],
    ids=["non-string", "whitespace", "empty"],
)
async def test_invalid_steps(default_grader: FakeGrader, output: str, reason: str):
    queue(default_grader, {"output": output, "tokenUsage": contains(3, 1, 2)})
    result = await matches_g_eval("Evaluate coherence", "Test input", "Test output", 0.7)
    assert result["reason"] == reason
    assert result["metadata"] == {"graderError": True}
    assert_usage(result, total=3, prompt=1, completion=2)
    assert len(default_grader.calls) == 1


async def test_steps_not_json(default_grader: FakeGrader):
    queue(default_grader, {"output": "no json here", "tokenUsage": contains(3, 1, 2)})
    result = await matches_g_eval("Evaluate coherence", "Test input", "Test output", 0.7)
    assert result["reason"] == "LLM-proposed evaluation steps are not in JSON format: no json here"
    assert result["metadata"] == {"graderError": True}


async def test_steps_wrapped_in_markdown_fence(default_grader: FakeGrader):
    queue(
        default_grader,
        {"output": '```json\n{"steps": ["Check clarity"]}\n```'},
        {"output": '{"score": 9, "reason": "good"}'},
    )
    result = await matches_g_eval("Evaluate coherence", "Test input", "Test output", 0.7)
    assert (result["pass"], result["score"], result["reason"]) == (True, 0.9, "good")


async def test_evaluation_result_not_json(default_grader: FakeGrader):
    result = await _evaluate_with(default_grader, "score is eight")
    assert result["reason"] == (
        "LLM-proposed evaluation result is not in JSON format: score is eight"
    )
    assert result["metadata"] == {"graderError": True}


async def test_accepts_numeric_string_scores(default_grader: FakeGrader):
    result = await _evaluate_with(default_grader, '{"score": "  7  ", "reason": "fine"}')
    assert result["pass"] is True
    assert result["score"] == pytest.approx(0.7, abs=1e-5)
    assert result["reason"] == "fine"
    assert "metadata" not in result


@pytest.mark.parametrize(
    ("output", "expected"),
    [('{"score": 0, "reason": "absent"}', 0), ('{"score": 10, "reason": "perfect"}', 1)],
    ids=["lower bound", "upper bound"],
)
async def test_accepts_score_bounds(default_grader: FakeGrader, output: str, expected: float):
    result = await _evaluate_with(default_grader, output)
    assert result["score"] == pytest.approx(expected, abs=1e-5)
    assert "metadata" not in result


async def test_tracks_token_usage_for_both_calls(default_grader: FakeGrader):
    result = await matches_g_eval("Evaluate coherence and clarity", "Test input", "Out", 0.7)
    usage = result["tokensUsed"]
    assert {k: v for k, v in usage.items() if k != "completionDetails"} == {
        "total": 25,
        "prompt": 13,
        "completion": 12,
        "cached": 0,
        "numRequests": 0,
    }
    assert isinstance(usage["completionDetails"], dict)


async def test_requires_input(default_grader: FakeGrader):
    with pytest.raises(ValueError, match="No source text to estimate reply"):
        await matches_g_eval("criteria", "", "out", 0.7)


async def test_input_and_output_json_are_stringified_in_prompt(default_grader: FakeGrader):
    queue(default_grader, clarity_steps(), {"output": '{"score": 5, "reason": "ok"}'})
    await matches_g_eval("criteria", '{"q": 1}', '{"a": [1, 2]}', 0.7)
    eval_prompt = default_grader.calls[1][0]
    assert '**Prompt**\n{"q":1}\n' in eval_prompt
    assert '**Reply**\n{"a":[1,2]}\n' in eval_prompt


# --- handleGEval ---------------------------------------------------------------------------------


class MatcherStub:
    def __init__(self, *results: dict[str, Any]) -> None:
        self.results = list(results)
        self.calls: list[tuple[Any, ...]] = []

    async def __call__(self, *args: Any) -> dict[str, Any]:
        self.calls.append(args)
        return self.results[0] if len(self.results) == 1 else self.results.pop(0)


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch):
    def install(*results: dict[str, Any]) -> MatcherStub:
        matcher = MatcherStub(*results)
        monkeypatch.setattr(llm_grading, "matches_g_eval", matcher)
        return matcher

    return install


def passing_grade(score: float = 0.8, reason: str = "test reason") -> dict[str, Any]:
    return {"pass": True, "score": score, "reason": reason}


def passing_geval_result() -> dict[str, Any]:
    return {
        "pass": True,
        "score": 0.8,
        "reason": "test reason 1",
        "tokensUsed": {"total": 10, "prompt": 6, "completion": 4},
    }


def failing_geval_result() -> dict[str, Any]:
    return {
        "pass": False,
        "score": 0.6,
        "reason": "test reason 2",
        "tokensUsed": {"total": 14, "prompt": 8, "completion": 6},
    }


def geval_assertion(type_: str = "g-eval", value: Any = "test criteria", **extra: Any):
    return {"type": type_, "value": value, "threshold": 0.7, **extra}


def make_params(
    assertion: dict[str, Any], rendered: Any, inverse: bool = False, prompt: Any = "test prompt"
) -> AssertionParams:
    return AssertionParams(
        assertion=assertion,
        base_type="g-eval",
        inverse=inverse,
        output="test output",
        output_string="test output",
        rendered_value=rendered,
        prompt=prompt,
        provider_response={"output": "test output"},
        test={"vars": {}, "assert": [], "options": {}},
        assertion_value_context={"prompt": prompt, "vars": {}},
    )


async def test_handler_string_value(stub):
    matcher = stub(passing_grade())
    result = await handle_g_eval(make_params(geval_assertion(), "test criteria"))
    assert result == {
        "assertion": {"type": "g-eval", "value": "test criteria", "threshold": 0.7},
        "pass": True,
        "score": 0.8,
        "reason": "test reason",
    }
    assert matcher.calls == [("test criteria", "test prompt", "test output", 0.7, {}, None)]


async def test_handler_array_value(stub):
    stub(passing_geval_result(), failing_geval_result())
    assertion = geval_assertion(value=["criteria1", "criteria2"])
    result = await handle_g_eval(make_params(assertion, ["criteria1", "criteria2"]))
    assert result["assertion"] == assertion
    assert result["pass"] is True
    assert result["score"] == pytest.approx(0.7)
    assert result["reason"] == "test reason 1\n\ntest reason 2"
    assert_usage(result, total=24, prompt=14, completion=10)


async def test_handler_empty_array(stub):
    matcher = stub(passing_grade())
    for type_, inverse in (("g-eval", False), ("not-g-eval", True)):
        assertion = geval_assertion(type_, [])
        result = await handle_g_eval(make_params(assertion, [], inverse))
        assert result == {
            "assertion": assertion,
            "pass": False,
            "score": 0,
            "reason": "G-Eval assertion requires at least one criterion string in the value array.",
        }
    assert matcher.calls == []


async def test_handler_default_threshold(stub):
    matcher = stub(passing_grade())
    await handle_g_eval(make_params({"type": "g-eval", "value": "test criteria"}, "test criteria"))
    assert matcher.calls == [("test criteria", "test prompt", "test output", 0.7, {}, None)]


@pytest.mark.parametrize("rendered", [None, ["valid criteria", 42]])
async def test_handler_rejects_invalid_value(stub, rendered: Any):
    stub(passing_grade())
    with pytest.raises(
        AssertionValueError,
        match="G-Eval assertion type must have a string or array of strings value",
    ):
        await handle_g_eval(make_params(geval_assertion(value=rendered), rendered))


async def test_handler_undefined_prompt(stub):
    matcher = stub(passing_grade())
    result = await handle_g_eval(make_params(geval_assertion(), "test criteria", prompt=None))
    assert result["pass"] is True
    assert matcher.calls == [("test criteria", "", "test output", 0.7, {}, None)]

    matcher = stub(passing_geval_result(), failing_geval_result())
    assertion = geval_assertion(value=["criteria1", "criteria2"])
    result = await handle_g_eval(make_params(assertion, ["criteria1", "criteria2"], prompt=None))
    assert result["score"] == pytest.approx(0.7)
    assert matcher.calls == [
        ("criteria1", "", "test output", 0.7, {}, None),
        ("criteria2", "", "test output", 0.7, {}, None),
    ]


@pytest.mark.parametrize(
    ("grade", "expected_pass", "expected_score"),
    [
        ({"pass": True, "score": 0.8, "reason": "matches the criterion"}, False, 0.2),
        ({"pass": False, "score": 0.3, "reason": "does not match the criterion"}, True, 0.7),
    ],
)
async def test_handler_inverse_string(stub, grade, expected_pass, expected_score):
    stub(grade)
    assertion = geval_assertion("not-g-eval")
    result = await handle_g_eval(make_params(assertion, "test criteria", inverse=True))
    assert set(result) == {"assertion", "pass", "score", "reason"}
    assert result["assertion"] == assertion
    assert result["pass"] is expected_pass
    assert result["score"] == pytest.approx(expected_score, abs=1e-5)
    assert result["reason"] == grade["reason"]


async def test_handler_inverse_array_passing(stub):
    stub(passing_geval_result(), failing_geval_result())
    assertion = geval_assertion("not-g-eval", ["criteria1", "criteria2"])
    result = await handle_g_eval(make_params(assertion, ["criteria1", "criteria2"], True))
    assert result["pass"] is False
    assert result["score"] == pytest.approx(0.3, abs=1e-5)
    assert result["reason"] == "test reason 1\n\ntest reason 2"
    assert_usage(result, total=24, prompt=14, completion=10)


async def test_handler_inverse_array_failing(stub):
    stub(
        {"pass": False, "score": 0.4, "reason": "test reason 1", "tokensUsed": contains(10, 6, 4)},
        {"pass": False, "score": 0.2, "reason": "test reason 2", "tokensUsed": contains(14, 8, 6)},
    )
    assertion = geval_assertion("not-g-eval", ["criteria1", "criteria2"])
    result = await handle_g_eval(make_params(assertion, ["criteria1", "criteria2"], True))
    assert result["pass"] is True
    assert result["score"] == pytest.approx(0.7, abs=1e-5)
    assert_usage(result, total=24, prompt=14, completion=10)


async def test_handler_threshold_boundary(stub):
    stub(passing_grade(0.7, "exactly at threshold"))
    result = await handle_g_eval(make_params(geval_assertion(), "test criteria"))
    assert (result["pass"], result["score"]) == (True, 0.7)
    result = await handle_g_eval(make_params(geval_assertion("not-g-eval"), "test criteria", True))
    assert result["pass"] is False
    assert result["score"] == pytest.approx(0.3, abs=1e-5)


@pytest.mark.parametrize(
    ("type_", "inverse", "reason", "usage"),
    [
        ("g-eval", False, "No output", contains(11, 5, 6)),
        (
            "not-g-eval",
            True,
            "LLM-proposed evaluation result is not in JSON format: oops",
            contains(21, 12, 9),
        ),
    ],
)
async def test_handler_propagates_string_grader_failure(stub, type_, inverse, reason, usage):
    stub(
        {
            "pass": False,
            "score": 0,
            "reason": reason,
            "tokensUsed": usage,
            "metadata": {"graderError": True},
        }
    )
    assertion = geval_assertion(type_)
    result = await handle_g_eval(make_params(assertion, "test criteria", inverse))
    assert result == {
        "assertion": assertion,
        "pass": False,
        "score": 0,
        "reason": reason,
        "tokensUsed": usage,
        "metadata": {"graderError": True},
    }


async def test_handler_grader_failure_threshold_zero(stub):
    stub({"pass": False, "score": 0, "reason": "No output", "metadata": {"graderError": True}})
    assertion = geval_assertion(threshold=0)
    result = await handle_g_eval(make_params(assertion, "test criteria"))
    # Upstream's `tokensUsed: resp.tokensUsed` is undefined here and drops out of JSON.
    assert {k: v for k, v in result.items() if v is not None} == {
        "assertion": assertion,
        "pass": False,
        "score": 0,
        "reason": "No output",
        "metadata": {"graderError": True},
    }


async def test_handler_array_grader_failure_names_criterion(stub):
    stub(
        {
            "pass": True,
            "score": 0.9,
            "reason": "first criterion matched",
            "tokensUsed": contains(10, 4, 6),
        },
        {
            "pass": False,
            "score": 0,
            "reason": 'G-Eval result has invalid or missing score: "bad"',
            "tokensUsed": contains(17, 9, 8),
            "metadata": {"graderError": True},
        },
    )
    assertion = geval_assertion("not-g-eval", ["criteria1", "criteria2"])
    result = await handle_g_eval(make_params(assertion, ["criteria1", "criteria2"], True))
    assert result["pass"] is False
    assert result["score"] == 0
    assert result["reason"] == (
        'G-Eval criterion 2/2 ("criteria2") failed: G-Eval result has invalid or missing score: '
        '"bad"'
    )
    assert result["metadata"] == {"graderError": True}
    assert_usage(result, total=27, prompt=13, completion=14)


async def test_handler_array_short_circuits_on_grader_failure(stub):
    matcher = stub(
        {
            "pass": False,
            "score": 0,
            "reason": "LLM-proposed evaluation result is not in JSON format: oops",
            "tokensUsed": contains(5, 3, 2),
            "metadata": {"graderError": True},
        }
    )
    values = ["criteria1", "criteria2", "criteria3"]
    result = await handle_g_eval(make_params(geval_assertion(value=values), values))
    assert (result["pass"], result["score"]) == (False, 0)
    assert len(matcher.calls) == 1


async def test_handler_inverts_real_zero_score(stub):
    stub({"pass": False, "score": 0, "reason": "criterion not observed at all"})
    assertion = geval_assertion("not-g-eval")
    result = await handle_g_eval(make_params(assertion, "test criteria", True))
    assert result == {
        "assertion": assertion,
        "pass": True,
        "score": 1,
        "reason": "criterion not observed at all",
    }


# --- runAssertion dispatch -----------------------------------------------------------------------


class TargetProvider(ApiProvider):
    def default_id(self) -> str:
        return "test-provider"

    async def call_api(self, prompt: str, context: Any = None, options: Any = None) -> Any:
        return {"output": "test output"}


async def test_run_assertion_inverts_not_g_eval(stub):
    matcher = stub(
        {"pass": True, "score": 0.9, "reason": "criterion met", "tokensUsed": contains(20, 12, 8)}
    )
    assertion = geval_assertion("not-g-eval", "The output is in English.")
    result = await run_assertion(
        prompt="Say hi.",
        provider=TargetProvider(),
        assertion=assertion,
        test={"vars": {}, "assert": [assertion], "options": {}},
        provider_response={"output": "Hello!"},
    )
    assert result["pass"] is False
    assert result["score"] == pytest.approx(0.1, abs=1e-5)
    assert result["reason"] == "criterion met"
    assert len(matcher.calls) == 1
    assert matcher.calls[0][:4] == ("The output is in English.", "Say hi.", "Hello!", 0.7)


async def test_run_assertion_does_not_flip_grader_failure(stub):
    stub(
        {
            "pass": False,
            "score": 0,
            "reason": "API error: 500 Internal Server Error",
            "tokensUsed": contains(3, 3, 0),
            "metadata": {"graderError": True},
        }
    )
    assertion = geval_assertion("not-g-eval", "The output is in English.")
    result = await run_assertion(
        prompt="Say hi.",
        provider=TargetProvider(),
        assertion=assertion,
        test={"vars": {}, "assert": [assertion], "options": {}},
        provider_response={"output": "Hello!"},
    )
    assert (result["pass"], result["score"]) == (False, 0)
    assert result["reason"] == "API error: 500 Internal Server Error"
    assert result["metadata"] == {"graderError": True}


async def test_run_assertion_end_to_end_with_fake_grader(default_grader: FakeGrader):
    """No stub: the real matcher runs against the fake default grader."""
    assertion = {"type": "g-eval", "value": "Evaluate coherence and clarity"}
    result = await run_assertion(
        prompt="Write a sentence.",
        provider=TargetProvider(),
        assertion=assertion,
        test={"vars": {}, "assert": [assertion], "options": {}},
        provider_response={"output": "A clear sentence."},
    )
    assert (result["pass"], result["score"]) == (True, 0.8)
    assert result["reason"] == "The response is well-structured and clear"
    assert default_grader.calls[1][0] == render_evaluate(
        "Evaluate coherence and clarity",
        ["Check clarity", "Evaluate coherence", "Assess grammar"],
        "Write a sentence.",
        "A clear sentence.",
    )
