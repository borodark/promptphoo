"""Synthesis prompts and helpers. Ports test/testCases/synthesis*.test.ts and
test/assertions/synthesis.test.ts cases; prompt texts are checked against
tests/fixtures/synthesis_golden.json, written by tools/synthesis_golden.mjs from upstream's own
template functions and the real npm dedent."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from promptphoo.synthesis import (
    InvariantError,
    dedent,
    parse_int,
    retry_with_deduplication,
    sample_array,
)
from promptphoo.synthesis import assertions as assertion_synthesis
from promptphoo.synthesis import dataset as dataset_synthesis

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "synthesis_golden.json").read_text())


@pytest.mark.parametrize("case", GOLDEN["personas"])
def test_generate_personas_prompt_golden(case):
    assert dataset_synthesis.generate_personas_prompt(*case["args"]) == case["out"]


@pytest.mark.parametrize("case", GOLDEN["testCases"])
def test_test_cases_prompt_golden(case):
    assert dataset_synthesis.test_cases_prompt(*case["args"]) == case["out"]


@pytest.mark.parametrize("case", GOLDEN["newQuestions"])
def test_new_questions_prompt_golden(case):
    assert assertion_synthesis.generate_new_questions_prompt(*case["args"]) == case["out"]


@pytest.mark.parametrize("case", GOLDEN["pythonPrompt"])
def test_python_prompt_golden(case):
    out = assertion_synthesis.convert_question_to_python_prompt(*case["args"])
    assert out == case["out"]
    # dedent unescapes `\b` in the raw template, as upstream does.
    assert "\\\b(join" in out


def test_dedent_basics():
    assert (
        dedent(
            (
                "\n    a\n      b ",
                "\n    c\n  ",
            ),
            1,
        )
        == "a\n  b 1\nc"
    )
    assert dedent(("x \\` \\$ \\{ y\\n",)) == "x ` $ { y\n"  # unescaped after trim
    assert dedent(("  line\\\n   cont",)) == "linecont"


def test_parse_int():
    assert parse_int("5") == 5
    assert parse_int(" 12abc") == 12
    assert math.isnan(parse_int("abc"))
    assert math.isnan(parse_int(None))


def test_sample_array():
    items = list(range(10))
    sample = sample_array(items, 3)
    assert len(sample) == 3 and set(sample) <= set(items)
    assert sorted(sample_array(items, 100)) == items
    assert sample_array(items, math.nan) == []


async def test_retry_with_deduplication():
    batches = [[{"a": 1}, {"a": 1}], [{"a": 1}], [{"b": 2}], [], [], []]

    async def op(current):
        return batches.pop(0)

    assert await retry_with_deduplication(op, 5) == [{"a": 1}, {"b": 2}]
    assert batches == []


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ('{"personas": ["A", "  B  ", ""]}', ["A", "B"]),
        ('["x", {"persona": "y"}, {"name": "z"}, {"description": "w"}, 5]', ["x", "y", "z", "w"]),
        ('```json\n{"user_personas": [{"name": "Dev"}]}\n```', ["Dev"]),
        ('Here you go: {"results": ["R1"]} thanks', ["R1"]),
        ('{"personas": []} {"persona_list": ["L"]}', ["L"]),
        ("nothing here", []),
    ],
)
def test_extract_personas(output, expected):
    assert dataset_synthesis.extract_personas(output) == expected


class FakeProvider:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.config: dict = {}

    def id(self):
        return "fake"

    async def call_api(self, prompt, context=None, options=None):
        self.prompts.append(prompt)
        out = self.outputs.pop(0)
        return out if isinstance(out, dict) else {"output": out}


@pytest.fixture
def fake_default(monkeypatch):
    def install(outputs):
        provider = FakeProvider(outputs)

        async def defaults(env=None):
            return {"synthesizeProvider": provider}

        monkeypatch.setattr("promptphoo.providers.defaults.get_default_providers", defaults)
        return provider

    return install


async def test_dataset_synthesize(fake_default):
    provider = fake_default(
        [
            '{"personas": ["Student", "Teacher"]}',
            '{"vars": [{"topic": "a"}, {"topic": "b"}, {"topic": "a"}]}',
            {"output": {"vars": [{"topic": "c"}, {"topic": "d"}]}},
        ]
    )
    out = await dataset_synthesis.synthesize_from_test_suite(
        {"prompts": [{"raw": "Tell me about {{topic}}"}], "tests": [{"vars": {"topic": "z"}}]},
        {"num_personas": 2, "num_test_cases_per_persona": 2},
    )
    assert out == [{"topic": "a"}, {"topic": "b"}, {"topic": "c"}, {"topic": "d"}]
    assert provider.prompts[0] == dataset_synthesis.generate_personas_prompt(
        ["Tell me about {{topic}}"], 2
    )
    # The second batch asks for the remaining two with the persona at index 2 % 2.
    assert provider.prompts[2] == dataset_synthesis.test_cases_prompt(
        ["Tell me about {{topic}}"], "Student", [{"vars": {"topic": "z"}}], 2, ["topic"], None
    )


async def test_dataset_synthesize_errors(fake_default):
    with pytest.raises(ValueError, match="requires at least one prompt"):
        await dataset_synthesis.synthesize(prompts=[], tests=[])
    fake_default(["no personas"])
    with pytest.raises(InvariantError, match="Expected at least one user persona"):
        await dataset_synthesis.synthesize(prompts=["p"], tests=[])
    fake_default(['{"personas": ["P"]}', '{"vars": "nope"}'])
    with pytest.raises(InvariantError, match='contain a "vars" array'):
        await dataset_synthesis.synthesize(prompts=["p"], tests=[])


async def test_assertions_synthesize(fake_default):
    questions = {
        "questions": [
            {"label": "Word Count", "question": "Is the output under 50 words?"},
            {"label": "Tone", "question": "Is the tone friendly?"},
        ]
    }
    provider = fake_default(
        [json.dumps(questions), "words = output.split()\nreturn len(words) < 50", " None "]
    )
    out = await assertion_synthesis.synthesize(
        prompts=["Answer {{q}}"], tests=[], num_questions=2, type="llm-rubric",
        instructions="Focus on brevity.",
    )  # fmt: skip
    by_metric = {a["metric"]: a for a in out}
    assert len(out) == 2
    # Each question is converted concurrently, so which question got which reply depends on
    # the sampled order; both shapes must be valid assertions.
    for assertion in by_metric.values():
        assert assertion["type"] in ("python", "llm-rubric")
    assert {a["type"] for a in out} == {"python", "llm-rubric"}
    assert provider.prompts[0] == (
        assertion_synthesis.generate_new_questions_prompt(["Answer {{q}}"], [], 2)
        + "\nFocus on brevity."
    )
    assert provider.config == {"maxTokens": 3000}


async def test_assertions_synthesize_bad_shape(fake_default):
    fake_default(['{"something": 1}'])
    with pytest.raises(InvariantError, match=r"of the form \{questions"):
        await assertion_synthesis.synthesize(prompts=["p"], tests=[])
    fake_default(["no json"])
    with pytest.raises(InvariantError, match="got 0"):
        await assertion_synthesis.synthesize(prompts=["p"], tests=[])
