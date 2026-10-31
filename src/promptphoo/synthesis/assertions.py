"""Assertion synthesis for `generate assertions`. Ports src/assertions/synthesis.ts.

The prompt texts are the raw segments of upstream's dedent templates in
synthesis/templates.py; `dedent` interpolates them as the npm tag does (including its
unescaping of sequences such as `\\b` inside the Python examples).
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from promptphoo.jsonutil import extract_json_objects
from promptphoo.jsvalues import js_string, json_stringify
from promptphoo.synthesis import dedent, invariant, sample_array
from promptphoo.synthesis.templates import NEW_QUESTIONS, PYTHON_PROMPT

logger = logging.getLogger("promptphoo")


def _prompts_block(prompts: list[str]) -> str:
    return "\n".join(
        f"      {'  ' if i > 0 else ''}<Prompt>\n          {prompt}\n        </Prompt>"
        for i, prompt in enumerate(prompts)
    )


def _num_gt_zero(value: Any) -> bool:
    return not (isinstance(value, float) and math.isnan(value)) and value > 0


def generate_new_questions_prompt(
    prompts: list[str], test_cases: list[dict[str, Any]], num_questions: Any
) -> str:
    all_assertions = [a for c in test_cases for a in (c.get("assert") or [])]
    positive = _num_gt_zero(num_questions)
    assertions_json = "\n".join(
        f"        {line}" if i > 0 else line
        for i, line in enumerate(json_stringify(all_assertions, 2).split("\n"))
    )
    return dedent(
        NEW_QUESTIONS,
        f"Create total {js_string(num_questions)} questions with:"
        if positive
        else "Create a comprehensive set of evaluation questions with:",
        "Do not generate any 'fully_newly_generated' questions if the implied questions alone "
        f"fulfill the requested {js_string(num_questions)}."
        if positive
        else "",
        "```",
        "s" if len(prompts) > 1 else "",
        _prompts_block(prompts),
        assertions_json,
    )


def convert_question_to_python_prompt(prompts: list[str], question: str) -> str:
    return dedent(PYTHON_PROMPT, _prompts_block(prompts), question)


async def synthesize(
    *,
    prompts: list[str],
    tests: list[dict[str, Any]],
    instructions: str | None = None,
    num_questions: Any = 5,
    provider: str | None = None,
    type: str = "pi",  # noqa: A002 - upstream's option name
) -> list[dict[str, Any]]:
    from promptphoo.synthesis.dataset import _load_provider

    if len(prompts) < 1:
        raise ValueError("Assertion synthesis requires at least one prompt.")
    logger.debug("Starting assertion synthesis. We'll begin by generating a set of questions")
    logger.debug(
        "Generating user personas from %d prompt%s...",
        len(prompts),
        "s" if len(prompts) > 1 else "",
    )
    provider_model = await _load_provider(provider)
    new_questions_prompt = generate_new_questions_prompt(prompts, tests, num_questions)
    if instructions:
        new_questions_prompt = f"{new_questions_prompt}\n{instructions}"
    logger.debug("Generated questions prompt:\n%s", new_questions_prompt)
    resp = await provider_model.call_api(new_questions_prompt)
    logger.debug("Received questions response:\n%s", resp.get("output"))
    invariant("output" in resp, "resp.output must be defined")
    output = resp["output"] if isinstance(resp["output"], str) else json_stringify(resp["output"])
    resp_objects = extract_json_objects(output)
    invariant(
        len(resp_objects) >= 1,
        f"Expected at least one JSON object in the response for questions, got {len(resp_objects)}",
    )
    wrapper = next(
        (
            obj
            for obj in resp_objects
            if isinstance(obj, dict)
            and isinstance(obj.get("questions"), list)
            and all(
                isinstance(q, dict)
                and isinstance(q.get("label"), str)
                and isinstance(q.get("question"), str)
                for q in obj["questions"]
            )
        ),
        None,
    )
    invariant(
        wrapper,
        lambda: (
            "Expected a JSON object of the form {questions: [...]} in the questions "
            "response, but the provider returned a different shape. Check the provider's "
            "transformResponse and that it returns the model's text content."
        ),
    )
    questions = sample_array(wrapper["questions"], num_questions)  # type: ignore[index]
    logger.debug(
        "Generated %d question%s:\n%s",
        len(questions), "" if len(questions) == 1 else "s",
        "\n".join(f"  - {q['question']}" for q in questions),
    )  # fmt: skip

    # Decide which questions a deterministic Python check can answer, and write it.
    provider_model.config = {"maxTokens": 3000}

    async def to_assertion(q: dict[str, Any]) -> dict[str, Any]:
        python_prompt = convert_question_to_python_prompt(prompts, q["question"])
        out = (await provider_model.call_api(python_prompt)).get("output")
        if not isinstance(out, str):
            raise TypeError("Cannot read properties of the python assertion response output")
        if out.lower().strip() == "none":
            return {"type": type, "metric": q["label"], "value": q["question"]}
        return {"type": "python", "metric": q["label"], "value": out}

    assertions = list(await asyncio.gather(*(to_assertion(q) for q in questions)))
    logger.debug("Generated %d new assertions", len(assertions))
    return assertions


async def synthesize_from_test_suite(
    test_suite: dict[str, Any], options: dict[str, Any]
) -> list[dict[str, Any]]:
    return await synthesize(
        **options,
        prompts=[prompt["raw"] for prompt in test_suite["prompts"]],
        tests=test_suite.get("tests") or [],
    )
