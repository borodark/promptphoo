"""Dataset synthesis for `generate dataset`. Ports src/testCase/synthesis.ts.

The prompt texts are the raw segments of upstream's dedent templates in
synthesis/templates.py; `dedent` interpolates them as the npm tag does.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from promptphoo import cli_state
from promptphoo.jsonutil import extract_json_objects
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.synthesis import (
    dedent,
    invariant,
    js_sort_key,
    retry_with_deduplication,
    sample_array,
)
from promptphoo.synthesis.templates import (
    PERSONAS,
    PERSONAS_PROMPTS,
    TC_EXISTING,
    TC_MAIN,
    TC_PROMPT,
    TC_PROMPTS,
    TC_TEST,
)
from promptphoo.templating import extract_variables_from_templates

logger = logging.getLogger("promptphoo")


def generate_personas_prompt(prompts: list[str], num_personas: Any) -> str:
    prompts_string = dedent(
        PERSONAS_PROMPTS, "\n".join(f"<Prompt>\n{prompt}\n</Prompt>" for prompt in prompts)
    )
    return dedent(
        PERSONAS,
        "s" if len(prompts) > 1 else "",
        prompts_string,
        num_personas,
        "these prompts" if len(prompts) > 1 else "this prompt",
    )


def test_cases_prompt(
    prompts: list[str],
    persona: str,
    tests: list[dict[str, Any]],
    num_test_cases_per_persona: Any,
    variables: list[str],
    instructions: str | None = None,
) -> str:
    prompts_string = dedent(TC_PROMPTS, "\n".join(dedent(TC_PROMPT, p) for p in prompts))
    rendered_tests = [
        dedent(TC_TEST, json_stringify(test.get("vars"), 2))
        for test in sample_array(tests, 100)
        if is_truthy(test.get("vars"))
    ]
    existing_tests = dedent(TC_EXISTING, "\n".join(sorted(rendered_tests, key=js_sort_key)))
    return dedent(
        TC_MAIN,
        "these prompts" if len(prompts) > 1 else "this prompt",
        prompts_string,
        persona,
        existing_tests,
        num_test_cases_per_persona,
        f" {instructions}" if instructions else "",
        ", ".join(f"{name}: string" for name in variables),
    )


def _parse_persona_item(item: Any) -> str | None:
    if isinstance(item, str):
        trimmed = item.strip()
        return trimmed or None
    if isinstance(item, dict):
        val = next(
            (item[k] for k in ("persona", "name", "description") if item.get(k) is not None), None
        )
        if isinstance(val, str):
            trimmed = val.strip()
            return trimmed or None
    return None


def _extract_from_list(items: list[Any]) -> list[str]:
    return [p for p in (_parse_persona_item(i) for i in items) if p is not None]


def _extract_from_record(obj: dict[str, Any]) -> list[str]:
    for key in ("personas", "user_personas", "persona_list", "personas_list", "results"):
        candidate = obj.get(key)
        if isinstance(candidate, list):
            extracted = _extract_from_list(candidate)
            if extracted:
                return extracted
    return []


_FENCE = re.compile(r"^```(?:json)?\s*\n([\s\S]*?)\n```$", re.IGNORECASE)


def extract_personas(output: str) -> list[str]:
    """Personas from a JSON response: {personas: [...]}, a bare list, or objects in text."""
    text = _FENCE.sub(r"\1", output.strip())
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            from_array = _extract_from_list(parsed)
            if from_array:
                return from_array
        elif isinstance(parsed, dict):
            from_obj = _extract_from_record(parsed)
            if from_obj:
                return from_obj
    except ValueError:
        pass
    resp_objects = extract_json_objects(output)
    for obj in resp_objects:
        if isinstance(obj, dict):
            from_obj = _extract_from_record(obj)
            if from_obj:
                return from_obj
    return _extract_from_list(resp_objects)


async def _load_provider(provider: str | None) -> Any:
    if provider is None:
        from promptphoo.providers.defaults import get_default_providers

        return (await get_default_providers())["synthesizeProvider"]
    from promptphoo.providers import load_api_provider

    return await load_api_provider(provider, base_path=cli_state.base_path())


async def synthesize(
    *,
    prompts: list[str],
    tests: list[dict[str, Any]],
    instructions: str | None = None,
    num_personas: Any = None,
    num_test_cases_per_persona: Any = None,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    if len(prompts) < 1:
        raise ValueError("Dataset synthesis requires at least one prompt.")
    num_personas = num_personas if is_truthy(num_personas) else 5
    num_test_cases_per_persona = (
        num_test_cases_per_persona if is_truthy(num_test_cases_per_persona) else 3
    )
    logger.debug(
        "Starting dataset synthesis. We'll begin by generating up to %s personas. Each persona "
        "will be used to generate %s test cases.",
        num_personas, num_test_cases_per_persona,
    )  # fmt: skip
    logger.debug(
        "Generating user personas from %d prompt%s...",
        len(prompts),
        "s" if len(prompts) > 1 else "",
    )
    provider_model = await _load_provider(provider)

    personas_prompt = generate_personas_prompt(prompts, num_personas)
    logger.debug("Generated personas prompt:\n%s", personas_prompt)
    resp = await provider_model.call_api(personas_prompt)
    logger.debug("Received personas response:\n%s", resp.get("output"))
    invariant("output" in resp, "resp.output must be defined")
    output = resp["output"] if isinstance(resp["output"], str) else json_stringify(resp["output"])
    personas = extract_personas(output)
    invariant(
        len(personas) > 0,
        "Expected at least one user persona in the response. Check the provider response format.",
    )
    logger.debug(
        "Generated %d persona%s:\n%s",
        len(personas), "" if len(personas) == 1 else "s", "\n".join(f"  - {p}" for p in personas),
    )  # fmt: skip

    variables = extract_variables_from_templates(prompts)
    logger.debug(
        "Extracted %d variable%s from prompt%s:\n%s",
        len(variables), "" if len(variables) == 1 else "s", "" if len(prompts) == 1 else "s",
        "\n".join(f"  - {v}" for v in variables),
    )  # fmt: skip

    batch_size = 20
    total_test_cases = num_personas * num_test_cases_per_persona

    async def generate_test_cases_for_persona(current: list[Any]) -> list[Any]:
        remaining = total_test_cases - len(current)
        current_batch_size = min(remaining, batch_size)
        persona = personas[len(current) % len(personas)]
        logger.debug(
            "Generating %s test cases for persona %d of %d...",
            current_batch_size, (len(current) % len(personas)) + 1, len(personas),
        )  # fmt: skip
        persona_prompt = test_cases_prompt(
            prompts, persona, tests, current_batch_size, variables, instructions
        )
        logger.debug("Generated persona prompt:\n%s", persona_prompt)
        persona_response = await provider_model.call_api(persona_prompt)
        logger.debug("Received persona response:\n%s", persona_response.get("output"))
        message = 'Expected the test case response to contain a "vars" array of objects.'
        invariant(persona_response.get("output") is not None, message)
        persona_output = persona_response["output"]
        if not isinstance(persona_output, str):
            persona_output = json_stringify(persona_output)
        wrapper = next(
            (
                obj
                for obj in extract_json_objects(persona_output)
                if isinstance(obj, dict)
                and isinstance(obj.get("vars"), list)
                and all(isinstance(v, dict) for v in obj["vars"])
            ),
            None,
        )
        invariant(wrapper, message)
        vars_ = wrapper["vars"]  # type: ignore[index]
        logger.debug("Received %d test cases", len(vars_))
        return vars_

    test_case_vars = await retry_with_deduplication(
        generate_test_cases_for_persona, total_test_cases
    )
    logger.debug("Generated %d test cases", len(test_case_vars))
    if len(test_case_vars) > total_test_cases:
        logger.debug(
            "Generated %d test cases, but only %s were requested. Sampling down to %s...",
            len(test_case_vars), total_test_cases, total_test_cases,
        )  # fmt: skip
        test_case_vars = sample_array(test_case_vars, total_test_cases)
    return test_case_vars


async def synthesize_from_test_suite(
    test_suite: dict[str, Any], options: dict[str, Any]
) -> list[dict[str, Any]]:
    return await synthesize(
        **options,
        prompts=[prompt["raw"] for prompt in test_suite["prompts"]],
        tests=test_suite.get("tests") or [],
    )
