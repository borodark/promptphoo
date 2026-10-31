"""LLM-judge matchers. Ports src/matchers/llmGrading.ts.

`matches_llm_rubric` always grades locally: upstream's remote branch only runs for redteam
configs, which promptphoo does not support (see D25). `matches_pi_score` needs Promptfoo's
remote Pi scoring API and raises a not-supported error.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from promptphoo.assertions.base import invariant
from promptphoo.assertions.refusal import JS_WHITESPACE, js_trim
from promptphoo.jsonutil import extract_first_json_object, js_json_parse
from promptphoo.jsvalues import UNDEFINED, is_truthy, json_stringify, number_to_string
from promptphoo.matchers.providers import call_provider_with_context, get_and_check_provider
from promptphoo.matchers.rubric import (
    LlmRubricProviderError,
    js_number,
    load_rubric_prompt,
    materialize_image_outputs_for_grading,
    render_llm_rubric_prompt,
    run_json_grading_prompt,
)
from promptphoo.matchers.shared import grader_fail, normalize_matcher_token_usage, try_parse
from promptphoo.prompts.grading import (
    DEFAULT_GRADING_PROMPT,
    OPENAI_CLOSED_QA_PROMPT,
    PROMPTFOO_FACTUALITY_PROMPT,
    TRAJECTORY_GOAL_SUCCESS_PROMPT,
)
from promptphoo.tokens import accumulate_token_usage

logger = logging.getLogger(__name__)

# Upstream defines these in src/prompts/index.ts (outside the vendored grading.ts), as plain
# template literals; the text is byte-identical.
GEVAL_PROMPT_STEPS = (
    "\n"
    "Given evaluation criteria that outline how you should judge a piece of text, "
    "generate 3-4 concise evaluation steps applicable to any text based on the criteria "
    "below and designed to check whether the criteria are satisfied by the text.\n"
    "\n"
    "**EVALUATION CRITERIA**\n"
    "{{criteria}}\n"
    "\n"
    "**OUTPUT FORMAT**\n"
    "IMPORTANT:\n"
    "- Return output ONLY as a minified JSON object (no code fences).\n"
    '- The JSON object must contain a single key, "steps", whose value is a list of '
    "strings.\n"
    "- Each string must represent one evaluation step.\n"
    "- Do NOT include any explanations, commentary, extra text, or additional "
    "formatting.\n"
    "\n"
    "Format:\n"
    '{"steps": <list_of_strings>}\n'
    "\n"
    "Example:\n"
    '{"steps":["<Evaluation Step 1>","<Evaluation Step 2>","<Evaluation Step '
    '3>","<Evaluation Step 4>"]}\n'
    "\n"
    "Here are the 3-4 concise evaluation steps, formatted as required in a minified "
    "JSON:\n"
    "JSON:\n"
)

GEVAL_PROMPT_EVALUATE = (
    "\n"
    "You will be given one Reply for a Prompt below. Your task is to rate the Reply on "
    "one metric.\n"
    "Please make sure you read and understand these instructions carefully. Please keep "
    "this document open while reviewing, and refer to it as needed.\n"
    "\n"
    "**Evaluation Criteria**\n"
    "{{criteria}}\n"
    "\n"
    "**Evaluation Steps**\n"
    "- {{steps}}\n"
    "Given the evaluation steps, return a JSON with two keys: \n"
    '  1) a "score" key that MUST be an integer from 0 to {{maxScore}}, where '
    "{{maxScore}} indicates that the condition described by the Evaluation Criteria is "
    "fully and clearly observed in the Reply according to the Evaluation Steps, and 0 "
    "indicates that it is not observed at all;\n"
    '  2) a "reason" key, a reason for the given score, but DO NOT QUOTE THE SCORE in '
    "your reason. Please mention specific information from Prompt and Reply in your "
    "reason, but be very concise with it!\n"
    "\n"
    "**Prompt**\n"
    "{{input}}\n"
    "\n"
    "**Reply**\n"
    "{{output}}\n"
    "\n"
    "**OUTPUT FORMAT**\n"
    "IMPORTANT: \n"
    "- Return output ONLY as a minified JSON object (no code fences).\n"
    '- The JSON object must contain exactly two keys: "score" and "reason".\n'
    "- No additional words, explanations, or formatting are needed.\n"
    "- Absolutely no additional text, explanations, line breaks, or formatting outside "
    "the JSON object are allowed.\n"
    "\n"
    "Example JSON:\n"
    '{"score":0,"reason":"The text of reply does not follow the evaluation criteria '
    'provided."}\n'
    "\n"
    "Here is the final evaluation in the required minified JSON format:\n"
    "JSON:\n"
)


NO_GRADING_CONFIG = (
    "Cannot grade output without grading config. Specify --grader option or grading config."
)
ATTACHED_IMAGE_OUTPUT_PLACEHOLDER = (
    "[Image output attached. Inspect the attached image directly for visual grading.]"
)
FACTUALITY_CATEGORY_DESCRIPTIONS = {
    "A": "The submitted answer is a subset of the expert answer and is fully consistent with it.",
    "B": "The submitted answer is a superset of the expert answer and is fully consistent with it.",
    "C": "The submitted answer contains all the same details as the expert answer.",
    "D": "There is a disagreement between the submitted answer and the expert answer.",
    "E": "The answers differ, but these differences don't matter from the perspective of "
    "factuality.",
}
_NOT_LT = "[^\n\r  ]"  # JS `.` without the s flag


def _nullish(value: Any, default: Any) -> Any:
    return default if value is None else value


def _get_factuality_score_lookup(grading: dict[str, Any]) -> dict[str, Any]:
    f = grading.get("factuality") or {}
    return {
        "A": _nullish(f.get("subset"), 1),
        "B": _nullish(f.get("superset"), 1),
        "C": _nullish(f.get("agree"), 1),
        "D": _nullish(f.get("disagree"), 0),
        "E": _nullish(f.get("differButFactual"), 1),
    }


def _build_factuality_result(
    option: str, reason: str, grading: dict[str, Any], resp: dict[str, Any]
) -> dict[str, Any]:
    lookup = _get_factuality_score_lookup(grading)
    passing = [k for k, v in lookup.items() if v > 0]
    failing = [k for k, v in lookup.items() if v == 0]
    passed = option in passing and option not in failing
    score = lookup.get(option)
    return {
        "pass": passed,
        "score": score if score is not None else (1 if passed else 0),
        "reason": reason,
        "tokensUsed": normalize_matcher_token_usage(resp.get("tokenUsage")),
    }


class _InvalidCategoryError(ValueError):
    pass


def _parse_factuality_json_response(response_text: str) -> dict[str, str] | None:
    try:
        json_data = extract_first_json_object(response_text)
        category = json_data.get("category") if isinstance(json_data, dict) else None
        if not category or not isinstance(category, str):
            return None
        option = js_trim(category).upper()
        if not re.fullmatch(r"[A-E]", option):
            raise _InvalidCategoryError(f"Invalid category value: {option}")
        reason = json_data.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise TypeError("jsonData.reason?.trim is not a function")
        reason = js_trim(reason) if reason else ""
        return {
            "option": option,
            "reason": reason or f"Category {option}: {FACTUALITY_CATEGORY_DESCRIPTIONS[option]}",
        }
    except _InvalidCategoryError:
        raise
    except Exception as err:
        logger.debug("JSON parsing failed: %s", err)
        return None


def _parse_legacy_factuality_response(response_text: str) -> dict[str, str]:
    answer = re.search(r"\s*\(?([a-eA-E])\)", response_text)
    if not answer:
        raise ValueError(
            f"Factuality checker output did not match expected format: {response_text}"
        )
    option = answer.group(1).upper()
    reason_match = re.search(r"\)\s*(.*)", response_text, re.DOTALL)
    return {
        "option": option,
        "reason": js_trim(reason_match.group(1))
        if reason_match and reason_match.group(1)
        else response_text,
    }


def _get_data_uri_payload(data: str) -> str | None:
    parts = js_trim(data).split(",")[:2]
    metadata, payload = parts[0], (parts[1] if len(parts) > 1 else "")
    if not payload or not metadata.lower().startswith("data:image/"):
        return None
    return payload


def _get_grading_output_for_images(llm_output: str, image_outputs: list[dict[str, Any]]) -> str:
    if not image_outputs:
        return llm_output
    trimmed = js_trim(llm_output)
    if not trimmed:
        return ATTACHED_IMAGE_OUTPUT_PLACEHOLDER
    if re.match(r"^data:image/[^;,]+;base64,", trimmed, re.IGNORECASE):
        return ATTACHED_IMAGE_OUTPUT_PLACEHOLDER
    for image in image_outputs:
        if not image.get("data"):
            continue
        data = js_trim(image["data"])
        if data == trimmed or _get_data_uri_payload(data) == trimmed:
            return ATTACHED_IMAGE_OUTPUT_PLACEHOLDER
    return llm_output


def _get_grading_output_for_audio(llm_output: str, audio: dict[str, Any] | None) -> str:
    if not audio or not audio.get("data"):
        return llm_output
    output_data = re.sub(
        r"\s", "", re.sub(r"^data:audio/[^;,]+;base64,", "", js_trim(llm_output), flags=re.I)
    )
    if output_data == re.sub(r"\s", "", audio["data"]):
        return audio.get("transcript") or "[Audio output]"
    return llm_output


async def matches_llm_rubric(
    rubric: Any,
    llm_output: str,
    grading: dict[str, Any] | None = None,
    vars: dict[str, Any] | None = None,  # noqa: A002 - Node's parameter name
    assertion: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    provider_call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if grading is None:
        raise ValueError(NO_GRADING_CONFIG)
    options = options or {}
    provider_response = options.get("providerResponse") or {}
    image_outputs = materialize_image_outputs_for_grading(provider_response.get("images"))[
        "imageOutputs"
    ]
    audio = provider_response.get("audio")
    grading_output = _get_grading_output_for_images(
        _get_grading_output_for_audio(llm_output, audio), image_outputs
    )
    # Upstream's remote-grading branch requires a redteam config (out of scope, D25).
    try:
        return await run_json_grading_prompt(
            assertion=assertion,
            check_name="llm-rubric check",
            default_prompt=DEFAULT_GRADING_PROMPT,
            grading=grading,
            label="llm-rubric",
            provider_call_context=provider_call_context,
            throw_on_error=bool(options.get("throwOnError")),
            images=image_outputs,
            audio=audio,
            vars={**(vars or {}), "output": try_parse(grading_output), "rubric": rubric},
        )
    except Exception as error:
        if options.get("throwOnError"):
            raise LlmRubricProviderError(str(error) or "No output") from error
        raise


async def matches_trajectory_goal_success(
    goal: str,
    trajectory: str,
    llm_output: str,
    grading: dict[str, Any] | None = None,
    vars: dict[str, Any] | None = None,  # noqa: A002
    assertion: dict[str, Any] | None = None,
    provider_call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if grading is None:
        raise ValueError(NO_GRADING_CONFIG)
    return await run_json_grading_prompt(
        assertion=assertion,
        check_name="trajectory:goal-success check",
        default_prompt=TRAJECTORY_GOAL_SUCCESS_PROMPT,
        grading=grading,
        label="trajectory:goal-success",
        provider_call_context=provider_call_context,
        vars={
            **(vars or {}),
            "goal": goal,
            "output": try_parse(llm_output),
            "trajectory": trajectory,
        },
    )


async def matches_pi_score(
    rendered_value: str,
    llm_input: str,
    llm_output: str,
    assertion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raise NotImplementedError(
        "The pi assertion is not supported by promptphoo: it requires Promptfoo's remote Pi "
        "scoring API (withpi.ai)."
    )


async def matches_factuality(
    input: str,  # noqa: A002 - Node's parameter name
    expected: str,
    output: str,
    grading: dict[str, Any] | None = None,
    vars: dict[str, Any] | None = None,  # noqa: A002
    provider_call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from promptphoo.providers.defaults import get_default_providers

    if grading is None:
        raise ValueError(NO_GRADING_CONFIG)
    template_vars = {
        **(vars or {}),
        "input": input,
        "ideal": expected,
        "completion": try_parse(output),
    }
    rubric_prompt = await load_rubric_prompt(
        grading.get("rubricPrompt"), PROMPTFOO_FACTUALITY_PROMPT
    )
    prompt = await render_llm_rubric_prompt(rubric_prompt, template_vars)
    final_provider = await get_and_check_provider(
        "text",
        grading.get("provider"),
        (await get_default_providers()).get("gradingProvider"),
        "factuality check",
    )
    resp = await call_provider_with_context(
        final_provider, prompt, "factuality", template_vars, provider_call_context
    )
    if resp.get("error") or not is_truthy(resp.get("output")):
        return grader_fail(resp.get("error") or "No output", resp.get("tokenUsage"))
    invariant(isinstance(resp["output"], str), "factuality produced malformed response")

    try:
        parsed_json = _parse_factuality_json_response(resp["output"])
        if parsed_json:
            return _build_factuality_result(
                parsed_json["option"], parsed_json["reason"], grading, resp
            )
    except Exception as err:
        return grader_fail(str(err), resp.get("tokenUsage"))

    logger.info("Falling back to legacy pattern matching for factuality check")
    try:
        legacy = _parse_legacy_factuality_response(resp["output"])
        return _build_factuality_result(legacy["option"], legacy["reason"], grading, resp)
    except Exception as err:
        return grader_fail(str(err), resp.get("tokenUsage"))


async def matches_closed_qa(
    input: str,  # noqa: A002 - Node's parameter name
    expected: str,
    output: str,
    grading: dict[str, Any] | None = None,
    vars: dict[str, Any] | None = None,  # noqa: A002
    provider_call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from promptphoo.providers.defaults import get_default_providers

    if grading is None:
        raise ValueError(NO_GRADING_CONFIG)
    template_vars = {
        **(vars or {}),
        "input": input,
        "criteria": expected,
        "completion": try_parse(output),
    }
    rubric_prompt = await load_rubric_prompt(grading.get("rubricPrompt"), OPENAI_CLOSED_QA_PROMPT)
    prompt = await render_llm_rubric_prompt(rubric_prompt, template_vars)
    final_provider = await get_and_check_provider(
        "text",
        grading.get("provider"),
        (await get_default_providers()).get("gradingProvider"),
        "model-graded-closedqa check",
    )
    resp = await call_provider_with_context(
        final_provider, prompt, "model-graded-closedqa", template_vars, provider_call_context
    )
    if resp.get("error") or not is_truthy(resp.get("output")):
        return grader_fail(resp.get("error") or "No output", resp.get("tokenUsage"))
    invariant(isinstance(resp["output"], str), "model-graded-closedqa produced malformed response")
    output_text: str = resp["output"]
    trimmed = output_text.rstrip(JS_WHITESPACE)
    passed = trimmed.endswith("Y")
    if passed:
        reason = f"The submission meets the criterion:\n{output_text}"
    elif trimmed.endswith("N"):
        reason = f"The submission does not meet the criterion:\n{output_text}"
    else:
        return grader_fail(
            f"Model grader produced a malformed response:\n{output_text}", resp.get("tokenUsage")
        )
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": reason,
        "tokensUsed": normalize_matcher_token_usage(resp.get("tokenUsage")),
    }


def is_grader_failure(resp: dict[str, Any]) -> bool:
    """A grader transport/parse failure (`metadata.graderError`): inverse-aware callers must
    propagate it verbatim instead of flipping it into a pass."""
    return ((resp or {}).get("metadata") or {}).get("graderError") is True


def _rubric_prompt_part(grading: dict[str, Any] | None, key: str) -> Any:
    rubric_prompt = (grading or {}).get("rubricPrompt")
    return rubric_prompt.get(key) if isinstance(rubric_prompt, dict) else None


async def matches_g_eval(
    criteria: str,
    input: str,  # noqa: A002 - Node's parameter name
    output: str,
    threshold: float,
    grading: dict[str, Any] | None = None,
    provider_call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from promptphoo.providers.defaults import get_default_providers

    if not input:
        raise ValueError("No source text to estimate reply")
    max_score = 10
    text_provider = await get_and_check_provider(
        "text",
        (grading or {}).get("provider"),
        (await get_default_providers()).get("gradingProvider"),
        "reply geval check",
    )
    tokens_used = normalize_matcher_token_usage(None)

    def fail_with_tokens(reason: str) -> dict[str, Any]:
        return grader_fail(reason, tokens_used)

    def fail_no_output(phase: str) -> dict[str, Any]:
        return fail_with_tokens(f"G-Eval {phase} call to {text_provider.id()} returned no output")

    steps_prompt = await load_rubric_prompt(
        _rubric_prompt_part(grading, "steps"), GEVAL_PROMPT_STEPS
    )
    prompt_steps = await render_llm_rubric_prompt(steps_prompt, {"criteria": criteria})
    resp_steps = await call_provider_with_context(
        text_provider, prompt_steps, "g-eval-steps", {"criteria": criteria}, provider_call_context
    )
    accumulate_token_usage(tokens_used, resp_steps.get("tokenUsage"))
    if resp_steps.get("error"):
        return fail_with_tokens(resp_steps["error"])
    steps_output = resp_steps.get("output")
    if not is_truthy(steps_output):
        return fail_no_output("step-generation")
    if not isinstance(steps_output, str):
        return fail_with_tokens("LLM-proposed evaluation steps response is not a string")

    try:
        steps_match = re.search(rf'\{{"steps"{_NOT_LT}+\}}', steps_output)
        if not steps_match:
            return fail_with_tokens(
                f"LLM-proposed evaluation steps are not in JSON format: {steps_output}"
            )
        parsed_steps = js_json_parse(steps_match.group(0))
        steps = parsed_steps.get("steps", UNDEFINED) if isinstance(parsed_steps, dict) else None
        if not isinstance(steps, list):
            return fail_with_tokens(
                f"G-Eval steps response has invalid or missing steps: {json_stringify(steps)}"
            )
        if len(steps) == 0:
            return fail_with_tokens("LLM does not propose any evaluation step")
        if not all(isinstance(s, str) and js_trim(s) != "" for s in steps):
            return fail_with_tokens(
                f"G-Eval steps response contains invalid steps: {json_stringify(steps)}"
            )
    except Exception as err:
        return fail_with_tokens(
            f"LLM-proposed evaluation steps are not in JSON format: {err}\n\n{steps_output}"
        )

    eval_prompt = await load_rubric_prompt(
        _rubric_prompt_part(grading, "evaluate"), GEVAL_PROMPT_EVALUATE
    )
    eval_vars = {
        "criteria": criteria,
        "steps": "\n- ".join(steps),
        "maxScore": str(max_score),
        "input": try_parse(input),
        "output": try_parse(output),
    }
    prompt_text = await render_llm_rubric_prompt(eval_prompt, eval_vars)
    resp = await call_provider_with_context(
        text_provider, prompt_text, "g-eval", eval_vars, provider_call_context
    )
    accumulate_token_usage(tokens_used, resp.get("tokenUsage"))
    if resp.get("error"):
        return fail_with_tokens(resp["error"])
    eval_output = resp.get("output")
    if not is_truthy(eval_output):
        return fail_no_output("evaluation")
    if not isinstance(eval_output, str):
        return fail_with_tokens("LLM-proposed evaluation result response is not a string")

    try:
        result_match = re.search(rf"\{{{_NOT_LT}+\}}", eval_output)
        if not result_match:
            return fail_with_tokens(
                f"LLM-proposed evaluation result is not in JSON format: {eval_output}"
            )
        result = js_json_parse(result_match.group(0))
    except Exception as err:
        return fail_with_tokens(
            f"LLM-proposed evaluation result is not in JSON format: {err}\n\n{eval_output}"
        )
    if not isinstance(result, dict):
        result = {}

    raw = result.get("score", UNDEFINED)
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        raw_score = raw
    elif isinstance(raw, str) and js_trim(raw) != "":
        raw_score = js_number(raw)
    else:
        raw_score = math.nan
    if not math.isfinite(raw_score):
        return fail_with_tokens(
            f"G-Eval result has invalid or missing score: {json_stringify(raw)}"
        )
    if raw_score < 0 or raw_score > max_score:
        return fail_with_tokens(
            f"G-Eval result score {number_to_string(raw_score)} is outside the expected "
            f"0-{max_score} range"
        )
    reason = result.get("reason", UNDEFINED)
    if not isinstance(reason, str) or js_trim(reason) == "":
        return fail_with_tokens(
            f"G-Eval result has invalid or missing reason: {json_stringify(reason)}"
        )
    score = raw_score / max_score
    return {
        "pass": score >= threshold,
        "score": score,
        "reason": reason,
        "tokensUsed": tokens_used,
    }
