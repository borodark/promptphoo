"""LLM-judge assertions. Ports src/assertions/{llmRubric,modelGradedClosedQa,factuality,geval,
agentRubric,pi}.ts and finalizeGradedAssertion from src/assertions/ragDefaults.ts.

`trajectory:goal-success` lives in assertions/trajectory.py, next to the other trajectory
assertions.
"""

from __future__ import annotations

from typing import Any

from promptphoo.assertions.base import AssertionParams, GradingResult, Handler, invariant
from promptphoo.assertions.rag_defaults import finalize_graded_assertion
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.tokens import accumulate_token_usage, create_empty_token_usage


def _stringify_object_rubric_prompt(params: AssertionParams) -> dict[str, Any] | None:
    options = params.test.get("options")
    if options and isinstance(options.get("rubricPrompt"), dict | list):
        options["rubricPrompt"] = json_stringify(options["rubricPrompt"])
    # Update the assertion value. This allows the web view to display the prompt.
    value = params.assertion.get("value")
    if not is_truthy(value):
        value = (options or {}).get("rubricPrompt")
    if value is not None:
        params.assertion["value"] = value
    return options


async def handle_llm_rubric(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.llm_grading import matches_llm_rubric

    rendered = params.rendered_value
    invariant(
        rendered is None or isinstance(rendered, str | dict | list),
        '"llm-rubric" assertion type must have a string or object value',
    )
    options = _stringify_object_rubric_prompt(params)
    provider_response = params.provider_response or {}
    multimodal = not params.assertion.get("transform") and (
        bool(provider_response.get("images")) or is_truthy(provider_response.get("audio"))
    )
    resp = await matches_llm_rubric(
        rendered if is_truthy(rendered) else "",
        params.output_string,
        options,
        params.test.get("vars"),
        params.assertion,
        {"providerResponse": provider_response} if multimodal else None,
        params.provider_call_context,
    )
    return finalize_graded_assertion(resp, params.assertion, params.inverse)


async def handle_agent_rubric(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.agent import matches_agent_rubric

    rendered = params.rendered_value
    invariant(
        rendered is None or isinstance(rendered, str | dict | list),
        '"agent-rubric" assertion type must have a string or object value',
    )
    options = _stringify_object_rubric_prompt(params)
    metadata = (params.provider_response or {}).get("metadata")
    resp = await matches_agent_rubric(
        rendered if is_truthy(rendered) else "",
        params.output_string,
        options,
        params.test.get("vars"),
        params.assertion,
        params.provider_call_context,
        metadata.get("workingDir") if isinstance(metadata, dict) else None,
    )
    return finalize_graded_assertion(resp, params.assertion, params.inverse)


async def handle_model_graded_closed_qa(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.llm_grading import matches_closed_qa

    invariant(
        isinstance(params.rendered_value, str),
        "model-graded-closedqa assertion type must have a string value",
    )
    invariant(params.prompt, "model-graded-closedqa assertion type must have a prompt")
    # rubricPrompt is rendered later in matches_closed_qa, with input/criteria/completion.
    resp = await matches_closed_qa(
        params.prompt or "",
        params.rendered_value,
        params.output_string,
        params.test.get("options"),
        params.test.get("vars"),
        params.provider_call_context,
    )
    return finalize_graded_assertion(resp, params.assertion, params.inverse, True)


async def handle_factuality(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.llm_grading import matches_factuality

    invariant(
        isinstance(params.rendered_value, str),
        "factuality assertion type must have a string value",
    )
    invariant(params.prompt, "factuality assertion type must have a prompt")
    # rubricPrompt is rendered later in matches_factuality, with input/ideal/completion.
    resp = await matches_factuality(
        params.prompt or "",
        params.rendered_value,
        params.output_string,
        params.test.get("options"),
        params.test.get("vars"),
        params.provider_call_context,
    )
    return finalize_graded_assertion(resp, params.assertion, params.inverse, True)


async def handle_g_eval(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.llm_grading import is_grader_failure, matches_g_eval

    assertion, inverse, rendered = params.assertion, params.inverse, params.rendered_value
    invariant(
        isinstance(rendered, str)
        or (isinstance(rendered, list) and all(isinstance(v, str) for v in rendered)),
        "G-Eval assertion type must have a string or array of strings value",
    )
    threshold = assertion.get("threshold")
    if threshold is None:
        threshold = 0.7
    options = params.test.get("options")

    if isinstance(rendered, list):
        if not rendered:
            return {
                "assertion": assertion,
                "pass": False,
                "score": 0,
                "reason": "G-Eval assertion requires at least one criterion string in the value "
                "array.",
            }
        # Serial, so the first grader failure short-circuits the remaining (paid) calls.
        responses: list[dict[str, Any]] = []
        failure: tuple[int, dict[str, Any]] | None = None
        for index, value in enumerate(rendered):
            resp = await matches_g_eval(
                value,
                params.prompt or "",
                params.output_string,
                threshold,
                options,
                params.provider_call_context,
            )
            responses.append(resp)
            if is_grader_failure(resp):
                failure = (index, resp)
                break
        tokens_used = create_empty_token_usage()
        for r in responses:
            accumulate_token_usage(tokens_used, r.get("tokensUsed"))
        if failure:
            index, resp = failure
            out: GradingResult = {
                "assertion": assertion,
                "pass": False,
                "score": 0,
                "reason": f"G-Eval criterion {index + 1}/{len(rendered)} "
                f"({json_stringify(rendered[index])}) failed: {resp.get('reason')}",
                "tokensUsed": tokens_used,
            }
            if resp.get("metadata") is not None:
                out["metadata"] = resp["metadata"]
            return out
        average = sum(r["score"] for r in responses) / len(responses)
        return {
            "assertion": assertion,
            "pass": (average >= threshold) != inverse,
            "score": 1 - average if inverse else average,
            "reason": "\n\n".join(r["reason"] for r in responses),
            "tokensUsed": tokens_used,
        }

    resp = await matches_g_eval(
        rendered,
        params.prompt or "",
        params.output_string,
        threshold,
        options,
        params.provider_call_context,
    )
    if is_grader_failure(resp):
        out = {"assertion": assertion, "pass": False, "score": 0, "reason": resp.get("reason")}
        for key in ("tokensUsed", "metadata"):
            if resp.get(key) is not None:
                out[key] = resp[key]
        return out
    return {
        "assertion": assertion,
        **resp,
        "pass": (resp["score"] >= threshold) != inverse,
        "score": 1 - resp["score"] if inverse else resp["score"],
    }


async def handle_pi_scorer(params: AssertionParams) -> GradingResult:
    from promptphoo.matchers.llm_grading import matches_pi_score

    invariant(
        isinstance(params.rendered_value, str), '"pi" assertion type must have a string value'
    )
    invariant(isinstance(params.prompt, str), '"pi" assertion must have a prompt that is a string')
    return await matches_pi_score(
        params.rendered_value, params.prompt or "", params.output_string, params.assertion
    )


HANDLERS: dict[str, Handler] = {
    "agent-rubric": handle_agent_rubric,
    "factuality": handle_factuality,
    "g-eval": handle_g_eval,
    "llm-rubric": handle_llm_rubric,
    "model-graded-closedqa": handle_model_graded_closed_qa,
    "model-graded-factuality": handle_factuality,
    "pi": handle_pi_scorer,
}
