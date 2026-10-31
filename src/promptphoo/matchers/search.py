"""search-rubric matcher. Ports src/matchers/search.ts (matchesSearchRubric)."""

from __future__ import annotations

import logging
from typing import Any

from promptphoo.jsonutil import extract_first_json_object
from promptphoo.jsvalues import is_truthy, js_string
from promptphoo.matchers.providers import call_provider_with_context, get_grading_provider
from promptphoo.matchers.rubric import load_rubric_prompt, render_llm_rubric_prompt
from promptphoo.matchers.shared import grader_fail, try_parse
from promptphoo.prompts.grading import DEFAULT_WEB_SEARCH_PROMPT
from promptphoo.providers.web_search_utils import (
    has_web_search_capability,
    load_web_search_provider,
)

logger = logging.getLogger(__name__)

# src/providers/anthropic/defaults.ts
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"


def _result(
    passed: bool, score: Any, reason: str, token_usage: Any, assertion: Any, metadata: Any
) -> dict[str, Any]:
    out: dict[str, Any] = {"pass": passed, "score": score, "reason": reason}
    if token_usage is not None:
        out["tokensUsed"] = token_usage
    out["assertion"] = assertion
    out["metadata"] = metadata
    return out


async def matches_search_rubric(
    rubric: str,
    llm_output: Any,
    grading: dict[str, Any] | None = None,
    vars: dict[str, Any] | None = None,
    assertion: dict[str, Any] | None = None,
    _provider: Any = None,
    provider_call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if grading is None:  # `!grading`: an empty {} is truthy in JS
        raise ValueError(
            "Cannot grade output without grading config. Specify --grader option or grading config."
        )

    # Search rubric assertion is like llm-rubric but with web search capabilities.
    from promptphoo.providers.defaults import get_default_providers

    defaults = await get_default_providers()
    default_search_providers = [
        defaults.get("webSearchProvider"),
        defaults.get("llmRubricProvider"),
        defaults.get("gradingProvider"),
    ]
    configured = (
        await get_grading_provider("text", grading["provider"], None)
        if grading.get("provider")
        else None
    )
    search_provider = configured or next((p for p in default_search_providers if p), None)

    # Prefer a known web-search default over a configured/default text grader that lacks search.
    if not has_web_search_capability(search_provider):
        web_default = next(
            (p for p in default_search_providers if has_web_search_capability(p)), None
        )
        if web_default:
            search_provider = web_default

    if not has_web_search_capability(search_provider):
        # For search-rubric, prefer Anthropic first.
        loaded = await load_web_search_provider(True)
        if loaded:
            search_provider = loaded

    if not search_provider or not has_web_search_capability(search_provider):
        raise ValueError(
            "search-rubric assertion requires a grading provider with web search capabilities. "
            "Use --grader with a web search provider (e.g., "
            f"anthropic:messages:{DEFAULT_ANTHROPIC_MODEL}, openai:responses:o4-mini with tools "
            "configured, perplexity:sonar) or configure one in defaultTest.options.provider"
        )

    rubric_prompt = await load_rubric_prompt(grading.get("rubricPrompt"), DEFAULT_WEB_SEARCH_PROMPT)
    template_vars = {**(vars or {}), "output": try_parse(llm_output), "rubric": rubric}
    prompt = await render_llm_rubric_prompt(rubric_prompt, template_vars)

    resp = await call_provider_with_context(
        search_provider, prompt, "search-rubric", template_vars, provider_call_context
    )

    if resp.get("error") or not is_truthy(resp.get("output")):
        # graderFail tags it so not-search-rubric never flips a transport failure into a pass.
        return {
            **grader_fail(
                f"Search rubric evaluation failed: {resp.get('error') or 'No output'}",
                resp.get("tokenUsage"),
            ),
            "assertion": assertion,
        }

    try:
        result = extract_first_json_object(js_string(resp["output"]))
        if not isinstance(result, dict) or not isinstance(result.get("pass"), bool):
            raise ValueError("Missing boolean search verdict")
        passed = result["pass"]
        raw_score = result.get("score")
        if isinstance(raw_score, int | float) and not isinstance(raw_score, bool):
            score = raw_score
        else:
            score = 1 if passed else 0
        threshold = (assertion or {}).get("threshold")
        if threshold is not None:
            passed = passed and score >= threshold
        return _result(
            passed,
            score,
            result.get("reason") or "No reason provided",
            resp.get("tokenUsage"),
            assertion,
            {
                "searchResults": result.get("searchResults") or [],
                "searchProvider": search_provider.id(),
            },
        )
    except Exception:
        logger.warning("[search-rubric] Could not parse a grading verdict from provider response")
        return _result(
            False,
            0,
            'Search rubric evaluation failed: Expected a JSON object with a boolean "pass" field',
            resp.get("tokenUsage"),
            assertion,
            {"graderError": True},
        )
