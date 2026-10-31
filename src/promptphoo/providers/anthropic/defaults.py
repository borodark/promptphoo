"""Default Anthropic grading providers. Ports src/providers/anthropic/defaults.ts.

Upstream reuses one bundle per `cliState.envScope` (a WeakMap keyed by the invocation scope).
promptphoo has no env scope object, so each call builds a fresh bundle, which is what upstream
does outside a scope (see D33).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from promptphoo import envars
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.anthropic.messages import AnthropicMessagesProvider

# Must support the rubric grader's forced tool_choice.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

_GRADE_OUTPUT_TOOL = {
    "name": "grade_output",
    "description": "Grade the given output based on specific criteria",
    "input_schema": {
        "type": "object",
        "properties": {
            "pass": {"type": "boolean", "description": "Whether the output passes the criteria"},
            "score": {"type": "number", "description": "The score assigned to the output"},
            "reason": {"type": "string", "description": "The reason for the given grade"},
        },
        "required": ["pass", "score", "reason"],
    },
}


class AnthropicLlmRubricProvider(AnthropicMessagesProvider):
    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        config = options.get("config") or {}
        super().__init__(
            model_name,
            {
                "env": options.get("env"),
                "config": {
                    "tool_choice": {"type": "tool", "name": "grade_output"},
                    "tools": [_GRADE_OUTPUT_TOOL],
                    **config,
                    # Keep the tool-use JSON parseable even if the caller enables thinking.
                    "showThinking": False,
                },
            },
        )

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = await super().call_api(prompt)
        output = result.get("output")
        if not isinstance(output, str):
            return {
                "error": "Anthropic LLM rubric grader - malformed non-string output\n\n"
                f"{json_stringify(output)}"
            }
        try:
            function_call = js_json_parse(output)
            return {"output": function_call["input"]}
        except (ValueError, TypeError, KeyError) as err:
            return {"error": f"Anthropic LLM rubric grader - invalid JSON: {err}\n\n{output}"}


def _env_value(env: Mapping[str, Any] | None, name: str) -> Any:
    value = env.get(name) if env else None
    return value if value is not None else envars.get_env_string(name)


def get_anthropic_providers(env: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create default Anthropic providers for the active environment.

    Every construction input is resolved up front, preserving explicit empty credentials and
    headers.
    """
    resolved_env: dict[str, Any] = {**(env or {})}
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS"):
        value = _env_value(env, name)
        if value is None:
            resolved_env.pop(name, None)
        else:
            resolved_env[name] = value
    grading_provider = AnthropicMessagesProvider(DEFAULT_ANTHROPIC_MODEL, {"env": resolved_env})
    llm_rubric_provider = AnthropicLlmRubricProvider(DEFAULT_ANTHROPIC_MODEL, {"env": resolved_env})
    web_search_provider = AnthropicMessagesProvider(
        DEFAULT_ANTHROPIC_MODEL,
        {
            "env": resolved_env,
            "config": {
                "tools": [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}]
            },
        },
    )
    return {
        "gradingJsonProvider": grading_provider,
        "gradingProvider": grading_provider,
        "llmRubricProvider": llm_rubric_provider,
        "suggestionsProvider": grading_provider,
        "synthesizeProvider": grading_provider,
        "webSearchProvider": web_search_provider,
    }
