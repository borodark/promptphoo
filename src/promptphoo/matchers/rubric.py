# SHIM: replaced by the grading-infrastructure branch at merge
"""Rubric prompt loading and rendering. Ports loadRubricPrompt / renderLlmRubricPrompt from
src/matchers/rubric.ts (the subset comparison.py needs)."""

from __future__ import annotations

import logging
import os
from typing import Any

from promptphoo import cli_state, envars
from promptphoo.files import is_javascript_file, maybe_load_from_external_file, parse_file_url
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import json_stringify
from promptphoo.templating import get_engine

logger = logging.getLogger(__name__)


async def load_rubric_prompt(rubric_prompt: Any, default_prompt: str) -> str:
    if not rubric_prompt:
        return default_prompt
    if isinstance(rubric_prompt, dict) and len(rubric_prompt) == 0:
        return default_prompt
    if isinstance(rubric_prompt, str) and rubric_prompt.startswith("file://"):
        base_path = cli_state.base_path() or ""
        rendered = get_engine().render_string(rubric_prompt, {})
        file_path, _function_name = parse_file_url(rendered)
        resolved = os.path.abspath(os.path.join(base_path, file_path))
        if is_javascript_file(file_path):
            raise ValueError(
                f"JavaScript rubric prompts are not supported by promptphoo: {rubric_prompt}. "
                "Use a .json, .yaml or .txt file instead."
            )
        try:
            with open(resolved, encoding="utf-8") as f:
                rubric_prompt = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File does not exist: {resolved}") from None
    else:
        rubric_prompt = maybe_load_from_external_file(rubric_prompt)
    if isinstance(rubric_prompt, dict | list):
        rubric_prompt = json_stringify(rubric_prompt)
    if not isinstance(rubric_prompt, str):
        raise ValueError("rubricPrompt must be a string")
    return rubric_prompt


def _process_context_for_templating(
    context: dict[str, Any], enable_object_access: bool
) -> dict[str, Any]:
    if enable_object_access:
        return context
    out: dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(value, list):
            out[key] = [
                json_stringify(item) if isinstance(item, dict | list) and item is not None else item
                for item in value
            ]
        elif isinstance(value, dict):
            out[key] = json_stringify(value)
        else:
            out[key] = value
    return out


def _render_strings(value: Any, render: Any) -> Any:
    if isinstance(value, str):
        return render(value)
    if isinstance(value, list):
        return [_render_strings(v, render) for v in value]
    if isinstance(value, dict):
        return {k: _render_strings(v, render) for k, v in value.items()}
    return value


async def render_llm_rubric_prompt(rubric_prompt: str, context: dict[str, Any]) -> str:
    enable_object_access = envars.get_env_bool("PROMPTFOO_DISABLE_OBJECT_STRINGIFY", False)
    processed = _process_context_for_templating(context, enable_object_access)
    engine = get_engine(None, False, True)
    try:
        # Render every string scalar within the JSON (values only, not keys)
        parsed = js_json_parse(rubric_prompt)
        rendered = _render_strings(parsed, lambda s: engine.render_string(s, processed))
        return json_stringify(rendered)
    except Exception as err:
        # Not valid JSON - fall through to legacy Nunjucks rendering below.
        logger.debug("[Rubric] Rubric prompt is not valid JSON, using Nunjucks rendering: %s", err)
    return engine.render_string(rubric_prompt, processed)
