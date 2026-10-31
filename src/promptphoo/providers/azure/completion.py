"""Azure OpenAI legacy completions. Ports src/providers/azure/completion.ts."""

from __future__ import annotations

import logging
from typing import Any

from promptphoo import envars, fetch
from promptphoo.finish_reason import normalize_finish_reason
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.azure.chat import (
    _first,
    _get,
    _tpl,
    azure_usage_cost,
    js_error_string,
)
from promptphoo.providers.azure.errors import CONTENT_FILTER_FALLBACK_MESSAGE
from promptphoo.providers.azure.generic import AzureGenericProvider
from promptphoo.providers.azure.tables import DEFAULT_AZURE_API_VERSION
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.shared import get_request_timeout_ms

logger = logging.getLogger(__name__)


def _req(obj: Any, key: str) -> Any:
    """`obj.key` without optional chaining: a TypeError when `obj` is undefined."""
    if not isinstance(obj, dict):
        raise TypeError(f"Cannot read properties of undefined (reading '{key}')")
    return obj.get(key)


def _any_filtered(results: Any) -> bool:
    return isinstance(results, dict) and any(
        isinstance(f, dict) and is_truthy(f.get("filtered")) for f in results.values()
    )


class AzureCompletionProvider(AzureGenericProvider):
    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        await self.ensure_initialized()
        if not self.auth_headers:
            raise RuntimeError("Invariant failed: auth headers are not initialized")
        if not self.get_api_base_url():
            raise ValueError("Azure API host must be set.")
        context = context or {}
        config = self.config

        stop_env = envars.get_env_string("OPENAI_STOP")
        try:
            stop = js_json_parse(stop_env) if stop_env else _first(config.get("stop"), "")
        except ValueError as err:
            raise ValueError(f"OPENAI_STOP is not a valid JSON string: SyntaxError: {err}") from err
        body: dict[str, Any] = {
            "model": self.deployment_name,
            "prompt": prompt,
            "max_tokens": _first(
                config.get("max_tokens"), envars.get_env_int("OPENAI_MAX_TOKENS", 1024)
            ),
            "temperature": _first(
                config.get("temperature"), envars.get_env_float("OPENAI_TEMPERATURE", 0)
            ),
            "top_p": _first(config.get("top_p"), envars.get_env_float("OPENAI_TOP_P", 1)),
            "presence_penalty": _first(
                config.get("presence_penalty"), envars.get_env_float("OPENAI_PRESENCE_PENALTY", 0)
            ),
            "frequency_penalty": _first(
                config.get("frequency_penalty"),
                envars.get_env_float("OPENAI_FREQUENCY_PENALTY", 0),
            ),
            "best_of": _first(config.get("best_of"), envars.get_env_int("OPENAI_BEST_OF", 1)),
        }
        if is_truthy(stop):
            body["stop"] = stop
        if isinstance(config.get("passthrough"), dict):
            body.update(config["passthrough"])

        data: Any = None
        cached = False
        try:
            result = await fetch.fetch_with_cache(
                f"{self.get_api_base_url()}/openai/deployments/{self.deployment_name}/completions"
                f"?api-version={config.get('apiVersion') or DEFAULT_AZURE_API_VERSION}",
                {
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        **(self.auth_headers or {}),
                        **(config.get("headers") or {}),
                    },
                    "body": json_stringify(body),
                },
                get_request_timeout_ms(),
                "json",
                _first(context.get("bustCache"), context.get("debug")),
                config.get("maxRetries"),
            )
            data, cached = result.data, result.cached
        except Exception as err:
            return {"error": f"API call error: {js_error_string(err)}"}

        try:
            error = _req(data, "error")
            if is_truthy(error):
                # Content filter errors: HTTP 400 with the content_filter code.
                if _get(error, "code") == "content_filter" and _get(error, "status") == 400:
                    return {
                        "output": _get(error, "message"),
                        "guardrails": {
                            "flagged": True,
                            "flaggedInput": True,
                            "flaggedOutput": False,
                        },
                    }
                return {
                    "error": f"API response error: {_tpl(error, 'code')} {_tpl(error, 'message')}"
                }

            choices = data.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else None
            finish_reason = normalize_finish_reason(_get(choice, "finish_reason"))
            content_filter_triggered = finish_reason == "content_filter"

            output = _get(choice, "text")
            if output is None:
                output = CONTENT_FILTER_FALLBACK_MESSAGE if content_filter_triggered else ""

            content_filter_results = _get(choice, "content_filter_results")
            prompt_filter_results = data.get("prompt_filter_results")
            flagged_input = isinstance(prompt_filter_results, list) and any(
                _any_filtered(_get(r, "content_filter_results")) for r in prompt_filter_results
            )
            flagged_output = content_filter_triggered or _any_filtered(content_filter_results)
            if flagged_output:
                logger.warning(
                    "Azure model %s output was flagged by content filter: %s",
                    self.deployment_name,
                    json_stringify(content_filter_results),
                )
            if flagged_input:
                logger.warning(
                    "Azure model %s input was flagged by content filter: %s",
                    self.deployment_name,
                    json_stringify(prompt_filter_results),
                )

            usage = data.get("usage")
            if cached:
                token_usage = {
                    "cached": _req(usage, "total_tokens"),
                    "total": _req(usage, "total_tokens"),
                }
            else:
                token_usage = {
                    "total": _req(usage, "total_tokens"),
                    "prompt": _req(usage, "prompt_tokens"),
                    "completion": _req(usage, "completion_tokens"),
                }
                cached_tokens = _get(usage, "prompt_tokens_details", "cached_tokens")
                if cached_tokens is not None:
                    token_usage["cached"] = cached_tokens
            response: ProviderResponse = {
                "output": output,
                "tokenUsage": {k: v for k, v in token_usage.items() if v is not None},
            }
            if finish_reason:
                response["finishReason"] = finish_reason
            cost = azure_usage_cost(self.deployment_name, config, usage)
            if cost is not None:
                response["cost"] = cost
            if flagged_input or flagged_output:
                response["guardrails"] = {
                    "flagged": True,
                    "flaggedInput": flagged_input,
                    "flaggedOutput": flagged_output,
                }
            return response
        except Exception as err:
            usage = _get(data, "usage")
            if cached:
                token_usage = {
                    "cached": _req(usage, "total_tokens"),
                    "total": _req(usage, "total_tokens"),
                }
            else:
                token_usage = {
                    "total": _get(usage, "total_tokens"),
                    "prompt": _get(usage, "prompt_tokens"),
                    "completion": _get(usage, "completion_tokens"),
                }
            return {
                "error": f"API response error: {js_error_string(err)}: {json_stringify(data)}",
                "tokenUsage": {k: v for k, v in token_usage.items() if v is not None},
            }
