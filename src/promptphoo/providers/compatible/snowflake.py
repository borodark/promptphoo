"""Snowflake Cortex REST COMPLETE endpoint (`/api/v2/cortex/inference:complete`). Ports
src/providers/snowflake.ts and the `snowflake:` factory in src/providers/registry.ts.

Not ported: abort signals and the response-header observer.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from promptphoo import envars, fetch
from promptphoo.finish_reason import normalize_finish_reason
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.compatible.common import serialize_provider
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.util import (
    calculate_safe_openai_cost,
    format_openai_error,
    get_chat_completion_refusal,
    get_token_usage_with_request_count,
    is_openai_error_only_response,
    js_error_string,
    parse_chat_completion_json_output,
    validate_chat_completion_message,
)
from promptphoo.providers.shared import get_request_timeout_ms

logger = logging.getLogger(__name__)

_ERROR_CODE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _get_snowflake_error_code(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    value = data.get("code")
    if value is None:
        value = data.get("error_code")
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    text = json_stringify(value) if not isinstance(value, str) else value
    return text if _ERROR_CODE.match(text) else None


def _get_snowflake_output(message: dict[str, Any]) -> Any:
    if message.get("functionCall") is not None:
        return message["functionCall"]
    if message.get("toolCalls") is not None:
        return message["toolCalls"]
    content = message.get("content")
    return content if isinstance(content, str) and content.strip() else ""


def _defined(response: dict[str, Any]) -> ProviderResponse:
    return {k: v for k, v in response.items() if v is not None}


class SnowflakeCortexProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        config = provider_options.get("config") or {}
        env = provider_options.get("env")
        account_identifier = config.get("accountIdentifier")
        if not account_identifier:
            value = envars.get_provider_env_string(env, "SNOWFLAKE_ACCOUNT_IDENTIFIER")
            account_identifier = (
                value
                if value is not None
                else envars.get_env_string("SNOWFLAKE_ACCOUNT_IDENTIFIER")
            )
        if not account_identifier and not config.get("apiBaseUrl"):
            raise ValueError(
                "Snowflake provider requires an account identifier. Set "
                "SNOWFLAKE_ACCOUNT_IDENTIFIER environment variable or specify accountIdentifier "
                "in config."
            )
        api_base_url = (
            config.get("apiBaseUrl") or f"https://{account_identifier}.snowflakecomputing.com"
        )
        super().__init__(
            model_name,
            {
                **provider_options,
                "config": {
                    **config,
                    "apiBaseUrl": api_base_url,
                    "apiKeyEnvar": "SNOWFLAKE_API_KEY",
                    "passthrough": {**(config.get("passthrough") or {})},
                },
            },
        )

    def default_id(self) -> str:
        return f"snowflake:{self.model_name}"

    def __str__(self) -> str:
        return f"[Snowflake Cortex Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "snowflake")

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        prepared = await self.get_openai_body(prompt, context, options)
        body, config = prepared["body"], prepared["config"]
        logger.debug(
            "[Snowflake Cortex] Calling API model=%s apiBaseUrl=%s",
            self.model_name,
            self.get_api_url(),
        )
        context = context or {}
        delete_from_cache = None
        try:
            bust = context.get("bustCache")
            result = await fetch.fetch_with_cache(
                f"{self.get_api_url()}/api/v2/cortex/inference:complete",
                {
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.get_api_key()}",
                        **(config.get("headers") or {}),
                    },
                    "body": json_stringify(body),
                },
                get_request_timeout_ms(),
                "json",
                bool(bust if bust is not None else context.get("debug")),
                self.config.get("maxRetries"),
            )
            data, cached, status, status_text = (
                result.data,
                result.cached,
                result.status,
                result.status_text,
            )
            latency_ms, delete_from_cache = result.latency_ms, result.delete_from_cache
            if status < 200 or status >= 300:
                text = data if isinstance(data, str) else json_stringify(data)
                return {"error": f"API error: {status} {status_text}\n{text}"}
            if is_openai_error_only_response(data):
                return {"error": format_openai_error(data)}
        except Exception as err:
            logger.error("[Snowflake Cortex] API call error: %s", js_error_string(err))
            return {"error": f"API call error: {js_error_string(err)}"}

        if isinstance(data, dict) and data.get("error"):
            return {"error": format_openai_error(data)}

        choices = data.get("choices") if isinstance(data, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        finish_reason = normalize_finish_reason(
            choice.get("finish_reason") if isinstance(choice, dict) else None
        )
        token_usage = get_token_usage_with_request_count(data, cached)
        cost = calculate_safe_openai_cost(self.model_name, config, data)
        accounting = {
            "tokenUsage": token_usage,
            "cached": cached,
            "latencyMs": latency_ms,
            "cost": cost,
        }

        if finish_reason == "error":
            if delete_from_cache:
                await delete_from_cache()
            return _defined(
                {
                    "error": "API error: Snowflake provider returned a generation error",
                    **accounting,
                    "finishReason": finish_reason,
                }
            )

        snowflake_error_code = _get_snowflake_error_code(data)
        if (
            not choice
            and isinstance(data, dict)
            and isinstance(data.get("message"), str)
            and snowflake_error_code
        ):
            if delete_from_cache:
                await delete_from_cache()
            return _defined(
                {
                    "error": (
                        f"API error: Snowflake provider returned error code {snowflake_error_code}"
                    ),
                    **accounting,
                    "metadata": {"snowflakeErrorCode": snowflake_error_code},
                }
            )

        message = validate_chat_completion_message(
            choice.get("message") if isinstance(choice, dict) else None,
            finish_reason=finish_reason,
        )
        if not message:
            if delete_from_cache:
                await delete_from_cache()
            return _defined(
                {
                    "error": "Malformed response data: expected choices[0].message",
                    **accounting,
                    "finishReason": finish_reason or None,
                }
            )

        refusal = get_chat_completion_refusal(message, finish_reason)
        if refusal:
            return _defined({**refusal, **accounting, "finishReason": finish_reason or None})

        output = _get_snowflake_output(message)
        response_format = config.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            output = parse_chat_completion_json_output(
                message, output, "[Snowflake Cortex] Failed to parse JSON output"
            )
        return _defined({"output": output, **accounting, "finishReason": finish_reason or None})


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `snowflake:` registry factory."""
    model_name = ":".join(provider_path.split(":")[1:])
    if not model_name:
        raise ValueError(
            "Snowflake provider requires a model name. Use format: snowflake:<model_name>"
        )
    return SnowflakeCortexProvider(model_name, {**options, "config": options.get("config") or {}})
