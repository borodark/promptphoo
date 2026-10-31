"""OpenRouter chat provider. Ports src/providers/openrouter.ts and the `openrouter:` factory in
src/providers/registry.ts.

Not ported: tracing spans, abort signals and the response-header observer.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from promptphoo import fetch
from promptphoo.finish_reason import FINISH_REASON_MAP, normalize_finish_reason
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.compatible.common import (
    configured_provider_options,
    serialize_provider,
)
from promptphoo.providers.openai.chat import (
    OpenAiChatCompletionProvider,
    get_openai_gateway_rate_limit_kind,
    get_openai_rate_limit_response,
)
from promptphoo.providers.openai.prepare import is_openai_json_schema_format
from promptphoo.providers.openai.util import (
    append_openai_api_path,
    format_openai_error,
    get_openai_chat_choice_error,
    get_openai_partial_output,
    get_openai_policy_refusal,
    get_token_usage_with_request_count,
    is_openai_error_only_response,
    js_error_string,
    validate_chat_completion_message,
)
from promptphoo.providers.openrouter_billing import (
    calculate_openrouter_response_cost,
    get_openrouter_billing_metadata,
)
from promptphoo.providers.shared import get_request_timeout_ms
from promptphoo.tracing.genai_tracer import extract_genai_response, get_test_index, with_genai_span

logger = logging.getLogger(__name__)

# OpenRouter's canonical error_type vocabulary. The availability types recover on their own;
# the permanent set describes a request that fails identically on retry.
TRANSIENT_ERROR_TYPES = frozenset(
    ["provider_overloaded", "provider_unavailable", "timeout", "server"]
)

PERMANENT_ERROR_TYPES = frozenset([
    "authentication", "permission_denied", "payment_required", "invalid_request",
    "invalid_prompt", "not_found", "precondition_failed", "payload_too_large", "unprocessable",
    "content_policy_violation", "refusal", "context_length_exceeded", "max_tokens_exceeded",
    "token_limit_exceeded", "string_too_long", "invalid_image", "image_too_large",
    "image_too_small", "unsupported_image_format", "image_not_found", "image_download_failed",
])  # fmt: skip

DEFAULT_OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"


def get_choice_error_kind(error: Any) -> dict[str, str] | None:
    """Classify a choice-level error code arriving in a 200 envelope."""
    record = error if isinstance(error, dict) else None
    metadata = (record or {}).get("metadata")
    metadata = metadata if isinstance(metadata, dict) else None
    error_type = (metadata or {}).get("error_type")
    error_type = error_type if isinstance(error_type, str) else None
    if error_type and error_type != "rate_limit_exceeded":
        if error_type in TRANSIENT_ERROR_TYPES:
            return {"retryableErrorKind": "transient_availability"}
        if error_type in PERMANENT_ERROR_TYPES:
            return None
        # Unknown or newly added error_type: fall through to the status code.
    code = (record or {}).get("code")
    status: float | None = None
    if isinstance(code, (int, float)) and not isinstance(code, bool):
        status = code
    elif isinstance(code, str) and re.fullmatch(r"\d{3}", code):
        status = int(code)
    if status == 429:
        return {"rateLimitKind": "rate_limit"}
    if status in (502, 503, 504):
        return {"retryableErrorKind": "transient_availability"}
    return None


def _with_cost(response: ProviderResponse, cost: Any) -> ProviderResponse:
    if cost is not None:
        response["cost"] = cost
    return response


def _bearer(value: Any) -> str:
    # Template-literal interpolation of an unset key.
    return f"Bearer {'undefined' if value is None else value}"


class OpenRouterProvider(OpenAiChatCompletionProvider):
    """Chat completions through OpenRouter, keeping `content` as the primary output with any
    `reasoning` shown as thinking."""

    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        config = provider_options.get("config") or {}
        passthrough: dict[str, Any] = {}
        # Pass through OpenRouter-specific options: https://openrouter.ai/docs/requests
        for key in ("transforms", "models", "route", "provider"):
            if is_truthy(config.get(key)):
                passthrough[key] = config[key]
        passthrough.update(config.get("passthrough") or {})
        super().__init__(
            model_name,
            {
                **provider_options,
                "config": {
                    **config,
                    "apiBaseUrl": config.get("apiBaseUrl") or DEFAULT_OPENROUTER_API_BASE_URL,
                    "apiKeyEnvar": config.get("apiKeyEnvar") or "OPENROUTER_API_KEY",
                    "passthrough": passthrough,
                },
            },
        )

    def default_id(self) -> str:
        return f"openrouter:{self.model_name}"

    def __str__(self) -> str:
        return f"[OpenRouter Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "openrouter")

    def calculate_response_cost(
        self, data: Any, config: dict[str, Any], cached: bool = False
    ) -> float | None:
        return calculate_openrouter_response_cost(data, config)

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        span_context = {
            "system": "openrouter",
            "operationName": "chat",
            "model": self.model_name,
            "providerId": self.id(),
            "testIndex": get_test_index(context),
            "promptLabel": ((context or {}).get("prompt") or {}).get("label"),
            "traceparent": (context or {}).get("traceparent"),
        }
        try:
            prepared = await self.get_openai_body(prompt, context, options)
        except Exception as error:
            failure = error

            async def rethrow(_span: Any) -> ProviderResponse:
                raise failure

            return await with_genai_span(
                span_context, rethrow, lambda response: extract_genai_response(response, True)
            )
        return await with_genai_span(
            {**span_context, **self.get_chat_tracing_request(prepared["body"])},
            lambda _span: self.execute_openrouter_call(prepared, context, options),
            lambda response: extract_genai_response(response, True),
        )

    async def execute_openrouter_call(
        self,
        prepared: dict[str, Any],
        context: dict[str, Any] | None = None,
        call_api_options: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        body, config = prepared["body"], prepared["config"]
        context = context or {}
        logger.debug("Calling OpenRouter API: model=%s", self.model_name)

        data: Any = None
        cached = False
        delete_from_cache = None
        response_headers: dict[str, str] | None = None
        try:
            organization = self.get_organization()
            bust = context.get("bustCache")
            result = await fetch.fetch_with_cache(
                append_openai_api_path(self.get_api_url(), "chat/completions"),
                {
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        "Authorization": _bearer(self.get_api_key()),
                        **({"OpenAI-Organization": organization} if organization else {}),
                        **(config.get("headers") or {}),
                    },
                    "body": json_stringify(body),
                },
                get_request_timeout_ms(),
                "json",
                bool(bust if bust is not None else context.get("debug")),
                None,
            )
            data, cached = result.data, result.cached
            status, status_text = result.status, result.status_text
            delete_from_cache, response_headers = result.delete_from_cache, result.headers
            http = {"http": {"status": status, "statusText": status_text,
                             "headers": response_headers or {}}}  # fmt: skip
            usage_fields = (
                {"tokenUsage": get_token_usage_with_request_count(data, cached)}
                if isinstance(data, dict) and data.get("usage")
                else {}
            )

            policy = get_openai_policy_refusal(data, True)
            if policy:
                response: ProviderResponse = {
                    "output": policy["message"]
                    if "partialOutput" not in policy
                    else get_openai_partial_output(
                        policy["partialOutput"],
                        is_openai_json_schema_format(body.get("response_format")),
                    ),
                    **usage_fields,
                    "cached": cached,
                }
                _with_cost(response, self.calculate_response_cost(data, config))
                response["isRefusal"] = True
                response["guardrails"] = {
                    "flagged": True,
                    **({"flaggedInput": True} if policy.get("flaggedInput") else {}),
                    "reason": policy["message"],
                }
                response["raw"] = data
                response["metadata"] = {
                    **(get_openrouter_billing_metadata(data) or {}),
                    **({"providerPolicy": {"code": policy["code"]}} if policy.get("code") else {}),
                    **http,
                }
                return response
            has_error = isinstance(data, dict) and is_truthy(data.get("error"))
            choice_error = None if has_error else get_openai_chat_choice_error(data)
            if choice_error:
                if delete_from_cache:
                    await delete_from_cache()
                rate_limit_kind = get_openai_gateway_rate_limit_kind(data)
                # The gateway classifier wins over the bare-code fallback: a billing-coded
                # 429 must stay 'quota', not be flattened to 'rate_limit'.
                choice_kind = (
                    None if rate_limit_kind else get_choice_error_kind(choice_error["error"])
                )
                response = {
                    "error": f"API error: {choice_error['error']['message']}",
                    **usage_fields,
                    "cached": cached,
                }
                _with_cost(response, self.calculate_response_cost(data, config))
                response["raw"] = data
                response["finishReason"] = "error"
                metadata: dict[str, Any] = {**(get_openrouter_billing_metadata(data) or {})}
                if rate_limit_kind:
                    metadata["rateLimitKind"] = rate_limit_kind
                elif choice_kind and choice_kind.get("rateLimitKind"):
                    metadata["rateLimitKind"] = choice_kind["rateLimitKind"]
                if choice_kind and choice_kind.get("retryableErrorKind"):
                    metadata["retryableErrorKind"] = choice_kind["retryableErrorKind"]
                response["metadata"] = {**metadata, **http}
                return response
            if status < 200 or status >= 300:
                rate_limit_kind = get_openai_gateway_rate_limit_kind(data)
                payload = data if isinstance(data, str) else json_stringify(data)
                return {
                    "error": f"API error: {status} {status_text}\n{payload}",
                    "metadata": {
                        **({"rateLimitKind": rate_limit_kind} if rate_limit_kind else {}),
                        **http,
                    },
                }
            if is_openai_error_only_response(data):
                return {"error": format_openai_error(data)}
        except Exception as err:
            logger.error("API call error: %s", js_error_string(err))
            rate_limit_response = get_openai_rate_limit_response(err, response_headers)
            if rate_limit_response:
                return rate_limit_response
            return {"error": f"API call error: {js_error_string(err)}"}

        if isinstance(data, dict) and is_truthy(data.get("error")):
            return {"error": format_openai_error(data)}

        choices = data.get("choices") if isinstance(data, dict) else None
        first = choices[0] if isinstance(choices, list) and choices else None
        # Guard against a 200 response with an empty or missing `choices` array.
        if not isinstance(first, dict) or not is_truthy(first.get("message")):
            # A malformed 200 must not be cached and replayed as the provider's answer.
            if delete_from_cache:
                await delete_from_cache()
            response = {
                "error": "Malformed response data: null"
                if data is None
                else "Malformed response data: expected choices[0].message",
                "tokenUsage": get_token_usage_with_request_count(data, cached),
                "cached": cached,
            }
            if is_truthy(data):
                _with_cost(response, self.calculate_response_cost(data, config))
                billing = get_openrouter_billing_metadata(data)
                if billing is not None:
                    response["metadata"] = billing
            return response

        def accounting(**fields: Any) -> ProviderResponse:
            response: ProviderResponse = {
                **fields,
                "tokenUsage": get_token_usage_with_request_count(data, cached),
                "cached": cached,
            }
            return _with_cost(response, self.calculate_response_cost(data, config))

        def with_metadata(response: ProviderResponse) -> ProviderResponse:
            billing = get_openrouter_billing_metadata(data)
            if billing is not None:
                response["metadata"] = billing
            if finish_reason:
                response["finishReason"] = finish_reason
            return response

        finish_reason = normalize_finish_reason(first.get("finish_reason"))
        if finish_reason == "error":
            # A failed generation carries partial output that must not be graded.
            if delete_from_cache:
                await delete_from_cache()
            return with_metadata(
                accounting(error="API error: OpenRouter provider returned a generation error")
            )
        message = validate_chat_completion_message(
            first["message"],
            allow_audio=True,
            allow_structured_content=True,
            finish_reason=finish_reason,
        )
        if not message:
            if delete_from_cache:
                await delete_from_cache()
            return with_metadata(
                accounting(error="Malformed response data: unusable choices[0].message")
            )
        if message["refusal"] or finish_reason == FINISH_REASON_MAP["content_filter"]:
            if message["structuredContent"]:
                output: Any = message["structuredContent"]
            elif message["content"]:
                output = get_openai_partial_output(
                    message["content"], is_openai_json_schema_format(body.get("response_format"))
                )
            else:
                output = message["refusal"] or "Content filtered by the model provider."
            response = accounting(output=output)
            response["isRefusal"] = True
            response["guardrails"] = {"flagged": True}
            response["raw"] = data
            return with_metadata(response)
        if message["audio"]:
            # Audio models answer with `content: null` and the payload on `message.audio`.
            audio = message["audio"]
            body_audio = body.get("audio") if isinstance(body.get("audio"), dict) else {}
            transcript = audio.get("transcript")
            audio_fields = {
                "id": audio.get("id") if isinstance(audio.get("id"), str) else None,
                "expiresAt": audio.get("expires_at")
                if isinstance(audio.get("expires_at"), (int, float))
                and not isinstance(audio.get("expires_at"), bool)
                else None,
                "data": audio.get("data") if isinstance(audio.get("data"), str) else None,
                "transcript": transcript if isinstance(transcript, str) else None,
                "format": audio["format"]
                if isinstance(audio.get("format"), str)
                else (body_audio.get("format") if body_audio.get("format") is not None else "wav"),
            }
            return with_metadata(
                accounting(
                    output=transcript if isinstance(transcript, str) else "",
                    audio={k: v for k, v in audio_fields.items() if v is not None},
                )
            )

        show_thinking = self.config.get("showThinking")
        show_thinking = True if show_thinking is None else show_thinking
        # Prioritize tool calls over content and reasoning.
        output = ""
        if message["functionCall"] or message["toolCalls"]:
            # Tool calls always take priority and never include thinking.
            output = message["functionCall"] or message["toolCalls"]
        elif message["structuredContent"]:
            # Content can be an array of parts (e.g. Gemini text plus image).
            output = message["structuredContent"]
        elif message["content"] and message["content"].strip():
            output = message["content"]
            if message["reasoning"] and show_thinking:
                output = f"Thinking: {message['reasoning']}\n\n{output}"
        elif message["reasoning"] and show_thinking:
            output = message["reasoning"]
        if is_openai_json_schema_format(body.get("response_format")):
            # Prefer the raw content so a "Thinking:" prefix does not break JSON.
            content = message["content"]
            candidate = (
                content if isinstance(content, str) else output if isinstance(output, str) else None
            )
            if candidate:
                try:
                    output = js_json_parse(candidate)
                except ValueError as error:
                    logger.warning(
                        "Failed to parse JSON output for json_schema: SyntaxError: %s", error
                    )
        return with_metadata(accounting(output=output))


def create_openrouter_provider(
    provider_path: str, options: dict[str, Any] | None = None
) -> OpenRouterProvider:
    options = options or {}
    model_name = ":".join(provider_path.split(":")[1:])
    provider_options: dict[str, Any] = dict(options.get("config") or {})
    if options.get("env") and not provider_options.get("env"):
        provider_options["env"] = options["env"]
    if options.get("id") and not provider_options.get("id"):
        provider_options["id"] = options["id"]
    return OpenRouterProvider(model_name, provider_options)


def create_provider(provider_path: str, options: dict[str, Any]) -> OpenRouterProvider:
    """The `openrouter:` registry factory (configuredProviderFactory)."""
    return create_openrouter_provider(provider_path, configured_provider_options(options))
