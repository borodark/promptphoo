"""Anthropic Messages API provider. Ports src/providers/anthropic/messages.ts.

Differences from upstream:
- Raw HTTP (POST {base}/v1/messages) instead of the SDK; the request body is the SDK params.
- `stream: true` is kept in the request params (and so in the cache key) but the request is sent
  non-streaming and the final message is read in one response. The SDK's refusal to send a
  non-streaming request that could take longer than ten minutes (max_tokens > 21,333) is kept.
- MCP (`config.mcp`) and Claude Code OAuth are not supported; MCP config is ignored with a warning.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from promptphoo import cache as cache_mod
from promptphoo import envars
from promptphoo.finish_reason import normalize_finish_reason
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.anthropic.generic import (
    AnthropicAPIError,
    AnthropicGenericProvider,
    hash_anthropic_cache_value,
    post_anthropic,
)
from promptphoo.providers.anthropic.util import (
    ANTHROPIC_MODELS,
    calculate_anthropic_cost,
    clamp_max_tokens_for_thinking_budget,
    claude_thinking_consumes_tokens,
    get_claude_model_warning_name,
    get_file_references,
    get_refusal_details,
    get_token_usage,
    is_always_on_adaptive_thinking_claude_model,
    is_between_tools_lowest_thinking_claude_model,
    is_claude_sonnet55_model,
    is_claude_thinking_enabled,
    is_disabled_thinking_rejected_at_effort,
    is_forced_tool_choice_unsupported_claude_model,
    is_sampling_params_deprecated_claude_model,
    normalize_anthropic_model_name,
    normalize_claude_thinking_config,
    output_from_message,
    parse_messages,
    process_anthropic_tools,
    resolve_claude_sampling_params,
)
from promptphoo.providers.shared import transform_tool_choice, transform_tools
from promptphoo.tokens import create_empty_token_usage
from promptphoo.tracing.genai_tracer import (
    extract_provider_response_attributes,
    get_test_index,
    with_genai_span,
)

logger = logging.getLogger(__name__)

# Each resume re-sends the turn so far, so a pausing run stops after this many.
MAX_PAUSE_TURN_RESUMES = 5
STREAMING_REQUIRED_MESSAGE = (
    "Streaming is required for operations that may take longer than 10 minutes. See "
    "https://github.com/anthropics/anthropic-sdk-typescript#long-requests for more details"
)


def _normalize_headers_for_cache_key(headers: dict[str, str]) -> list[dict[str, str]] | None:
    if not headers:
        return None
    entries = [
        {"name": name.lower(), "valueHash": hash_anthropic_cache_value(value)}
        for name, value in headers.items()
    ]
    return sorted(entries, key=lambda e: (e["name"], e["valueHash"]))


def _with_turn_container(params: dict[str, Any], responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Reuse the turn's container unless the caller pinned a different one."""
    requested = params.get("container")
    if isinstance(requested, str) or (
        isinstance(requested, dict) and requested.get("id") is not None
    ):
        return params
    for response in reversed(responses):
        container = response.get("container")
        if container:
            return {
                **params,
                "container": {**requested, "id": container["id"]} if requested else container["id"],
            }
    return params


def _merge_anthropic_usage(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    entries = [m["usage"] for m in messages if m.get("usage") is not None]
    if not entries:
        return None
    acc = entries[0]
    for usage in entries[1:]:
        acc_cc, usage_cc = acc.get("cache_creation"), usage.get("cache_creation")
        acc_otd, usage_otd = acc.get("output_tokens_details"), usage.get("output_tokens_details")
        acc = {
            **usage,
            "input_tokens": (acc.get("input_tokens") or 0) + (usage.get("input_tokens") or 0),
            "output_tokens": (acc.get("output_tokens") or 0) + (usage.get("output_tokens") or 0),
            "cache_creation_input_tokens": (acc.get("cache_creation_input_tokens") or 0)
            + (usage.get("cache_creation_input_tokens") or 0),
            "cache_creation": {
                "ephemeral_5m_input_tokens": ((acc_cc or {}).get("ephemeral_5m_input_tokens") or 0)
                + ((usage_cc or {}).get("ephemeral_5m_input_tokens") or 0),
                "ephemeral_1h_input_tokens": ((acc_cc or {}).get("ephemeral_1h_input_tokens") or 0)
                + ((usage_cc or {}).get("ephemeral_1h_input_tokens") or 0),
            }
            if acc_cc or usage_cc
            else None,
            "cache_read_input_tokens": (acc.get("cache_read_input_tokens") or 0)
            + (usage.get("cache_read_input_tokens") or 0),
            "output_tokens_details": {
                "thinking_tokens": ((acc_otd or {}).get("thinking_tokens") or 0)
                + ((usage_otd or {}).get("thinking_tokens") or 0),
            }
            if acc_otd or usage_otd
            else None,
        }
    return acc


def _with_merged_usage(response: dict[str, Any], responses: list[dict[str, Any]]) -> dict[str, Any]:
    usage = _merge_anthropic_usage(responses)
    return {**response, "usage": usage} if usage else response


def _to_cached_message(message: dict[str, Any], responses: list[dict[str, Any]]) -> dict[str, Any]:
    if len(responses) == 1:
        return message
    return {
        **message,
        "fileReferences": [
            ref
            for r in responses
            for block in r.get("content") or []
            for ref in get_file_references(block)
        ],
        "finalText": "".join(
            b.get("text", "") for b in responses[-1].get("content") or [] if b.get("type") == "text"
        ),
        "billedCalls": [
            {
                "stop_details": r.get("stop_details"),
                "stop_reason": r.get("stop_reason"),
                "usage": r.get("usage"),
            }
            for r in responses
        ],
    }


def _maybe_load_response_format(output_format: Any, vars: dict[str, Any] | None) -> Any:
    from promptphoo.files import maybe_load_response_format_from_external_file

    return maybe_load_response_format_from_external_file(output_format, vars)


def _error_message(err: BaseException) -> str:
    return str(err) if str(err) else type(err).__name__


class AnthropicMessagesProvider(AnthropicGenericProvider):
    SUPPORTS_CLAUDE_CODE_OAUTH = True
    WARNS_ON_UNKNOWN_MODEL = True
    ANTHROPIC_MODELS = ANTHROPIC_MODELS
    ANTHROPIC_MODELS_NAMES = [m["id"] for m in ANTHROPIC_MODELS]

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, options)
        self._sampling_params_deprecation_warned = False
        self._manual_thinking_conversion_warned = False
        self._disabled_thinking_removal_warned = False
        if (
            type(self).WARNS_ON_UNKNOWN_MODEL
            and normalize_anthropic_model_name(model_name) not in self.ANTHROPIC_MODELS_NAMES
        ):
            logger.warning("Using unknown Anthropic model: %s", model_name)
        if (self.config.get("mcp") or {}).get("enabled"):
            logger.warning(
                "[anthropic] MCP servers are not supported by promptphoo; config.mcp is ignored."
            )

    # --- cost --------------------------------------------------------------------------------

    def calculate_message_cost(
        self, config: dict[str, Any], message: dict[str, Any], model_name: str | None = None
    ) -> float | None:
        usage = message.get("usage") or {}
        # Since September 24, 2026, only these categories bill refusals before any output.
        if (
            message.get("stop_reason") == "refusal"
            and usage.get("output_tokens") == 0
            and ((message.get("stop_details") or {}).get("category") or "")
            not in ("bio", "frontier_llm", "reasoning_extraction")
        ):
            return 0
        return calculate_anthropic_cost(
            model_name or self.model_name,
            config,
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("cache_read_input_tokens"),
            usage.get("cache_creation_input_tokens"),
            (usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens"),
            usage.get("inference_geo"),
        )

    def _calculate_message_costs(
        self, config: dict[str, Any], calls: list[dict[str, Any]]
    ) -> float | None:
        total: float | None = 0
        for call in calls:
            call_cost = self.calculate_message_cost(config, call)
            total = total + call_cost if total is not None and call_cost is not None else None
        return total

    # --- hooks for compatible adapters -------------------------------------------------------

    def sanitize_request_headers(self, headers: dict[str, str]) -> dict[str, str]:
        return headers

    def allows_claude_generation_fallback(self) -> bool:
        from urllib.parse import urlparse

        try:
            return (urlparse(self.request_base_url()).hostname or "").lower() == "api.anthropic.com"
        except ValueError:
            return False

    def validate_authentication(self) -> None:
        if not self.api_key and not self.using_claude_code_oauth:
            raise ValueError(
                "Anthropic API key is not set. Set the ANTHROPIC_API_KEY environment variable or "
                "add `apiKey` to the provider config. Alternatively, if you have an active Claude "
                "Code session, set `apiKeyRequired: false` in the provider config to authenticate "
                "via Claude Code."
            )

    def should_cache_responses(self) -> bool:
        return True

    # --- request ------------------------------------------------------------------------------

    async def _create_message(
        self, params: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        body = dict(params)
        if body.get("stream"):
            # Not streamed: read the final message in one response (see module docstring).
            body["stream"] = False
        max_tokens = body.get("max_tokens")
        if (
            not params.get("stream")
            and isinstance(max_tokens, int | float)
            and 60 * 60 * max_tokens / 128000 > 10 * 60
        ):
            raise ValueError(STREAMING_REQUIRED_MESSAGE)
        return await post_anthropic(self, "/v1/messages", body, headers)

    async def _send_message(
        self, params: dict[str, Any], headers: dict[str, str], responses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Resume paused server-tool turns, retaining each request for usage and cost."""

        async def send(request_params: dict[str, Any]) -> dict[str, Any]:
            request_params = _with_turn_container(request_params, responses)
            message = await self._create_message(request_params, headers)
            responses.append(message)
            return message

        message = await send(params)
        if message.get("stop_reason") != "pause_turn":
            return message
        turn_content = list(message.get("content") or [])
        resumes = 0
        while message.get("stop_reason") == "pause_turn":
            if resumes == MAX_PAUSE_TURN_RESUMES:
                logger.warning(
                    "Claude's turn was still paused (stop_reason: pause_turn) after %d resumes, "
                    "so the output may be incomplete. Lower the tool max_uses or split the task.",
                    MAX_PAUSE_TURN_RESUMES,
                )
                break
            try:
                message = await send(
                    {
                        **params,
                        "messages": [
                            *params["messages"],
                            {"role": "assistant", "content": turn_content},
                        ],
                    }
                )
            except Exception as err:
                logger.warning(
                    "Could not resume a paused Claude turn, so the output may be incomplete: %s",
                    _error_message(err),
                )
                break
            turn_content = [*turn_content, *(message.get("content") or [])]
            resumes += 1
        return {**message, "content": turn_content}

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.validate_authentication()
        if not self.model_name:
            raise ValueError("Anthropic model name is not set. Please provide a valid model name.")
        span_context = {
            "system": self.get_gen_ai_system(),
            "operationName": "chat",
            "model": self.model_name,
            "providerId": self.id(),
            "maxTokens": self.config.get("max_tokens"),
            "temperature": self.config.get("temperature"),
            "testIndex": get_test_index(context),
            "promptLabel": ((context or {}).get("prompt") or {}).get("label"),
            "traceparent": (context or {}).get("traceparent"),
            "requestBody": prompt,
        }
        return await with_genai_span(
            span_context,
            lambda _span: self._call_api_internal(prompt, context),
            extract_provider_response_attributes,
        )

    def get_gen_ai_system(self) -> str:
        return "anthropic"

    def _resolve_model_thinking(
        self,
        requested: Any,
        effort: str | None,
        *,
        sampling_params_deprecated: bool,
        always_on_adaptive_thinking: bool,
        model_warning_name: str,
    ) -> tuple[Any, bool, bool]:
        """Returns (thinking, thinkingEnabled, thinkingConsumesTokens)."""
        requested_type = requested.get("type") if isinstance(requested, dict) else None
        if (
            sampling_params_deprecated or always_on_adaptive_thinking
        ) and requested_type == "enabled":
            if not self._manual_thinking_conversion_warned:
                logger.warning(
                    f"Adaptive thinking is always on for {model_warning_name}. Manual thinking "
                    "budgets have been removed; use effort to control reasoning depth."
                    if always_on_adaptive_thinking
                    else f'Manual extended thinking (thinking.type "enabled") is not supported on '
                    f"{model_warning_name} and has been converted to adaptive thinking. Use "
                    'thinking: { type: "adaptive" } with effort to control reasoning depth.'
                )
                self._manual_thinking_conversion_warned = True
        elif requested_type == "disabled" and not self._disabled_thinking_removal_warned:
            between_tools = is_between_tools_lowest_thinking_claude_model(self.model_name)
            if always_on_adaptive_thinking:
                logger.warning(
                    f"Adaptive thinking is always on for {model_warning_name}. thinking.type "
                    '"disabled" has been omitted.'
                )
                self._disabled_thinking_removal_warned = True
            elif is_disabled_thinking_rejected_at_effort(self.model_name, effort):
                lowest = "between_tools" if between_tools else "disabled"
                logger.warning(
                    f'{model_warning_name} only accepts thinking.type "{lowest}" at effort "high" '
                    f'or below (got "{effort}"), so thinking.type "disabled" has been omitted. '
                    'Lower effort to "high" if you need thinking off.'
                )
                self._disabled_thinking_removal_warned = True
            elif between_tools:
                logger.warning(
                    f'{model_warning_name} does not accept thinking.type "disabled", so it has '
                    'been sent as "between_tools", the model\'s lowest setting, which turns off '
                    'up-front thinking. Set thinking.type "between_tools" to silence this warning.'
                )
                self._disabled_thinking_removal_warned = True
        elif (
            requested_type == "between_tools"
            and not self._disabled_thinking_removal_warned
            and is_between_tools_lowest_thinking_claude_model(self.model_name)
            and is_disabled_thinking_rejected_at_effort(self.model_name, effort)
        ):
            logger.warning(
                f'{model_warning_name} only accepts thinking.type "between_tools" at effort "high" '
                f'or below (got "{effort}"), so it has been omitted and the model thinks '
                'adaptively. Lower effort to "high" to turn off up-front thinking.'
            )
            self._disabled_thinking_removal_warned = True

        resolved = normalize_claude_thinking_config(
            self.model_name,
            requested,
            effort,
            allow_generation_fallback=sampling_params_deprecated,
        )
        thinking_enabled = always_on_adaptive_thinking or is_claude_thinking_enabled(resolved)
        consumes = claude_thinking_consumes_tokens(self.model_name, resolved)
        return resolved, thinking_enabled, consumes

    def _build_message_response(
        self,
        message: dict[str, Any],
        config: dict[str, Any],
        processed_output_format: Any,
        cached: bool,
    ) -> dict[str, Any]:
        finish_reason = normalize_finish_reason(message.get("stop_reason"))
        show_thinking = config.get("showThinking")
        output: Any = output_from_message(message, True if show_thinking is None else show_thinking)
        is_structured = (
            isinstance(processed_output_format, dict)
            and processed_output_format.get("type") == "json_schema"
        )
        content = message.get("content") or []
        file_references = message.get("fileReferences")
        if file_references is None:
            file_references = [ref for block in content for ref in get_file_references(block)]

        if is_structured:
            has_pending_tools = message.get("stop_reason") == "pause_turn" or any(
                b.get("type") == "tool_use" for b in content
            )
            if has_pending_tools:
                text = ""
            else:
                text = message.get("finalText")
                if text is None:
                    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            try:
                output = json.loads(text or output)
            except ValueError as error:
                logger.error(
                    "Failed to parse JSON output from structured outputs: SyntaxError: %s", error
                )

        refusal_details = get_refusal_details(message)
        if refusal_details and not cached:
            logger.warning(refusal_details)

        response: dict[str, Any] = {"output": output}
        if file_references:
            response["metadata"] = {"fileReferences": file_references}
        response["tokenUsage"] = get_token_usage(message, cached)
        if finish_reason:
            response["finishReason"] = finish_reason
        if refusal_details:
            response["guardrails"] = {"flagged": True, "reason": refusal_details}
        cost = self._calculate_message_costs(config, message.get("billedCalls") or [message])
        if cost is not None:
            response["cost"] = cost
        if cached:
            response["cached"] = True
        return response

    async def _call_api_internal(
        self, prompt: str, context: dict[str, Any] | None
    ) -> dict[str, Any]:
        from promptphoo.assertions.tools import maybe_load_tools_from_external_file

        context = context or {}
        prompt_config = (context.get("prompt") or {}).get("config") or {}
        config: dict[str, Any] = {**self.config, **prompt_config}
        vars = context.get("vars")

        parsed = parse_messages(prompt)
        system = parsed.get("system")
        extracted_messages = parsed["extractedMessages"]
        prompt_thinking = parsed.get("thinking")
        if system and not extracted_messages:
            # The API rejects a request with a system prompt and no messages, which is what
            # system-only prompts (closed QA's and select-best's grading prompts) parse to.
            # Send the system content as the user turn, as upstream's Bedrock Claude path does
            # (D68).
            extracted_messages = [{"role": "user", "content": system}]
            system = None

        loaded_tools = await maybe_load_tools_from_external_file(config.get("tools"), vars) or []
        config_tools = transform_tools(loaded_tools, "anthropic")
        processed = process_anthropic_tools(config_tools)
        all_tools = processed["processedTools"]
        required_beta_features = processed["requiredBetaFeatures"]

        processed_output_format = _maybe_load_response_format(config.get("output_format"), vars)

        sampling_params_deprecated = is_sampling_params_deprecated_claude_model(
            self.model_name, allow_generation_fallback=self.allows_claude_generation_fallback()
        )
        always_on = is_always_on_adaptive_thinking_claude_model(self.model_name)
        model_warning_name = get_claude_model_warning_name(self.model_name) or "this Claude model"
        requested_thinking = config.get("thinking")
        if requested_thinking is None:
            requested_thinking = prompt_thinking
        resolved_thinking, thinking_enabled, thinking_consumes_tokens = (
            self._resolve_model_thinking(
                requested_thinking,
                config.get("effort"),
                sampling_params_deprecated=sampling_params_deprecated,
                always_on_adaptive_thinking=always_on,
                model_warning_name=model_warning_name,
            )
        )

        model_rejects_forced = is_forced_tool_choice_unsupported_claude_model(self.model_name)
        forced_rejected = (
            isinstance(resolved_thinking, dict) and resolved_thinking.get("type") == "enabled"
        ) or model_rejects_forced
        resolved_tool_choice: Any = None
        if is_truthy(config.get("tool_choice")):
            transformed = transform_tool_choice(config["tool_choice"], "anthropic")
            transformed_type = transformed.get("type") if isinstance(transformed, dict) else None
            if forced_rejected and transformed_type in ("any", "tool"):
                logger.warning(
                    f"tool_choice type '{transformed_type}' (forced tool use) is not supported on "
                    f"{model_warning_name} and will be omitted. Use 'auto' or 'none' instead."
                    if model_rejects_forced
                    else f"tool_choice type '{transformed_type}' (forced tool use) is incompatible "
                    "with extended thinking and will be omitted. Use 'auto' or remove tool_choice."
                )
            else:
                resolved_tool_choice = transformed

        env_temperature_raw = (self.env or {}).get("ANTHROPIC_TEMPERATURE")
        if env_temperature_raw is None:
            env_temperature_raw = envars.get_env_string("ANTHROPIC_TEMPERATURE")
        env_temperature = envars.parse_env_float(
            None if env_temperature_raw is None else str(env_temperature_raw)
        )
        sampling, sampling_warnings = resolve_claude_sampling_params(
            config,
            thinking_enabled=thinking_enabled,
            sampling_params_deprecated=sampling_params_deprecated,
            default_temperature=env_temperature if env_temperature is not None else 0,
        )
        for warning in sampling_warnings:
            logger.warning(warning)

        explicit_sampling = (
            config.get("temperature") is not None
            or config.get("top_p") is not None
            or config.get("top_k") is not None
            or env_temperature is not None
        )
        if (
            sampling_params_deprecated
            and explicit_sampling
            and not self._sampling_params_deprecation_warned
        ):
            logger.warning(
                f"temperature, top_p, and top_k are not supported on {model_warning_name} and "
                "will be omitted. Remove these sampling parameters from your config (or unset "
                "ANTHROPIC_TEMPERATURE) to silence this warning."
                if always_on
                else f"temperature is deprecated on {model_warning_name} and will be omitted "
                "(along with top_p and top_k). Remove these sampling parameters from your config "
                "(or unset ANTHROPIC_TEMPERATURE) to silence this warning."
            )
            self._sampling_params_deprecation_warned = True

        should_stream = config.get("stream")
        if should_stream is None:
            should_stream = False
        max_tokens = config.get("max_tokens")
        if max_tokens is None:
            max_tokens = envars.get_env_int(
                "ANTHROPIC_MAX_TOKENS", 2048 if thinking_consumes_tokens else 1024
            )

        params: dict[str, Any] = {"model": self.model_name}
        if system:
            params["system"] = system
        params["max_tokens"] = clamp_max_tokens_for_thinking_budget(max_tokens, resolved_thinking)
        params["messages"] = extracted_messages
        params["stream"] = should_stream
        params.update(sampling)
        if is_truthy(config.get("cache_control")):
            params["cache_control"] = config["cache_control"]
        if is_truthy(config.get("service_tier")):
            params["service_tier"] = config["service_tier"]
        if config.get("stop_sequences"):
            params["stop_sequences"] = config["stop_sequences"]
        if is_truthy(config.get("metadata")):
            params["metadata"] = config["metadata"]
        if all_tools:
            params["tools"] = all_tools
        if resolved_tool_choice:
            params["tool_choice"] = resolved_tool_choice
        if resolved_thinking:
            params["thinking"] = resolved_thinking
        if is_truthy(processed_output_format) or is_truthy(config.get("effort")):
            output_config: dict[str, Any] = {}
            if is_truthy(processed_output_format):
                output_config["format"] = processed_output_format
            if is_truthy(config.get("effort")):
                output_config["effort"] = config["effort"]
            params["output_config"] = output_config
        extra_body = config.get("extra_body")
        if isinstance(extra_body, dict):
            params.update(extra_body)

        headers = self.sanitize_request_headers(dict(config.get("headers") or {}))
        all_beta: list[str] = [*(config.get("beta") or []), *required_beta_features]
        user_beta = (config.get("headers") or {}).get("anthropic-beta")
        if isinstance(user_beta, str) and user_beta:
            all_beta.extend(e.strip() for e in user_beta.split(",") if e.strip())
        if is_truthy(processed_output_format) and "structured-outputs-2025-11-13" not in all_beta:
            all_beta.append("structured-outputs-2025-11-13")
        all_beta = list(dict.fromkeys(all_beta))
        if all_beta:
            headers["anthropic-beta"] = ",".join(all_beta)

        should_use_cache = (
            cache_mod.is_cache_enabled()
            and (config.get("mcp") or {}).get("enabled") is not True
            and self.should_cache_responses()
            and not self.has_custom_headers()
            and not config.get("headers")
        )
        cache = cache_mod.get_cache() if should_use_cache else None
        cache_key_params = {k: v for k, v in params.items() if k != "metadata"}
        cache_key_headers = _normalize_headers_for_cache_key(headers)
        cache_key = None
        if should_use_cache:
            key_value = {
                **cache_key_params,
                **({"headers": cache_key_headers} if cache_key_headers else {}),
            }
            cache_key = (
                f"anthropic:messages:{self.model_name}:{self.get_cache_identity_hash()}:"
                f"{self.get_cache_namespace()}:{hash_anthropic_cache_value(key_value)}"
            )
        ephemeral_key = cache_mod.get_scoped_cache_key(cache_key) if cache_key else None
        clear_generation = cache_mod.get_cache_clear_generation()

        if cache is not None and cache_key and ephemeral_key:
            cached_response = await self.get_cached_response(
                cache, cache_key, ephemeral_key, clear_generation
            )
            if cached_response:
                try:
                    cached_message = json.loads(cached_response)
                    if not isinstance(cached_message, dict):
                        raise TypeError("legacy cache entry")
                    if cached_message.get("stop_reason") != "pause_turn":
                        logger.debug("Returning cached Anthropic Messages response")
                        return self._build_message_response(
                            cached_message, config, processed_output_format, True
                        )
                except (ValueError, TypeError, AttributeError):
                    # An old cache item that was just the text content.
                    return {
                        "output": cached_response,
                        "cached": True,
                        "tokenUsage": create_empty_token_usage(),
                    }

        responses: list[dict[str, Any]] = []
        try:
            resolved_message = _with_merged_usage(
                await self._send_message(params, headers, responses), responses
            )
            message = _to_cached_message(resolved_message, responses)
            if (
                cache is not None
                and cache_key
                and ephemeral_key
                and (message.get("stop_reason") != "pause_turn")
            ):
                try:
                    await self.set_cached_response(
                        cache,
                        cache_key,
                        ephemeral_key,
                        clear_generation,
                        cache_mod.get_cache_ttl_ms(),
                        json_stringify(message),
                    )
                except Exception as err:
                    logger.error("Failed to cache response: %s", err)

            output_message = message
            if is_claude_sonnet55_model(self.model_name):
                content = message.get("content") or []
                earlier = [
                    block
                    for turn in responses[:-1]
                    for block in turn.get("content") or []
                    if block.get("type") == "thinking" and block not in content
                ]
                output_message = {**message, "content": [*earlier, *content]}
            return self._build_message_response(
                output_message, config, processed_output_format, False
            )
        except Exception as err:
            logger.error("Anthropic Messages API call error: %s", _error_message(err))
            error = f"API call error: {_error_message(err)}"
            if isinstance(err, AnthropicAPIError) and isinstance(err.error, dict):
                details = err.error.get("error")
                if isinstance(details, dict):
                    error = (
                        f"API call error: {details.get('message')}, status {err.status}, "
                        f"type {details.get('type')}"
                    )
            result: dict[str, Any] = {"error": error}
            if responses:
                result["tokenUsage"] = get_token_usage(
                    _with_merged_usage(responses[-1], responses), False
                )
                cost = self._calculate_message_costs(config, responses)
                if cost is not None:
                    result["cost"] = cost
            return result
