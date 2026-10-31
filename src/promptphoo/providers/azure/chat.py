"""Azure OpenAI chat completions, including "on your data" (`data_sources`). Ports
src/providers/azure/chat.ts.

Calls run inside a GenAI tracing span (see D57). Not ported: MCP tool injection (rejected with
a clear error).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from promptphoo import envars, fetch
from promptphoo.assertions.tools import maybe_load_tools_from_external_file
from promptphoo.fetch.errors import HttpRateLimitError, format_rate_limit_error_message
from promptphoo.files import (
    maybe_load_from_external_file,
    maybe_load_response_format_from_external_file,
)
from promptphoo.finish_reason import FINISH_REASON_MAP, normalize_finish_reason
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.anthropic.util import (
    is_forced_tool_choice_unsupported_claude_model,
    is_sampling_params_deprecated_claude_model,
)
from promptphoo.providers.azure.generic import AzureGenericProvider
from promptphoo.providers.azure.tables import DEFAULT_AZURE_API_VERSION
from promptphoo.providers.azure.util import calculate_azure_cost
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.function_callbacks import FunctionCallbackHandler
from promptphoo.providers.openai import gpt6
from promptphoo.providers.shared import get_request_timeout_ms, parse_chat_prompt, transform_tools
from promptphoo.templating import render_vars_in_object
from promptphoo.tracing.genai_tracer import get_test_index, with_genai_span

logger = logging.getLogger(__name__)

_MISSING = gpt6._MISSING


def _first(*values: Any) -> Any:
    """JS `a ?? b ?? c`."""
    return next((v for v in values if v is not None), None)


def _get(obj: Any, *path: str) -> Any:
    """JS optional chaining: `obj?.a?.b`, None for any missing step."""
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _tpl(obj: Any, key: str) -> str:
    """A value as a JS template literal prints it (`undefined` for a missing key)."""
    from promptphoo.jsvalues import js_string

    if not isinstance(obj, dict) or key not in obj:
        return "undefined"
    return js_string(obj[key])


def js_error_message(err: BaseException) -> str:
    return getattr(err, "message", None) or str(err)


def js_error_string(err: BaseException) -> str:
    """`String(err)`: `<name>: <message>`."""
    name = getattr(err, "name", None)
    if not isinstance(name, str):
        name = "TypeError" if isinstance(err, (TypeError, AttributeError, KeyError)) else "Error"
    message = js_error_message(err)
    return f"{name}: {message}" if message else name


def azure_usage_cost(model_name: str, config: dict[str, Any], usage: Any) -> float | None:
    return calculate_azure_cost(
        model_name,
        config,
        _get(usage, "prompt_tokens"),
        _get(usage, "completion_tokens"),
        _get(usage, "prompt_tokens_details", "cached_tokens"),
        _get(usage, "prompt_tokens_details", "audio_tokens"),
        _get(usage, "completion_tokens_details", "audio_tokens"),
        _get(usage, "prompt_tokens_details", "image_tokens"),
        _get(usage, "prompt_tokens_details", "cached_tokens_details", "audio_tokens"),
        _get(usage, "prompt_tokens_details", "cached_tokens_details", "image_tokens"),
        _get(usage, "completion_tokens_details", "image_tokens"),
    )


def _drop_none(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if v is not None}


def _azure_span_result(response: ProviderResponse) -> dict[str, Any]:
    """Azure chat's span result: four usage totals, prediction details, the finish reason."""
    result: dict[str, Any] = {}
    usage = response.get("tokenUsage")
    if usage:
        details = usage.get("completionDetails") or {}
        result["tokenUsage"] = {
            "prompt": usage.get("prompt"),
            "completion": usage.get("completion"),
            "total": usage.get("total"),
            "cached": usage.get("cached"),
            "completionDetails": {
                "reasoning": details.get("reasoning"),
                "acceptedPrediction": details.get("acceptedPrediction"),
                "rejectedPrediction": details.get("rejectedPrediction"),
            },
        }
    if response.get("finishReason"):
        result["finishReasons"] = [response["finishReason"]]
    return result


class AzureChatCompletionProvider(AzureGenericProvider):
    def __init__(self, deployment_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(deployment_name, options)
        mcp = self.config.get("mcp")
        if isinstance(mcp, dict) and mcp.get("enabled"):
            raise ValueError(
                "MCP tools (config.mcp) are not supported by promptphoo's Azure provider."
            )
        self.function_callback_handler = FunctionCallbackHandler()

    def is_reasoning_model(self, model_name: str | None = None) -> bool:
        """Reasoning models use max_completion_tokens instead of max_tokens, don't take
        temperature, and may take reasoning_effort."""
        if model_name is None:
            model_name = _first(self.config.get("modelName"), self.deployment_name)
        if is_truthy(self.config.get("isReasoningModel")) or is_truthy(self.config.get("o1")):
            return True
        lower = str(model_name).lower()
        return (
            lower.startswith("o1")
            or "-o1" in lower
            or lower.startswith("o3")
            or "-o3" in lower
            or lower.startswith("o4")
            or "-o4" in lower
            or lower.startswith("gpt-5")
            or "-gpt-5" in lower
            or lower == "gpt-chat-latest"
            or lower.startswith("gpt-chat-latest-")
            or gpt6.is_gpt6_model(lower)
            or "deepseek-r1" in lower
            or "deepseek_r1" in lower
            or "phi-4-reasoning" in lower
            or "phi-4-mini-reasoning" in lower
            # MAI-Code-* are fast coding models on the standard chat surface.
            or "mai-thinking" in lower
            or "mai-ds-r1" in lower
            or "mai-reasoning" in lower
            or ("grok" in lower and "reasoning" in lower)
        )

    def is_grok4_or_newer_model(self) -> bool:
        """Grok 4+ deployments reject presence_penalty, frequency_penalty and stop."""
        return re.search(r"grok-(?:[4-9]|\d{2,})", self.deployment_name.lower()) is not None

    def is_sampling_params_deprecated_claude_model(
        self, config: dict[str, Any] | None = None
    ) -> bool:
        """Claude Opus 4.7+ reject temperature/top_p but keep the standard max_tokens body."""
        config = self.config if config is None else config
        return bool(is_truthy(config.get("isClaudeOpus47OrLater"))) or (
            is_sampling_params_deprecated_claude_model(
                _first(config.get("modelName"), self.deployment_name),
                allow_generation_fallback=False,
            )
        )

    async def get_openai_body(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        call_api_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Returns `{"body", "config"}`: the wire request and the prompt-merged config."""
        context = context or {}
        vars = context.get("vars")
        prompt_config = (context.get("prompt") or {}).get("config")
        config: dict[str, Any] = {**self.config, **(prompt_config or {})}

        messages = parse_chat_prompt(prompt, [{"role": "user", "content": prompt}])

        if is_truthy(config.get("systemPrompt")):
            system = {"role": "system", "content": config["systemPrompt"]}
            index = next(
                (
                    i
                    for i, msg in enumerate(messages)
                    if isinstance(msg, dict) and msg.get("role") == "system"
                ),
                -1,
            )
            if index >= 0:
                messages[index] = system
            else:
                messages = [system, *messages]

        response_format = (
            {
                "response_format": maybe_load_response_format_from_external_file(
                    config["response_format"], vars
                )
            }
            if is_truthy(config.get("response_format"))
            else {}
        )

        passthrough = config.get("passthrough")
        passthrough_model = passthrough.get("model") if isinstance(passthrough, dict) else None
        capability_model_name = (
            passthrough_model
            if isinstance(passthrough_model, str)
            else _first(config.get("modelName"), self.deployment_name)
        ).lower()
        is_reasoning = self.is_reasoning_model(capability_model_name)
        is_fixed_reasoning = re.match(r"^gpt-chat-latest(?:-|$)", capability_model_name) is not None
        gpt6_variant = gpt6.get_gpt6_variant(capability_model_name)
        use_model_defaults = gpt6_variant in ("sol", "luna", "6.1-sol")
        sampling_params_deprecated = self.is_sampling_params_deprecated_claude_model(config)
        grok_sampling_restricted = self.is_grok4_or_newer_model()
        omit_defaults = is_truthy(config.get("omitDefaults"))

        if omit_defaults or use_model_defaults:
            max_tokens_default = (
                None
                if envars.get_env_string("OPENAI_MAX_TOKENS") is None
                else envars.get_env_int("OPENAI_MAX_TOKENS")
            )
        else:
            max_tokens_default = envars.get_env_int("OPENAI_MAX_TOKENS", 1024)
        max_tokens = _first(config.get("max_tokens"), max_tokens_default)
        max_completion_tokens = _first(
            config.get("max_completion_tokens"),
            envars.get_env_int("OPENAI_MAX_COMPLETION_TOKENS"),
            max_tokens,
        )

        if omit_defaults:
            temperature_default = (
                None
                if envars.get_env_string("OPENAI_TEMPERATURE") is None
                else envars.get_env_float("OPENAI_TEMPERATURE")
            )
        else:
            temperature_default = envars.get_env_float("OPENAI_TEMPERATURE", 0)
        temperature = _first(config.get("temperature"), temperature_default)

        def sampling(key: str, envar: str, default: float) -> Any:
            if omit_defaults:
                return _first(config.get(key), envars.get_env_float(envar))
            return _first(config.get(key), envars.get_env_float(envar, default))

        top_p = sampling("top_p", "OPENAI_TOP_P", 1)
        presence_penalty = sampling("presence_penalty", "OPENAI_PRESENCE_PENALTY", 0)
        frequency_penalty = sampling("frequency_penalty", "OPENAI_FREQUENCY_PENALTY", 0)

        # Reasoning effort. `_MISSING` is JS undefined; None is an explicit null.
        render = gpt6.make_render(lambda value: render_vars_in_object(value, vars))
        configured_gpt6_effort = (
            gpt6.get_gpt6_chat_reasoning_effort(self.config, prompt_config, render)
            if gpt6_variant
            else _MISSING
        )
        default_effort: Any = _MISSING if omit_defaults or use_model_defaults else "medium"
        if is_fixed_reasoning:
            reasoning_effort: Any = _MISSING
        elif gpt6_variant:
            reasoning_effort = (
                default_effort if configured_gpt6_effort is _MISSING else configured_gpt6_effort
            )
        else:
            configured = config.get("reasoning_effort")
            reasoning_effort = default_effort if configured is None else configured
        if not is_reasoning:
            rendered_effort: Any = _MISSING
        elif gpt6_variant or reasoning_effort is _MISSING:
            rendered_effort = reasoning_effort
        else:
            rendered_effort = render_vars_in_object(reasoning_effort, vars)

        loaded_tools = (
            (await maybe_load_tools_from_external_file(config["tools"], vars)) or []
            if is_truthy(config.get("tools"))
            else []
        )
        all_tools = list(transform_tools(loaded_tools, "openai"))

        body: dict[str, Any] = {"model": self.deployment_name, "messages": messages}
        if is_reasoning:
            if rendered_effort is not _MISSING:
                body["reasoning_effort"] = rendered_effort
            if max_completion_tokens is not None:
                body["max_completion_tokens"] = max_completion_tokens
            if gpt6.is_gpt6_model(capability_model_name) and temperature is not None:
                body["temperature"] = temperature
        else:
            if max_tokens is not None:
                body["max_tokens"] = max_tokens
            if temperature is not None and not sampling_params_deprecated:
                body["temperature"] = temperature
        # Grok takes reasoning_effort without being a reasoning model here; forward an
        # explicitly configured value.
        if grok_sampling_restricted and not is_reasoning and "reasoning_effort" in config:
            body["reasoning_effort"] = render_vars_in_object(config["reasoning_effort"], vars)
        if top_p is not None and not sampling_params_deprecated:
            body["top_p"] = top_p
        if presence_penalty is not None and not grok_sampling_restricted:
            body["presence_penalty"] = presence_penalty
        if frequency_penalty is not None and not grok_sampling_restricted:
            body["frequency_penalty"] = frequency_penalty
        if "seed" in config:
            body["seed"] = config["seed"]
        if is_truthy(config.get("functions")):
            body["functions"] = maybe_load_from_external_file(
                render_vars_in_object(config["functions"], vars)
            )
        if is_truthy(config.get("function_call")):
            body["function_call"] = config["function_call"]
        if all_tools:
            body["tools"] = all_tools
        if is_truthy(config.get("tool_choice")):
            body["tool_choice"] = config["tool_choice"]
        if is_truthy(config.get("deployment_id")):
            body["deployment_id"] = config["deployment_id"]
        # Legacy support for API versions before 2024-02-15-preview.
        if is_truthy(config.get("dataSources")):
            body["dataSources"] = config["dataSources"]
        if is_truthy(config.get("data_sources")):
            body["data_sources"] = config["data_sources"]
        body.update(response_format)
        include_log_probs = (call_api_options or {}).get("includeLogProbs")
        if is_truthy(include_log_probs):
            body["logprobs"] = include_log_probs
        if is_truthy(config.get("stop")) and not grok_sampling_restricted:
            body["stop"] = config["stop"]
        if isinstance(passthrough, dict):
            body.update(passthrough)

        if (
            is_forced_tool_choice_unsupported_claude_model(
                _first(config.get("modelName"), self.deployment_name)
            )
            and is_truthy(body.get("tool_choice"))
            and body["tool_choice"] not in ("auto", "none")
        ):
            logger.warning(
                "Forced tool choice is not supported on %s and will be omitted. Use 'auto' or "
                "'none' instead.",
                self.deployment_name,
            )
            del body["tool_choice"]

        if is_fixed_reasoning:
            body.pop("reasoning_effort", None)

        if gpt6_variant:
            if rendered_effort is _MISSING:
                body.pop("reasoning_effort", None)
            else:
                body["reasoning_effort"] = rendered_effort
        if use_model_defaults:
            output_cap = gpt6.resolve_gpt6_chat_output_cap(
                self.config,
                prompt_config,
                False,
                max_completion_tokens=envars.get_env_int("OPENAI_MAX_COMPLETION_TOKENS"),
                max_tokens=envars.get_env_int("OPENAI_MAX_TOKENS"),
            )
            if output_cap is None:
                body.pop("max_completion_tokens", None)
            else:
                body["max_completion_tokens"] = output_cap
        gpt6.apply_gpt6_request_rules(body, capability_model_name, "chat")

        return {
            "body": body,
            "config": {**config, "modelName": capability_model_name}
            if isinstance(passthrough_model, str)
            else config,
        }

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
        span_context = {
            "system": "azure",
            "operationName": "chat",
            "model": self.deployment_name,
            "providerId": self.id(),
            "maxTokens": self.config.get("max_tokens"),
            "temperature": self.config.get("temperature"),
            "topP": self.config.get("top_p"),
            "stopSequences": self.config.get("stop"),
            "frequencyPenalty": self.config.get("frequency_penalty"),
            "presencePenalty": self.config.get("presence_penalty"),
            "testIndex": get_test_index(context),
            "promptLabel": ((context or {}).get("prompt") or {}).get("label"),
            "traceparent": (context or {}).get("traceparent"),
        }
        return await with_genai_span(
            span_context,
            lambda _span: self._call_api_internal(prompt, context, options),
            _azure_span_result,
        )

    def get_chat_url(self, config: dict[str, Any]) -> str:
        api_version = config.get("apiVersion") or DEFAULT_AZURE_API_VERSION
        path = (
            "extensions/chat/completions"
            if is_truthy(config.get("dataSources"))
            else ("chat/completions")
        )
        return (
            f"{self.get_api_base_url()}/openai/deployments/{self.deployment_name}/{path}"
            f"?api-version={api_version}"
        )

    async def _call_api_internal(
        self,
        prompt: str,
        context: dict[str, Any] | None,
        options: dict[str, Any] | None,
    ) -> ProviderResponse:
        prepared = await self.get_openai_body(prompt, context, options)
        body, config = prepared["body"], prepared["config"]
        context = context or {}

        data: Any = None
        cached = False
        latency_ms: Any = None
        try:
            result = await fetch.fetch_with_cache(
                self.get_chat_url(config),
                {
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        **(self.auth_headers or {}),
                        **(self.config.get("headers") or {}),
                    },
                    "body": json_stringify(body),
                },
                get_request_timeout_ms(),
                "json",
                _first(context.get("bustCache"), context.get("debug")),
                self.config.get("maxRetries"),
            )
            cached = result.cached
            latency_ms = result.latency_ms
            if isinstance(result.data, str):
                try:
                    data = js_json_parse(result.data)
                except ValueError:
                    return {
                        "error": f"API returned invalid JSON response (status {result.status}): "
                        f"{result.data}\n\nRequest body: {json_stringify(body, 2)}"
                    }
            else:
                data = result.data
        except HttpRateLimitError as err:
            # Keep the structured signal so the scheduler does not retry hard quotas.
            return {
                "error": format_rate_limit_error_message(err),
                "metadata": {
                    "rateLimitKind": err.kind,
                    "http": {
                        "status": err.status,
                        "statusText": err.status_text,
                        "headers": err.headers or {},
                    },
                },
            }
        except Exception as err:
            return {"error": f"API call error: {js_error_message(err)}"}

        # Inputs and outputs can be flagged by content filters.
        flagged_input = False
        flagged_output = False
        is_refusal = False
        output: Any = ""
        log_probs: Any = None
        finish_reason: str | None = None

        try:
            if not isinstance(data, dict):
                raise TypeError(
                    f"Cannot read properties of {json_stringify(data)} (reading 'error')"
                )
            error = data.get("error")
            if is_truthy(error):
                # Was the input prompt deemed inappropriate?
                if (
                    _get(error, "status") == 400
                    and _get(error, "code") == FINISH_REASON_MAP["content_filter"]
                ):
                    flagged_input = True
                    output = _get(error, "message")
                    finish_reason = FINISH_REASON_MAP["content_filter"]
                else:
                    return {
                        "error": f"API response error: {_tpl(error, 'code')} "
                        f"{_tpl(error, 'message')}"
                    }
            else:
                has_data_sources = is_truthy(config.get("dataSources")) or is_truthy(
                    config.get("data_sources")
                )
                choices = data.get("choices")
                if has_data_sources:
                    choice = (
                        next(
                            (c for c in choices if c["message"]["role"] == "assistant"),
                            None,
                        )
                        if choices is not None
                        else None
                    )
                else:
                    choice = choices[0] if isinstance(choices, list) and choices else None
                message = _get(choice, "message")

                # NOTE: `n` > 1 is not supported; only the first choice's finish_reason counts.
                finish_reason = normalize_finish_reason(_get(choice, "finish_reason"))

                refusal = _get(message, "refusal")
                is_refusal = is_truthy(refusal)
                output = refusal if is_refusal else _get(message, "content")

                filter_error = _get(choice, "content_filter_results", "error")
                if is_truthy(filter_error):
                    logger.warning(
                        "Content filtering system is down or otherwise unable to complete the "
                        "request in time: %s %s",
                        _tpl(filter_error, "code"),
                        _tpl(filter_error, "message"),
                    )
                else:
                    flagged_output = finish_reason == FINISH_REASON_MAP["content_filter"]

                if output is None:
                    tool_calls = _get(message, "tool_calls")
                    function_call = _get(message, "function_call")
                    callbacks = config.get("functionToolCallbacks")
                    has_calls = is_truthy(tool_calls) or is_truthy(function_call)
                    if is_truthy(callbacks) and has_calls:
                        all_calls: list[Any] = []
                        if is_truthy(tool_calls):
                            all_calls.extend(
                                tool_calls if isinstance(tool_calls, list) else [tool_calls]
                            )
                        if is_truthy(function_call):
                            all_calls.append(function_call)
                        output = await self.function_callback_handler.process_calls(
                            all_calls[0] if len(all_calls) == 1 else all_calls, callbacks
                        )
                    else:
                        # A confirmed content-filter block is intentionally empty.
                        output = _first(tool_calls, function_call, "" if flagged_output else None)
                elif not is_refusal and _get(config.get("response_format"), "type") in (
                    "json_schema",
                    "json_object",
                ):
                    try:
                        output = js_json_parse(output) if isinstance(output, str) else output
                    except ValueError as err:
                        logger.error(
                            "Failed to parse JSON output: SyntaxError: %s. Output was: %s",
                            err,
                            output,
                        )

                content = _get(choice, "logprobs", "content")
                if isinstance(content, list):
                    log_probs = [_get(entry, "logprob") for entry in content]

            usage = data.get("usage")
            if cached:
                token_usage = _drop_none(
                    {
                        "cached": _get(usage, "total_tokens"),
                        "total": _get(usage, "total_tokens"),
                    }
                )
            else:
                token_usage = _drop_none(
                    {
                        "total": _get(usage, "total_tokens"),
                        "prompt": _get(usage, "prompt_tokens"),
                        "completion": _get(usage, "completion_tokens"),
                    }
                )
                cached_tokens = _get(usage, "prompt_tokens_details", "cached_tokens")
                if cached_tokens is not None:
                    token_usage["cached"] = cached_tokens
                details = _get(usage, "completion_tokens_details")
                if is_truthy(details):
                    token_usage["completionDetails"] = _drop_none(
                        {
                            "reasoning": details.get("reasoning_tokens"),
                            "acceptedPrediction": details.get("accepted_prediction_tokens"),
                            "rejectedPrediction": details.get("rejected_prediction_tokens"),
                        }
                    )

            response: ProviderResponse = {}
            if output is not None:
                response["output"] = output
            response["tokenUsage"] = token_usage
            response["cached"] = cached
            if latency_ms is not None:
                response["latencyMs"] = latency_ms
            if log_probs is not None:
                response["logProbs"] = log_probs
            if finish_reason is not None:
                response["finishReason"] = finish_reason
            if is_refusal:
                response["isRefusal"] = True
            cost = azure_usage_cost(
                _first(config.get("modelName"), self.deployment_name), config, usage
            )
            if cost is not None:
                response["cost"] = cost
            response["guardrails"] = {
                "flagged": flagged_input or flagged_output or is_refusal,
                "flaggedInput": flagged_input,
                "flaggedOutput": flagged_output,
            }
            return response
        except Exception as err:
            return {"error": f"API response error: {js_error_string(err)}: {json_stringify(data)}"}
