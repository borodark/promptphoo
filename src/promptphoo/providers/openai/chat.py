"""OpenAI Chat Completions provider. Ports src/providers/openai/chat.ts.

Not ported: MCP tool injection (rejected with a clear error), tracing spans, abort signals and
the streaming response-header observer.
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any, ClassVar

from promptphoo import envars, fetch
from promptphoo.finish_reason import FINISH_REASON_MAP, normalize_finish_reason
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.function_callbacks import execute_provider_function_callback
from promptphoo.providers.openai import gpt6
from promptphoo.providers.openai.billing import calculate_openai_usage_cost
from promptphoo.providers.openai.generic import OpenAiGenericProvider, strip_fine_tune_prefix
from promptphoo.providers.openai.prepare import (
    is_openai_json_schema_format,
    prepare_openai_functions,
    prepare_openai_reasoning,
    prepare_openai_response_format,
    prepare_openai_tools,
)
from promptphoo.providers.openai.util import (
    OPENAI_CHAT_MODELS,
    append_openai_api_path,
    assert_openai_api_model,
    format_openai_error,
    get_openai_chat_choice_error,
    get_openai_effective_service_tier,
    get_openai_gateway_error_type,
    get_openai_gateway_provider_code,
    get_openai_partial_output,
    get_openai_policy_refusal,
    get_token_usage,
    is_custom_openai_endpoint,
    is_openai_error_only_response,
    js_error_string,
    normalize_openai_billing_model_name,
    normalize_openai_service_tier_for_wire,
    validate_function_call,
)
from promptphoo.providers.openrouter_billing import (
    calculate_openrouter_response_cost,
    get_openrouter_billing_metadata,
    is_openrouter_endpoint,
)
from promptphoo.providers.shared import (
    get_request_timeout_ms,
    parse_chat_prompt,
    transform_tool_choice,
)
from promptphoo.templating import render_vars_in_object
from promptphoo.tracing.genai_tracer import (
    extract_provider_response_attributes,
    get_test_index,
    with_genai_span,
)

logger = logging.getLogger(__name__)

# util/fetch/errors.ts: codes that unambiguously describe the account's billing state.
DEFINITIVE_BILLING_ERROR_CODES = frozenset(
    ["credit_balance_exhausted", "billing_hard_limit_reached", "billing_not_active",
     "access_terminated"]
)  # fmt: skip


def _first(*values: Any) -> Any:
    """JS `a ?? b ?? c`."""
    return next((v for v in values if v is not None), None)


def _or(*values: Any) -> Any:
    """JS `a || b || c`."""
    for value in values[:-1]:
        if is_truthy(value):
            return value
    return values[-1]


def get_openai_gateway_rate_limit_kind(data: Any) -> str | None:
    provider_code = get_openai_gateway_provider_code(data)
    if provider_code and provider_code.lower() in DEFINITIVE_BILLING_ERROR_CODES:
        return "quota"
    return "rate_limit" if get_openai_gateway_error_type(data) == "rate_limit_exceeded" else None


def get_openai_rate_limit_response(
    error: Any, response_headers: dict[str, str] | None = None
) -> ProviderResponse | None:
    """Structured error for the transport's HttpRateLimitError, when fetch raises one."""
    error_class = getattr(fetch, "HttpRateLimitError", None)
    if error_class is None or not isinstance(error, error_class):
        return None
    formatter = getattr(fetch, "format_rate_limit_error_message", None)
    message = formatter(error) if formatter else str(error)
    return {
        "error": message,
        "metadata": {
            "rateLimitKind": getattr(error, "kind", None),
            "http": {
                "status": getattr(error, "status", None),
                "statusText": getattr(error, "status_text", None),
                "headers": getattr(error, "headers", None) or response_headers or {},
            },
        },
    }


def _get_chat_search_citations(annotations: Any, output: Any) -> list[dict[str, str]]:
    if not isinstance(annotations, list):
        return []
    citations: list[dict[str, str]] = []
    for annotation in annotations:
        annotation = annotation if isinstance(annotation, dict) else {}
        citation = (
            annotation.get("url_citation") if annotation.get("type") == "url_citation" else None
        )
        citation = citation if isinstance(citation, dict) else {}
        url = citation.get("url")
        if not isinstance(url, str) or not re.match(r"^https?://", url, re.I):
            continue
        start, end = citation.get("start_index"), citation.get("end_index")
        excerpt = (
            output[start : end + 1].strip()
            if isinstance(output, str)
            and isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            else ""
        )
        title = citation["title"].strip() if isinstance(citation.get("title"), str) else ""
        content = ": ".join(p for p in (title, excerpt) if p) or url
        citations.append({"url": url, "content": content})
    return citations


def _get_chat_search_surcharge(model_name: str) -> float:
    billing_model = normalize_openai_billing_model_name(model_name)
    if "/" in billing_model:
        return 0
    if re.match(r"^gpt-5-search-api(?:-|$)", billing_model):
        return 0.01
    if re.match(r"^gpt-4o(?:-mini)?-search-preview(?:-|$)", billing_model):
        return 0.025
    return 0


# --- OpenRouter GPT-6 reasoning reconciliation -------------------------------------------------

_MISSING = object()


def _get_openrouter_passthrough(config: dict[str, Any] | None) -> dict[str, Any]:
    passthrough = (config or {}).get("passthrough")
    return passthrough if isinstance(passthrough, dict) else {}


def _get_openrouter_reasoning_control(
    config: dict[str, Any] | None, vars: dict[str, Any] | None
) -> dict[str, Any] | None:
    config = config or {}
    passthrough = _get_openrouter_passthrough(config)
    reasoning = passthrough.get("reasoning", _MISSING)
    nested_effort = (
        None
        if reasoning is None
        else reasoning.get("effort", _MISSING)
        if isinstance(reasoning, dict)
        else _MISSING
    )
    efforts = [
        ("flat", config.get("reasoning_effort", _MISSING)),
        ("flat", passthrough.get("reasoning_effort", _MISSING)),
        ("nested", nested_effort),
    ]
    for kind, configured in efforts:
        if configured is _MISSING:
            continue
        value = render_vars_in_object(configured, vars or {})
        if value is None or value == "":
            return {"kind": "reset"}
        return {"kind": kind, "value": value}
    reasoning_dict = reasoning if isinstance(reasoning, dict) else {}
    if "max_tokens" in reasoning_dict:
        budget = render_vars_in_object(reasoning_dict["max_tokens"], vars or {})
        if budget is None:
            return {"kind": "reset"}
        return {"kind": "budget", "value": budget}
    enabled = render_vars_in_object(reasoning_dict.get("enabled"), vars or {})
    if isinstance(enabled, bool):
        return {"kind": "enabled", "value": enabled}
    return None


def _reconcile_openrouter_reasoning(
    body: dict[str, Any],
    provider_config: dict[str, Any],
    prompt_config: dict[str, Any] | None,
    vars: dict[str, Any] | None,
) -> None:
    provider_reasoning = _get_openrouter_passthrough(provider_config).get("reasoning", _MISSING)
    prompt_reasoning = _get_openrouter_passthrough(prompt_config).get("reasoning", _MISSING)
    has_reasoning = prompt_reasoning is not None and (
        provider_reasoning not in (None, _MISSING) or prompt_reasoning not in (None, _MISSING)
    )
    reasoning: dict[str, Any] | None = (
        {
            **(provider_reasoning if isinstance(provider_reasoning, dict) else {}),
            **(prompt_reasoning if isinstance(prompt_reasoning, dict) else {}),
        }
        if has_reasoning
        else None
    )
    control = _get_openrouter_reasoning_control(
        prompt_config, vars
    ) or _get_openrouter_reasoning_control(provider_config, vars)
    if not control:
        if callable(body.get("reasoning_effort")):
            del body["reasoning_effort"]
        if reasoning is not None:
            for key in ("effort", "max_tokens"):
                if callable(reasoning.get(key)):
                    del reasoning[key]
            body["reasoning"] = reasoning
        return
    kind = control["kind"]
    if kind == "reset":
        body.pop("reasoning_effort", None)
        if reasoning is not None:
            for key in ("effort", "enabled", "max_tokens"):
                reasoning.pop(key, None)
        if reasoning:
            body["reasoning"] = reasoning
        else:
            body.pop("reasoning", None)
        return
    if kind == "enabled":
        normalized = {**(reasoning or {}), "enabled": control["value"]}
        normalized.pop("effort", None)
        normalized.pop("max_tokens", None)
        body["reasoning"] = normalized
        body.pop("reasoning_effort", None)
        return
    if kind == "budget":
        normalized = {**(reasoning or {}), "max_tokens": control["value"]}
        normalized.pop("effort", None)
        if normalized.get("enabled") is False:
            del normalized["enabled"]
        body["reasoning"] = normalized
        body.pop("reasoning_effort", None)
        return

    effort = control["value"]
    if reasoning is not None:
        reasoning.pop("max_tokens", None)
        if isinstance(reasoning.get("enabled"), bool) and reasoning["enabled"] == (
            effort == "none"
        ):
            del reasoning["enabled"]
    # OpenRouter rejects conflicting aliases; retain nested options when canonicalizing.
    if reasoning is not None and (kind == "nested" or "effort" in reasoning):
        body["reasoning"] = {**reasoning, "effort": effort}
        body.pop("reasoning_effort", None)
    else:
        body["reasoning_effort"] = effort
        if reasoning:
            body["reasoning"] = reasoning
        elif reasoning is not None or prompt_reasoning is None:
            body.pop("reasoning", None)


# --- provider ----------------------------------------------------------------------------------


class OpenAiChatCompletionProvider(OpenAiGenericProvider):
    """`openai:chat:<model>`. Subclass and override `get_api_url_default()`/`default_id()` (or
    pass `config.apiBaseUrl`/`config.apiKeyEnvar`) for OpenAI-compatible endpoints."""

    OPENAI_CHAT_MODELS: ClassVar[list[dict[str, Any]]] = OPENAI_CHAT_MODELS
    OPENAI_CHAT_MODEL_NAMES: ClassVar[list[str]] = [m["id"] for m in OPENAI_CHAT_MODELS]

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        if model_name not in self.OPENAI_CHAT_MODEL_NAMES:
            logger.debug("Using unknown chat model: %s", model_name)
        super().__init__(model_name, options)
        self.loaded_function_callbacks: dict[str, Any] = {}
        mcp = self.config.get("mcp")
        if isinstance(mcp, dict) and mcp.get("enabled"):
            raise ValueError(
                "MCP tools (config.mcp) are not supported by promptphoo's OpenAI provider."
            )

    def _uses_openrouter(self) -> bool:
        system = self.get_gen_ai_system()
        return system == "openrouter" or (
            system == "openai" and is_openrouter_endpoint(self.get_api_url())
        )

    def _uses_gateway_error_format(self) -> bool:
        system = self.get_gen_ai_system()
        return system == "openrouter" or (
            system == "openai" and is_custom_openai_endpoint(self.get_api_url())
        )

    def get_audio_input_format(self) -> str | None:
        passthrough = self.config.get("passthrough")
        model = _first(
            passthrough.get("model") if isinstance(passthrough, dict) else None, self.model_name
        )
        return (
            "openai"
            if isinstance(model, str)
            and re.match(r"^gpt-(?:audio|4o(?:-mini)?-audio)(?:-|$)", model)
            else None
        )

    def validate_function_tool_call(self, output: Any, vars: dict[str, Any] | None = None) -> None:
        validate_function_call(output, self.config.get("functions"), vars)

    def is_reasoning_capability_model(self, model_name: str) -> bool:
        return OpenAiGenericProvider.is_reasoning_model(self, model_name)

    def supports_temperature_for_capability_model(self, model_name: str) -> bool:
        return not self.is_reasoning_capability_model(model_name)

    async def execute_function_callback(
        self,
        function_name: str,
        args: Any,
        config: dict[str, Any],
        call_id: str | None = None,
    ) -> str:
        return await execute_provider_function_callback(
            function_name=function_name,
            args=args,
            call_id=call_id,
            callbacks=config.get("functionToolCallbacks"),
            cache=self.loaded_function_callbacks,
        )

    async def get_openai_body(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        call_api_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Returns `{"body", "config"}`: the wire request and the prompt-merged config."""
        context = context or {}
        vars = context.get("vars") or {}
        prompt_config = (context.get("prompt") or {}).get("config")
        merged: dict[str, Any] = {**self.config, **(prompt_config or {})}
        passthrough = dict(merged["passthrough"]) if is_truthy(merged.get("passthrough")) else None
        config: dict[str, Any] = {**merged, "passthrough": passthrough}
        effective_service_tier = get_openai_effective_service_tier(self.config, prompt_config)

        messages = parse_chat_prompt(prompt, [{"role": "user", "content": prompt}])

        passthrough_model = (
            passthrough["model"]
            if passthrough is not None and isinstance(passthrough.get("model"), str)
            else None
        )
        capability_model = strip_fine_tune_prefix(
            self.normalize_capability_model_name(
                passthrough_model or self.get_capability_model_name()
            )
        )
        # Repeating the configured model must preserve subclass capabilities.
        uses_configured_capabilities = capability_model == strip_fine_tune_prefix(
            self.normalize_capability_model_name(self.get_capability_model_name())
        )
        is_gpt5 = self.is_gpt5_model(capability_model)
        is_o_series = capability_model.startswith(("o1", "o3", "o4")) or any(
            s in capability_model for s in ("/o1", "/o3", "/o4")
        )
        is_gpt6 = gpt6.is_gpt6_model(capability_model)
        is_openrouter_gpt6 = is_gpt6 and self._uses_openrouter()
        is_reasoning = (
            self.is_reasoning_model()
            if uses_configured_capabilities
            else self.is_reasoning_capability_model(capability_model)
        )
        max_completion_tokens = (
            _first(
                config.get("max_completion_tokens"),
                envars.get_env_int("OPENAI_MAX_COMPLETION_TOKENS"),
            )
            if is_reasoning
            else None
        )
        if config.get("omitDefaults"):
            max_tokens_default = (
                None
                if envars.get_env_string("OPENAI_MAX_TOKENS") is None
                else envars.get_env_int("OPENAI_MAX_TOKENS")
            )
        else:
            max_tokens_default = envars.get_env_int("OPENAI_MAX_TOKENS", 1024)
        max_tokens = (
            None
            if is_reasoning or is_gpt5
            else _first(config.get("max_tokens"), max_tokens_default)
        )

        if config.get("omitDefaults"):
            temperature_default = (
                None
                if envars.get_env_string("OPENAI_TEMPERATURE") is None
                else envars.get_env_float("OPENAI_TEMPERATURE")
            )
        else:
            temperature_default = envars.get_env_float("OPENAI_TEMPERATURE", 0)
        supports_temperature = is_gpt6 or (
            self.supports_temperature()
            if uses_configured_capabilities
            else self.supports_temperature_for_capability_model(capability_model)
        )
        temperature = (
            _first(config.get("temperature"), temperature_default) if supports_temperature else None
        )
        reasoning_effort = (
            render_vars_in_object(config.get("reasoning_effort"), vars)
            if is_reasoning and not is_gpt6
            else None
        )

        loaded_tools = (await prepare_openai_tools(config.get("tools"), vars)) or []
        functions = prepare_openai_functions(config.get("functions"), vars)
        response_format = prepare_openai_response_format(config.get("response_format"), vars)
        all_tools = list(loaded_tools)

        body: dict[str, Any] = {"model": self.model_name, "messages": messages}
        if "seed" in config:
            body["seed"] = config["seed"]
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if max_completion_tokens is not None:
            body["max_completion_tokens"] = max_completion_tokens
        if is_truthy(reasoning_effort):
            body["reasoning_effort"] = reasoning_effort
        if temperature is not None:
            body["temperature"] = temperature
        for key, envar, default in (
            ("top_p", "OPENAI_TOP_P", 1),
            ("presence_penalty", "OPENAI_PRESENCE_PENALTY", 0),
            ("frequency_penalty", "OPENAI_FREQUENCY_PENALTY", 0),
        ):
            if key in config or envars.get_env_string(envar):
                body[key] = _first(config.get(key), envars.get_env_float(envar, default))
        if is_truthy(config.get("functions")):
            body["functions"] = functions
        if is_truthy(config.get("function_call")):
            body["function_call"] = config["function_call"]
        if all_tools:
            body["tools"] = all_tools
        if is_truthy(config.get("tool_choice")):
            body["tool_choice"] = transform_tool_choice(config["tool_choice"], "openai")
        if is_truthy(config.get("tool_resources")):
            body["tool_resources"] = config["tool_resources"]
        if is_truthy(config.get("response_format")):
            body["response_format"] = response_format
        if is_truthy((call_api_options or {}).get("includeLogProbs")):
            body["logprobs"] = call_api_options["includeLogProbs"]  # type: ignore[index]
        if is_truthy(config.get("stop")):
            body["stop"] = config["stop"]
        for key in (
            "safety_identifier",
            "prompt_cache_key",
            "prompt_cache_options",
            "prompt_cache_retention",
        ):
            if key in config:
                body[key] = config[key]
        body.update(passthrough or {})
        if effective_service_tier is not None:
            body["service_tier"] = normalize_openai_service_tier_for_wire(
                effective_service_tier, self.get_api_url()
            )
        if "audio" in capability_model:
            body["modalities"] = _or(config.get("modalities"), ["text", "audio"])
            body["audio"] = _or(config.get("audio"), {"voice": "alloy", "format": "wav"})
        if (is_gpt5 or is_gpt6) and is_truthy(config.get("verbosity")):
            body["verbosity"] = config["verbosity"]
        assert_openai_api_model(body.get("model"), self.get_api_url())

        # Handle reasoning_effort and reasoning parameters for reasoning models.
        if (
            not is_gpt6
            and is_truthy(config.get("reasoning_effort"))
            and (is_reasoning or "gpt-oss" in capability_model)
        ):
            body["reasoning_effort"] = render_vars_in_object(config["reasoning_effort"], vars)
        if is_truthy(config.get("reasoning")) and is_o_series:
            body["reasoning"] = prepare_openai_reasoning(config["reasoning"], vars)

        if is_truthy(config.get("user")):
            body["user"] = config["user"]
        if is_truthy(config.get("metadata")):
            body["metadata"] = config["metadata"]
        if "store" in config:
            body["store"] = config["store"]

        # Models that reject max_tokens, even when it arrives through passthrough.
        if (is_reasoning or is_gpt5) and "max_tokens" in body:
            del body["max_tokens"]

        render = gpt6.make_render(lambda value: render_vars_in_object(value, vars))
        gpt6_effort: Any = gpt6._MISSING
        if is_gpt6 and not is_openrouter_gpt6:
            gpt6_effort = gpt6.get_gpt6_chat_reasoning_effort(self.config, prompt_config, render)
        if is_gpt6:
            output_cap = gpt6.resolve_gpt6_chat_output_cap(
                self.config,
                prompt_config,
                is_openrouter_gpt6,
                max_completion_tokens=envars.get_env_int("OPENAI_MAX_COMPLETION_TOKENS"),
                # OPENAI_MAX_TOKENS applies to GPT-6 only when reasoning is turned off.
                max_tokens=envars.get_env_int("OPENAI_MAX_TOKENS")
                if is_openrouter_gpt6 or gpt6_effort == "none"
                else None,
            )
            if output_cap is None:
                body.pop("max_completion_tokens", None)
            else:
                body["max_completion_tokens"] = output_cap
        if is_openrouter_gpt6:
            _reconcile_openrouter_reasoning(body, self.config, prompt_config, vars)
        elif is_gpt6:
            if config.get("reasoning") is not None:
                raise ValueError(
                    "GPT-6 Chat Completions requests use reasoning_effort. Configure "
                    "reasoning_effort, or use the Responses API for config.reasoning."
                )
            if gpt6.is_missing(gpt6_effort):
                body.pop("reasoning_effort", None)
            else:
                body["reasoning_effort"] = gpt6_effort
        # OpenRouter can translate Chat tools to the upstream Responses API.
        gpt6.apply_gpt6_request_rules(
            body, capability_model, "chat", is_openrouter=is_openrouter_gpt6
        )

        return {
            "body": body,
            "config": {
                **config,
                "functions": functions,
                "response_format": response_format,
                "service_tier": effective_service_tier,
            },
        }

    def get_billing_model_name(self, config: dict[str, Any]) -> str:
        passthrough = config.get("passthrough")
        model = passthrough.get("model") if isinstance(passthrough, dict) else None
        return model if isinstance(model, str) else super().get_billing_model_name(config)

    def get_billing_region(self) -> str | None:
        return None

    def calculate_response_cost(
        self, data: Any, config: dict[str, Any], cached: bool
    ) -> float | None:
        """Cost from the raw usage payload. Compatible providers override this hook when their
        API exposes authoritative billing data."""
        if self._uses_openrouter():
            return calculate_openrouter_response_cost(data, config)
        model_name = self.get_billing_model_name(config)
        data = data if isinstance(data, dict) else {}
        token_cost = calculate_openai_usage_cost(
            normalize_openai_billing_model_name(model_name),
            config,
            data.get("usage"),
            api_url=self.get_api_url(),
            cached_response=cached,
            provider=self.get_gen_ai_system(),
            region=self.get_billing_region(),
            service_tier=_first(data.get("service_tier"), config.get("service_tier")),
        )
        search_cost = 0 if cached else _get_chat_search_surcharge(model_name)
        if token_cost is None:
            return search_cost or None
        return token_cost + search_cost

    def get_provider_response_metadata(self, data: Any) -> dict[str, Any]:
        """Provider-specific fields extracted while the raw response is still available."""
        if self._uses_openrouter() and isinstance(data, dict):
            return get_openrouter_billing_metadata(data) or {}
        return {}

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        api_key = self.get_api_key()
        if self.requires_api_key() and not api_key:
            raise ValueError(self.get_missing_api_key_error_message())
        span_context = {
            "system": self.get_gen_ai_system(),
            "operationName": "chat",
            "openaiApiType": "chat_completions",
            "model": self.model_name,
            "providerId": self.id(),
            "evalId": (context or {}).get("evaluationId")
            or (((context or {}).get("test") or {}).get("metadata") or {}).get("evaluationId"),
            "testIndex": get_test_index(context),
            "promptLabel": ((context or {}).get("prompt") or {}).get("label"),
            "traceparent": (context or {}).get("traceparent"),
            "requestBody": prompt,
        }
        try:
            prepared = await self.get_openai_body(prompt, context, options)
        except Exception as error:
            failure = error

            async def rethrow(_span: Any) -> ProviderResponse:
                raise failure

            return await with_genai_span(
                span_context, rethrow, extract_provider_response_attributes
            )
        return await with_genai_span(
            {**span_context, **self.get_chat_tracing_request(prepared["body"])},
            lambda _span: self._call_api_internal(prepared, context, options, api_key),
            extract_provider_response_attributes,
        )

    def get_chat_tracing_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """getChatTracingRequest: the request parameters the span records."""

        def as_number(value: Any) -> Any:
            return value if isinstance(value, int | float) and not isinstance(value, bool) else None

        max_tokens = as_number(body.get("max_completion_tokens"))
        if max_tokens is None:
            max_tokens = as_number(body.get("max_tokens"))
        stop = body.get("stop")
        return {
            "maxTokens": max_tokens,
            "temperature": as_number(body.get("temperature")),
            "topP": as_number(body.get("top_p")),
            "stopSequences": stop
            if isinstance(stop, list)
            else [stop]
            if isinstance(stop, str)
            else None,
        }

    def _http(self, status: Any, status_text: Any, headers: Any) -> dict[str, Any]:
        return {"http": {"status": status, "statusText": status_text, "headers": headers or {}}}

    def _get_refusal_response(
        self,
        data: Any,
        config: dict[str, Any],
        cached: bool,
        status: Any,
        status_text: Any,
        headers: Any,
        latency_ms: Any,
    ) -> ProviderResponse | None:
        choices = data.get("choices") if isinstance(data, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        finish_reason = normalize_finish_reason((choice or {}).get("finish_reason"))
        if not message or (
            not is_truthy(message.get("refusal"))
            and finish_reason != FINISH_REASON_MAP["content_filter"]
        ):
            return None
        cost = self.calculate_response_cost(data, config, cached)
        response: ProviderResponse = {
            "output": _or(
                message.get("refusal"), message.get("content"), "Content filtered by provider"
            ),
            "tokenUsage": get_token_usage(data, cached),
            "cached": cached,
        }
        if latency_ms is not None:
            response["latencyMs"] = latency_ms
        if cost is not None:
            response["cost"] = cost
        response["isRefusal"] = True
        if finish_reason:
            response["finishReason"] = finish_reason
        response["guardrails"] = {"flagged": True}
        response["metadata"] = {
            **self.get_provider_response_metadata(data),
            **self._http(status, status_text, headers),
        }
        return response

    async def _call_api_internal(
        self,
        prepared: dict[str, Any],
        context: dict[str, Any] | None,
        call_api_options: dict[str, Any] | None,
        api_key: str | None,
    ) -> ProviderResponse:
        body, config = prepared["body"], prepared["config"]
        get_auth_headers = self.get_request_authentication()
        data: Any = None
        status: Any = None
        status_text: Any = None
        cached = False
        latency_ms: Any = None
        delete_from_cache = None
        response_headers: dict[str, str] | None = None

        try:
            headers = {
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {api_key}"}
                    if api_key and not get_auth_headers
                    else {}
                ),
                **self.get_openai_request_headers(config.get("headers")),
            }
            if get_auth_headers:
                # Our transport has no per-attempt auth hook; resolve the headers once.
                auth_headers = get_auth_headers()
                if inspect.isawaitable(auth_headers):
                    auth_headers = await auth_headers
                headers.update(auth_headers or {})
            result = await fetch.fetch_with_cache(
                append_openai_api_path(self.get_api_url(), "chat/completions"),
                {"method": "POST", "headers": headers, "body": json_stringify(body)},
                get_request_timeout_ms(),
                "json",
                bool(self.should_bust_cache(context)),
                self.config.get("maxRetries"),
            )
            data, cached, status, status_text = (
                result.data,
                result.cached,
                result.status,
                result.status_text,
            )
            latency_ms, delete_from_cache = result.latency_ms, result.delete_from_cache
            response_headers = result.headers

            gateway_error_format = self._uses_gateway_error_format()
            policy = get_openai_policy_refusal(data, gateway_error_format)
            if policy:
                cost = self.calculate_response_cost(data, config, cached)
                response: ProviderResponse = {
                    "output": policy["message"]
                    if "partialOutput" not in policy
                    else get_openai_partial_output(
                        policy["partialOutput"],
                        is_openai_json_schema_format(body.get("response_format")),
                    ),
                }
                if isinstance(data, dict) and data.get("usage"):
                    response["tokenUsage"] = get_token_usage(data, cached)
                response["cached"] = cached
                if latency_ms is not None:
                    response["latencyMs"] = latency_ms
                if cost is not None:
                    response["cost"] = cost
                response["isRefusal"] = True
                response["guardrails"] = {
                    "flagged": True,
                    **({"flaggedInput": True} if policy.get("flaggedInput") else {}),
                    "reason": policy["message"],
                }
                response["raw"] = data
                response["metadata"] = {
                    **self.get_provider_response_metadata(data),
                    **({"providerPolicy": {"code": policy["code"]}} if policy.get("code") else {}),
                    **self._http(status, status_text, response_headers),
                }
                return response

            choice_error = (
                get_openai_chat_choice_error(data)
                if (
                    self._uses_openrouter()
                    or (gateway_error_format and get_openai_gateway_error_type(data) is not None)
                )
                and not (isinstance(data, dict) and is_truthy(data.get("error")))
                else None
            )
            if choice_error:
                if delete_from_cache:
                    await delete_from_cache()
                cost = self.calculate_response_cost(data, config, cached)
                rate_limit_kind = get_openai_gateway_rate_limit_kind(data)
                response = {"error": f"API error: {choice_error['error']['message']}"}
                if isinstance(data, dict) and data.get("usage"):
                    response["tokenUsage"] = get_token_usage(data, cached)
                response["cached"] = cached
                if latency_ms is not None:
                    response["latencyMs"] = latency_ms
                if cost is not None:
                    response["cost"] = cost
                response["raw"] = data
                response["metadata"] = {
                    **self.get_provider_response_metadata(data),
                    **({"rateLimitKind": rate_limit_kind} if rate_limit_kind else {}),
                    **self._http(status, status_text, response_headers),
                }
                return response

            if status < 200 or status >= 300:
                error_message = (
                    f"API error: {status} {status_text}\n"
                    f"{data if isinstance(data, str) else json_stringify(data)}"
                )
                rate_limit_kind = get_openai_gateway_rate_limit_kind(data)
                # OpenRouter also uses invalid_prompt for request errors; refusals are marked.
                if (
                    isinstance(data, dict)
                    and isinstance(data.get("error"), dict)
                    and data["error"].get("code") == "invalid_prompt"
                    and not self._uses_openrouter()
                    and (not gateway_error_format or get_openai_gateway_error_type(data) is None)
                ):
                    cost = self.calculate_response_cost(data, config, cached)
                    response = {"output": error_message}
                    if data.get("usage"):
                        response["tokenUsage"] = get_token_usage(data, cached)
                    response["cached"] = cached
                    if latency_ms is not None:
                        response["latencyMs"] = latency_ms
                    if cost is not None:
                        response["cost"] = cost
                    response["isRefusal"] = True
                    # This error specifically indicates the input was rejected.
                    response["guardrails"] = {"flagged": True, "flaggedInput": True}
                    response["metadata"] = {
                        **self.get_provider_response_metadata(data),
                        **self._http(status, status_text, response_headers),
                    }
                    return response
                return {
                    "error": error_message,
                    "metadata": {
                        **({"rateLimitKind": rate_limit_kind} if rate_limit_kind else {}),
                        **self._http(status, status_text, response_headers),
                    },
                }
        except Exception as err:
            logger.error("API call error: %s", js_error_string(err))
            if delete_from_cache:
                await delete_from_cache()
            rate_limit_response = get_openai_rate_limit_response(err, response_headers)
            if rate_limit_response:
                return rate_limit_response
            return {
                "error": f"API call error: {js_error_string(err)}",
                "metadata": self._http(0, "Error", response_headers),
            }

        error_origin: str | None = None
        completed_accounting: dict[str, Any] | None = None
        try:
            refusal = self._get_refusal_response(
                data, config, cached, status, status_text, response_headers, latency_ms
            )
            if refusal:
                return refusal
            # A completed error-only envelope.
            if is_openai_error_only_response(data):
                return {
                    "error": format_openai_error(data),
                    "metadata": self._http(status, status_text, response_headers),
                }
            choice = data["choices"][0]
            message = choice["message"]
            finish_reason = normalize_finish_reason(choice.get("finish_reason"))
            cost = self.calculate_response_cost(data, config, cached)
            provider_metadata = self.get_provider_response_metadata(data)
            completed_accounting = {
                "tokenUsage": get_token_usage(data, cached),
                "cost": cost,
                "cached": cached,
                "latencyMs": latency_ms,
            }
            content_filtered = finish_reason == FINISH_REASON_MAP["content_filter"]

            reasoning = ""
            output: Any = ""
            content = message.get("content")
            function_call = message.get("function_call")
            tool_calls = message.get("tool_calls")
            if is_truthy(message.get("reasoning")):
                reasoning = message["reasoning"]
                output = content
            elif is_truthy(content) and (is_truthy(function_call) or is_truthy(tool_calls)):
                output = content if isinstance(tool_calls, list) and not tool_calls else message
            elif content is None or (content == "" and is_truthy(tool_calls)):
                output = _or(function_call, tool_calls)
            else:
                output = content
            logprobs = choice.get("logprobs")
            log_probs = (
                [entry.get("logprob") for entry in logprobs["content"]]
                if isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list)
                else None
            )

            # Structured output.
            if is_openai_json_schema_format(body.get("response_format")) and isinstance(
                output, str
            ):
                try:
                    output = js_json_parse(output)
                except ValueError as error:
                    logger.error("Failed to parse JSON output: SyntaxError: %s", error)
            show_thinking = self.config.get("showThinking")
            show_thinking = True if show_thinking is None else show_thinking
            if reasoning and isinstance(output, str) and show_thinking:
                output = f"Thinking: {reasoning}\n\n{output}"

            def finish(extra_metadata: dict[str, Any], **fields: Any) -> ProviderResponse:
                response: ProviderResponse = {
                    **fields,
                    "tokenUsage": get_token_usage(data, cached),
                    "cached": cached,
                }
                if latency_ms is not None:
                    response["latencyMs"] = latency_ms
                if log_probs is not None:
                    response["logProbs"] = log_probs
                if finish_reason:
                    response["finishReason"] = finish_reason
                if cost is not None:
                    response["cost"] = cost
                response["guardrails"] = {"flagged": content_filtered}
                response["metadata"] = {
                    **provider_metadata,
                    **self._http(status, status_text, response_headers),
                    **extra_metadata,
                }
                return response

            # Function tool callbacks.
            function_calls = [function_call] if is_truthy(function_call) else tool_calls
            callbacks = config.get("functionToolCallbacks")
            if is_truthy(function_calls) and is_truthy(callbacks):
                results: list[str] = []
                has_successful_callback = False
                for call in function_calls:
                    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                    function_name = _or(call.get("name"), fn.get("name"))
                    if isinstance(callbacks, dict) and is_truthy(callbacks.get(function_name)):
                        try:
                            result = await self.execute_function_callback(
                                function_name,
                                _or(call.get("arguments"), fn.get("arguments")),
                                config,
                                _first(call.get("call_id"), call.get("id")),
                            )
                            results.append(result)
                            has_successful_callback = True
                        except Exception as error:
                            # Fall back to returning the function call itself.
                            logger.debug(
                                "Function callback failed for %s with error %s, falling back "
                                "to original output",
                                function_name,
                                error,
                            )
                            has_successful_callback = False
                            break
                if has_successful_callback and results:
                    return finish({}, output="\n".join(results))

            # DeepSeek's reasoning_content is prepended as thinking.
            reasoning_content = message.get("reasoning_content")
            if (
                is_truthy(reasoning_content)
                and isinstance(reasoning_content, str)
                and isinstance(output, str)
                and show_thinking
            ):
                output = f"Thinking: {reasoning_content}\n\n{output}"
            audio = message.get("audio")
            if is_truthy(audio):
                body_audio = body.get("audio") if isinstance(body.get("audio"), dict) else {}
                audio_fields = {
                    "id": audio.get("id"),
                    "expiresAt": audio.get("expires_at"),
                    "data": audio.get("data"),
                    "transcript": audio.get("transcript"),
                    "format": _or(audio.get("format"), body_audio.get("format"), "wav"),
                }
                return finish(
                    {},
                    output=_or(audio.get("transcript"), ""),
                    audio={k: v for k, v in audio_fields.items() if v is not None},
                )

            annotations = message.get("annotations")
            citations = _get_chat_search_citations(annotations, output)
            extra: dict[str, Any] = {}
            # Include all choices for multi-response requests (n > 1).
            if len(data["choices"]) > 1:
                extra["choices"] = data["choices"]
            if isinstance(annotations, list) and annotations:
                extra["annotations"] = annotations
            if citations:
                extra["citations"] = citations
            return finish(extra, output=output)
        except Exception as err:
            if error_origin != "tool" and delete_from_cache:
                await delete_from_cache()
            response = {"error": f"API error: {js_error_string(err)}: {json_stringify(data)}"}
            if error_origin == "tool" and completed_accounting:
                response.update(completed_accounting)
            response["metadata"] = {
                **({"errorOrigin": error_origin} if error_origin else {}),
                **self._http(status, status_text, response_headers),
            }
            return response
