"""OpenAI Responses API provider (`openai:responses:<model>`). Ports
src/providers/openai/responses.ts.

Not ported (see D23): streaming (`stream: true`) and background-mode polling are rejected with a
clear error, and abort signals are omitted. Calls run inside a GenAI tracing span (see D57). A
background request that completes synchronously, or comes back cancelled/failed, is still handled as
upstream handles it. Hosted tools (web_search, file_search, code_interpreter, remote `type: mcp`)
pass through to the API unchanged, as upstream does; the processor renders their output items.
"""

from __future__ import annotations

import inspect
import logging
import math
import re
from typing import Any, ClassVar

from promptphoo import envars, fetch
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import is_truthy, js_string, json_stringify
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.function_callbacks import FunctionCallbackHandler
from promptphoo.providers.openai import gpt6
from promptphoo.providers.openai.billing import (
    calculate_observable_openai_tool_cost,
    calculate_openai_usage_cost,
    uses_azure_openai_billing,
)
from promptphoo.providers.openai.chat import get_openai_rate_limit_response
from promptphoo.providers.openai.generic import OpenAiGenericProvider, strip_fine_tune_prefix
from promptphoo.providers.openai.prepare import (
    prepare_openai_reasoning,
    prepare_openai_response_format,
    prepare_openai_tools,
)
from promptphoo.providers.openai.tables import OPENAI_RESPONSES_MODEL_NAMES
from promptphoo.providers.openai.urls import url_hostname
from promptphoo.providers.openai.util import (
    OPENAI_BILLING_MODELS,
    RETIRED_OPENAI_MODEL_IDS,
    append_openai_api_path,
    assert_openai_api_model,
    flatten_response_tool,
    format_openai_error,
    get_openai_effective_service_tier,
    get_openai_gateway_error_type,
    get_openai_partial_output,
    get_openai_policy_refusal,
    is_azure_openai_endpoint,
    is_custom_openai_endpoint,
    js_error_string,
    normalize_openai_billing_model_name,
    normalize_openai_service_tier_for_wire,
)
from promptphoo.providers.openrouter_billing import (
    calculate_openrouter_response_cost,
    get_openrouter_billing_metadata,
    is_openrouter_endpoint,
)
from promptphoo.providers.responses import (
    ResponsesProcessor,
    get_responses_output_text,
    get_responses_token_usage,
    normalize_responses_input,
)
from promptphoo.providers.shared import LONG_RUNNING_MODEL_TIMEOUT_MS, get_request_timeout_ms
from promptphoo.templating import render_vars_in_object
from promptphoo.tracing.genai_tracer import (
    build_chat_span_context,
    extract_provider_response_attributes,
    with_genai_span,
)

logger = logging.getLogger(__name__)

_GPT5_PRO = re.compile(r"(^|/)gpt-5(?:\.\d+)?-pro(?:-|$)")
_BEDROCK_MANTLE = re.compile(r"^bedrock-mantle\.[a-z0-9-]+\.api\.aws$")
_BEDROCK_RUNTIME = re.compile(
    r"^bedrock-runtime(?:-fips)?\.[a-z0-9-]+\.(?:amazonaws\.com|api\.aws)$"
)
_BEDROCK_RUNTIME_PROFILE = re.compile(r"^(global|us)\.openai\.(.+)$")


def _first(*values: Any) -> Any:
    """JS `a ?? b ?? c`."""
    return next((v for v in values if v is not None), None)


def has_unpriced_azure_responses_tool_usage(data: Any) -> bool:
    """Azure bills hosted tool calls that the usage ledger does not price."""
    data = data if isinstance(data, dict) else {}
    output = data.get("output") if isinstance(data.get("output"), list) else []
    item_types = [item.get("type") if isinstance(item, dict) else None for item in output]
    if any(
        t in ("file_search_call", "code_interpreter_call", "image_generation_call")
        for t in item_types
    ):
        return True
    tool_usage = data.get("tool_usage")
    web_search = tool_usage.get("web_search") if isinstance(tool_usage, dict) else None
    requests = web_search.get("num_requests") if isinstance(web_search, dict) else None
    if (
        isinstance(requests, int | float)
        and not isinstance(requests, bool)
        and math.isfinite(requests)
        and requests >= 0
    ):
        return requests > 0
    return web_search is not None or "web_search_call" in item_types


def _http(status: Any, status_text: Any, headers: Any) -> dict[str, Any]:
    return {"http": {"status": status, "statusText": status_text, "headers": headers or {}}}


class OpenAiResponsesProvider(OpenAiGenericProvider):
    """`openai:responses:<model>`. Azure Responses and other compatible backends subclass this
    and override `get_api_url_default()`, `get_gen_ai_system()` (via `default_id()`),
    `get_billing_usage()` and `get_billing_region()`."""

    OPENAI_RESPONSES_MODEL_NAMES: ClassVar[list[str]] = [
        m for m in OPENAI_RESPONSES_MODEL_NAMES if m not in RETIRED_OPENAI_MODEL_IDS
    ]

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, options)
        self.function_callback_handler = FunctionCallbackHandler()
        self.processor = ResponsesProcessor(
            model_name=self.model_name,
            provider_type="openai",
            function_callback_handler=self.function_callback_handler,
            cost_calculator=lambda *_args: None,
        )

    # -- capabilities -------------------------------------------------------------------------

    def is_reasoning_model(self, model_name: str | None = None) -> bool:
        model = model_name or self.get_capability_model_name()
        return model == "codex-mini-latest" or super().is_reasoning_model(model)

    def is_reasoning_capability_model(self, model_name: str) -> bool:
        configured = strip_fine_tune_prefix(self.get_capability_model_name())
        if model_name == configured:
            return self.is_reasoning_model()
        return model_name == "codex-mini-latest" or OpenAiGenericProvider.is_reasoning_model(
            self, model_name
        )

    def supports_temperature_for_capability_model(self, model_name: str) -> bool:
        configured = strip_fine_tune_prefix(self.get_capability_model_name())
        if model_name == configured:
            return self.supports_temperature()
        return not self.is_reasoning_capability_model(model_name)

    def _get_effective_model_name(self, config: dict[str, Any]) -> str:
        passthrough = config.get("passthrough")
        model = passthrough.get("model") if isinstance(passthrough, dict) else None
        return (
            self.normalize_capability_model_name(model)
            if isinstance(model, str)
            else self.get_capability_model_name()
        )

    def get_billing_model_name(self, config: dict[str, Any]) -> str:
        return self._get_effective_model_name(config)

    def _uses_gateway_error_format(self) -> bool:
        return self.get_gen_ai_system() == "openai" and is_custom_openai_endpoint(
            self.get_api_url()
        )

    def get_billing_usage(self, data: Any, config: dict[str, Any]) -> Any:
        return data.get("usage") if isinstance(data, dict) else None

    def get_billing_region(self) -> str | None:
        return None

    def _get_bedrock_endpoint(self) -> str | None:
        hostname = url_hostname(self.get_api_url())
        if not hostname:
            return None
        if _BEDROCK_MANTLE.match(hostname):
            return "mantle"
        if _BEDROCK_RUNTIME.match(hostname):
            return "runtime"
        return None

    def _supports_persisted_gpt6_effort_updates(self) -> bool:
        return self.get_gen_ai_system() != "bedrock" and self._get_bedrock_endpoint() is None

    # -- billing --------------------------------------------------------------------------------

    def apply_billing(
        self, result: ProviderResponse, data: Any, config: dict[str, Any], cached: bool
    ) -> ProviderResponse:
        data = data if isinstance(data, dict) else {}
        if self.get_gen_ai_system() == "openai" and is_openrouter_endpoint(self.get_api_url()):
            unbilled = {k: v for k, v in result.items() if k != "cost"}
            cost = calculate_openrouter_response_cost(data, config, "responses")
            billing_metadata = get_openrouter_billing_metadata(data)
            return {
                **unbilled,
                **({} if cost is None else {"cost": cost}),
                **(
                    {"metadata": {**(result.get("metadata") or {}), **billing_metadata}}
                    if billing_metadata
                    else {}
                ),
            }
        service_tier = _first(data.get("service_tier"), config.get("service_tier"))
        unprefixed = normalize_openai_billing_model_name(self.get_billing_model_name(config))
        bedrock_endpoint = self._get_bedrock_endpoint()
        is_bedrock = self.get_gen_ai_system() == "bedrock" or bedrock_endpoint is not None
        runtime_profile = (
            _BEDROCK_RUNTIME_PROFILE.match(unprefixed) if bedrock_endpoint == "runtime" else None
        )
        if runtime_profile:
            billing_model_name = runtime_profile.group(2)
        elif is_bedrock and bedrock_endpoint != "runtime":
            billing_model_name = re.sub(r"^openai\.", "", unprefixed)
        else:
            billing_model_name = unprefixed
        regional_processing = (
            bool(runtime_profile and runtime_profile.group(1) != "global")
            if bedrock_endpoint == "runtime"
            else is_bedrock or self.model_name.startswith("openai.")
        )
        lookup_model = normalize_openai_billing_model_name(billing_model_name)
        response_cost = calculate_openai_usage_cost(
            lookup_model,
            config,
            self.get_billing_usage(data, config),
            api_url=self.get_api_url(),
            cached_response=cached,
            provider="bedrock" if is_bedrock else self.get_gen_ai_system(),
            region=self.get_billing_region(),
            regional_processing=regional_processing,
            service_tier=service_tier,
        )
        variant = gpt6.get_gpt6_variant(billing_model_name)
        if variant in ("sol", "luna", "6.1-sol") and uses_azure_openai_billing(
            config, self.get_api_url(), self.get_gen_ai_system()
        ):
            unbilled = {k: v for k, v in result.items() if k != "cost"}
            if response_cost is None or (
                not cached and has_unpriced_azure_responses_tool_usage(data)
            ):
                return unbilled
            return {**unbilled, "cost": response_cost}
        tool_cost = (
            0 if cached else calculate_observable_openai_tool_cost(data, lookup_model, config)
        )
        if response_cost is None:
            return dict(result)
        return {**result, "cost": response_cost + tool_cost}

    def _get_policy_response(
        self,
        data: Any,
        config: dict[str, Any],
        cached: bool,
        status: Any,
        status_text: Any,
        headers: dict[str, str] | None,
    ) -> ProviderResponse | None:
        policy = get_openai_policy_refusal(data, self._uses_gateway_error_format())
        if not policy:
            return None
        response = data.get("response") if isinstance(data.get("response"), dict) else data
        billing_data = {**response, "usage": _first(response.get("usage"), data.get("usage"))}
        partial_output = get_responses_output_text(response)
        response_format = config.get("response_format")
        is_json_schema = (
            isinstance(response_format, dict) and response_format.get("type") == "json_schema"
        )
        guardrails: dict[str, Any] = {"flagged": True}
        if not partial_output and policy.get("flaggedInput"):
            guardrails["flaggedInput"] = True
        guardrails["reason"] = policy["message"]
        metadata: dict[str, Any] = {}
        if response.get("id"):
            metadata["responseId"] = response["id"]
        if response.get("model"):
            metadata["model"] = response["model"]
        if policy.get("code"):
            metadata["providerPolicy"] = {"code": policy["code"]}
        metadata.update(_http(status, status_text, headers))
        result: ProviderResponse = {
            "output": policy["message"]
            if partial_output is None
            else get_openai_partial_output(partial_output, is_json_schema),
            **(
                {"tokenUsage": get_responses_token_usage(billing_data, cached)}
                if is_truthy(billing_data.get("usage"))
                else {}
            ),
            "cached": cached,
            "isRefusal": True,
            "guardrails": guardrails,
            "raw": response,
            "metadata": metadata,
        }
        return self.apply_billing(result, billing_data, config, cached)

    # -- request --------------------------------------------------------------------------------

    def _get_deployment_capabilities(self, config: dict[str, Any]) -> dict[str, bool]:
        effective_model = self._get_effective_model_name(config)
        capability_model = strip_fine_tune_prefix(effective_model)
        is_effective_gpt5 = capability_model.startswith("gpt-5") or "/gpt-5" in capability_model
        is_effective_reasoning = self.is_reasoning_capability_model(capability_model)
        supports_temperature = self.supports_temperature_for_capability_model(capability_model)
        # Azure model ids can be opaque deployment names. Preserve explicit capability hints
        # for those overrides; known OpenAI models supply their own capabilities instead.
        passthrough = config.get("passthrough")
        passthrough_model = passthrough.get("model") if isinstance(passthrough, dict) else None
        normalized = normalize_openai_billing_model_name(effective_model)
        lookup = normalized.split(":")[1] if normalized.startswith("ft:") else normalized
        is_known_override = isinstance(passthrough_model, str) and any(
            m["id"] == lookup for m in OPENAI_BILLING_MODELS
        )
        has_azure_host = not is_known_override and any(
            is_azure_openai_endpoint(endpoint)
            for endpoint in (config.get("apiHost"), config.get("apiBaseUrl"), self.get_api_url())
        )
        is_gpt6 = gpt6.is_gpt6_model(capability_model)
        azure_reasoning = has_azure_host and ("reasoning" in config or "reasoning_effort" in config)
        azure_verbosity = has_azure_host and "verbosity" in config
        # Only reasoning config promotes a custom deployment to a reasoning model; verbosity
        # alone must not change the max_output_tokens defaults.
        return {
            "isGPT6Model": is_gpt6,
            "isAzureResponsesDeploymentWithReasoningConfig": azure_reasoning,
            "isReasoningModel": is_gpt6 or is_effective_reasoning or azure_reasoning,
            "supportsVerbosity": is_gpt6 or is_effective_gpt5 or azure_verbosity,
            "supportsTemperature": is_gpt6 or supports_temperature,
        }

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

        # Chat-format content parts are translated so chat-authored multimodal prompts work.
        input: Any = prompt
        try:
            parsed = js_json_parse(prompt)
            if isinstance(parsed, list):
                input = normalize_responses_input(parsed)
        except (ValueError, TypeError):
            pass

        caps = self._get_deployment_capabilities(config)
        is_gpt6, is_reasoning = caps["isGPT6Model"], caps["isReasoningModel"]
        if config.get("omitDefaults"):
            max_output_default = (
                None
                if envars.get_env_string("OPENAI_MAX_TOKENS") is None
                else envars.get_env_int("OPENAI_MAX_TOKENS")
            )
        else:
            max_output_default = envars.get_env_int("OPENAI_MAX_TOKENS", 1024)
        reasoning_max_output_default = _first(
            envars.get_env_int("OPENAI_MAX_COMPLETION_TOKENS"),
            envars.get_env_int("OPENAI_MAX_TOKENS"),
        )
        max_output_tokens = _first(
            config.get("max_output_tokens"),
            reasoning_max_output_default if is_reasoning else max_output_default,
        )

        render = gpt6.make_render(lambda value: render_vars_in_object(value, vars))
        gpt6_reasoning = (
            gpt6.get_gpt6_responses_reasoning(self.config, prompt_config, render)
            if is_gpt6
            else None
        )
        rendered_reasoning = (
            gpt6_reasoning if is_gpt6 else prepare_openai_reasoning(config.get("reasoning"), vars)
        )
        rendered_reasoning_effort = (
            render_vars_in_object(config.get("reasoning_effort"), vars)
            if is_reasoning and not is_gpt6
            else None
        )
        effective_effort = _first(
            (rendered_reasoning or {}).get("effort"), rendered_reasoning_effort
        )
        has_azure_reasoning_effort = (
            caps["isAzureResponsesDeploymentWithReasoningConfig"]
            and effective_effort is not None
            and effective_effort != "none"
        )
        if config.get("omitDefaults") or is_gpt6:
            temperature_default = (
                None
                if envars.get_env_string("OPENAI_TEMPERATURE") is None
                else envars.get_env_float("OPENAI_TEMPERATURE")
            )
        else:
            temperature_default = envars.get_env_float("OPENAI_TEMPERATURE", 0)
        temperature = (
            _first(config.get("temperature"), temperature_default)
            if caps["supportsTemperature"] and (is_gpt6 or not has_azure_reasoning_effort)
            else None
        )
        reasoning_effort = effective_effort if is_reasoning else None
        instructions = config.get("instructions")

        response_format = prepare_openai_response_format(config.get("response_format"), vars)
        text_format: dict[str, Any] = {"format": {"type": "text"}}
        if isinstance(response_format, dict):
            if response_format.get("type") == "json_object":
                # json_object requires the word "json" in the input prompt.
                text_format = {"format": {"type": "json_object"}}
            elif response_format.get("type") == "json_schema":
                json_schema = response_format.get("json_schema")
                json_schema = json_schema if isinstance(json_schema, dict) else {}
                schema = response_format.get("schema")
                if not is_truthy(schema):
                    schema = json_schema.get("schema")
                schema_name = (
                    json_schema.get("name") or response_format.get("name") or "response_schema"
                )
                text_format = {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        **({} if schema is None else {"schema": schema}),
                        "strict": True,
                    }
                }
        if caps["supportsVerbosity"] and is_truthy(config.get("verbosity")):
            text_format = {**text_format, "verbosity": config["verbosity"]}

        loaded_tools = await prepare_openai_tools(config.get("tools"), vars)
        responses_tools = (
            [flatten_response_tool(t) for t in loaded_tools]
            if isinstance(loaded_tools, list)
            else loaded_tools
        )
        tool_choice = flatten_response_tool(config.get("tool_choice"))

        body: dict[str, Any] = {"model": self.model_name, "input": input}
        if "access_programs" in config:
            body["access_programs"] = config["access_programs"]
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        if is_truthy(reasoning_effort):
            body["reasoning"] = {"effort": reasoning_effort}
        if temperature is not None:
            body["temperature"] = temperature
        if is_truthy(instructions):
            body["instructions"] = instructions
        if (is_gpt6 or not is_truthy(reasoning_effort) or reasoning_effort == "none") and (
            "top_p" in config or envars.get_env_string("OPENAI_TOP_P")
        ):
            body["top_p"] = _first(config.get("top_p"), envars.get_env_float("OPENAI_TOP_P", 1))
        if is_truthy(responses_tools):
            body["tools"] = responses_tools
        if is_truthy(tool_choice):
            body["tool_choice"] = tool_choice
        if is_truthy(config.get("max_tool_calls")):
            body["max_tool_calls"] = config["max_tool_calls"]
        if "include" in config:
            body["include"] = config["include"]
        if is_truthy(config.get("previous_response_id")):
            body["previous_response_id"] = config["previous_response_id"]
        body["text"] = text_format
        for key in ("truncation", "metadata"):
            if is_truthy(config.get(key)):
                body[key] = config[key]
        if "parallel_tool_calls" in config:
            body["parallel_tool_calls"] = is_truthy(config["parallel_tool_calls"])
        if is_truthy(config.get("stream")):
            body["stream"] = config["stream"]
        if "store" in config:
            body["store"] = is_truthy(config["store"])
        for key in ("background", "webhook_url", "user"):
            if is_truthy(config.get(key)):
                body[key] = config[key]
        if "safety_identifier" in config:
            body["safety_identifier"] = config["safety_identifier"]
        if is_truthy(config.get("service_tier")):
            body["service_tier"] = config["service_tier"]
        for key in ("prompt_cache_key", "prompt_cache_options", "prompt_cache_retention"):
            if key in config:
                body[key] = config[key]
        body.update(passthrough or {})
        if effective_service_tier is not None:
            body["service_tier"] = normalize_openai_service_tier_for_wire(
                effective_service_tier, self.get_api_url()
            )
        elif not (
            passthrough and "service_tier" in passthrough and passthrough["service_tier"] is None
        ):
            # A cleared tier must not leave a stale string on the wire.
            body.pop("service_tier", None)
        assert_openai_api_model(body.get("model"), self.get_api_url())

        # reasoning_effort is deprecated in favour of reasoning.effort; merge so that extra
        # config.reasoning fields (e.g. summary) don't drop the effort.
        if is_gpt6:
            if gpt6_reasoning:
                body["reasoning"] = gpt6_reasoning
            else:
                body.pop("reasoning", None)
        elif isinstance(rendered_reasoning, dict) and is_reasoning:
            body["reasoning"] = {**(body.get("reasoning") or {}), **rendered_reasoning}

        # The Responses API uses max_output_tokens; strip a max_tokens leaked via passthrough.
        body.pop("max_tokens", None)

        gpt6.apply_gpt6_request_rules(
            body,
            _first((passthrough or {}).get("model"), self.get_capability_model_name()),
            "responses",
            default_responses_temperature=None if config.get("omitDefaults") else 0,
            supports_persisted_effort_updates=self._supports_persisted_gpt6_effort_updates(),
        )

        return {
            "body": body,
            "config": {
                **config,
                "service_tier": effective_service_tier,
                # The effective tools, for downstream validation.
                "tools": body["tools"] if isinstance(body.get("tools"), list) else loaded_tools,
                "response_format": response_format,
            },
        }

    # -- call -----------------------------------------------------------------------------------

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        api_key = self.get_api_key()
        if self.requires_api_key() and not api_key:
            raise ValueError(self.get_missing_api_key_error_message())
        prepared = await self.get_openai_body(prompt, context, options)
        if is_truthy(prepared["body"].get("stream")):
            raise ValueError(
                "Streaming Responses API requests (stream: true) are not supported by "
                "promptphoo. Remove `stream` from the provider config."
            )
        effective_body = prepared["body"]

        def as_number(value: Any) -> Any:
            return value if isinstance(value, int | float) and not isinstance(value, bool) else None

        stop = effective_body.get("stop")
        span_context = build_chat_span_context(
            system=self.get_gen_ai_system(),
            model=js_string(effective_body.get("model")),
            provider_id=self.id(),
            prompt=prompt,
            context=context,
            request={
                "maxTokens": as_number(effective_body.get("max_output_tokens")),
                "temperature": as_number(effective_body.get("temperature")),
                "topP": as_number(effective_body.get("top_p")),
                "stopSequences": stop if isinstance(stop, list) else None,
            },
        )
        return await with_genai_span(
            {**span_context, "openaiApiType": "responses"},
            lambda _span: self._call_api_internal(context, prepared, api_key),
            extract_provider_response_attributes,
        )

    def _validate_deep_research(self, model_name: str, config: dict[str, Any]) -> str | None:
        raw_tools = config.get("tools")
        tools = [t for t in raw_tools if isinstance(t, dict)] if isinstance(raw_tools, list) else []
        has_data_source = any(
            t.get("type") in ("web_search", "web_search_preview", "mcp")
            or (
                t.get("type") == "file_search"
                and isinstance(t.get("vector_store_ids"), list)
                and len(t["vector_store_ids"]) > 0
            )
            for t in tools
        )
        if not has_data_source:
            return (
                f"Deep research model {model_name} requires at least one data source. Configure "
                "web_search, web_search_preview, file_search with vector_store_ids, or an MCP "
                "tool."
            )
        for tool in tools:
            if tool.get("type") == "mcp" and tool.get("require_approval") != "never":
                return (
                    f"Deep research model {model_name} requires MCP tools to have "
                    "require_approval: 'never'. Update your MCP tool configuration:\ntools:\n"
                    "  - type: mcp\n    require_approval: never"
                )
        return None

    async def _call_api_internal(
        self, context: dict[str, Any] | None, prepared: dict[str, Any], api_key: str | None
    ) -> ProviderResponse:
        body, config = prepared["body"], prepared["config"]
        effective_model = self._get_effective_model_name(config)

        is_deep_research = "deep-research" in effective_model
        if is_deep_research:
            error = self._validate_deep_research(effective_model, config)
            if error:
                return {"error": error}

        timeout = get_request_timeout_ms()
        if is_deep_research or _GPT5_PRO.search(effective_model) or body.get("background") is True:
            eval_timeout = envars.get_env_int("PROMPTFOO_EVAL_TIMEOUT_MS", 0)
            timeout = eval_timeout if eval_timeout > 0 else LONG_RUNNING_MODEL_TIMEOUT_MS
            logger.debug(
                "Using timeout of %sms for long-running model %s", timeout, effective_model
            )

        data: Any = None
        status: Any = None
        status_text: Any = None
        cached = False
        delete_from_cache = None
        response_headers: dict[str, str] | None = None
        try:
            custom_headers = self.get_openai_request_headers(config.get("headers"))
            lower = {k.lower() for k in custom_headers}
            get_auth_headers = self.get_request_authentication()
            key = None if get_auth_headers else api_key
            headers: dict[str, str] = {
                **({} if "content-type" in lower else {"Content-Type": "application/json"}),
                **(
                    {"Authorization": f"Bearer {key}"}
                    if key and "authorization" not in lower
                    else {}
                ),
                **custom_headers,
            }
            if get_auth_headers:
                # Our transport has no per-attempt auth hook; resolve the headers once.
                auth_headers = get_auth_headers()
                if inspect.isawaitable(auth_headers):
                    auth_headers = await auth_headers
                headers.update(auth_headers or {})
            result = await fetch.fetch_with_cache(
                append_openai_api_path(self.get_api_url(), "responses"),
                {"method": "POST", "headers": headers, "body": json_stringify(body)},
                timeout,
                "json",
                bool(self.should_bust_cache(context)),
                config.get("maxRetries"),
            )
            data, cached, status, status_text = (
                result.data,
                result.cached,
                result.status,
                result.status_text,
            )
            delete_from_cache, response_headers = result.delete_from_cache, result.headers

            policy_response = self._get_policy_response(
                data if isinstance(data, dict) else {},
                config,
                cached,
                status,
                status_text,
                response_headers,
            )
            if policy_response:
                return policy_response
            if status < 200 or status >= 300:
                error_message = (
                    f"API error: {status} {status_text}\n"
                    f"{data if isinstance(data, str) else json_stringify(data)}"
                )
                # OpenRouter reuses invalid_prompt for malformed requests; refusals are marked.
                error = data.get("error") if isinstance(data, dict) else None
                if (
                    isinstance(error, dict)
                    and error.get("code") == "invalid_prompt"
                    and not is_openrouter_endpoint(self.get_api_url())
                    and (
                        not self._uses_gateway_error_format()
                        or get_openai_gateway_error_type(data) is None
                    )
                ):
                    response: ProviderResponse = {"output": error_message}
                    if isinstance(data, dict) and is_truthy(data.get("usage")):
                        response["tokenUsage"] = get_responses_token_usage(data, cached)
                    response["isRefusal"] = True
                    response["metadata"] = _http(status, status_text, response_headers)
                    return response
                return {
                    "error": error_message,
                    "metadata": _http(status, status_text, response_headers),
                }

            if body.get("background") and isinstance(data, dict):
                background_error = None
                if data.get("status") in ("queued", "in_progress") and not data.get("id"):
                    background_error = "Background response is missing its response ID."
                elif data.get("status") in ("queued", "in_progress"):
                    background_error = (
                        f"Background response {data.get('id')} is {data['status']}; polling "
                        "background responses is not supported by promptphoo. Remove "
                        "`background` from the provider config."
                    )
                elif data.get("status") in ("cancelled", "failed"):
                    upstream = (data.get("error") or {}).get("message")
                    background_error = (
                        f"Background response {data.get('id')} {data['status']}: {upstream}"
                        if upstream
                        else f"Background response {data.get('id')} was {data['status']}."
                    )
                if background_error:
                    if delete_from_cache:
                        await delete_from_cache()
                    return {
                        "error": background_error,
                        "metadata": _http(status, status_text, response_headers),
                    }
        except Exception as err:
            rate_limit_response = get_openai_rate_limit_response(err)
            if rate_limit_response:
                return rate_limit_response
            logger.error("API call error: %s", js_error_string(err))
            if delete_from_cache:
                await delete_from_cache()
            return {
                "error": f"API call error: {js_error_string(err)}",
                "metadata": _http(0, "Error", response_headers),
            }

        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict) and is_truthy(error.get("message")):
            if delete_from_cache:
                await delete_from_cache()
            return {
                "error": format_openai_error(data),
                "metadata": _http(status, status_text, response_headers),
            }

        processed = await self.processor.process_response_output(
            data if isinstance(data, dict) else {}, config, cached
        )
        billed = self.apply_billing(processed, data, config, cached)
        return {
            **billed,
            "metadata": {
                **(billed.get("metadata") or {}),
                **_http(status, status_text, response_headers),
            },
        }
