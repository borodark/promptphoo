"""Azure OpenAI Responses API provider (`azure:responses:<deployment>`). Ports
src/providers/azure/responses.ts.

As upstream, this extends the Azure base (auth, endpoint resolution) and composes the shared
ResponsesProcessor; it does not inherit from the OpenAI Responses provider. Requests go to
`<base>/openai/v1/responses?api-version=<apiVersion|preview>` with the deployment name as the
body's `model`.

Not ported (see D23): `stream: true` is rejected with the OpenAI Responses provider's error, and a
background request that comes back queued or in progress returns an error instead of polling.
Tracing spans and abort signals are omitted.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from promptphoo import envars, fetch
from promptphoo.assertions.tools import maybe_load_tools_from_external_file
from promptphoo.files import maybe_load_response_format_from_external_file
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import is_truthy, js_string, json_stringify
from promptphoo.providers.azure.chat import js_error_message, js_error_string
from promptphoo.providers.azure.generic import AzureGenericProvider
from promptphoo.providers.azure.util import calculate_azure_cost
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.function_callbacks import FunctionCallbackHandler
from promptphoo.providers.openai import gpt6
from promptphoo.providers.openai.util import flatten_response_tool
from promptphoo.providers.responses import ResponsesProcessor
from promptphoo.providers.shared import LONG_RUNNING_MODEL_TIMEOUT_MS, get_request_timeout_ms
from promptphoo.templating import render_vars_in_object

logger = logging.getLogger(__name__)

# Azure Responses API uses the v1 preview API version
AZURE_RESPONSES_API_VERSION = "preview"

_FIXED_REASONING = re.compile(r"^gpt-chat-latest(?:-|$)")


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


def azure_responses_cost(
    model_name: str, usage: Any, config: Any = None, response_data: Any = None
) -> float | None:
    """calculateAzureCost over Chat- or Responses-style usage. The tier Azure actually served is
    mapped into its passthrough-based cost contract, falling back to the requested tier."""
    config = config if isinstance(config, dict) else {}
    requested_service_tier = config.get("service_tier")
    cost_config = {k: v for k, v in config.items() if k != "service_tier"}
    service_tier = _first(_get(response_data, "service_tier"), requested_service_tier)
    passthrough = cost_config.get("passthrough")
    cost_config["passthrough"] = {
        **(passthrough if isinstance(passthrough, dict) else {}),
        **({} if service_tier is None else {"service_tier": service_tier}),
    }

    def tokens(*paths: tuple[str, ...]) -> Any:
        return _first(*(_get(usage, *path) for path in paths))

    return calculate_azure_cost(
        config["model"] if isinstance(config.get("model"), str) else model_name,
        cost_config,
        tokens(("prompt_tokens",), ("input_tokens",)),
        tokens(("completion_tokens",), ("output_tokens",)),
        tokens(
            ("prompt_tokens_details", "cached_tokens"), ("input_tokens_details", "cached_tokens")
        ),
        tokens(("prompt_tokens_details", "audio_tokens"), ("input_tokens_details", "audio_tokens")),
        tokens(
            ("completion_tokens_details", "audio_tokens"),
            ("output_tokens_details", "audio_tokens"),
        ),
        tokens(("prompt_tokens_details", "image_tokens"), ("input_tokens_details", "image_tokens")),
        tokens(
            ("prompt_tokens_details", "cached_tokens_details", "audio_tokens"),
            ("input_tokens_details", "cached_tokens_details", "audio_tokens"),
        ),
        tokens(
            ("prompt_tokens_details", "cached_tokens_details", "image_tokens"),
            ("input_tokens_details", "cached_tokens_details", "image_tokens"),
        ),
        tokens(
            ("completion_tokens_details", "image_tokens"),
            ("output_tokens_details", "image_tokens"),
        ),
    )


class AzureResponsesProvider(AzureGenericProvider):
    def __init__(self, deployment_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(deployment_name, options)
        self.function_callback_handler = FunctionCallbackHandler()
        self.processor = ResponsesProcessor(
            model_name=self.deployment_name,
            provider_type="azure",
            function_callback_handler=self.function_callback_handler,
            cost_calculator=azure_responses_cost,
        )

    def _default_capability_model(self) -> str:
        return _first(self.config.get("modelName"), self.deployment_name)

    def is_reasoning_model(self, model_name: str | None = None) -> bool:
        """Reasoning models use max_completion_tokens instead of max_tokens, don't take
        temperature, and may take a configurable reasoning effort."""
        if model_name is None:
            model_name = self._default_capability_model()
        # Explicit config flags first (as chat does).
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
            or ("grok" in lower and "reasoning" in lower)
        )

    def supports_temperature(self, model_name: str | None = None) -> bool:
        return not self.is_reasoning_model(model_name)

    async def get_azure_responses_body(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        call_api_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = context or {}
        vars = context.get("vars")
        prompt_config = (context.get("prompt") or {}).get("config")
        config: dict[str, Any] = {**self.config, **(prompt_config or {})}

        input: Any = prompt
        try:
            parsed = js_json_parse(prompt)
            if isinstance(parsed, list):
                input = parsed
        except (ValueError, TypeError):
            pass

        passthrough_model = _get(config.get("passthrough"), "model")
        capability_model_name = (
            passthrough_model
            if isinstance(passthrough_model, str)
            else js_string(_first(config.get("modelName"), self.deployment_name))
        ).lower()
        is_reasoning_model = self.is_reasoning_model(capability_model_name)
        is_fixed_reasoning_model = bool(_FIXED_REASONING.search(capability_model_name))
        is_gpt6_model = gpt6.is_gpt6_model(capability_model_name)
        if config.get("omitDefaults"):
            max_output_tokens_default = (
                None
                if envars.get_env_string("OPENAI_MAX_TOKENS") is None
                else envars.get_env_int("OPENAI_MAX_TOKENS")
            )
        else:
            max_output_tokens_default = envars.get_env_int("OPENAI_MAX_TOKENS", 1024)
        reasoning_max_output_tokens_default = _first(
            envars.get_env_int("OPENAI_MAX_COMPLETION_TOKENS"),
            envars.get_env_int("OPENAI_MAX_TOKENS"),
        )
        max_output_tokens = _first(
            config.get("max_output_tokens"),
            reasoning_max_output_tokens_default
            if is_reasoning_model
            else max_output_tokens_default,
        )

        if config.get("omitDefaults") or is_gpt6_model:
            temperature_default = (
                None
                if envars.get_env_string("OPENAI_TEMPERATURE") is None
                else envars.get_env_float("OPENAI_TEMPERATURE")
            )
        else:
            temperature_default = envars.get_env_float("OPENAI_TEMPERATURE", 0)
        temperature = (
            _first(config.get("temperature"), temperature_default)
            if is_gpt6_model or self.supports_temperature(capability_model_name)
            else None
        )
        render = gpt6.make_render(lambda value: render_vars_in_object(value, vars))
        gpt6_reasoning = (
            gpt6.get_gpt6_responses_reasoning(self.config, prompt_config, render)
            if is_gpt6_model
            else None
        )
        reasoning_effort = (
            render_vars_in_object(config.get("reasoning_effort"), vars)
            if is_reasoning_model and not is_gpt6_model and not is_fixed_reasoning_model
            else None
        )

        instructions = config.get("instructions")

        # Load response_format from an external file if needed (including a nested schema).
        try:
            response_format = maybe_load_response_format_from_external_file(
                config.get("response_format"), vars
            )
        except Exception as error:
            raw_format = config.get("response_format")
            file = f" file: {raw_format}" if isinstance(raw_format, str) else ""
            raise ValueError(
                f"Failed to load response_format{file}\n"
                f"Error: {js_error_message(error)}\n"
                "Make sure the file exists and contains valid JSON schema format."
            ) from error

        text_format: dict[str, Any] = {"format": {"type": "text"}}
        if is_truthy(response_format) and isinstance(response_format, dict):
            if response_format.get("type") == "json_object":
                text_format = {"format": {"type": "json_object"}}
            elif response_format.get("type") == "json_schema":
                # The schema is already loaded by maybe_load_response_format_from_external_file.
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

        # Verbosity for reasoning models, if configured.
        if is_reasoning_model and is_truthy(config.get("verbosity")):
            text_format = {**text_format, "verbosity": config["verbosity"]}

        loaded_tools = (
            await maybe_load_tools_from_external_file(config["tools"], vars)
            if is_truthy(config.get("tools"))
            else None
        )
        tools = (
            [flatten_response_tool(t) for t in loaded_tools]
            if isinstance(loaded_tools, list)
            else loaded_tools
        )
        raw_tool_choice = config.get("tool_choice")
        tool_choice = (
            {"type": "function", "name": raw_tool_choice["function"]["name"]}
            if isinstance(raw_tool_choice, dict)
            and raw_tool_choice.get("type") == "function"
            and "function" in raw_tool_choice
            and is_truthy(_get(raw_tool_choice, "function", "name"))
            else raw_tool_choice
        )

        # Azure Responses API uses the `model` field for the deployment name.
        body: dict[str, Any] = {"model": self.deployment_name, "input": input}
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        if is_truthy(reasoning_effort):
            body["reasoning"] = {"effort": reasoning_effort}
        if temperature is not None:
            body["temperature"] = temperature
        if is_truthy(instructions):
            body["instructions"] = instructions
        if "top_p" in config or envars.get_env_string("OPENAI_TOP_P"):
            body["top_p"] = _first(config.get("top_p"), envars.get_env_float("OPENAI_TOP_P", 1))
        if is_truthy(tools):
            body["tools"] = tools
        if is_truthy(tool_choice):
            body["tool_choice"] = tool_choice
        if is_truthy(config.get("max_tool_calls")):
            body["max_tool_calls"] = config["max_tool_calls"]
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
        if config.get("service_tier") is not None:
            body["service_tier"] = config["service_tier"]
        passthrough = config.get("passthrough")
        if is_truthy(passthrough) and isinstance(passthrough, dict):
            body.update(passthrough)

        reasoning = body.get("reasoning")
        if is_fixed_reasoning_model and isinstance(reasoning, dict):
            reasoning = {k: v for k, v in reasoning.items() if k != "effort"}
            if reasoning:
                body["reasoning"] = reasoning
            else:
                del body["reasoning"]

        if is_gpt6_model:
            if gpt6_reasoning:
                body["reasoning"] = gpt6_reasoning
            else:
                body.pop("reasoning", None)
        gpt6.apply_gpt6_request_rules(
            body,
            capability_model_name,
            "responses",
            default_responses_temperature=None if config.get("omitDefaults") else 0,
        )

        logger.debug("Azure Responses API request body: %s", json_stringify(body))
        return body

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        await self.ensure_initialized()
        if self.auth_headers is None:
            raise RuntimeError("Invariant failed: auth headers are not initialized")

        if not self.get_api_base_url():
            raise ValueError(
                "Azure API configuration missing. Set AZURE_API_HOST environment variable or "
                "configure apiHost in provider config.\n"
                "Example: AZURE_API_HOST=your-resource.openai.azure.com"
            )
        if not self.auth_headers.get("api-key") and not self.auth_headers.get("Authorization"):
            raise ValueError(
                "Azure API authentication failed. Set AZURE_API_KEY environment variable or "
                "configure apiKey in provider config.\n"
                "You can also use Microsoft Entra ID authentication."
            )

        body = await self.get_azure_responses_body(prompt, context, options)
        if is_truthy(body.get("stream")):
            # See D23: the SSE reader is not ported.
            raise ValueError(
                "Streaming Responses API requests (stream: true) are not supported by "
                "promptphoo. Remove `stream` from the provider config."
            )
        context = context or {}

        # Deep research models get the long-running timeout.
        timeout = get_request_timeout_ms()
        if "deep-research" in self.deployment_name:
            eval_timeout = envars.get_env_int("PROMPTFOO_EVAL_TIMEOUT_MS", 0)
            timeout = eval_timeout if eval_timeout > 0 else LONG_RUNNING_MODEL_TIMEOUT_MS
            logger.debug(
                "Using timeout of %sms for deep research model %s", timeout, self.deployment_name
            )

        data: Any = None
        cached = False
        try:
            # The Azure Responses URL has no deployment name in it.
            api_version = self.config.get("apiVersion") or AZURE_RESPONSES_API_VERSION
            url = f"{self.get_api_base_url()}/openai/v1/responses?api-version={api_version}"
            result = await fetch.fetch_with_cache(
                url,
                {
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        **self.auth_headers,
                        **(self.config.get("headers") or {}),
                    },
                    "body": json_stringify(body),
                },
                timeout,
                "json",
                _first(context.get("bustCache"), context.get("debug")),
                self.config.get("maxRetries"),
            )
            data, cached, status, status_text = (
                result.data,
                result.cached,
                result.status,
                result.status_text,
            )

            if status < 200 or status >= 300:
                return {
                    "error": f"API error: {status} {status_text}\n"
                    f"{data if isinstance(data, str) else json_stringify(data)}"
                }

            # See D23: background responses are not polled.
            if (
                body.get("background")
                and isinstance(data, dict)
                and data.get("status") in ("queued", "in_progress")
            ):
                if result.delete_from_cache:
                    await result.delete_from_cache()
                if not data.get("id"):
                    return {"error": "Background response is missing its response ID."}
                return {
                    "error": f"Background response {data.get('id')} is {data['status']}; "
                    "polling background responses is not supported by promptphoo. Remove "
                    "`background` from the provider config."
                }
        except Exception as err:
            logger.error("API call error: %s", js_error_string(err))
            return {"error": f"API call error: {js_error_string(err)}"}

        logger.debug("\tAzure Responses API response: %s", json_stringify(data))

        # The shared response processor does all response processing.
        processed = await self.processor.process_response_output(
            data if isinstance(data, dict) else {}, body, cached
        )
        response_usage = _get(data, "usage")
        cached_input_tokens = _first(
            _get(response_usage, "prompt_tokens_details", "cached_tokens"),
            _get(response_usage, "input_tokens_details", "cached_tokens"),
        )
        token_usage = processed.get("tokenUsage")
        if not cached and isinstance(token_usage, dict) and cached_input_tokens is not None:
            processed["tokenUsage"] = {**token_usage, "cached": cached_input_tokens}
        return processed
