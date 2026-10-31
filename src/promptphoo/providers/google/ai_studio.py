"""Google AI Studio (Gemini API) chat and embedding providers. Ports
src/providers/google/ai.studio.ts.
"""

from __future__ import annotations

import logging
from email.utils import parsedate_to_datetime
from typing import Any

from promptphoo.jsvalues import UNDEFINED, json_stringify
from promptphoo.providers.google.base import GoogleGenericProvider
from promptphoo.providers.google.models import CHAT_MODELS
from promptphoo.providers.google.util import (
    calculate_google_cost,
    calculate_google_cost_from_usage,
    collect_grounding_metadata,
    collect_thought_signatures,
    create_auth_cache_discriminator,
    format_candidate_contents,
    gemini_format_and_system_instructions,
    get_candidate,
    get_google_api_base_url,
    get_google_guardrails,
    get_google_response_service_tier,
    get_last_prompt_safety_ratings,
    is_non_candidate_stream_chunk,
    merge_google_completion_options,
    merge_google_request_tools,
    merge_parts,
    normalize_gemini_audio,
    normalize_google_service_tier,
    normalize_safety_settings,
    parse_config_response_schema,
    remove_deprecated_gemini_generation_params,
    remove_google_function_declarations,
    resolve_google_tool_config,
)
from promptphoo.providers.shared import (
    get_request_timeout_ms,
    js_error_string,
    parse_chat_prompt,
    resolve_provider_env,
    should_bust_provider_cache,
    with_response_cache_metadata,
)

logger = logging.getLogger(__name__)

DEFAULT_API_HOST = "generativelanguage.googleapis.com"
GENERATE_CONTENT_MODEL_PREFIXES = ("gemini", "gemma", "codegemma", "paligemma")
MISSING_API_KEY_ERROR = (
    "Google API key is not set. Set the GOOGLE_API_KEY or GEMINI_API_KEY environment variable "
    "or add `apiKey` to the provider config."
)


def uses_generate_content_api(model_name: str) -> bool:
    return model_name.startswith(GENERATE_CONTENT_MODEL_PREFIXES)


def _defined(**kw: Any) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


def _date_ms(headers: Any) -> float | None:
    """Date.parse(headers.date): NaN (None) when missing or unparseable."""
    value = (headers or {}).get("date") if isinstance(headers, dict) else None
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).timestamp() * 1000
    except (TypeError, ValueError):
        return None


def prompt_base_path_of(prompt_config: dict[str, Any], config: dict[str, Any]) -> Any:
    """promptConfig?.basePath ?? this.config.basePath"""
    value = prompt_config.get("basePath")
    return config.get("basePath") if value is None else value


def js_typeof(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if callable(value):
        return "function"
    return "object"


def gemini_token_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """tokenUsage for a generateContent response (AI Studio shape)."""
    usage = usage or {}
    prompt = usage.get("promptTokenCount")
    token_usage: dict[str, Any] = {}
    if prompt is not None:
        token_usage["prompt"] = prompt + (usage.get("toolUsePromptTokenCount") or 0)
    if usage.get("candidatesTokenCount") is not None:
        token_usage["completion"] = usage["candidatesTokenCount"]
    if usage.get("totalTokenCount") is not None:
        token_usage["total"] = usage["totalTokenCount"]
    token_usage["numRequests"] = 1
    if usage.get("cachedContentTokenCount") is not None:
        token_usage["cached"] = usage["cachedContentTokenCount"]
    if usage.get("thoughtsTokenCount") is not None:
        token_usage["completionDetails"] = {
            "reasoning": usage["thoughtsTokenCount"],
            "acceptedPrediction": 0,
            "rejectedPrediction": 0,
        }
    return token_usage


def completion_tokens_for_cost(usage: dict[str, Any] | None) -> Any:
    """Thinking tokens are billed as output tokens."""
    usage = usage or {}
    if usage.get("candidatesTokenCount") is None:
        return None
    return usage["candidatesTokenCount"] + (usage.get("thoughtsTokenCount") or 0)


def build_generation_config(model_name: str, config: dict[str, Any], vertex: bool) -> Any:
    generation_config = config.get("generationConfig") or {}
    if vertex:
        base: dict[str, Any] = _defined(
            context=config.get("context"),
            examples=config.get("examples"),
            stopSequences=config.get("stopSequences"),
            temperature=config.get("temperature"),
            maxOutputTokens=config.get("maxOutputTokens"),
            topP=config.get("topP"),
            topK=config.get("topK"),
        )
    else:
        base = _defined(
            temperature=config.get("temperature"),
            topP=config.get("topP"),
            topK=config.get("topK"),
            stopSequences=config.get("stopSequences"),
            maxOutputTokens=config.get("maxOutputTokens"),
        )
    result = {**base, **generation_config}
    if "-tts" in model_name:
        result.pop("response_modalities", None)
        modalities = generation_config.get("responseModalities")
        if modalities is None and generation_config.get("response_modalities") is not None:
            modalities = [m.upper() for m in generation_config["response_modalities"]]
        result["responseModalities"] = modalities if modalities is not None else ["AUDIO"]
        result["speechConfig"] = generation_config.get("speechConfig") or {
            "voiceConfig": {"voice": "Kore"}
            if model_name.startswith("gemini-3.8-")
            else {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
        }
    return result


def split_passthrough(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Returns (service tier, passthrough tools or UNDEFINED, remaining passthrough)."""
    passthrough = dict(config.get("passthrough") or {})
    tier = passthrough.pop("service_tier", None)
    camel_tier = passthrough.pop("serviceTier", None)
    tools = passthrough.pop("tools", UNDEFINED)
    # resolve_google_tool_config already folds these in.
    passthrough.pop("toolConfig", None)
    passthrough.pop("tool_config", None)
    service_tier = normalize_google_service_tier(
        tier
        if tier is not None
        else camel_tier
        if camel_tier is not None
        else config.get("service_tier")
    )
    return service_tier, tools, passthrough


class AIStudioChatProvider(GoogleGenericProvider):
    """Google AI Studio provider for Gemini models."""

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        if model_name not in CHAT_MODELS:
            logger.debug("Using unknown Google chat model: %s", model_name)
        options = options or {}
        # Force non-vertex mode for AI Studio.
        super().__init__(
            model_name, {**options, "config": {**(options.get("config") or {}), "vertexai": False}}
        )

    def get_api_endpoint(self, action: str | None = None) -> str:
        suffix = f":{action}" if action else ""
        return (
            f"{self._get_api_base_url()}/{self._get_api_version()}/models/{self.model_name}{suffix}"
        )

    def _get_api_version(self) -> str:
        if self.config.get("apiVersion"):
            return self.config["apiVersion"]
        return "v1alpha" if self.model_name == "gemini-2.0-flash-thinking-exp" else "v1beta"

    def get_api_host(self) -> str:
        from promptphoo.templating import render_string

        api_host = (
            self.config.get("apiHost")
            or (resolve_provider_env(self.env, ["GOOGLE_API_HOST", "PALM_API_HOST"]) or {}).get(
                "value"
            )
            or DEFAULT_API_HOST
        )
        return render_string(api_host, {})

    def _get_api_base_url(self) -> str:
        return get_google_api_base_url(self.config, self.env)

    async def get_auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **(self.config.get("headers") or {})}
        api_key = self.get_api_key()
        if api_key:
            headers["x-goog-api-key"] = api_key
        return headers

    async def _fetch(self, url: str, body: dict[str, Any], context: Any) -> Any:
        from promptphoo.fetch import fetch_with_cache

        headers = await self.get_auth_headers()
        auth_discriminator = create_auth_cache_discriminator(headers)
        return await fetch_with_cache(
            url,
            {
                "method": "POST",
                "headers": headers,
                "body": json_stringify(body),
                **({"_authHash": auth_discriminator} if auth_discriminator else {}),
            },
            get_request_timeout_ms(),
            "json",
            should_bust_provider_cache(context),
        )

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.get_api_key():
            raise ValueError(MISSING_API_KEY_ERROR)
        if uses_generate_content_api(self.model_name):
            return await self.call_gemini(prompt, context)

        # Legacy PaLM API path.
        prompt_config = ((context or {}).get("prompt") or {}).get("config")
        config = merge_google_completion_options(self.config, prompt_config)
        messages = parse_chat_prompt(prompt, [{"content": prompt}])
        body = _defined(
            prompt={"messages": messages},
            temperature=config.get("temperature"),
            topP=config.get("topP"),
            topK=config.get("topK"),
            safetySettings=normalize_safety_settings(config.get("safetySettings")),
            stopSequences=config.get("stopSequences"),
            maxOutputTokens=config.get("maxOutputTokens"),
        )
        try:
            response = await self._fetch(
                f"{self._get_api_base_url()}/v1beta3/models/{self.model_name}:generateMessage",
                body,
                context,
            )
            data, cached = response.data, response.cached
        except Exception as err:
            return {"error": f"API call error: {js_error_string(err)}"}

        if not isinstance(data, dict) or not data.get("candidates"):
            return {"error": f"API did not return any candidate responses: {json_stringify(data)}"}
        try:
            output = data["candidates"][0].get("content")
            usage = data.get("usageMetadata") or {}
            cost = calculate_google_cost_from_usage(
                self.model_name,
                config,
                usage.get("promptTokenCount"),
                completion_tokens_for_cost(usage),
                False,
                data.get("usageMetadata"),
            )
            return with_response_cache_metadata(
                {
                    "output": output,
                    "tokenUsage": gemini_token_usage(usage),
                    **({"cost": cost} if cost is not None else {}),
                    "raw": data,
                },
                cached,
            )
        except Exception as err:
            return {"error": f"API response error: {js_error_string(err)}: {json_stringify(data)}"}

    async def call_gemini(self, prompt: str, context: dict[str, Any] | None = None) -> dict:
        if not self.get_api_key():
            raise ValueError(MISSING_API_KEY_ERROR)
        context = context or {}
        prompt_config = (context.get("prompt") or {}).get("config") or {}
        config = merge_google_completion_options(self.config, prompt_config)
        prompt_base_path = prompt_base_path_of(prompt_config, self.config)

        formatted = gemini_format_and_system_instructions(
            prompt,
            context.get("vars"),
            config.get("systemInstruction"),
            base_path=self.config.get("basePath")
            if prompt_config.get("systemInstruction") is None
            else prompt_base_path,
            use_assistant_role=config.get("useAssistantRole"),
        )
        contents, system_instruction = formatted["contents"], formatted["systemInstruction"]

        resolved_tools = resolve_google_tool_config(config)
        tool_config, tools_disabled = (
            resolved_tools.get("toolConfig"),
            resolved_tools["toolsDisabled"],
        )
        all_tools = await self.get_all_tools(context, skip_executable_tool_files=tools_disabled)
        request_tools = (
            remove_google_function_declarations(all_tools) if tools_disabled else all_tools
        )
        service_tier, passthrough_tools, passthrough = split_passthrough(config)
        if tools_disabled and passthrough_tools is not UNDEFINED:
            passthrough_tools = remove_google_function_declarations(passthrough_tools)
        merged_tools = merge_google_request_tools(
            request_tools, None if passthrough_tools is UNDEFINED else passthrough_tools
        )

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": build_generation_config(self.model_name, config, vertex=False),
            "safetySettings": normalize_safety_settings(config.get("safetySettings")),
            **({"toolConfig": tool_config} if tool_config else {}),
            **({"tools": merged_tools} if merged_tools else {}),
            **({"system_instruction": system_instruction} if system_instruction else {}),
            **({"service_tier": service_tier} if service_tier else {}),
            **passthrough,
        }
        if body["safetySettings"] is None:
            # undefined in Node, dropped by JSON.stringify (unless passthrough set it).
            del body["safetySettings"]
        body["generationConfig"] = remove_deprecated_gemini_generation_params(
            self.model_name, body["generationConfig"]
        )

        if config.get("responseSchema"):
            if (body["generationConfig"] or {}).get("response_schema"):
                raise ValueError(
                    "`responseSchema` provided but `generationConfig.response_schema` already set."
                )
            schema = parse_config_response_schema(
                config["responseSchema"],
                context.get("vars"),
                self.config.get("basePath")
                if prompt_config.get("responseSchema") is None
                else prompt_base_path,
            )
            body["generationConfig"]["response_schema"] = schema
            body["generationConfig"]["response_mime_type"] = "application/json"

        try:
            endpoint = self.get_api_endpoint(
                "streamGenerateContent" if config.get("streaming") is True else "generateContent"
            )
            response = await self._fetch(endpoint, body, context)
            data = response.data
            cached = response.cached
            response_headers = response.headers
            response_timestamp = _date_ms(response.headers)
        except Exception as err:
            return {"error": f"API call error: {js_error_string(err)}"}

        data_with_response = data if isinstance(data, list) else [data]
        last_data = data_with_response[-1] if data_with_response else None
        if not last_data:
            return {"error": f"No response data found in response: {json_stringify(data)}"}
        output: Any = None
        candidate: dict[str, Any] | None = None
        try:
            for datum in data_with_response:
                if isinstance(data, list) and is_non_candidate_stream_chunk(datum):
                    continue
                chunk_candidate = get_candidate(datum)
                candidate = {**(candidate or {}), **chunk_candidate}
                if not (chunk_candidate.get("content") or {}).get("parts") and chunk_candidate.get(
                    "finishReason"
                ) in ("STOP", "MAX_TOKENS"):
                    continue
                output = merge_parts(output, format_candidate_contents(chunk_candidate))
            if output is None or output == "" or candidate is None:
                raise ValueError(f"No output found in response: {json_stringify(data)}")
        except Exception as err:
            return {"error": js_error_string(err)}

        try:
            guardrails = get_google_guardrails(
                get_last_prompt_safety_ratings(data_with_response), candidate.get("safetyRatings")
            )
            grounding = collect_grounding_metadata(data_with_response)
            thought_signatures = collect_thought_signatures(data_with_response)
            usage = last_data.get("usageMetadata")
            actual_service_tier = get_google_response_service_tier(response_headers, usage)
            cost = calculate_google_cost_from_usage(
                self.model_name,
                config,
                (usage or {}).get("promptTokenCount"),
                completion_tokens_for_cost(usage),
                False,
                usage,
                actual_service_tier,
                None,
                None,
                response_timestamp,
            )
            audio = normalize_gemini_audio(output)
            result = with_response_cache_metadata(
                {
                    "output": output,
                    **({"audio": audio} if audio else {}),
                    "tokenUsage": gemini_token_usage(usage),
                    **({"cost": cost} if cost is not None else {}),
                    "raw": data,
                    **({"guardrails": guardrails} if guardrails else {}),
                    "metadata": {
                        **grounding,
                        **({"thoughtSignatures": thought_signatures} if thought_signatures else {}),
                        **({"serviceTier": actual_service_tier} if actual_service_tier else {}),
                    },
                },
                cached,
            )
            try:
                result["output"] = await self.execute_function_tool_callbacks(
                    output,
                    {
                        **config,
                        "basePath": self.config.get("basePath")
                        if prompt_config.get("functionToolCallbacks") is None
                        else prompt_base_path,
                    },
                    tools_disabled,
                )
            except Exception as error:
                return {**{k: v for k, v in result.items() if k != "output"},
                        "error": js_error_string(error)}  # fmt: skip
            return result
        except Exception as err:
            return {"error": f"API response error: {js_error_string(err)}: {json_stringify(data)}"}


class AIStudioEmbeddingProvider(AIStudioChatProvider):
    """Calls the Gemini API `:embedContent` endpoint."""

    supports_embedding_cancellation = True

    def default_id(self) -> str:
        return f"google:embedding:{self.model_name}"

    def __str__(self) -> str:
        return f"[Google AI Studio Embedding Provider {self.model_name}]"

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "error": f"Provider {self.id()} is an embedding provider; use a non-embedding google: "
            "provider for chat completions."
        }

    async def call_embedding_api(
        self,
        text: Any,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.get_api_key():
            return {"error": MISSING_API_KEY_ERROR}
        if not isinstance(text, str):
            return {
                "error": "Invalid input type for embedding API. Expected string, got "
                f"{js_typeof(text)}."
            }
        config = self.config
        body = {
            "content": {"parts": [{"text": text}]},
            **_defined(
                taskType=config.get("taskType"),
                outputDimensionality=config.get("outputDimensionality"),
                title=config.get("title"),
            ),
        }
        try:
            response = await self._fetch(self.get_api_endpoint("embedContent"), body, context)
            data, cached = response.data, response.cached
        except Exception as err:
            logger.error("Google AI Studio embedding API call error: %s", err)
            return {"error": f"API call error: {js_error_string(err)}"}

        values = (
            ((data or {}).get("embedding") or {}).get("values") if isinstance(data, dict) else None
        )
        if not values:
            return {
                "error": f"No embedding found in Google AI Studio response: {json_stringify(data)}"
            }
        prompt_tokens = (data.get("usageMetadata") or {}).get("promptTokenCount")
        cost = (
            None
            if prompt_tokens is None
            else calculate_google_cost(self.model_name, self.config, prompt_tokens, 0)
        )
        return with_response_cache_metadata(
            {
                "embedding": values,
                "tokenUsage": {
                    **({"total": prompt_tokens} if prompt_tokens is not None else {}),
                    "numRequests": 1,
                },
                **({"cost": cost} if cost is not None else {}),
            },
            cached,
        )


DEFAULT_AI_STUDIO_MODEL = "gemini-3.8-flash"


def get_google_ai_studio_providers(env: dict[str, Any] | None = None) -> dict[str, Any]:
    grading_provider = AIStudioChatProvider(DEFAULT_AI_STUDIO_MODEL, {"env": env})
    return {
        "gradingJsonProvider": AIStudioChatProvider(
            DEFAULT_AI_STUDIO_MODEL,
            {
                "env": env,
                "config": {"generationConfig": {"response_mime_type": "application/json"}},
            },
        ),
        "gradingProvider": grading_provider,
        "llmRubricProvider": AIStudioChatProvider(DEFAULT_AI_STUDIO_MODEL, {"env": env}),
        "suggestionsProvider": grading_provider,
        "synthesizeProvider": grading_provider,
    }
