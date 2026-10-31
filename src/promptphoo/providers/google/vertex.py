"""Vertex AI chat (Gemini, Llama, PaLM 2) and embedding providers. Ports
src/providers/google/vertex.ts.

Express mode (an API key, no OAuth config) posts directly with `x-goog-api-key`. Otherwise
requests go through `GoogleAuthClient` (google-auth ADC, service-account JSON or keyFilename).
Claude-on-Vertex needs the Anthropic message helpers and is not ported yet.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from typing import Any

from promptphoo import cache as cache_module
from promptphoo import cli_state
from promptphoo.jsvalues import UNDEFINED, json_stringify
from promptphoo.providers.base import ApiProvider
from promptphoo.providers.google.ai_studio import (
    build_generation_config,
    completion_tokens_for_cost,
    prompt_base_path_of,
    split_passthrough,
)
from promptphoo.providers.google.auth import GoogleAuthManager, get_google_client, load_credentials
from promptphoo.providers.google.base import GoogleGenericProvider
from promptphoo.providers.google.models import get_vertex_api_host_for_region
from promptphoo.providers.google.util import (
    VERTEX_SHARED_REQUEST_TYPE_HEADER,
    calculate_google_cost,
    calculate_google_cost_from_usage,
    collect_grounding_metadata,
    collect_thought_signatures,
    format_candidate_contents,
    gemini_format_and_system_instructions,
    get_candidate,
    get_google_response_service_tier,
    get_vertex_service_tier_headers,
    is_non_candidate_stream_chunk,
    merge_google_completion_options,
    merge_google_request_tools,
    merge_parts,
    normalize_gemini_audio,
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

VERTEX_LLAMA_4_MODELS = frozenset(
    [
        "llama-4-maverick-17b-128e-instruct-maas",
        "llama-4-scout-17b-16e-instruct-maas",
    ]
)

SAFETY_FINISH_REASONS = ("SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII",
                         "IMAGE_SAFETY")  # fmt: skip


def _defined(**kw: Any) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


def get_required_vertex_llama_region(model_name: str) -> str:
    return "us-east5" if model_name in VERTEX_LLAMA_4_MODELS else "us-central1"


def get_vertex_llama_region_error(model_name: str, region: str) -> str | None:
    required = get_required_vertex_llama_region(model_name)
    if region == required:
        return None
    subject = (
        f"Llama model {model_name} is"
        if model_name in VERTEX_LLAMA_4_MODELS
        else "Llama models are"
    )
    return (
        f"{subject} only available in the {required} region. Current region: {region}. "
        f"Please set region: '{required}' in your configuration."
    )


def get_vertex_llama_safety_request_config(model_name: str, llama_config: Any) -> dict[str, Any]:
    safety = (llama_config or {}).get("safetySettings") or {}
    if model_name in VERTEX_LLAMA_4_MODELS:
        if safety.get("enabled") is True or "llama_guard_settings" in safety:
            return {
                "error": f"Llama model {model_name} does not support Llama Guard safety settings. "
                "Remove llamaConfig.safetySettings or set enabled: false without "
                "llama_guard_settings."
            }
        return {"bodyFields": {}, "enabled": False, "settingCount": 0}
    guard = safety.get("llama_guard_settings", UNDEFINED)
    if guard is not UNDEFINED and not isinstance(guard, dict):
        kind = "object" if guard is None or isinstance(guard, list) else type(guard).__name__
        kind = {"str": "string", "int": "number", "float": "number", "bool": "boolean"}.get(
            kind, kind
        )
        return {
            "error": f"Invalid llama_guard_settings: must be an object, received {kind}",
        }
    model_safety = {
        "enabled": safety.get("enabled") is not False,
        "llama_guard_settings": guard or {},
    }
    return {
        "bodyFields": {"extra_body": {"google": {"model_safety_settings": model_safety}}},
        "enabled": model_safety["enabled"],
        "settingCount": len(model_safety["llama_guard_settings"]),
    }


def get_vertex_api_host(
    region: str, config_api_host: str | None = None, env: dict[str, Any] | None = None
) -> str:
    return (
        config_api_host
        or (resolve_provider_env(env, ["VERTEX_API_HOST"]) or {}).get("value")
        or get_vertex_api_host_for_region(region)
    )


def get_vertex_body_cache_key(prefix: str, body: Any, api_host: str) -> str:
    serialized = body if isinstance(body, str) else json_stringify(body)
    digest = hmac.new(
        b"promptfoo:vertex:cache-key:v1",
        api_host.encode() + b"\0" + serialized.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{prefix}:{digest}"


def get_vertex_client_options(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "credentials": load_credentials(config.get("credentials")),
        "googleAuthOptions": config.get("googleAuthOptions"),
        "scopes": config.get("scopes"),
        "keyFilename": config.get("keyFilename"),
    }


async def _cache_get(key: str) -> Any:
    return await cache_module.get_cache().get(key)


async def _cache_set(key: str, response: dict[str, Any]) -> None:
    await cache_module.get_cache().set(key, json_stringify(response))


def _cached_response(raw: Any) -> dict[str, Any]:
    from promptphoo.jsonutil import js_json_parse

    return with_response_cache_metadata(js_json_parse(raw) if isinstance(raw, str) else raw, True)


class VertexChatProvider(GoogleGenericProvider):
    """Vertex AI provider for Gemini, Llama and PaLM 2 models."""

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        # Force vertex mode for Vertex AI provider.
        super().__init__(
            model_name, {**options, "config": {**(options.get("config") or {}), "vertexai": True}}
        )

    def get_api_host(self) -> str:
        return get_vertex_api_host(self.get_region(), self.config.get("apiHost"), self.env)

    def _get_api_version(self) -> str:
        return (
            self.config.get("apiVersion")
            or (resolve_provider_env(self.env, ["VERTEX_API_VERSION"]) or {}).get("value")
            or "v1"
        )

    def _get_publisher(self) -> str:
        return (
            self.config.get("publisher")
            or (resolve_provider_env(self.env, ["VERTEX_PUBLISHER"]) or {}).get("value")
            or "google"
        )

    def get_api_endpoint(self, action: str | None = None) -> str:
        suffix = f":{action}" if action else ""
        return (
            f"https://{self.get_api_host()}/{self._get_api_version()}/publishers/"
            f"{self._get_publisher()}/models/{self.model_name}{suffix}"
        )

    async def get_auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **(self.config.get("headers") or {})}
        if self.is_express_mode():
            api_key = self.get_api_key()
            if api_key:
                headers["x-goog-api-key"] = api_key
        return headers

    async def get_client_with_credentials(self) -> Any:
        client, _ = await get_google_client(get_vertex_client_options(self.config))
        return client

    def is_express_mode(self) -> bool:
        """An API key without explicitly configured OAuth credentials (OAuth takes priority)."""
        has_api_key = bool(self.get_api_key())
        disabled = self.config.get("expressMode") is False
        auth_options = self.config.get("googleAuthOptions") or {}
        has_oauth = bool(
            self.config.get("credentials")
            or self.config.get("keyFilename")
            or auth_options.get("keyFilename")
            or auth_options.get("credentials")
        )
        return has_api_key and not disabled and not has_oauth

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if "claude" in self.model_name:
            return await self.call_claude_api(prompt, context)
        if "gemini" in self.model_name:
            return await self.call_gemini_api(prompt, context, options)
        if "llama" in self.model_name:
            return await self.call_llama_api(prompt, context)
        return await self.call_palm2_api(prompt, context)

    async def call_claude_api(self, prompt: str, context: dict[str, Any] | None = None) -> dict:
        return {
            "error": f"Claude models on Vertex AI ({self.id()}) are not supported by promptphoo "
            "yet. Use the anthropic: provider instead."
        }

    async def call_gemini_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        tier_headers = get_vertex_service_tier_headers(service_tier, self.config.get("headers"))
        opaque_service_tier = (
            service_tier
            if service_tier and service_tier not in ("standard", "priority", "flex")
            else None
        )
        if tools_disabled and passthrough_tools is not UNDEFINED:
            passthrough_tools = remove_google_function_declarations(passthrough_tools)
        merged_tools = merge_google_request_tools(
            request_tools, None if passthrough_tools is UNDEFINED else passthrough_tools
        )
        model_armor = config.get("modelArmor") or {}
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": build_generation_config(self.model_name, config, vertex=True),
            **(
                {"safetySettings": normalize_safety_settings(config["safetySettings"])}
                if config.get("safetySettings")
                else {}
            ),
            **({"toolConfig": tool_config} if tool_config else {}),
            **({"tools": merged_tools} if merged_tools else {}),
            **({"systemInstruction": system_instruction} if system_instruction else {}),
            **({"serviceTier": opaque_service_tier} if opaque_service_tier else {}),
            **passthrough,
            **(
                {
                    "model_armor_config": {
                        **_defined(
                            prompt_template_name=model_armor.get("promptTemplate") or None,
                            response_template_name=model_armor.get("responseTemplate") or None,
                        )
                    }
                }
                if model_armor.get("promptTemplate") or model_armor.get("responseTemplate")
                else {}
            ),
        }
        if self.model_name in ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts") and (
            body.get("systemInstruction") or body.get("system_instruction")
        ):
            return {
                "error": f"{self.model_name} on Vertex does not support system instructions. "
                "Put delivery instructions in speechMetadata.style."
            }
        body["generationConfig"] = remove_deprecated_gemini_generation_params(
            self.model_name, body["generationConfig"]
        )
        if config.get("responseSchema"):
            if (body["generationConfig"] or {}).get("response_schema"):
                raise ValueError(
                    "`responseSchema` provided but `generationConfig.response_schema` already set."
                )
            body["generationConfig"]["response_schema"] = parse_config_response_schema(
                config["responseSchema"],
                context.get("vars"),
                self.config.get("basePath")
                if prompt_config.get("responseSchema") is None
                else prompt_base_path,
            )
            body["generationConfig"]["response_mime_type"] = "application/json"

        use_cache = cache_module.is_cache_enabled() and not should_bust_provider_cache(context)
        api_host = self.get_api_host()
        tier_header = next(
            (
                v
                for k, v in tier_headers.items()
                if k.lower() == VERTEX_SHARED_REQUEST_TYPE_HEADER.lower()
            ),
            None,
        )
        cache_body = (
            body if tier_header is None else {"requestBody": body, "serviceTierHeader": tier_header}
        )
        # Arbitrary provider headers can select a tenant or contain secrets.
        cacheable = use_cache and all(
            name.lower() == VERTEX_SHARED_REQUEST_TYPE_HEADER.lower()
            for name in (self.config.get("headers") or {})
        )
        cache_key = (
            get_vertex_body_cache_key(f"vertex:{self.model_name}", cache_body, api_host)
            if cacheable
            else None
        )

        response: Any = None
        if cache_key:
            cached = await _cache_get(cache_key)
            if cached:
                logger.debug("Returning cached Vertex Gemini response for %s", self.model_name)
                response = _cached_response(cached)

        if response is None:
            result = await self._request_gemini(config, api_host, body, tier_headers)
            if "error" in result:
                return result
            data, response_headers, requested_service_tier = (
                result["data"], result["headers"], result["requestedServiceTier"]
            )  # fmt: skip
            try:
                parsed = self._parse_gemini_response(
                    data, config, response_headers, requested_service_tier
                )
            except Exception as err:
                return {
                    "error": f"Gemini API response error: {js_error_string(err)}. "
                    f"Response data: {json_stringify(data)}"
                }
            if parsed.get("final"):
                return parsed["response"]
            response = parsed["response"]
            if cache_key:
                await _cache_set(cache_key, response)

        try:
            response["output"] = await self.execute_function_tool_callbacks(
                response.get("output"),
                {
                    **config,
                    "basePath": self.config.get("basePath")
                    if prompt_config.get("functionToolCallbacks") is None
                    else prompt_base_path,
                },
                tools_disabled,
            )
        except Exception as error:
            return {**{k: v for k, v in response.items() if k != "output"},
                    "error": js_error_string(error)}  # fmt: skip
        return response

    async def _request_gemini(
        self,
        config: dict[str, Any],
        api_host: str,
        body: dict[str, Any],
        tier_headers: dict[str, str],
    ) -> dict[str, Any]:
        from promptphoo.fetch import fetch_with_proxy

        try:
            # Default to non-streaming: Model Armor floor settings only work there.
            endpoint = (
                "streamGenerateContent" if config.get("streaming") is True else "generateContent"
            )
            request_headers = {**tier_headers, **(await self.get_auth_headers())}
            requested_service_tier = next(
                (
                    v
                    for k, v in request_headers.items()
                    if k.lower() == VERTEX_SHARED_REQUEST_TYPE_HEADER.lower()
                ),
                None,
            )
            if self.is_express_mode():
                url = (
                    f"https://{api_host}/{self._get_api_version()}/publishers/"
                    f"{self._get_publisher()}/models/{self.model_name}:{endpoint}"
                )
                res = await fetch_with_proxy(
                    url,
                    {
                        "method": "POST",
                        "headers": request_headers,
                        "body": json_stringify(body),
                        "timeout": get_request_timeout_ms() / 1000,
                    },
                )
                if not res.ok:
                    try:
                        error_data = res.json()
                    except ValueError:
                        error_data = None
                    logger.debug("Gemini API express mode error: %s", error_data)
                    detail = f": {json_stringify(error_data)}" if error_data is not None else ""
                    return {"error": f"API call error: {res.status} {res.status_text}{detail}"}
                data = res.json()
                response_headers: Any = res.headers
            else:
                client = await self.get_client_with_credentials()
                project_id = await self.get_project_id()
                url = (
                    f"https://{api_host}/{self._get_api_version()}/projects/{project_id}/locations/"
                    f"{self.get_region()}/publishers/{self._get_publisher()}/models/"
                    f"{self.model_name}:{endpoint}"
                )
                res = await client.request(
                    url=url,
                    method="POST",
                    data=body,
                    headers=request_headers,
                    timeout=get_request_timeout_ms(),
                )
                data, response_headers = res["data"], res["headers"]
        except Exception as err:
            gaxios_data = (getattr(err, "response", None) or {}).get("data")
            first = gaxios_data[0] if isinstance(gaxios_data, list) and gaxios_data else None
            if isinstance(first, dict) and first.get("error"):
                details = first["error"]
                logger.error("Gemini API error: %s", json_stringify(details))
                return {
                    "error": f"API call error: Status {details.get('status')}, Code "
                    f"{details.get('code')}, Message:\n\n{details.get('message')}"
                }
            logger.debug("Gemini API error: %s", err)
            return {"error": f"API call error: {js_error_string(err)}"}
        return {
            "data": data,
            "headers": response_headers,
            "requestedServiceTier": requested_service_tier,
        }

    def _parse_gemini_response(
        self,
        data: Any,
        config: dict[str, Any],
        response_headers: Any,
        requested_service_tier: Any,
    ) -> dict[str, Any]:
        """Returns {response, final}; `final` responses bypass the cache and callbacks."""
        normalized = data if isinstance(data, list) else [data]
        error = (
            normalized[0].get("error") if normalized and isinstance(normalized[0], dict) else None
        )
        if error:
            return {
                "final": True,
                "response": {"error": f"Error {error.get('code')}: {error.get('message')}"},
            }
        output: Any = None
        for datum in normalized:
            feedback = datum.get("promptFeedback") or {}
            usage = datum.get("usageMetadata") or {}
            if feedback.get("blockReason"):
                is_model_armor = feedback["blockReason"] == "MODEL_ARMOR"
                reason = feedback.get("blockReasonMessage") or (
                    "Content was blocked due to "
                    f"{'Model Armor' if is_model_armor else 'safety settings'}: "
                    f"{feedback['blockReason']}"
                )
                return {
                    "final": True,
                    "response": {
                        "output": reason,
                        "tokenUsage": {
                            "total": usage.get("totalTokenCount") or 0,
                            "prompt": usage.get("promptTokenCount") or 0,
                            "completion": usage.get("candidatesTokenCount") or 0,
                        },
                        "guardrails": {
                            "flagged": True,
                            "flaggedInput": True,
                            "flaggedOutput": False,
                            "reason": reason,
                        },
                        "metadata": {
                            **(
                                {
                                    "modelArmor": {
                                        "blockReason": feedback["blockReason"],
                                        **(
                                            {"blockReasonMessage": feedback["blockReasonMessage"]}
                                            if feedback.get("blockReasonMessage")
                                            else {}
                                        ),
                                    }
                                }
                                if is_model_armor
                                else {}
                            )
                        },
                    },
                }

            if isinstance(data, list) and is_non_candidate_stream_chunk(datum):
                continue

            candidate = get_candidate(datum)
            finish_reason = candidate.get("finishReason")
            if finish_reason and finish_reason in SAFETY_FINISH_REASONS:
                message = (
                    "Content was blocked due to safety settings with finish reason: "
                    f"{finish_reason}."
                )
                token_usage = {
                    "total": usage.get("totalTokenCount") or 0,
                    "prompt": usage.get("promptTokenCount") or 0,
                    "completion": usage.get("candidatesTokenCount") or 0,
                }
                guardrails = {
                    "flagged": True,
                    "flaggedInput": False,
                    "flaggedOutput": True,
                    "reason": message,
                }
                if (cli_state.config() or {}).get("redteam"):
                    # Refusals are not errors during redteams.
                    final = {"output": message, "tokenUsage": token_usage, "guardrails": guardrails}
                    return {"final": True, "response": final}
                return {"final": True, "response": {"error": message, "guardrails": guardrails}}
            if finish_reason == "MAX_TOKENS":
                # MAX_TOKENS is treated as a successful completion with the generated output.
                if (candidate.get("content") or {}).get("parts"):
                    output = merge_parts(output, format_candidate_contents(candidate))
                logger.debug("Gemini API: MAX_TOKENS reached")
            elif finish_reason and finish_reason != "STOP":
                logger.error("Gemini API error due to finish reason: %s.", finish_reason)
                return {
                    "final": True,
                    "response": {"error": f"Finish reason {finish_reason}: {json_stringify(data)}"},
                }
            elif (candidate.get("content") or {}).get("parts"):
                output = merge_parts(output, format_candidate_contents(candidate))
            elif finish_reason != "STOP":
                return {
                    "final": True,
                    "response": {"error": f"No output found in response: {json_stringify(data)}"},
                }

        if output is None or output == "":
            return {
                "final": True,
                "response": {"error": f"No output found in response: {json_stringify(data)}"},
            }

        last = normalized[-1]
        usage = last.get("usageMetadata") or {}
        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")
        thoughts = usage.get("thoughtsTokenCount")
        token_usage: dict[str, Any] = {
            "total": usage.get("totalTokenCount") or 0,
            "prompt": (prompt_tokens or 0) + (usage.get("toolUsePromptTokenCount") or 0),
            "completion": completion_tokens or 0,
        }
        if usage.get("cachedContentTokenCount") is not None:
            token_usage["cached"] = usage["cachedContentTokenCount"]
        if thoughts is not None:
            token_usage["completionDetails"] = {
                "reasoning": thoughts,
                "acceptedPrediction": 0,
                "rejectedPrediction": 0,
            }
        actual_service_tier = get_google_response_service_tier(
            response_headers, last.get("usageMetadata"), True
        )
        cost = calculate_google_cost_from_usage(
            self.model_name,
            config,
            prompt_tokens,
            completion_tokens_for_cost(usage),
            True,
            last.get("usageMetadata"),
            actual_service_tier,
            self.get_region(),
            requested_service_tier,
        )
        audio = normalize_gemini_audio(output)
        thought_signatures = collect_thought_signatures(normalized)
        response: dict[str, Any] = {
            "cached": False,
            "output": output,
            **({"audio": audio} if audio else {}),
            "tokenUsage": token_usage,
            **({"cost": cost} if cost is not None else {}),
            "metadata": {
                **({"thoughtSignatures": thought_signatures} if thought_signatures else {}),
                **({"serviceTier": actual_service_tier} if actual_service_tier else {}),
                **(
                    {"trafficType": usage["trafficType"]}
                    if isinstance(usage.get("trafficType"), str)
                    else {}
                ),
            },
        }
        grounding = collect_grounding_metadata(normalized)
        if grounding:
            response["metadata"] = {**response["metadata"], **grounding}
        return {"final": False, "response": response}

    async def call_palm2_api(self, prompt: str, context: dict[str, Any] | None = None) -> dict:
        instances = parse_chat_prompt(
            prompt, [{"messages": [{"author": "user", "content": prompt}]}]
        )
        c = self.config
        body = {
            "instances": instances,
            "parameters": _defined(
                context=c.get("context"),
                examples=c.get("examples"),
                safetySettings=normalize_safety_settings(c.get("safetySettings")),
                stopSequences=c.get("stopSequences"),
                temperature=c.get("temperature"),
                maxOutputTokens=c.get("maxOutputTokens"),
                topP=c.get("topP"),
                topK=c.get("topK"),
            ),
        }
        api_host = self.get_api_host()
        use_cache = cache_module.is_cache_enabled() and not should_bust_provider_cache(context)
        cache_key = (
            get_vertex_body_cache_key(f"vertex:palm2:{self.model_name}", body, api_host)
            if use_cache
            else None
        )
        if cache_key:
            cached = await _cache_get(cache_key)
            if cached:
                return _cached_response(cached)
        try:
            client = await self.get_client_with_credentials()
            project_id = await self.get_project_id()
            url = (
                f"https://{api_host}/{self._get_api_version()}/projects/{project_id}/locations/"
                f"{self.get_region()}/publishers/{self._get_publisher()}/models/"
                f"{self.model_name}:predict"
            )
            res = await client.request(
                url=url,
                method="POST",
                headers={"Content-Type": "application/json"},
                data=body,
                timeout=get_request_timeout_ms(),
            )
            data = res["data"]
        except Exception as err:
            return {"error": f"API call error: {json_stringify(_error_json(err))}"}
        try:
            if isinstance(data, dict) and data.get("error"):
                return {
                    "error": f"Error {data['error'].get('code')}: {data['error'].get('message')}"
                }
            predictions = data.get("predictions") if isinstance(data, dict) else None
            prediction = predictions[0] if predictions else None
            if not prediction or not prediction.get("candidates"):
                return {"error": f"No valid predictions returned from API: {json_stringify(data)}"}
            response = {"output": prediction["candidates"][0].get("content"), "cached": False}
            if cache_key:
                await _cache_set(cache_key, response)
            return response
        except Exception as err:
            return {"error": f"API response error: {js_error_string(err)}: {json_stringify(data)}"}

    async def call_llama_api(self, prompt: str, context: dict[str, Any] | None = None) -> dict:
        region = self.get_region()
        region_error = get_vertex_llama_region_error(self.model_name, region)
        if region_error:
            return {"error": region_error}
        messages = parse_chat_prompt(prompt, [{"role": "user", "content": prompt}])
        safety = get_vertex_llama_safety_request_config(
            self.model_name, self.config.get("llamaConfig")
        )
        if "error" in safety:
            return {"error": safety["error"]}
        body = {
            "model": f"meta/{self.model_name}",
            "messages": messages,
            "max_tokens": self.config.get("maxOutputTokens") or 1024,
            "stream": False,
            **_defined(
                temperature=self.config.get("temperature"),
                top_p=self.config.get("topP"),
                top_k=self.config.get("topK"),
            ),
            **safety["bodyFields"],
        }
        api_host = self.get_api_host()
        use_cache = cache_module.is_cache_enabled() and not should_bust_provider_cache(context)
        cache_key = (
            get_vertex_body_cache_key(f"vertex:llama:{self.model_name}", body, api_host)
            if use_cache
            else None
        )
        if cache_key:
            cached = await _cache_get(cache_key)
            if cached:
                return _cached_response(cached)
        try:
            client = await self.get_client_with_credentials()
            project_id = await self.get_project_id()
            url = (
                f"https://{api_host}/v1beta1/projects/{project_id}/locations/{region}/endpoints/"
                "openapi/chat/completions"
            )
            res = await client.request(
                url=url,
                method="POST",
                headers={"Content-Type": "application/json; charset=utf-8"},
                data=body,
                timeout=get_request_timeout_ms(),
            )
            data = res["data"]
        except Exception as err:
            response_data = (getattr(err, "response", None) or {}).get("data")
            if response_data:
                return {"error": f"API call error: {json_stringify(response_data)}"}
            return {"error": f"API call error: {js_error_string(err)}"}
        try:
            choices = data.get("choices") if isinstance(data, dict) else None
            output = (choices[0].get("message") or {}).get("content") if choices else ""
            if not output:
                return {"error": f"No output found in Llama API response: {json_stringify(data)}"}
            usage = data.get("usage") or {}
            response = {
                "cached": False,
                "output": output,
                "tokenUsage": {
                    "total": usage.get("total_tokens") or 0,
                    "prompt": usage.get("prompt_tokens") or 0,
                    "completion": usage.get("completion_tokens") or 0,
                    "numRequests": 1,
                },
            }
            if cache_key:
                await _cache_set(cache_key, response)
            return response
        except Exception as err:
            return {
                "error": f"Llama API response error: {js_error_string(err)}. "
                f"Response data: {json_stringify(data)}"
            }


def _error_json(err: BaseException) -> Any:
    """JSON.stringify(err) of a gaxios error: its enumerable fields (an Error stringifies {})."""
    response = getattr(err, "response", None)
    return {"response": response} if response else {}


class VertexEmbeddingProvider(ApiProvider):
    """Vertex AI text embeddings (`:predict`) and Gemini Embedding 2 (`:embedContent`)."""

    supports_embedding_cancellation = True

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.model_name = model_name

    def id(self) -> str:
        # Upstream ignores a custom id here.
        return f"vertex:{self.model_name}"

    def default_id(self) -> str:
        return self.id()

    async def get_client_with_credentials(self) -> Any:
        client, _ = await get_google_client(get_vertex_client_options(self.config))
        return client

    def _uses_embed_content(self) -> bool:
        return re.match(r"^gemini-embedding-2(?:$|-)", self.model_name) is not None

    def get_region(self) -> str:
        if self._uses_embed_content():
            return GoogleAuthManager.resolve_region(self.config, self.env, None, "global")
        return GoogleAuthManager.resolve_region(self.config, self.env)

    def get_api_version(self) -> str:
        return self.config.get("apiVersion") or "v1"

    def get_api_host(self) -> str:
        return get_vertex_api_host(self.get_region(), self.config.get("apiHost"), self.env)

    async def get_project_id(self) -> str:
        return await GoogleAuthManager.resolve_project_id(self.config, self.env)

    async def call_api(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("Vertex API does not provide text inference.")

    async def call_embedding_api(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        embed_content = self._uses_embed_content()
        if embed_content:
            for key in ("taskType", "title", "autoTruncate"):
                if self.config.get(key) is not None:
                    raise ValueError(
                        f"{self.model_name} does not support {key}. Use instructions in the "
                        "input text for embedding tasks."
                    )
        dimensions = self.config.get("outputDimensionality")
        if embed_content:
            body: dict[str, Any] = {
                "content": {"parts": [{"text": text}]},
                **(
                    {"embedContentConfig": {"outputDimensionality": dimensions}}
                    if dimensions is not None
                    else {}
                ),
            }
        else:
            auto_truncate = self.config.get("autoTruncate")
            body = {
                "instances": [
                    {
                        "content": text,
                        **_defined(
                            task_type=self.config.get("taskType"), title=self.config.get("title")
                        ),
                    }
                ],
                "parameters": {
                    "autoTruncate": False if auto_truncate is None else auto_truncate,
                    **({"outputDimensionality": dimensions} if dimensions is not None else {}),
                },
            }
        try:
            client = await self.get_client_with_credentials()
            project_id = await self.get_project_id()
            url = (
                f"https://{self.get_api_host()}/{self.get_api_version()}/projects/{project_id}/"
                f"locations/{self.get_region()}/publishers/google/models/{self.model_name}:"
                f"{'embedContent' if embed_content else 'predict'}"
            )
            res = await client.request(
                url=url, method="POST", data=body, timeout=get_request_timeout_ms()
            )
            data = res["data"] or {}
        except Exception as err:
            logger.error("Vertex API call error: %s", err)
            raise

        predictions = data.get("predictions") if isinstance(data, dict) else None
        embedding_data = (predictions[0] or {}).get("embeddings") if predictions else None
        values = (
            (data.get("embedding") or {}).get("values")
            if embed_content
            else (embedding_data or {}).get("values")
        )
        if not values:
            message = f"No valid embeddings returned from API: {json_stringify(data)}"
            logger.error(message)
            raise ValueError(message)

        usage = data.get("usageMetadata") or {}
        prompt_tokens = usage.get("promptTokenCount") if embed_content else None
        result: dict[str, Any] = {"embedding": values}
        if embed_content:
            cost = (
                None
                if prompt_tokens is None
                else calculate_google_cost(self.model_name, self.config, prompt_tokens, 0, True)
            )
            if cost is not None:
                result["cost"] = cost
        total = (
            _first(usage.get("totalTokenCount"), usage.get("promptTokenCount"))
            if embed_content
            else _first(((embedding_data or {}).get("statistics") or {}).get("token_count"), 0)
        )
        result["tokenUsage"] = {
            **({"prompt": prompt_tokens} if embed_content and prompt_tokens is not None else {}),
            **({"total": total} if total is not None else {}),
            "numRequests": 1,
        }
        return result


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


DEFAULT_VERTEX_MODEL = "gemini-3.8-flash"
DEFAULT_VERTEX_REGION = "global"
DEFAULT_VERTEX_EMBEDDING_MODEL = "gemini-embedding-001"


def get_google_vertex_embedding_provider(env: dict[str, Any] | None = None):
    return VertexEmbeddingProvider(DEFAULT_VERTEX_EMBEDDING_MODEL, {"env": env})


def get_google_vertex_providers(env: dict[str, Any] | None = None) -> dict[str, Any]:
    grading = VertexChatProvider(
        DEFAULT_VERTEX_MODEL, {"env": env, "config": {"region": DEFAULT_VERTEX_REGION}}
    )
    return {
        "embeddingProvider": get_google_vertex_embedding_provider(env),
        "gradingJsonProvider": grading,
        "gradingProvider": grading,
        "suggestionsProvider": grading,
        "synthesizeProvider": grading,
    }
