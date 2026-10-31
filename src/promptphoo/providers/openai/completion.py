"""OpenAI legacy Completions provider. Ports src/providers/openai/completion.ts."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from promptphoo import envars, fetch
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.openai.billing import calculate_openai_usage_cost
from promptphoo.providers.openai.generic import OpenAiGenericProvider
from promptphoo.providers.openai.util import (
    OPENAI_COMPLETION_MODELS,
    append_openai_api_path,
    assert_openai_api_model,
    format_openai_error,
    get_openai_effective_service_tier,
    get_token_usage,
    is_openai_first_party_api_url,
    js_error_string,
    normalize_openai_billing_model_name,
)
from promptphoo.providers.shared import get_request_timeout_ms

logger = logging.getLogger(__name__)


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


class OpenAiCompletionProvider(OpenAiGenericProvider):
    OPENAI_COMPLETION_MODELS: ClassVar[list[dict[str, Any]]] = OPENAI_COMPLETION_MODELS
    OPENAI_COMPLETION_MODEL_NAMES: ClassVar[list[str]] = [m["id"] for m in OPENAI_COMPLETION_MODELS]

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, options)
        if (
            model_name not in self.OPENAI_COMPLETION_MODEL_NAMES
            and self.get_api_url() == self.get_api_url_default()
        ):
            logger.warning("FYI: Using unknown OpenAI completion model: %s", model_name)

    def get_billing_model_name(self, config: dict[str, Any]) -> str:
        passthrough = config.get("passthrough")
        model = passthrough.get("model") if isinstance(passthrough, dict) else None
        return model if isinstance(model, str) else super().get_billing_model_name(config)

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        if self.requires_api_key() and not self.get_api_key():
            raise ValueError(self.get_missing_api_key_error_message())

        openai_stop = envars.get_env_string("OPENAI_STOP")
        try:
            stop = (
                js_json_parse(openai_stop or "")
                if openai_stop
                else (self.config.get("stop") or ["<|im_end|>", "<|endoftext|>"])
            )
        except ValueError as err:
            raise ValueError(f"OPENAI_STOP is not a valid JSON string: SyntaxError: {err}") from err
        prompt_config = ((context or {}).get("prompt") or {}).get("config")
        prompt_replaces_passthrough = (
            isinstance(prompt_config, dict) and "passthrough" in prompt_config
        )
        effective_passthrough = (
            prompt_config["passthrough"]  # type: ignore[index]
            if prompt_replaces_passthrough
            else self.config.get("passthrough")
        )
        effective_service_tier = get_openai_effective_service_tier(self.config, prompt_config)
        is_first_party = is_openai_first_party_api_url(self.get_api_url())
        request_service_tier = None if is_first_party else effective_service_tier

        config = self.config
        body: dict[str, Any] = {"model": self.model_name, "prompt": prompt}
        if "seed" in config:
            body["seed"] = config["seed"]
        body.update(
            {
                "max_tokens": _first(
                    config.get("max_tokens"), envars.get_env_int("OPENAI_MAX_TOKENS", 1024)
                ),
                "temperature": _first(
                    config.get("temperature"), envars.get_env_float("OPENAI_TEMPERATURE", 0)
                ),
                "top_p": _first(config.get("top_p"), envars.get_env_float("OPENAI_TOP_P", 1)),
                "presence_penalty": _first(
                    config.get("presence_penalty"),
                    envars.get_env_float("OPENAI_PRESENCE_PENALTY", 0),
                ),
                "frequency_penalty": _first(
                    config.get("frequency_penalty"),
                    envars.get_env_float("OPENAI_FREQUENCY_PENALTY", 0),
                ),
                "best_of": _first(config.get("best_of"), envars.get_env_int("OPENAI_BEST_OF", 1)),
            }
        )
        if is_truthy((options or {}).get("includeLogProbs")):
            body["logprobs"] = options["includeLogProbs"]  # type: ignore[index]
        if is_truthy(stop):
            body["stop"] = stop
        if isinstance(effective_passthrough, dict):
            body.update(effective_passthrough)
        if request_service_tier is not None:
            body["service_tier"] = request_service_tier
        # OpenAI's legacy /v1/completions schema rejects service_tier.
        if is_first_party:
            body.pop("service_tier", None)
        assert_openai_api_model(body.get("model"), self.get_api_url())
        return await self._call_api_internal(body, context)

    async def _call_api_internal(
        self, body: dict[str, Any], context: dict[str, Any] | None
    ) -> ProviderResponse:
        context = context or {}
        cached = False
        try:
            api_key = self.get_api_key()
            result = await fetch.fetch_with_cache(
                append_openai_api_path(self.get_api_url(), "completions"),
                {
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
                        **self.get_openai_request_headers(),
                    },
                    "body": json_stringify(body),
                },
                get_request_timeout_ms(),
                "json",
                bool(_first(context.get("bustCache"), context.get("debug"))),
                self.config.get("maxRetries"),
            )
            data, cached, latency_ms = result.data, result.cached, result.latency_ms
        except Exception as err:
            logger.error("API call error: %s", js_error_string(err))
            return {"error": f"API call error: {js_error_string(err)}"}

        if isinstance(data, dict) and is_truthy(data.get("error")):
            return {"error": format_openai_error(data)}
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices or not is_truthy(choices[0]):
            return {"error": f"Malformed response data: {json_stringify(data)}", "cached": cached}
        try:
            billing_model = self.get_billing_model_name(
                {**self.config, "passthrough": {"model": body.get("model")}}
            )
            response: ProviderResponse = {}
            if "text" in choices[0]:
                response["output"] = choices[0]["text"]
            response["tokenUsage"] = get_token_usage(data, cached)
            response["cached"] = cached
            if latency_ms is not None:
                response["latencyMs"] = latency_ms
            cost = calculate_openai_usage_cost(
                normalize_openai_billing_model_name(billing_model),
                self.config,
                data.get("usage"),
                cached_response=cached,
                service_tier=_first(data.get("service_tier"), body.get("service_tier")),
                api_url=self.get_api_url(),
            )
            if cost is not None:
                response["cost"] = cost
            return response
        except Exception as err:
            return {"error": f"API error: {js_error_string(err)}: {json_stringify(data)}"}
