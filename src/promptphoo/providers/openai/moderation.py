"""OpenAI moderation provider. Ports src/providers/openai/moderation.ts."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from promptphoo import cache, fetch
from promptphoo.jsvalues import UNDEFINED, json_stringify
from promptphoo.providers.openai.generic import OpenAiGenericProvider
from promptphoo.providers.openai.util import (
    RETIRED_OPENAI_MODEL_IDS,
    append_openai_api_path,
    assert_openai_api_model,
    js_error_string,
)
from promptphoo.providers.shared import get_request_timeout_ms

logger = logging.getLogger(__name__)

_OPENAI_MODERATION_AND_RETIRED_MODELS = [
    {"id": "omni-moderation-latest", "maxTokens": 32768, "capabilities": ["text", "image"]},
    {"id": "omni-moderation-2024-09-26", "maxTokens": 32768, "capabilities": ["text", "image"]},
]
OPENAI_MODERATION_MODELS = [
    m for m in _OPENAI_MODERATION_AND_RETIRED_MODELS if m["id"] not in RETIRED_OPENAI_MODEL_IDS
]

OPENAI_MODERATION_CACHE_HASH_KEY = "promptfoo:openai:moderation-cache-key:v1"
_INFLIGHT_REQUESTS: dict[str, asyncio.Future[Any]] = {}


def _hmac_hex(key: str, message: str) -> str:
    return hmac.new(key.encode(), message.encode(), hashlib.sha256).hexdigest()


def hash_moderation_cache_value(value: Any) -> str:
    serialized = value if isinstance(value, str) else json_stringify(value)
    return _hmac_hex(OPENAI_MODERATION_CACHE_HASH_KEY, serialized)


def get_openai_moderation_auth_cache_namespace(api_key: str) -> str:
    return _hmac_hex(api_key, OPENAI_MODERATION_CACHE_HASH_KEY)


async def _fetch_with_dedupe(key: str, fetcher: Callable[[], Awaitable[Any]]) -> Any:
    inflight = _INFLIGHT_REQUESTS.get(key)
    if inflight is None:
        inflight = asyncio.ensure_future(fetcher())
        _INFLIGHT_REQUESTS[key] = inflight
        inflight.add_done_callback(lambda _: _INFLIGHT_REQUESTS.pop(key, None))
    return await asyncio.shield(inflight)


def is_text_input(item: dict[str, Any]) -> bool:
    return item.get("type") == "text"


def is_image_input(item: dict[str, Any]) -> bool:
    return item.get("type") == "image_url"


def parse_openai_moderation_response(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results")
    if not results:
        return {"flags": []}
    # Unique flag codes with their highest confidence score.
    flag_map: dict[str, Any] = {}
    for result in results:
        if result.get("flagged"):
            for category, flagged in (result.get("categories") or {}).items():
                if flagged:
                    existing = flag_map.get(category)
                    current = (result.get("category_scores") or {}).get(category)
                    if category not in flag_map or (
                        current is not None and existing is not None and current > existing
                    ):
                        flag_map[category] = current
    return {
        "flags": [
            {"code": code, "description": code, "confidence": confidence}
            for code, confidence in flag_map.items()
        ]
    }


def _handle_api_error(err: Any, data: Any = None) -> dict[str, Any]:
    logger.error("OpenAI moderation API error: %s", err)
    err_text = err if isinstance(err, str) else js_error_string(err)
    if data:
        payload = data if isinstance(data, str) else json_stringify(data)
        return {"error": f"API error: {err_text}: {payload}"}
    return {"error": f"API call error: {err_text}"}


def get_moderation_cache_key(
    model_name: str,
    config: dict[str, Any] | None,
    content: Any,
    api_key: str | None = None,
    api_url: str | None = None,
    organization: str | None = None,
) -> str:
    headers = (config or {}).get("headers")
    cache_config = {
        **(config or {}),
        "apiKey": UNDEFINED,
        "apiKeyEnvar": UNDEFINED,
        "apiUrl": UNDEFINED if api_url is None else api_url,
        "organization": UNDEFINED if organization is None else organization,
        "headers": hash_moderation_cache_value(
            [[key, hash_moderation_cache_value(headers[key])] for key in sorted(headers)]
        )
        if isinstance(headers, dict) and headers
        else UNDEFINED,
    }
    namespace = get_openai_moderation_auth_cache_namespace(api_key) if api_key else "no-api-key"
    return (
        f"openai:moderation:{model_name}:{hash_moderation_cache_value(cache_config)}:"
        f"{namespace}:{hash_moderation_cache_value(content)}"
    )


def supports_image_input(model_name: str) -> bool:
    model = next((m for m in OPENAI_MODERATION_MODELS if m["id"] == model_name), None)
    return "image" in model["capabilities"] if model else False


def format_moderation_input(content: Any, supports_images: bool) -> Any:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if supports_images else content
    if not supports_images:
        logger.warning(
            "Using image inputs with a text-only moderation model. Images will be ignored."
        )
        return " ".join(item.get("text", "") for item in content if is_text_input(item))
    return content


class OpenAiModerationProvider(OpenAiGenericProvider):
    MODERATION_MODELS: ClassVar[list[dict[str, Any]]] = OPENAI_MODERATION_MODELS
    MODERATION_MODEL_IDS: ClassVar[list[str]] = [m["id"] for m in OPENAI_MODERATION_MODELS]

    def __init__(
        self, model_name: str = "omni-moderation-latest", options: dict[str, Any] | None = None
    ) -> None:
        super().__init__(model_name, options)
        assert_openai_api_model(model_name, self.get_api_url())
        if model_name not in self.MODERATION_MODEL_IDS:
            logger.warning("Using unknown OpenAI moderation model: %s", model_name)

    async def call_moderation_api(
        self, user_prompt: str, assistant_response: Any
    ) -> dict[str, Any]:
        api_key = self.get_api_key()
        if self.requires_api_key() and not api_key:
            return _handle_api_error(self.get_missing_api_key_error_message())

        use_cache = cache.is_cache_enabled()
        content = format_moderation_input(assistant_response, supports_image_input(self.model_name))
        cache_key = get_moderation_cache_key(
            self.model_name,
            self.config,
            content,
            api_key=api_key,
            api_url=self.get_api_url(),
            organization=self.get_organization(),
        )
        if use_cache:
            cached = await cache.get_cache().get(cache_key)
            if cached:
                logger.debug("Returning cached moderation response")
                return {**json.loads(cached), "cached": True}

        logger.debug("Calling OpenAI moderation API with model %s", self.model_name)
        request_body = json_stringify({"model": self.model_name, "input": content})
        headers = {
            "Content-Type": "application/json",
            "x-promptfoo-silent": "true",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            **self.get_openai_request_headers(),
        }
        try:

            async def fetcher() -> Any:
                return await fetch.fetch_with_cache(
                    append_openai_api_path(self.get_api_url(), "moderations"),
                    {"method": "POST", "headers": headers, "body": request_body},
                    get_request_timeout_ms(),
                    "json",
                    True,
                    self.config.get("maxRetries"),
                )

            result = await _fetch_with_dedupe(cache.get_scoped_cache_key(cache_key), fetcher)
            data, status, status_text = result.data, result.status, result.status_text
            if status < 200 or status >= 300:
                return _handle_api_error(
                    f"{status} {status_text}",
                    data if isinstance(data, str) else json_stringify(data),
                )
            logger.debug("\tOpenAI moderation API response: %s", json_stringify(data))
            response = parse_openai_moderation_response(data)
            if use_cache:
                await cache.get_cache().set(cache_key, json_stringify(response))
            return response
        except Exception as err:
            return _handle_api_error(err)

    async def call_api(self, prompt, context=None, options=None):
        raise NotImplementedError("Not implemented")
