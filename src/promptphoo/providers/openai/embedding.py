"""OpenAI embeddings provider. Ports src/providers/openai/embedding.ts."""

from __future__ import annotations

import base64
import binascii
import logging
import math
import struct
from typing import Any

from promptphoo import fetch
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.openai.billing import calculate_openai_usage_cost
from promptphoo.providers.openai.generic import OpenAiGenericProvider
from promptphoo.providers.openai.util import (
    append_openai_api_path,
    assert_openai_api_model,
    get_token_usage,
    js_error_string,
    normalize_openai_billing_model_name,
)
from promptphoo.providers.shared import (
    get_request_timeout_ms,
    should_bust_provider_cache,
    with_response_cache_metadata,
)

logger = logging.getLogger(__name__)

_INVALID = "Invalid base64 embedding in OpenAI embeddings API response"


def decode_base64_embedding(value: str) -> list[float]:
    """Decode a float32 little-endian base64 embedding, rejecting non-canonical input."""
    try:
        raw = base64.b64decode(value + "=" * (-len(value) % 4), validate=False)
    except (binascii.Error, ValueError) as err:
        raise ValueError(_INVALID) from err
    canonical = base64.b64encode(raw).decode("ascii")
    if not raw or len(raw) % 4 != 0 or (value != canonical and value != canonical.rstrip("=")):
        raise ValueError(_INVALID)
    embedding = [float(x) for x in struct.unpack(f"<{len(raw) // 4}f", raw)]
    if any(not math.isfinite(x) for x in embedding):
        raise ValueError(_INVALID)
    return embedding


def _js_typeof(value: Any) -> str:
    if value is None:
        return "object"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "object"


class OpenAiEmbeddingProvider(OpenAiGenericProvider):
    supports_embedding_cancellation = True

    def get_billing_model_name(self, config: dict[str, Any]) -> str:
        passthrough = config.get("passthrough")
        model = passthrough.get("model") if isinstance(passthrough, dict) else None
        return model if isinstance(model, str) else super().get_billing_model_name(config)

    async def call_embedding_api(
        self,
        text: Any,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.requires_api_key() and not self.get_api_key():
            return {"error": self.get_missing_api_key_error_message()}
        if not isinstance(text, str):
            return {
                "error": f"Invalid input type for embedding API. Expected string, got "
                f"{_js_typeof(text)}. Input: {json_stringify(text)}"
            }
        body = {"input": text, "model": self.model_name, **(self.config.get("passthrough") or {})}
        assert_openai_api_model(body.get("model"), self.get_api_url())

        data: Any = None
        delete_from_cache = None
        cached = False
        latency_ms: Any = None
        try:
            api_key = self.get_api_key()
            result = await fetch.fetch_with_cache(
                append_openai_api_path(self.get_api_url(), "embeddings"),
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
                should_bust_provider_cache(context),
                self.config.get("maxRetries"),
            )
            data, cached, status, status_text = (
                result.data,
                result.cached,
                result.status,
                result.status_text,
            )
            latency_ms, delete_from_cache = result.latency_ms, result.delete_from_cache
            if status and (status < 200 or status >= 300):
                payload = data if isinstance(data, str) else json_stringify(data)
                return {"error": f"API error: {status} {status_text or 'Unknown error'}\n{payload}"}
        except Exception as err:
            logger.error("API call error: %s", js_error_string(err))
            if delete_from_cache:
                await delete_from_cache()
            return {"error": f"API call error: {js_error_string(err)}"}

        try:
            items = data.get("data") if isinstance(data, dict) else None
            first = items[0] if isinstance(items, list) and items else None
            embedding = first.get("embedding") if isinstance(first, dict) else None
            if not embedding:
                return {"error": "No embedding found in OpenAI embeddings API response"}
            billing_model = normalize_openai_billing_model_name(
                self.get_billing_model_name(self.config)
            )
            response: dict[str, Any] = {
                "embedding": decode_base64_embedding(embedding)
                if isinstance(embedding, str)
                else embedding
            }
            if latency_ms is not None:
                response["latencyMs"] = latency_ms
            response["tokenUsage"] = get_token_usage(data, False)
            cost = calculate_openai_usage_cost(billing_model, self.config, data.get("usage"))
            if cost is not None:
                response["cost"] = cost
            return with_response_cache_metadata(response, cached)
        except Exception as err:
            logger.error("Response parsing error: %s", js_error_string(err))
            if delete_from_cache:
                await delete_from_cache()
            return {"error": f"API error: {js_error_string(err)}"}

    async def call_api(self, prompt, context=None, options=None):
        raise NotImplementedError("Not implemented")
