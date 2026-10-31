"""Azure OpenAI embeddings. Ports src/providers/azure/embedding.ts."""

from __future__ import annotations

from typing import Any

from promptphoo import fetch
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.azure.chat import _get, js_error_string
from promptphoo.providers.azure.generic import AzureGenericProvider
from promptphoo.providers.azure.tables import DEFAULT_AZURE_API_VERSION
from promptphoo.providers.shared import get_request_timeout_ms, throw_if_aborted


def _usage(data: Any, cached: bool) -> dict[str, Any]:
    usage = _get(data, "usage")
    if cached:
        token_usage = {
            "cached": _get(usage, "total_tokens"),
            "total": _get(usage, "total_tokens"),
            "numRequests": 1,
        }
    else:
        token_usage = {
            "total": _get(usage, "total_tokens"),
            "prompt": _get(usage, "prompt_tokens"),
            "completion": _get(usage, "completion_tokens"),
            "numRequests": 1,
        }
    return {k: v for k, v in token_usage.items() if v is not None}


class AzureEmbeddingProvider(AzureGenericProvider):
    supports_embedding_cancellation = True

    async def call_embedding_api(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self.ensure_initialized()
        if not self.auth_headers:
            raise RuntimeError("Invariant failed: auth headers are not initialized")
        if not self.get_api_base_url():
            raise ValueError("Azure API host must be set.")

        body: dict[str, Any] = {"input": text, "model": self.deployment_name}
        if "dimensions" in self.config:
            body["dimensions"] = self.config["dimensions"]
        abort_signal = (options or {}).get("abortSignal")
        data: Any = None
        cached = False
        try:
            result = await fetch.fetch_with_cache(
                f"{self.get_api_base_url()}/openai/deployments/{self.deployment_name}/embeddings"
                f"?api-version={self.config.get('apiVersion') or DEFAULT_AZURE_API_VERSION}",
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
                max_retries=self.config.get("maxRetries"),
            )
            data, cached = result.data, result.cached
        except Exception as err:
            throw_if_aborted(abort_signal)
            return {
                "error": f"API call error: {js_error_string(err)}",
                "tokenUsage": {"total": 0, "prompt": 0, "completion": 0, "numRequests": 1},
            }

        try:
            items = _get(data, "data")
            embedding = _get(items[0], "embedding") if isinstance(items, list) and items else None
            if not is_truthy(embedding):
                raise ValueError("No embedding returned")
            # Surface the cache hit so downstream skips rate-limit delays.
            return {"embedding": embedding, "cached": cached, "tokenUsage": _usage(data, cached)}
        except Exception as err:
            return {
                "error": f"API response error: {js_error_string(err)}: {json_stringify(data)}",
                "tokenUsage": _usage(data, cached),
            }
