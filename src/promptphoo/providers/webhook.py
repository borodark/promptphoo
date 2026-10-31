"""The `webhook:<url>` provider. Ports src/providers/webhook.ts and its factory in
src/providers/registry.ts.

POSTs `{"prompt": ..., "config": ...}` as JSON and expects `{"output": "<string>"}` back.
"""

from __future__ import annotations

from typing import Any

from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.base import ApiProvider, ProviderResponse
from promptphoo.providers.shared import get_request_timeout_ms, js_error_string


class WebhookProvider(ApiProvider):
    def __init__(self, webhook_url: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.webhook_url = webhook_url
        # Upstream keeps `config` as given, so an empty `{}` (truthy in JS) is still sent.
        self.raw_config: Any = (options or {}).get("config")

    def default_id(self) -> str:
        return f"webhook:{self.webhook_url}"

    def __str__(self) -> str:
        return f"[Webhook Provider {self.webhook_url}]"

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        from promptphoo.fetch import fetch_with_cache

        params: dict[str, Any] = {"prompt": prompt}
        if is_truthy(self.raw_config):
            params["config"] = self.raw_config
        try:
            response = await fetch_with_cache(
                self.webhook_url,
                {
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": json_stringify(params),
                },
                get_request_timeout_ms(),
                "json",
            )
        except Exception as err:
            return {"error": f"Webhook call error: {js_error_string(err)}"}
        data = response.data
        if isinstance(data, dict) and isinstance(data.get("output"), str):
            out: ProviderResponse = {"output": data["output"], "cached": response.cached}
            if response.latency_ms is not None:
                out["latencyMs"] = response.latency_ms
            return out
        return {
            "error": f"Webhook response error: Unexpected response format: {json_stringify(data)}"
        }


def create_provider(provider_path: str, options: dict[str, Any]) -> WebhookProvider:
    return WebhookProvider(provider_path[len("webhook:") :], options)
