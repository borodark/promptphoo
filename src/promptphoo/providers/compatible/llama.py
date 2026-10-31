"""llama.cpp server (`/completion`) provider. Ports src/providers/llama.ts and the `llama` /
`llama:` entry of src/providers/registry.ts.
"""

from __future__ import annotations

from typing import Any

from promptphoo import fetch
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.base import ApiProvider, ProviderResponse
from promptphoo.providers.credentials import resolve_provider_env
from promptphoo.providers.shared import get_request_timeout_ms, js_error_string

# Body keys after `prompt` and `n_predict`, in upstream's order (undefined ones are dropped).
LLAMA_OPTION_KEYS = (
    "temperature", "top_k", "top_p", "n_keep", "stop", "repeat_penalty", "repeat_last_n",
    "penalize_nl", "presence_penalty", "frequency_penalty", "mirostat", "mirostat_tau",
    "mirostat_eta", "seed", "ignore_eos", "logit_bias",
)  # fmt: skip


class LlamaProvider(ApiProvider):
    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        super().__init__(options)
        self.model_name = model_name
        self.config = options.get("config")  # type: ignore[assignment]

    @property
    def modelName(self) -> str:  # noqa: N802 - upstream attribute name
        return self.model_name

    def default_id(self) -> str:
        return f"llama:{self.model_name}"

    def __str__(self) -> str:
        return f"[Llama Provider {self.model_name}]"

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        config = self.config or {}
        n_predict = config.get("n_predict")
        body: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": 512 if n_predict is None else n_predict,
        }
        for key in LLAMA_OPTION_KEYS:
            if config.get(key) is not None:
                body[key] = config[key]
        resolved = resolve_provider_env(self.env, ["LLAMA_BASE_URL"])
        url = (resolved["value"] if resolved else None) or "http://localhost:8080"
        try:
            result = await fetch.fetch_with_cache(
                f"{url}/completion",
                {
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": json_stringify(body),
                },
                get_request_timeout_ms(),
            )
        except Exception as err:
            return {"error": f"API call error: {js_error_string(err)}"}
        data = result.data
        try:
            response: ProviderResponse = {
                "output": data.get("content") if isinstance(data, dict) else None,
                "cached": result.cached,
            }
            if result.latency_ms is not None:
                response["latencyMs"] = result.latency_ms
            return response
        except Exception as err:
            return {"error": f"API response error: {js_error_string(err)}: {json_stringify(data)}"}


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `llama` / `llama:` registry entry (`providerPath.slice('llama:'.length)`)."""
    return LlamaProvider(provider_path[len("llama:") :], options)
