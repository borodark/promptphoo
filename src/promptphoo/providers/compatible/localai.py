"""LocalAI chat, completion and embedding providers. Ports src/providers/localai.ts and the
`localai:` entry of src/providers/registry.ts.
"""

from __future__ import annotations

from typing import Any

from promptphoo import envars, fetch
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.base import ApiProvider, ProviderResponse
from promptphoo.providers.credentials import resolve_provider_env
from promptphoo.providers.shared import get_request_timeout_ms, js_error_string, parse_chat_prompt


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def _bust(context: dict[str, Any] | None) -> bool:
    context = context or {}
    return bool(_first(context.get("bustCache"), context.get("debug")))


class LocalAiGenericProvider(ApiProvider):
    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        super().__init__(options)
        self.model_name = model_name
        config = options.get("config")
        resolved = resolve_provider_env(self.env, ["LOCALAI_BASE_URL"])
        self.api_base_url = (
            (config or {}).get("apiBaseUrl")
            or (resolved["value"] if resolved else None)
            or "http://localhost:8080/v1"
        )
        self.config = config or {}

    @property
    def modelName(self) -> str:  # noqa: N802 - upstream attribute name
        return self.model_name

    def default_id(self) -> str:
        return f"localai:{self.model_name}"

    def __str__(self) -> str:
        return f"[LocalAI Provider {self.model_name}]"

    def _temperature(self) -> Any:
        return _first(
            self.config.get("temperature"),
            envars.parse_env_float((self.env or {}).get("LOCALAI_TEMPERATURE")),
            envars.get_env_float("LOCALAI_TEMPERATURE"),
            0.7,
        )

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        raise NotImplementedError("Not implemented")


class LocalAiChatProvider(LocalAiGenericProvider):
    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        messages = parse_chat_prompt(prompt, [{"role": "user", "content": prompt}])
        body = {"model": self.model_name, "messages": messages, "temperature": self._temperature()}
        try:
            result = await fetch.fetch_with_cache(
                f"{self.api_base_url}/chat/completions",
                {
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": json_stringify(body),
                },
                get_request_timeout_ms(),
                "json",
                _bust(context),
            )
            data = result.data
        except Exception as err:
            return {"error": f"API call error: {js_error_string(err)}"}
        choices = data.get("choices") if isinstance(data, dict) else None
        first = choices[0] if isinstance(choices, list) and choices else None
        if not (isinstance(first, dict) and first.get("message")):
            return {"error": f"Malformed response data: {json_stringify(data)}"}
        try:
            return {"output": first["message"].get("content")}
        except Exception as err:
            return {"error": f"API response error: {js_error_string(err)}: {json_stringify(data)}"}


class LocalAiEmbeddingProvider(LocalAiGenericProvider):
    supports_embedding_cancellation = True

    async def call_embedding_api(self, text, context=None, options=None) -> dict[str, Any]:
        body = {"input": text, "model": self.model_name}
        try:
            result = await fetch.fetch_with_cache(
                f"{self.api_base_url}/embeddings",
                {
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": json_stringify(body),
                },
                get_request_timeout_ms(),
            )
            data = result.data
        except Exception as err:
            return {"error": f"API call error: {js_error_string(err)}"}
        try:
            items = data.get("data") if isinstance(data, dict) else None
            first = items[0] if isinstance(items, list) and items else None
            embedding = first.get("embedding") if isinstance(first, dict) else None
            if not embedding:
                raise ValueError("No embedding found in LocalAI embeddings API response")
            return {"embedding": embedding}
        except Exception as err:
            return {"error": f"API response error: {js_error_string(err)}: {json_stringify(data)}"}


class LocalAiCompletionProvider(LocalAiGenericProvider):
    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        body = {"model": self.model_name, "prompt": prompt, "temperature": self._temperature()}
        try:
            result = await fetch.fetch_with_cache(
                f"{self.api_base_url}/completions",
                {
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": json_stringify(body),
                },
                get_request_timeout_ms(),
                "json",
                _bust(context),
            )
            data = result.data
        except Exception as err:
            return {"error": f"API call error: {js_error_string(err)}"}
        choices = data.get("choices") if isinstance(data, dict) else None
        first = choices[0] if isinstance(choices, list) and choices else None
        if not first:
            return {"error": f"Malformed response data: {json_stringify(data)}"}
        try:
            return {"output": first.get("text")}
        except Exception as err:
            return {"error": f"API response error: {js_error_string(err)}: {json_stringify(data)}"}


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `localai:` registry entry."""
    splits = provider_path.split(":")
    model_type = splits[1] if len(splits) > 1 else ""
    model_name = ":".join(splits[2:])
    if model_type == "chat":
        return LocalAiChatProvider(model_name, options)
    if model_type == "completion":
        return LocalAiCompletionProvider(model_name, options)
    if model_type in ("embedding", "embeddings"):
        return LocalAiEmbeddingProvider(model_name, options)
    return LocalAiChatProvider(":".join(splits[1:]), options)
