"""Base class for Anthropic providers. Ports src/providers/anthropic/generic.ts.

Upstream builds an `@anthropic-ai/sdk` client; promptphoo talks to the HTTP API directly through
`fetch.fetch_with_cache`, reproducing the SDK's request (`x-api-key`, `anthropic-version`,
`ANTHROPIC_CUSTOM_HEADERS`) and error shapes. Claude Code OAuth (`apiKeyRequired: false`) is not
supported: there is no credential loader, so such a provider warns and stays unauthenticated.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from promptphoo import envars
from promptphoo.jsvalues import js_string, json_stringify
from promptphoo.providers.base import ApiProvider
from promptphoo.providers.credentials import resolve_provider_api_key

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_CACHE_HASH_CONTEXT = "promptfoo:anthropic:cache-key:v1"
MAX_EPHEMERAL_RESPONSE_CACHE_ENTRIES = 512


def _layer_value(layer: Mapping[str, Any] | None, key: str) -> str | None:
    if layer and layer.get(key) is not None:
        return js_string(layer[key])
    return None


def get_scoped_custom_headers(env: Mapping[str, Any] | None = None) -> str | None:
    for layer in (
        env,
        envars.get_env_overrides("provider"),
        env,
        envars.get_env_overrides("suite"),
        envars.get_env_overrides("file"),
    ):
        value = _layer_value(layer, "ANTHROPIC_CUSTOM_HEADERS")
        if value is not None:
            return value
    return None


def parse_anthropic_custom_headers(value: str | None) -> dict[str, str]:
    """Newline-separated `Name: value` lines, as the Anthropic SDK parses them."""
    headers: dict[str, str] = {}
    for line in (value or "").split("\n") if value is not None else []:
        colon = line.find(":")
        if colon >= 0:
            name = line[:colon].strip()
            if name:
                headers[name] = line[colon + 1 :].strip()
    return headers


def _canonicalize(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return [_canonicalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _canonicalize(value[k]) for k in sorted(value, key=_utf16_key)}
    return value


def _utf16_key(key: Any) -> Any:
    # JS Array.prototype.sort compares UTF-16 code units.
    return str(key).encode("utf-16-be")


def hash_anthropic_cache_value(value: Any) -> str:
    canonical = _canonicalize(value)
    serialized = canonical if isinstance(canonical, str) else json_stringify(canonical)
    h = hashlib.sha256()
    h.update(ANTHROPIC_CACHE_HASH_CONTEXT.encode())
    h.update(b"\0")
    h.update(serialized.encode("utf-8"))
    return h.hexdigest()


class AnthropicGenericProvider(ApiProvider):
    """Shared API key / base URL resolution and the response cache helpers."""

    SUPPORTS_CLAUDE_CODE_OAUTH = False

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        super().__init__(options)
        self.model_name = model_name
        self.config = options.get("config") or {}
        self._client_has_custom_headers = bool(self.config.get("headers")) or bool(
            parse_anthropic_custom_headers(self._custom_header_value())
        )
        self.api_key = self.get_api_key()
        self.using_claude_code_oauth = False
        self._ephemeral_cache_namespace = str(uuid.uuid4())
        self._ephemeral_response_cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._ephemeral_cache_clear_generation = 0
        if (
            not self.api_key
            and self.config.get("apiKeyRequired") is False
            and type(self).SUPPORTS_CLAUDE_CODE_OAUTH
        ):
            logger.warning(
                "[anthropic] apiKeyRequired is false but no Claude Code credential was found. "
                "Run `claude /login` to create one, or set ANTHROPIC_API_KEY."
            )

    def _custom_header_value(self) -> str | None:
        scoped = get_scoped_custom_headers(self.env)
        return scoped if scoped is not None else envars.get_env_string("ANTHROPIC_CUSTOM_HEADERS")

    def default_id(self) -> str:
        return f"anthropic:{self.model_name}"

    def __str__(self) -> str:
        return f"[Anthropic Provider {self.model_name}]"

    def requires_api_key(self) -> bool:
        return not (
            type(self).SUPPORTS_CLAUDE_CODE_OAUTH and self.config.get("apiKeyRequired") is False
        )

    def get_api_key(self) -> str | None:
        return resolve_provider_api_key(self.config, self.env, ["ANTHROPIC_API_KEY"])

    def get_api_base_url(self) -> str | None:
        env_url = envars.get_provider_env_string(self.env, "ANTHROPIC_BASE_URL")
        if env_url is None:
            env_url = envars.get_env_string("ANTHROPIC_BASE_URL")
        return self.config.get("apiBaseUrl") or (
            DEFAULT_ANTHROPIC_BASE_URL if env_url == "" else env_url
        )

    def request_base_url(self) -> str:
        """The base URL the SDK client would send to (its default when nothing is set)."""
        return (self.get_api_base_url() or DEFAULT_ANTHROPIC_BASE_URL).rstrip("/")

    def build_request_headers(self, request_headers: Mapping[str, str]) -> dict[str, str]:
        """SDK default headers, then ANTHROPIC_CUSTOM_HEADERS, then per-request headers."""
        headers: dict[str, str] = {
            "accept": "application/json",
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        custom = parse_anthropic_custom_headers(self._custom_header_value())
        scoped = get_scoped_custom_headers(self.env)
        for name, value in custom.items():
            _set_header(headers, name, value)
        if scoped is not None and self.api_key and "x-api-key" not in {k.lower() for k in custom}:
            _set_header(headers, "x-api-key", self.api_key)
        for name, value in request_headers.items():
            _set_header(headers, name, value)
        return headers

    def get_cache_identity_hash(self) -> str:
        base_url = self.get_api_base_url()
        # JSON.stringify drops an undefined apiBaseUrl.
        return hash_anthropic_cache_value({} if base_url is None else {"apiBaseUrl": base_url})

    def has_custom_headers(self) -> bool:
        return self._client_has_custom_headers or bool(
            parse_anthropic_custom_headers(self._custom_header_value())
        )

    def get_cache_namespace(self) -> str:
        # Labels are stable non-secret tenant namespaces; unlabeled providers reuse responses
        # only within their own process lifetime.
        return hash_anthropic_cache_value(
            {
                "providerId": self.id(),
                "providerLabel": self.label or self._ephemeral_cache_namespace,
            }
        )

    async def get_cached_response(
        self, cache: Any, cache_key: str, ephemeral_cache_key: str, clear_generation: int
    ) -> Any:
        if self.label:
            return await cache.get(cache_key)
        self._sync_ephemeral_cache(clear_generation)
        entry = self._ephemeral_response_cache.get(ephemeral_cache_key)
        if entry is None:
            return None
        response, expires_at = entry
        if expires_at <= time.time() * 1000:
            del self._ephemeral_response_cache[ephemeral_cache_key]
            return None
        return response

    async def set_cached_response(
        self,
        cache: Any,
        cache_key: str,
        ephemeral_cache_key: str,
        clear_generation: int,
        ttl_ms: int,
        response: str,
    ) -> None:
        if self.label:
            await cache.set(cache_key, response)
            return
        self._sync_ephemeral_cache(clear_generation)
        store = self._ephemeral_response_cache
        if ephemeral_cache_key not in store and len(store) >= MAX_EPHEMERAL_RESPONSE_CACHE_ENTRIES:
            store.popitem(last=False)
        store[ephemeral_cache_key] = (
            response,
            float("inf") if ttl_ms <= 0 else time.time() * 1000 + ttl_ms,
        )

    def _sync_ephemeral_cache(self, clear_generation: int) -> None:
        if self._ephemeral_cache_clear_generation != clear_generation:
            self._ephemeral_response_cache.clear()
            self._ephemeral_cache_clear_generation = clear_generation

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("Not implemented: callApi must be implemented by subclasses")


def _set_header(headers: dict[str, str], name: str, value: str) -> None:
    """Set a header, replacing any existing spelling of the same name."""
    for existing in [k for k in headers if k.lower() == name.lower()]:
        del headers[existing]
    headers[name] = value


class AnthropicAPIError(Exception):
    """An HTTP error from the Anthropic API, shaped like the SDK's APIError."""

    def __init__(self, status: int, error: Any, text: str | None = None) -> None:
        self.status = status
        self.error = error
        super().__init__(_make_api_error_message(status, error, text))


def _make_api_error_message(status: int | None, error: Any, text: str | None) -> str:
    if isinstance(error, dict) and error.get("message"):
        msg = (
            error["message"]
            if isinstance(error["message"], str)
            else json_stringify(error["message"])
        )
    elif error is not None:
        msg = json_stringify(error)
    else:
        msg = text or ""
    if status and msg:
        return f"{status} {msg}"
    if status:
        return f"{status} status code (no body)"
    return msg or "(no status code or body)"


async def post_anthropic(
    provider: AnthropicGenericProvider,
    path: str,
    body: dict[str, Any],
    request_headers: Mapping[str, str],
    timeout_ms: int = 600_000,
) -> dict[str, Any]:
    """POST a JSON body to the Anthropic API and return the parsed JSON response.

    Raises AnthropicAPIError for non-2xx responses and ConnectionError when no response came
    back (the SDK's APIConnectionError / APIConnectionTimeoutError).
    """
    from promptphoo.fetch import fetch_with_cache

    url = f"{provider.request_base_url()}{path}"
    try:
        result = await fetch_with_cache(
            url,
            {
                "method": "POST",
                "headers": provider.build_request_headers(request_headers),
                "body": json_stringify(body),
                # Upstream calls Anthropic through its SDK, which never adds x-promptfoo-version.
                "sdkTransport": True,
            },
            timeout_ms,
            "json",
            bust=True,
            max_retries=2,
        )
    except (RuntimeError, OSError, TimeoutError) as err:
        message = str(err)
        raise ConnectionError(
            "Request timed out." if "timed out" in message.lower() else "Connection error."
        ) from err
    if not 200 <= result.status < 300:
        data = result.data
        if isinstance(data, dict | list):
            raise AnthropicAPIError(result.status, data)
        raise AnthropicAPIError(result.status, None, data if isinstance(data, str) else None)
    if not isinstance(result.data, dict):
        raise ValueError(f"Unexpected Anthropic API response: {json_stringify(result.data)}")
    return result.data
