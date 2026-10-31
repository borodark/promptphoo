"""The HTTP provider. Ports HttpProvider and the body helpers of src/providers/http.ts.

Streaming, websockets, Promptfoo Cloud targets and `promptfoo://` targets are not ported.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from promptphoo import cli_state, python_hooks
from promptphoo import fetch as _fetch
from promptphoo.files import maybe_load_config_from_external_file, maybe_load_from_external_file
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import UNDEFINED, is_truthy, js_string, json_stringify
from promptphoo.providers.base import ApiProvider, ProviderResponse
from promptphoo.providers.http import signature as _signature
from promptphoo.providers.http.config import parse_http_provider_config, validate_multipart_config
from promptphoo.providers.http.metadata import (
    format_raw_request_for_debug_metadata,
    log_transformed_prompt,
    sanitize_multipart_fields,
    sanitize_multipart_files,
    sanitize_request_body_for_metadata,
    sanitize_transformed_request_for_metadata,
)
from promptphoo.providers.http.multipart import render_http_multipart_body
from promptphoo.providers.http.raw import (
    escape_json_variables,
    extract_body_from_raw_request,
    parse_raw_request,
    render_raw_request_with_nunjucks,
)
from promptphoo.providers.http.tls import create_https_agent
from promptphoo.providers.http.transforms import (
    create_session_parser,
    create_transform_request,
    create_transform_response,
    create_validate_status,
    load_transform_module,
    normalize_response_transform_result,
)
from promptphoo.providers.shared import (
    get_request_timeout_ms,
    transform_tool_choice,
    transform_tools,
)
from promptphoo.sanitizer import safe_json_stringify, sanitize_object, sanitize_url
from promptphoo.templating import get_engine, render_vars_in_object
from promptphoo.tokens import create_empty_token_usage
from promptphoo.whatwg_url import URL, InvalidUrlError, form_urlencoded_serialize

logger = logging.getLogger(__name__)

AUTH_TOKEN_CACHE_HMAC_CONTEXT = "promptfoo:http-auth-token-cache-key"
AUTH_TOKEN_CACHE_HMAC_KEY = secrets.token_bytes(32)
MAX_AUTH_CACHE_ENTRIES = 256
TOKEN_REFRESH_BUFFER_MS = 60000

# JSON.stringify(new LRUCache({ max: 256 })) in lru-cache 11.
_LRU_CACHE_JSON = {
    "ttl": 0, "ttlResolution": 1, "ttlAutopurge": False, "updateAgeOnGet": False,
    "updateAgeOnHas": False, "allowStale": False, "noDisposeOnSet": False, "noUpdateTTL": False,
    "maxEntrySize": 0, "noDeleteOnFetchRejection": False, "noDeleteOnStaleGet": False,
    "allowStaleOnFetchAbort": False, "allowStaleOnFetchRejection": False,
    "ignoreFetchAbort": False, "backgroundFetchSize": 1,
}  # fmt: skip


def _render(template: Any, vars: dict[str, Any]) -> str:
    return get_engine().render_string(
        template if isinstance(template, str) else js_string(template), vars
    )


def _now_ms() -> int:
    return _signature.now_ms()


# --- body helpers ----------------------------------------------------------------------------


def render_json_template(template: str, vars: dict[str, Any]) -> Any:
    """Render a JSON-looking template; retry with JSON-escaped vars if the result won't parse."""
    rendered = render_vars_in_object(template, vars)
    try:
        return js_json_parse(rendered)
    except ValueError:
        re_rendered = render_vars_in_object(template, escape_json_variables(vars))
        return js_json_parse(re_rendered)


def process_json_body(body: Any, vars: dict[str, Any]) -> Any:
    """Render templates in a JSON body; nested strings holding JSON objects/arrays are parsed."""
    rendered = render_vars_in_object(body, vars)

    if isinstance(rendered, dict | list):

        def process_nested(obj: Any) -> Any:
            if isinstance(obj, list):
                return [process_nested(v) for v in obj]
            if isinstance(obj, dict):
                return {k: process_nested(v) for k, v in obj.items()}
            if isinstance(obj, str):
                trimmed = obj.strip()
                if trimmed.startswith(("{", "[")):
                    try:
                        parsed = js_json_parse(obj)
                    except ValueError:
                        return obj
                    return parsed if isinstance(parsed, dict | list) else obj
            return obj

        return process_nested(rendered)

    if isinstance(rendered, str):
        try:
            return js_json_parse(rendered)
        except ValueError as err:
            trimmed = rendered.strip()
            if trimmed.startswith(("{", "[")) and isinstance(body, str):
                try:
                    return render_json_template(body, vars)
                except ValueError:
                    pass
            logger.debug(
                "[HTTP Provider] Body is a string that failed JSON parsing, using as-is: %s", err
            )
            return rendered
    return rendered


def process_text_body(body: Any, vars: dict[str, Any]) -> Any:
    """Render templates in a text body."""
    if body is None:
        return body
    if isinstance(body, dict | list):
        raise AssertionError(
            "Invariant failed: Expected body to be a string when content type is not "
            "application/json"
        )
    try:
        return render_vars_in_object(body, vars)
    except Exception as err:
        logger.warning("Error rendering body template: %s", err)
        return body


def determine_request_body(
    content_type: bool, parsed_prompt: Any, config_body: Any, vars: dict[str, Any]
) -> Any:
    actual_config_body = config_body
    if isinstance(config_body, str) and content_type:
        try:
            actual_config_body = js_json_parse(config_body)
            logger.debug("[HTTP Provider] Parsed stringified config body to object")
        except ValueError:
            pass
    if content_type:
        if isinstance(parsed_prompt, dict | list):
            # Object.assign({}, configBody || {}, parsedPrompt)
            merged: dict[str, Any] = {}
            for source in (
                actual_config_body if is_truthy(actual_config_body) else {},
                parsed_prompt,
            ):
                merged.update(_own_entries(source))
            return merged
        return process_json_body(actual_config_body, {**vars, "prompt": parsed_prompt})
    return process_text_body(actual_config_body, {**vars, "prompt": parsed_prompt})


def _own_entries(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {str(i): v for i, v in enumerate(value)}
    if isinstance(value, str):
        return {str(i): c for i, c in enumerate(value)}
    return {}


def estimate_token_count(text: Any, multiplier: float = 1.3) -> int:
    """Word-based token estimate: ceil(words * multiplier)."""
    if not text or not isinstance(text, str):
        return 0
    words = [w for w in text.strip().split() if len(w) > 0]
    import math

    return math.ceil(len(words) * multiplier)


def content_type_is_json(headers: dict[str, str] | None) -> bool:
    if not headers:
        return False
    return any(
        key.lower().startswith("content-type") and "application/json" in js_string(value)
        for key, value in headers.items()
    )


def _remove_multipart_content_type(headers: dict[str, str]) -> None:
    for key in list(headers):
        if key.lower() == "content-type":
            del headers[key]


# --- auth cache keys -----------------------------------------------------------------------------


def stable_serialize(value: Any, seen: set[int] | None = None) -> str:
    seen = seen if seen is not None else set()
    if value is UNDEFINED:
        return "undefined"
    if value is None:
        return "null"
    if callable(value) and not isinstance(value, dict | list):
        return json.dumps({"function": getattr(value, "__name__", "") or "anonymous"})
    if not isinstance(value, dict | list | tuple):
        try:
            return json_stringify(value)
        except Exception:
            return js_string(value)
    if id(value) in seen:
        return '"[Circular]"'
    seen.add(id(value))
    try:
        if isinstance(value, list | tuple):
            return "[" + ",".join(stable_serialize(v, seen) for v in value) + "]"
        return (
            "{"
            + ",".join(
                f"{json.dumps(str(k), ensure_ascii=False)}:{stable_serialize(value[k], seen)}"
                for k in sorted(value, key=str)
            )
            + "}"
        )
    finally:
        seen.discard(id(value))


def digest_auth_cache_input(value: Any) -> str:
    mac = hmac.new(AUTH_TOKEN_CACHE_HMAC_KEY, digestmod=hashlib.sha256)
    mac.update(AUTH_TOKEN_CACHE_HMAC_CONTEXT.encode())
    mac.update(b"\0")
    mac.update(stable_serialize(value).encode())
    return mac.hexdigest()


def _get_provider_cache_identity(provider: Any) -> Any:
    if provider is None:
        return UNDEFINED
    try:
        provider_id = provider.id()
    except Exception as err:
        provider_id = f"error:{err}"
    return {
        "type": type(provider).__name__,
        "id": provider_id,
        "label": getattr(provider, "label", None),
        "config": getattr(provider, "config", None),
    }


def _file_auth_context_cache_input(context: dict[str, Any]) -> dict[str, Any]:
    serializable = {
        k: v
        for k, v in context.items()
        if k not in ("filters", "getCache", "logger", "originalProvider")
    }
    filters = context.get("filters")
    return {
        **serializable,
        "filters": {k: stable_serialize(filters[k]) for k in sorted(filters)}
        if filters
        else UNDEFINED,
        "getCache": UNDEFINED
        if context.get("getCache") is None
        else stable_serialize(context["getCache"]),
        "logger": UNDEFINED if context.get("logger") is None else True,
        "originalProvider": _get_provider_cache_identity(context.get("originalProvider")),
    }


def _format_file_auth_freshness(expiration: Any) -> str:
    if expiration is None:
        return "never expires"
    import math

    return f"expires in {max(0, math.ceil((expiration - _now_ms()) / 1000))} seconds"


def _parse_file_auth_result(result: Any) -> dict[str, Any]:
    """FileAuthResultSchema.parse: {token: non-empty string, expiration?: finite number|null}."""
    issues: list[dict[str, Any]] = []
    if not isinstance(result, dict):
        raise ValueError(
            json.dumps(
                [
                    {
                        "expected": "object",
                        "code": "invalid_type",
                        "path": [],
                        "message": "Invalid input: expected object, received "
                        + ("undefined" if result is None else type(result).__name__),
                    }
                ],
                indent=2,
            )
        )
    token = result.get("token")
    if not isinstance(token, str):
        issues.append(
            {
                "expected": "string",
                "code": "invalid_type",
                "path": ["token"],
                "message": "Invalid input: expected string, received "
                + ("undefined" if token is None else type(token).__name__),
            }
        )
    elif len(token) < 1:
        issues.append(
            {
                "origin": "string",
                "code": "too_small",
                "minimum": 1,
                "inclusive": True,
                "path": ["token"],
                "message": "Too small: expected string to have >=1 characters",
            }
        )
    expiration = result.get("expiration")
    if expiration is not None and (
        isinstance(expiration, bool)
        or not isinstance(expiration, int | float)
        or expiration != expiration
        or expiration in (float("inf"), float("-inf"))
    ):
        issues.append(
            {
                "expected": "number",
                "code": "invalid_type",
                "path": ["expiration"],
                "message": "Invalid input: expected number",
            }
        )
    if issues:
        raise ValueError(json.dumps(issues, indent=2))
    out = {"token": token}
    if "expiration" in result:
        out["expiration"] = expiration
    return out


# --- the provider ------------------------------------------------------------------------------


class HttpProvider(ApiProvider):
    """`http(s)://...` provider: templated JSON/text/raw requests with transforms and auth."""

    def __init__(self, url: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        super().__init__(options)
        self.config = parse_http_provider_config(options.get("config", UNDEFINED))
        validate_multipart_config(self.config)
        redteam = (cli_state.config() or {}).get("redteam", UNDEFINED)
        if not is_truthy(self.config.get("tokenEstimation")) and is_truthy(redteam):
            self.config["tokenEstimation"] = {"enabled": True, "multiplier": 1.3}
        self.url: str = self.config.get("url") or url

        self._transform_response: Any = None
        self._session_parser: Any = None
        self._transform_request: Any = None
        self._validate_status: Any = None
        self._signature_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._auth_token_cache: dict[str, dict[str, Any]] = {}
        self._token_refresh_locks: dict[str, dict[str, Any]] = {}
        self._https_agent: Any = None
        self._https_agent_lock = asyncio.Lock()
        self._fetched_sessions: set[str] = set()
        self._session_endpoint_parser: Any = None

        if self.config.get("request"):
            self.config["request"] = maybe_load_from_external_file(self.config["request"])
        elif not (
            is_truthy(self.config.get("body"))
            or is_truthy(self.config.get("multipart"))
            or self.config.get("method") == "GET"
        ):
            raise AssertionError(
                f"Invariant failed: Expected HTTP provider {self.url} to have a config containing "
                f"{{body}}, but instead got {safe_json_stringify(self.config)}"
            )
        if is_truthy(self.config.get("body")):
            self.config["body"] = maybe_load_config_from_external_file(self.config["body"])

    # identity -------------------------------------------------------------------------------

    def default_id(self) -> str:
        return self.url

    def id(self) -> str:
        return self.url

    def __str__(self) -> str:
        return f"[HTTP Provider {self.url}]"

    def to_json(self) -> dict[str, Any]:
        """JSON.stringify(provider): the instance's own enumerable fields, as in Node."""
        out: dict[str, Any] = {
            "url": self.url,
            "config": self.config,
            "transformResponse": {},
            "sessionParser": {},
            "transformRequest": {},
            "validateStatus": {},
            "signatureCache": dict(_LRU_CACHE_JSON),
            "authTokenCache": {},
            "tokenRefreshLocks": {},
            "fetchedSessions": {},
        }
        if self.config.get("session"):
            out["sessionEndpointParser"] = {}
        for key in ("transform", "delay", "inputs", "label", "prompts"):
            value = getattr(self, key, None)
            if value is not None:
                out[key] = value
        return out

    # lazily-built transforms (Node builds these promises in the constructor) ------------------

    async def _get_transform_response(self) -> Callable[..., Awaitable[dict[str, Any]]]:
        if self._transform_response is None:
            parser = self.config.get("transformResponse") or self.config.get("responseParser")
            self._transform_response = create_transform_response(
                load_transform_module(parser, "transform_response")
            )
        return self._transform_response

    async def _get_transform_request(self) -> Callable[..., Awaitable[Any]]:
        if self._transform_request is None:
            self._transform_request = create_transform_request(
                load_transform_module(self.config.get("transformRequest"), "transform_request")
            )
        return self._transform_request

    async def _get_session_parser(self) -> Callable[..., Awaitable[Any]]:
        if self._session_parser is None:
            self._session_parser = create_session_parser(self.config.get("sessionParser"))
        return self._session_parser

    async def _get_validate_status(self) -> Callable[..., Awaitable[Any]]:
        if self._validate_status is None:
            self._validate_status = create_validate_status(self.config.get("validateStatus"))
        return self._validate_status

    async def _get_session_endpoint_parser(self) -> Callable[..., Awaitable[Any]] | None:
        if not self.config.get("session"):
            return None
        if self._session_endpoint_parser is None:
            self._session_endpoint_parser = create_session_parser(
                self.config["session"].get("responseParser")
            )
        return self._session_endpoint_parser

    # token estimation -------------------------------------------------------------------------

    def _estimate_token_usage(
        self, prompt_text: Any, completion_text: Any
    ) -> dict[str, Any] | None:
        estimation = self.config.get("tokenEstimation") or {}
        if not estimation.get("enabled"):
            return None
        try:
            multiplier = estimation.get("multiplier", 1.3)
            prompt_tokens = estimate_token_count(prompt_text, multiplier)
            completion_tokens = estimate_token_count(completion_text, multiplier)
            return {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
                "numRequests": 1,
            }
        except Exception as err:
            logger.warning("Failed to estimate tokens: %s", err)
            return None

    # OAuth / file auth token cache ------------------------------------------------------------

    async def _refresh_oauth_token_if_needed(self, vars: dict[str, Any]) -> dict[str, Any]:
        auth = self.config.get("auth")
        if not auth or auth.get("type") != "oauth":
            raise ValueError("OAuth auth is not configured")
        base_config: dict[str, Any] = {
            **auth,
            "clientId": _render(auth["clientId"], vars) if auth.get("clientId") else None,
            "clientSecret": _render(auth["clientSecret"], vars)
            if auth.get("clientSecret")
            else None,
            "tokenUrl": _render(auth.get("tokenUrl") or "", vars),
            "scopes": [_render(s, vars) for s in auth["scopes"]] if auth.get("scopes") else None,
        }
        if auth.get("grantType") == "password" and "username" in auth:
            oauth_config = {
                **base_config,
                "username": _render(auth["username"], vars) if auth.get("username") else None,
                "password": _render(auth["password"], vars) if auth.get("password") else None,
            }
        else:
            oauth_config = base_config
        cache_key = f"oauth:{digest_auth_cache_input(oauth_config)}"
        cached = self._get_valid_cached_token(cache_key)
        if cached:
            logger.debug("[HTTP Provider Auth]: Using cached OAuth token")
            return cached
        return await self._refresh_token_with_lock(
            cache_key, lambda: self._perform_token_refresh(oauth_config, cache_key)
        )

    async def _perform_token_refresh(
        self, oauth_config: dict[str, Any], cache_key: str
    ) -> dict[str, Any]:
        try:
            pairs = [["grant_type", js_string(oauth_config.get("grantType"))]]
            if oauth_config.get("clientId"):
                pairs.append(["client_id", oauth_config["clientId"]])
            if oauth_config.get("clientSecret"):
                pairs.append(["client_secret", oauth_config["clientSecret"]])
            if oauth_config.get("grantType") == "password":
                if not oauth_config.get("username") or not oauth_config.get("password"):
                    raise ValueError("Username and password are required for password grant type")
                pairs.append(["username", oauth_config["username"]])
                pairs.append(["password", oauth_config["password"]])
            if oauth_config.get("scopes"):
                pairs.append(["scope", " ".join(oauth_config["scopes"])])

            fetch_options: dict[str, Any] = {
                "method": "POST",
                "headers": {"Content-Type": "application/x-www-form-urlencoded"},
                "body": form_urlencoded_serialize(pairs),
            }
            agent = await self._get_https_agent()
            if agent is not None:
                fetch_options["dispatcher"] = agent
            response = await _fetch.fetch_with_cache(
                oauth_config["tokenUrl"], fetch_options, get_request_timeout_ms(), "text", True, 0
            )
            if response.status < 200 or response.status >= 300:
                raise ValueError(
                    f"OAuth token request failed with status {response.status} "
                    f"{response.status_text}: {response.data}"
                )
            token_data = js_json_parse(response.data)
            if not token_data.get("access_token"):
                raise ValueError("OAuth token response missing access_token")
            expires_in = token_data.get("expires_in") or 3600
            expires_at = _now_ms() + expires_in * 1000
            cached = self._cache_token(cache_key, token_data["access_token"], expires_at)
            logger.debug("[HTTP Provider Auth]: Successfully refreshed OAuth token")
            return cached
        except Exception as err:
            logger.error("[HTTP Provider Auth]: Failed to refresh OAuth token: Error: %s", err)
            raise ValueError(f"Failed to refresh OAuth token: Error: {err}") from err

    def _cache_token(self, cache_key: str, token: str, expires_at: Any = None) -> dict[str, Any]:
        cached = {"token": token, "expiresAt": expires_at}
        self._prune_expired_auth_tokens()
        self._auth_token_cache.pop(cache_key, None)
        self._auth_token_cache[cache_key] = cached
        while len(self._auth_token_cache) > MAX_AUTH_CACHE_ENTRIES:
            del self._auth_token_cache[next(iter(self._auth_token_cache))]
        return cached

    def _get_valid_cached_token(self, cache_key: str) -> dict[str, Any] | None:
        cached = self._auth_token_cache.get(cache_key)
        if not cached:
            return None
        if cached["expiresAt"] is None or _now_ms() + TOKEN_REFRESH_BUFFER_MS < cached["expiresAt"]:
            del self._auth_token_cache[cache_key]
            self._auth_token_cache[cache_key] = cached
            return cached
        del self._auth_token_cache[cache_key]
        return None

    def _prune_expired_auth_tokens(self) -> None:
        now = _now_ms()
        for key, cached in list(self._auth_token_cache.items()):
            if (
                cached["expiresAt"] is not None
                and now + TOKEN_REFRESH_BUFFER_MS >= cached["expiresAt"]
            ):
                del self._auth_token_cache[key]

    async def _wait_for_in_flight_token_refresh(self, cache_key: str) -> dict[str, Any] | None:
        lock = self._token_refresh_locks.get(cache_key)
        if lock is None:
            return None
        logger.debug("[HTTP Provider Auth]: Token refresh already in progress, waiting...")
        try:
            return await asyncio.shield(lock["promise"])
        except Exception:
            if self._token_refresh_locks.get(cache_key) is lock:
                del self._token_refresh_locks[cache_key]
            logger.debug("[HTTP Provider Auth]: Previous token refresh failed, retrying...")
        return None

    async def _refresh_token_with_lock(
        self,
        cache_key: str,
        refresh: Callable[[], Awaitable[dict[str, Any]]],
        on_lock_acquired: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        while cache_key in self._token_refresh_locks:
            refreshed = await self._wait_for_in_flight_token_refresh(cache_key)
            if refreshed:
                return refreshed
        task = asyncio.ensure_future(refresh())
        lock = {"promise": task}
        self._token_refresh_locks[cache_key] = lock
        if on_lock_acquired:
            on_lock_acquired()
        try:
            return await asyncio.shield(task)
        finally:
            if self._token_refresh_locks.get(cache_key) is lock:
                del self._token_refresh_locks[cache_key]

    async def _refresh_file_token_if_needed(
        self, prompt: str, vars: dict[str, Any], context: dict[str, Any] | None
    ) -> dict[str, Any]:
        auth = self.config.get("auth")
        if not auth or auth.get("type") != "file":
            raise ValueError("File auth is not configured")
        auth_context = {
            **(context or {}),
            "prompt": (context or {}).get("prompt") or {"raw": prompt, "label": prompt},
            "vars": vars,
        }
        cache_key = "file:" + digest_auth_cache_input(
            {"path": auth["path"], "context": _file_auth_context_cache_input(auth_context)}
        )
        cached = self._get_valid_cached_token(cache_key)
        if cached:
            logger.debug("[HTTP Provider Auth]: Using cached file auth token")
            return cached
        return await self._refresh_token_with_lock(
            cache_key,
            lambda: self._perform_file_token_refresh(auth_context, cache_key),
            lambda: logger.debug("[HTTP Provider Auth]: Acquired file auth refresh lock"),
        )

    async def _perform_file_token_refresh(
        self, auth_context: dict[str, Any], cache_key: str
    ) -> dict[str, Any]:
        auth = self.config["auth"]
        path: str = auth["path"]
        function_name: str | None = None
        if path.startswith("file://"):
            from promptphoo.files import parse_file_url

            path, function_name = parse_file_url(path)
        default_function_name = "get_auth" if path.endswith(".py") else "default"
        try:
            logger.debug(
                "[HTTP Provider Auth]: Invoking file auth function %s#%s",
                path,
                function_name or default_function_name,
            )
            if not path.endswith(".py"):
                raise ValueError(
                    f"JavaScript auth files are not supported by promptphoo: {auth['path']}. "
                    "Use a Python file (file://auth.py:get_auth) instead."
                )
            full = os.path.join(cli_state.base_path() or "", path)
            context_arg = {
                k: v
                for k, v in auth_context.items()
                if k not in ("getCache", "logger", "filters", "originalProvider")
            }
            raw = await python_hooks.run_python(
                full, function_name or default_function_name, [context_arg]
            )
            result = _parse_file_auth_result(raw)
            cached = self._cache_token(cache_key, result["token"], result.get("expiration"))
            logger.info(
                "[HTTP Provider Auth]: Successfully refreshed file auth token (%s)",
                _format_file_auth_freshness(result.get("expiration")),
            )
            return cached
        except Exception as err:
            logger.error("[HTTP Provider Auth]: Failed to refresh file auth token: Error: %s", err)
            raise ValueError(f"Failed to refresh file auth token: Error: {err}") from err

    # signature auth -------------------------------------------------------------------------------

    def _get_signature(self, vars: dict[str, Any]) -> dict[str, Any]:
        signature_auth = self.config["signatureAuth"]
        rendered_config = {
            "type": "pem",
            **signature_auth,
            "privateKey": _render(signature_auth["privateKey"], vars)
            if signature_auth.get("privateKey")
            else None,
        }
        cache_key = digest_auth_cache_input({"signature": rendered_config})
        cached = self._signature_cache.get(cache_key)
        if cached is not None:
            self._signature_cache.move_to_end(cache_key)
            if not _signature.needs_signature_refresh(
                cached["timestamp"],
                signature_auth.get("signatureValidityMs", 300000),
                signature_auth.get("signatureRefreshBufferMs"),
            ):
                return cached
        timestamp = _now_ms()
        task = asyncio.ensure_future(_signature.generate_signature(rendered_config, timestamp))
        entry = {"timestamp": timestamp, "signature": task}

        def on_done(t: asyncio.Future[str]) -> None:
            if not t.cancelled() and t.exception() is not None:
                if self._signature_cache.get(cache_key) is entry:
                    del self._signature_cache[cache_key]

        task.add_done_callback(on_done)
        self._signature_cache[cache_key] = entry
        self._signature_cache.move_to_end(cache_key)
        while len(self._signature_cache) > MAX_AUTH_CACHE_ENTRIES:
            self._signature_cache.popitem(last=False)
        return entry

    # session endpoint -----------------------------------------------------------------------------

    async def _resolve_session_id(self, vars: dict[str, Any]) -> str | None:
        if not self.config.get("session"):
            return None
        context_session_id = vars.get("sessionId")
        if context_session_id and context_session_id in self._fetched_sessions:
            logger.debug(
                "[HTTP Provider Session]: Reusing existing session from context: %s",
                context_session_id,
            )
            return context_session_id
        logger.debug("[HTTP Provider Session]: Fetching new session from endpoint")
        new_session_id = await self._fetch_session_from_endpoint(vars)
        self._fetched_sessions.add(new_session_id)
        return new_session_id

    async def _fetch_session_from_endpoint(self, vars: dict[str, Any]) -> str:
        session = self.config["session"]
        parser = await self._get_session_endpoint_parser()
        assert parser is not None
        url = _render(session["url"], vars)
        headers: dict[str, str] = {}
        for key, value in (session.get("headers") or {}).items():
            headers[key.lower()] = _render(value, vars)
        body: str | None = None
        if session.get("body") and session.get("method") != "GET":
            if isinstance(session["body"], str):
                body = _render(session["body"], vars)
            else:
                body = json_stringify(render_vars_in_object(session["body"], vars))
                headers.setdefault("content-type", "application/json")
        method = session.get("method") or "POST"
        logger.debug(
            "[HTTP Provider Session]: Calling session endpoint %s %s", method, sanitize_url(url)
        )
        fetch_options: dict[str, Any] = {"method": method, "headers": headers}
        if body:
            fetch_options["body"] = body
        agent = await self._get_https_agent()
        if agent is not None:
            fetch_options["dispatcher"] = agent
        response = await _fetch.fetch_with_cache(
            url,
            fetch_options,
            get_request_timeout_ms(),
            "text",
            True,
            self.config.get("maxRetries"),
        )
        if response.status < 200 or response.status >= 300:
            raise ValueError(
                f"Session endpoint request failed with status {response.status} "
                f"{response.status_text}: {response.data}"
            )
        raw_text = response.data
        try:
            parsed_data = js_json_parse(raw_text)
        except (ValueError, TypeError):
            parsed_data = None
        session_id = await parser(
            {
                "headers": response.headers,
                "body": parsed_data if parsed_data is not None else raw_text,
            }
        )
        if not is_truthy(session_id):
            shown = parsed_data if parsed_data is not None else raw_text
            raise ValueError(
                "Session endpoint did not return a session ID. Response: "
                f"{safe_json_stringify(sanitize_object(shown, context='session response'))}"
            )
        return session_id

    # TLS ---------------------------------------------------------------------------------

    async def _get_https_agent(self) -> Any:
        if not self.config.get("tls"):
            return None
        if self._https_agent is not None:
            return self._https_agent
        async with self._https_agent_lock:
            if self._https_agent is None:
                self._https_agent = await asyncio.to_thread(create_https_agent, self.config["tls"])
                logger.debug("[HTTP Provider] HTTPS agent created successfully")
        return self._https_agent

    async def cleanup(self) -> None:
        if self._https_agent is not None:
            await self._https_agent.aclose()
            self._https_agent = None

    # headers -----------------------------------------------------------------------------

    def get_default_headers(self, body: Any) -> dict[str, str]:
        if self.config.get("method") == "GET":
            return {}
        if self.config.get("multipart"):
            return {}
        if isinstance(body, dict | list):
            return {"content-type": "application/json"}
        if isinstance(body, str):
            return {"content-type": "application/x-www-form-urlencoded"}
        return {}

    def validate_content_type_and_body(self, headers: dict[str, str], body: Any) -> None:
        if body is None:
            return
        if isinstance(body, dict | list) and not content_type_is_json(headers):
            raise ValueError(
                "Content-Type is not application/json, but body is an object or array. The body "
                "must be a string if the Content-Type is not application/json."
            )
        if isinstance(body, str) and content_type_is_json(headers):
            try:
                js_json_parse(body)
            except ValueError:
                logger.warning(
                    "[HTTP Provider] Content-Type is application/json, but body is a string. "
                    "This is likely to cause unexpected results. It should be an object or "
                    "array. Body: %s headers: %s",
                    body,
                    safe_json_stringify(headers),
                )

    def _apply_auth_headers(self, headers: dict[str, str], vars: dict[str, Any]) -> None:
        auth = self.config.get("auth") or {}
        auth_type = auth.get("type")
        if auth_type == "oauth" and isinstance(vars.get("token"), str):
            headers["authorization"] = f"Bearer {vars['token']}"
        if auth_type == "bearer":
            headers["authorization"] = f"Bearer {_render(auth.get('token'), vars)}"
        if auth_type == "basic":
            username = _render(auth.get("username"), vars)
            password = _render(auth.get("password"), vars)
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["authorization"] = f"Basic {credentials}"
        if auth_type == "api_key" and auth.get("placement") == "header":
            key_name = _render(auth.get("keyName"), vars)
            headers[key_name.lower()] = _render(auth.get("value"), vars)

    async def get_headers(
        self, default_headers: dict[str, str], vars: dict[str, Any]
    ) -> dict[str, str]:
        config_headers = {k.lower(): v for k, v in (self.config.get("headers") or {}).items()}
        all_headers = {
            key: _render(value, vars)
            for key, value in {**default_headers, **config_headers}.items()
        }
        self._apply_auth_headers(all_headers, vars)
        return all_headers

    # calls ----------------------------------------------------------------------------------------

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        context = context or {}
        prompt_config = (context.get("prompt") or {}).get("config") or {}
        raw_tools = prompt_config.get("tools")
        if raw_tools is None:
            raw_tools = self.config.get("tools")
        raw_tool_choice = prompt_config.get("tool_choice")
        if raw_tool_choice is None:
            raw_tool_choice = self.config.get("tool_choice")
        fmt = self.config.get("transformToolsFormat")
        tools = transform_tools(raw_tools, fmt) if fmt else raw_tools
        tool_choice = transform_tool_choice(raw_tool_choice, fmt) if fmt else raw_tool_choice
        if is_truthy(tool_choice) and (
            not is_truthy(tools) or (isinstance(tools, list) and len(tools) == 0)
        ):
            logger.warning(
                "[HTTP Provider]: tool_choice is set but tools is empty or undefined. This may "
                "cause API errors."
            )

        def serialize_for_template(value: Any) -> Any:
            return json_stringify(value) if isinstance(value, dict | list) else value

        vars: dict[str, Any] = {**(context.get("vars") or {}), "prompt": prompt}
        if context.get("evaluationId"):
            vars["evaluationId"] = context["evaluationId"]
        if tools is not None:
            vars["tools"] = serialize_for_template(tools)
        if tool_choice is not None:
            vars["tool_choice"] = serialize_for_template(tool_choice)

        auth = self.config.get("auth") or {}
        if auth.get("type") == "oauth":
            token = await self._refresh_oauth_token_if_needed(vars)
            if "token" in vars:
                logger.warning(
                    "[HTTP Provider Auth]: `token` is already defined in vars and will be "
                    "overwritten"
                )
            vars["token"] = token["token"]
        elif auth.get("type") == "file":
            token = await self._refresh_file_token_if_needed(prompt, vars, context)
            if "token" in vars:
                logger.warning(
                    "[HTTP Provider Auth]: `token` is already defined in vars and will be "
                    "overwritten"
                )
            if "expiration" in vars:
                logger.warning(
                    "[HTTP Provider Auth]: `expiration` is already defined in vars and will be "
                    "overwritten"
                )
            vars["token"] = token["token"]
            vars["expiration"] = token["expiresAt"]

        if self.config.get("signatureAuth"):
            pending = self._get_signature(vars)
            signature = await asyncio.shield(pending["signature"])
            for key in ("signature", "signatureTimestamp"):
                if is_truthy(vars.get(key)):
                    logger.warning(
                        "[HTTP Provider Auth]: `%s` is already defined in vars and will be "
                        "overwritten",
                        key,
                    )
            vars["signature"] = signature
            vars["signatureTimestamp"] = pending["timestamp"]

        if self.config.get("session"):
            resolved = await self._resolve_session_id(vars)
            if resolved:
                vars["sessionId"] = resolved

        if self.config.get("request"):
            return await self._call_api_with_raw_request(vars, context, options)

        body_config = self.config.get("body")
        headers = await self.get_headers(self.get_default_headers(body_config), vars)
        if context.get("traceparent"):
            headers["traceparent"] = context["traceparent"]
        if context.get("tracestate"):
            headers["tracestate"] = context["tracestate"]

        multipart = self.config.get("multipart")
        if multipart:
            _remove_multipart_content_type(headers)
        else:
            self.validate_content_type_and_body(headers, body_config)

        transformed_prompt = await (await self._get_transform_request())(prompt, vars, context)
        log_transformed_prompt(transformed_prompt, prompt, headers)

        url_template = self.url
        method_template = self.config.get("method") or ("POST" if multipart else "GET")
        rendered_url = _render(url_template, vars)
        rendered_method = _render(method_template, vars)
        rendered_body = (
            None
            if multipart
            else determine_request_body(
                content_type_is_json(headers), transformed_prompt, body_config, vars
            )
        )
        query_params = {
            key: _render(value, vars)
            for key, value in (self.config.get("queryParams") or {}).items()
        }
        if auth.get("type") == "api_key" and auth.get("placement") == "query":
            query_params[_render(auth.get("keyName"), vars)] = _render(auth.get("value"), vars)

        method = rendered_method or "POST"
        if multipart and method.upper() in ("GET", "HEAD"):
            raise ValueError(f"HTTP provider {method} requests cannot use multipart")

        url = rendered_url
        if query_params:
            try:
                url_obj = URL(url)
                for key, value in query_params.items():
                    url_obj.search_params_append(key, value)
                url = url_obj.href
            except InvalidUrlError as err:
                logger.warning(
                    "[HTTP Provider]: Failed to construct URL object: TypeError: %s", err
                )
                query_string = form_urlencoded_serialize([[k, v] for k, v in query_params.items()])
                url = f"{url}{'&' if '?' in url else '?'}{query_string}"

        logger.debug("[HTTP Provider]: Calling %s with config.", sanitize_url(url))

        multipart_body = (
            render_http_multipart_body(multipart, {**vars, "prompt": transformed_prompt})
            if multipart
            else None
        )

        fetch_options: dict[str, Any] = {"method": rendered_method, "headers": headers}
        if method != "GET" and multipart_body:
            fetch_options["body"] = multipart_body["body"].content
            fetch_options["headers"] = {
                **headers,
                "content-type": multipart_body["body"].content_type,
            }
        elif method != "GET" and rendered_body is not None:
            if content_type_is_json(headers):
                fetch_options["body"] = (
                    rendered_body
                    if isinstance(rendered_body, str)
                    else json_stringify(rendered_body)
                )
            else:
                fetch_options["body"] = (
                    rendered_body.strip()
                    if isinstance(rendered_body, str)
                    else js_string(rendered_body)
                )
        agent = await self._get_https_agent()
        if agent is not None:
            fetch_options["dispatcher"] = agent
            logger.debug("[HTTP Provider]: Using custom HTTPS agent for TLS connection")

        bust = True if multipart_body else _bust(context)
        response = await _fetch.fetch_with_cache(
            url,
            fetch_options,
            get_request_timeout_ms(),
            "text",
            bust,
            self.config.get("maxRetries"),
        )
        data, cached, status = response.data, response.cached, response.status
        status_text, response_headers = response.status_text, response.headers
        latency_ms = response.latency_ms

        if not is_truthy(await (await self._get_validate_status())(status)):
            raise ValueError(f"HTTP call failed with status {status} {status_text}: {data}")

        http_meta: dict[str, Any] = {
            "status": status,
            "statusText": status_text,
            "headers": sanitize_object(response_headers, context="response headers"),
        }
        if context.get("debug"):
            http_meta["requestHeaders"] = sanitize_object(headers, context="request headers")
        ret: dict[str, Any] = {"raw": data}
        if latency_ms is not None:
            ret["latencyMs"] = latency_ms
        ret["cached"] = cached
        ret["metadata"] = {"http": http_meta}
        if context.get("debug"):
            ret["metadata"]["transformedRequest"] = sanitize_transformed_request_for_metadata(
                transformed_prompt, headers
            )
            if multipart_body:
                ret["metadata"]["multipart"] = {
                    "fields": sanitize_multipart_fields(multipart_body["fields"]),
                    "files": sanitize_multipart_files(multipart_body["files"]),
                }
            else:
                ret["metadata"]["finalRequestBody"] = sanitize_request_body_for_metadata(
                    rendered_body, headers
                )

        return await self._finish_response(ret, vars, data, response, transformed_prompt, prompt)

    async def _finish_response(
        self,
        ret: dict[str, Any],
        vars: dict[str, Any],
        data: Any,
        response: Any,
        transformed_prompt: Any,
        prompt: str,
    ) -> ProviderResponse:
        raw_text = data
        try:
            parsed_data = js_json_parse(raw_text) if isinstance(raw_text, str) else None
        except ValueError:
            parsed_data = None

        if vars.get("sessionId") and self.config.get("session"):
            ret["sessionId"] = vars["sessionId"]

        response_headers = response.headers
        try:
            session_id = await (await self._get_session_parser())(
                {
                    "headers": response_headers,
                    "body": parsed_data if parsed_data is not None else raw_text,
                }
            )
            if is_truthy(session_id):
                ret["sessionId"] = session_id
        except Exception as err:
            logger.error(
                "Error parsing session ID: Error: %s. Got headers: %s and parsed body: %s",
                err,
                safe_json_stringify(sanitize_object(response_headers, context="response headers")),
                safe_json_stringify(sanitize_object(parsed_data, context="response body")),
            )
            raise

        transform_context: dict[str, Any] = {
            "response": {
                "data": data,
                "status": response.status,
                "statusText": response.status_text,
                "headers": response_headers,
                "cached": response.cached,
            }
        }
        if response.latency_ms is not None:
            transform_context["response"]["latencyMs"] = response.latency_ms
        parsed_output = await (await self._get_transform_response())(
            parsed_data, raw_text, transform_context
        )
        return self._process_response_with_token_estimation(
            ret, parsed_output, raw_text, transformed_prompt, prompt, parsed_data
        )

    async def _call_api_with_raw_request(
        self,
        vars: dict[str, Any],
        context: dict[str, Any],
        options: dict[str, Any] | None,
    ) -> ProviderResponse:
        request_template = self.config["request"]
        prompt = vars.get("prompt")
        transformed_prompt = await (await self._get_transform_request())(prompt, vars, context)
        escaped_vars = escape_json_variables({**vars, "prompt": transformed_prompt})
        rendered_request = render_raw_request_with_nunjucks(request_template, escaped_vars)
        parsed_request = parse_raw_request(rendered_request.strip())
        log_transformed_prompt(transformed_prompt, prompt, parsed_request["headers"])

        protocol = (
            "https" if self.url.startswith("https") or self.config.get("useHttps") else "http"
        )
        host = parsed_request["headers"].get("host")
        url = URL(
            parsed_request["url"],
            f"{protocol}://{js_string(host) if host is not None else 'undefined'}",
        ).href

        parsed_request["headers"].pop("content-length", None)
        if context.get("traceparent"):
            parsed_request["headers"]["traceparent"] = context["traceparent"]
        if context.get("tracestate"):
            parsed_request["headers"]["tracestate"] = context["tracestate"]
        self._apply_auth_headers(parsed_request["headers"], vars)

        auth = self.config.get("auth") or {}
        if auth.get("type") == "api_key" and auth.get("placement") == "query":
            try:
                key_name = _render(auth.get("keyName"), vars)
                value = _render(auth.get("value"), vars)
                url_obj = URL(url)
                url_obj.search_params_append(key_name, value)
                url = url_obj.href
                url_path = url_obj.pathname + url_obj.search
                request_lines = rendered_request.split("\n")
                first_line = request_lines[0]
                req_method = first_line.split(" ")[0]
                req_protocol = first_line.split(" ")[-1]
                request_lines[0] = f"{req_method} {url_path} {req_protocol}"
                parsed_request.update(parse_raw_request("\n".join(request_lines).strip()))
            except Exception as err:
                logger.warning(
                    "[HTTP Provider]: Failed to add API key to query params in raw request: %s", err
                )

        body_content: str | None = None
        parsed_body = parsed_request.get("body") or {}
        if parsed_body.get("text"):
            body_content = parsed_body["text"].strip()
        elif parsed_body.get("params") is not None:
            body_content = extract_body_from_raw_request(rendered_request)

        logger.debug(
            "[HTTP Provider]: Calling %s with raw request: %s",
            sanitize_url(url),
            parsed_request["method"],
        )

        fetch_options: dict[str, Any] = {
            "method": parsed_request["method"],
            "headers": parsed_request["headers"],
        }
        if body_content:
            fetch_options["body"] = body_content
        agent = await self._get_https_agent()
        if agent is not None:
            fetch_options["dispatcher"] = agent

        response = await _fetch.fetch_with_cache(
            url,
            fetch_options,
            get_request_timeout_ms(),
            "text",
            _bust(context),
            self.config.get("maxRetries"),
        )
        data, status, status_text = response.data, response.status, response.status_text
        if not is_truthy(await (await self._get_validate_status())(status)):
            raise ValueError(f"HTTP call failed with status {status} {status_text}: {data}")

        ret: dict[str, Any] = {}
        if response.latency_ms is not None:
            ret["latencyMs"] = response.latency_ms
        ret["cached"] = response.cached

        # sessionId / sessionParser are applied before the metadata in raw mode.
        raw_text = data
        try:
            parsed_data = js_json_parse(raw_text) if isinstance(raw_text, str) else None
        except ValueError:
            parsed_data = None
        if vars.get("sessionId") and self.config.get("session"):
            ret["sessionId"] = vars["sessionId"]
        try:
            session_id = await (await self._get_session_parser())(
                {
                    "headers": response.headers,
                    "body": parsed_data if parsed_data is not None else raw_text,
                }
            )
            if is_truthy(session_id):
                ret["sessionId"] = session_id
        except Exception as err:
            logger.error(
                "Error parsing session ID: Error: %s. Got headers: %s and parsed body: %s",
                err,
                safe_json_stringify(sanitize_object(response.headers, context="response headers")),
                safe_json_stringify(sanitize_object(parsed_data, context="response body")),
            )
            raise

        ret["metadata"] = {"http": {"status": status, "statusText": status_text}}
        if context.get("debug"):
            ret["raw"] = data
            headers_meta = sanitize_object(response.headers, context="response headers")
            has_transform = bool(self.config.get("transformRequest"))
            ret["metadata"] = {
                **ret["metadata"],
                "headers": headers_meta,
                "transformedRequest": sanitize_transformed_request_for_metadata(
                    transformed_prompt, parsed_request["headers"]
                )
                if has_transform
                else format_raw_request_for_debug_metadata(parsed_request, body_content),
                "finalRequestBody": sanitize_transformed_request_for_metadata(
                    body_content, parsed_request["headers"]
                )
                if has_transform
                else sanitize_request_body_for_metadata(body_content, parsed_request["headers"]),
                "http": {
                    "status": status,
                    "statusText": status_text,
                    "headers": headers_meta,
                    "requestHeaders": sanitize_object(
                        parsed_request["headers"], context="request headers"
                    ),
                },
            }

        transform_context: dict[str, Any] = {
            "response": {
                "data": data,
                "status": status,
                "statusText": status_text,
                "headers": response.headers,
                "cached": response.cached,
            }
        }
        if response.latency_ms is not None:
            transform_context["response"]["latencyMs"] = response.latency_ms
        parsed_output = await (await self._get_transform_response())(
            parsed_data, raw_text, transform_context
        )
        return self._process_response_with_token_estimation(
            ret, parsed_output, raw_text, transformed_prompt, prompt, parsed_data
        )

    # response post-processing ------------------------------------------------------------------

    @staticmethod
    def _get_completion_text(parsed_output: Any, raw_text: Any) -> Any:
        if isinstance(parsed_output, str):
            return parsed_output
        if (
            isinstance(parsed_output, dict)
            and isinstance(parsed_output.get("output"), str)
            and parsed_output["output"]
        ):
            return parsed_output["output"]
        return raw_text

    def _process_response_with_token_estimation(
        self,
        ret: dict[str, Any],
        parsed_output: Any,
        raw_text: Any,
        transformed_prompt: Any,
        prompt: Any,
        original_response: Any = None,
    ) -> ProviderResponse:
        original_token_usage = (
            original_response["tokenUsage"]
            if isinstance(original_response, dict)
            and isinstance(original_response.get("tokenUsage"), dict)
            else None
        )
        estimated: dict[str, Any] | None = None
        if (self.config.get("tokenEstimation") or {}).get("enabled"):
            prompt_text = transformed_prompt if isinstance(transformed_prompt, str) else prompt
            estimated = self._estimate_token_usage(
                prompt_text, self._get_completion_text(parsed_output, raw_text)
            )
        result = {**ret, **normalize_response_transform_result(parsed_output)}
        if not is_truthy(result.get("tokenUsage")):
            if original_token_usage:
                result["tokenUsage"] = original_token_usage
            elif estimated:
                result["tokenUsage"] = estimated
            else:
                result["tokenUsage"] = {**create_empty_token_usage(), "numRequests": 1}
        return result


def _bust(context: dict[str, Any]) -> bool:
    value = context.get("bustCache")
    if value is None:
        value = context.get("debug")
    return is_truthy(value)


def create_provider(path: str, options: dict[str, Any]) -> HttpProvider:
    """Registry factory for `http:` / `https:` provider ids."""
    return HttpProvider(path, options)
