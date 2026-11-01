"""Response cache and fetchWithCache. Ports src/cache.ts.

Node's KeyvFile cache (<configDir>/cache/cache.json) is rewritten whole without locking, so
sharing it between runtimes would lose writes. promptphoo keeps the same envelope format
({"value", "expires"} per key, absolute ms expiry) in its own file, cache-py.json.

`fetch_with_cache` lives here as it does upstream, and `promptphoo.fetch` re-exports it. Not
ported: folding Cloud auth headers into fetch cache keys, and abort signals (cancel the awaiting
task instead).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import shutil
import time
import weakref
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from promptphoo import envars
from promptphoo.jsvalues import json_stringify, number_to_string
from promptphoo.paths import get_config_directory_path
from promptphoo.private_files import make_private_dirs, open_private, restrict

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL_SECONDS = 60 * 60 * 24 * 14
CACHE_FILE_NAME = "cache-py.json"

_enabled = envars.get_env_bool("PROMPTFOO_CACHE_ENABLED", True)
_enabled_override: ContextVar[bool | None] = ContextVar("cache_enabled", default=None)
_namespace: ContextVar[str | None] = ContextVar("cache_namespace", default=None)
_cache_clear_generation = 0


def get_cache_ttl_ms() -> int:
    return envars.get_env_int("PROMPTFOO_CACHE_TTL", DEFAULT_CACHE_TTL_SECONDS) * 1000


def _cache_type() -> str:
    return envars.get_env_string("PROMPTFOO_CACHE_TYPE") or (
        "memory" if envars.get_env_string("NODE_ENV") == "test" else "disk"
    )


def _cache_path() -> str:
    return envars.get_env_string("PROMPTFOO_CACHE_PATH") or os.path.join(
        get_config_directory_path(), "cache"
    )


class Cache:
    """In-memory map with optional JSON-file persistence."""

    def __init__(self, path: str | None, ttl_ms: int) -> None:
        self.path = path
        self.ttl_ms = ttl_ms
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._lock = asyncio.Lock()
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("[Cache] Ignoring unreadable cache file %s: %s", path, e)

    async def get(self, key: str) -> Any:
        entry = self._data.get("keyv:" + key)
        if entry is None:
            return None
        expires = entry.get("expires")
        if expires is not None and expires < time.time() * 1000:
            self._data.pop("keyv:" + key, None)
            return None
        return entry.get("value")

    async def set(self, key: str, value: Any, ttl_ms: int | None = None) -> None:
        ttl = self.ttl_ms if ttl_ms is None else ttl_ms
        self._data["keyv:" + key] = {
            "value": value,
            "expires": int(time.time() * 1000 + ttl) if ttl else None,
        }
        self._dirty = True
        await self.flush()

    async def delete(self, key: str) -> None:
        if self._data.pop("keyv:" + key, None) is not None:
            self._dirty = True
            await self.flush()

    async def delete_many(self, keys: list[str]) -> None:
        removed = [self._data.pop("keyv:" + key, None) for key in keys]
        if any(entry is not None for entry in removed):
            self._dirty = True
            await self.flush()

    def stored_keys(self) -> list[str]:
        return [k[len("keyv:") :] for k in self._data if k.startswith("keyv:")]

    async def clear(self) -> None:
        global _cache_clear_generation
        self._data.clear()
        self._dirty = True
        await self.flush()
        _cache_clear_generation += 1

    async def flush(self) -> None:
        if not self.path or not self._dirty:
            return
        async with self._lock:
            tmp = self.path + ".tmp"
            with open_private(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            restrict(tmp)  # a leftover .tmp from an older run keeps its mode on reopen
            os.replace(tmp, self.path)
            self._dirty = False


class _NamespacedCache:
    def __init__(self, cache: Cache, namespace: str) -> None:
        self._cache = cache
        self._ns = namespace

    async def get(self, key: str) -> Any:
        return await self._cache.get(f"{self._ns}:{key}")

    async def set(self, key: str, value: Any, ttl_ms: int | None = None) -> None:
        await self._cache.set(f"{self._ns}:{key}", value, ttl_ms)

    async def delete(self, key: str) -> None:
        await self._cache.delete(f"{self._ns}:{key}")

    async def clear(self) -> None:
        """Clear only this namespace's keys (clearNamespacedCache)."""
        global _cache_clear_generation
        prefix = f"{self._ns}:"
        keys = [k for k in self._cache.stored_keys() if k.startswith(prefix)]
        if keys:
            try:
                await self._cache.delete_many(keys)
            except Exception as err:
                raise RuntimeError(
                    f'[Cache] Failed to clear {len(keys)} keys for namespace "{self._ns}": {err}'
                ) from err
        _cache_clear_generation += 1


_instance: Cache | None = None


def _get_instance() -> Cache:
    global _instance
    if _instance is None:
        path = None
        if _cache_type() == "disk" and _enabled:
            directory = _cache_path()
            try:
                if not os.path.exists(directory):
                    logger.info(f"Creating cache folder at {directory}.")
                    make_private_dirs(directory)
                path = os.path.join(directory, CACHE_FILE_NAME)
            except OSError as err:
                logger.warning(
                    f"[Cache] Failed to initialize disk cache: {err}. Using memory cache instead."
                )
        _instance = Cache(path, get_cache_ttl_ms())
    return _instance


def get_cache() -> Cache | _NamespacedCache:
    ns = _namespace.get()
    return _NamespacedCache(_get_instance(), ns) if ns else _get_instance()


def get_scoped_cache_key(key: str, namespace: str | None = None) -> str:
    namespace = _namespace.get() if namespace is None else namespace
    return f"{namespace}:{key}" if namespace else key


def get_cache_clear_generation() -> int:
    return _cache_clear_generation


def _current_namespace_includes_repeat_index(repeat_index: Any) -> bool:
    namespace = _namespace.get()
    parts = namespace.split(":") if namespace else []
    return any(
        part == "repeat" and i + 1 < len(parts) and parts[i + 1] == number_to_string(repeat_index)
        for i, part in enumerate(parts)
    )


def _should_apply_repeat_cache_suffix(repeat_index: Any) -> bool:
    return (
        isinstance(repeat_index, int | float)
        and not isinstance(repeat_index, bool)
        and repeat_index > 0
        and not _current_namespace_includes_repeat_index(repeat_index)
    )


def _repeat_suffix(repeat_index: Any) -> str:
    if not _should_apply_repeat_cache_suffix(repeat_index):
        return ""
    return f":repeat{number_to_string(repeat_index)}"


def is_cache_enabled() -> bool:
    override = _enabled_override.get()
    return _enabled if override is None else override


def enable_cache() -> None:
    global _enabled
    _enabled = True


def disable_cache() -> None:
    global _enabled
    _enabled = False


_claimed_cache_keys: set[str] = set()


def claim_cache_key_once(cache_key: str) -> bool:
    """Atomically claim a cache-scoped one-time action. Disk-backed claims use an exclusive file
    so separate eval processes cannot both attribute the same background response's usage."""
    scoped = get_scoped_cache_key(cache_key)
    if scoped in _claimed_cache_keys:
        return False
    if _cache_type() == "disk" and is_cache_enabled():
        claims_path = os.path.join(_cache_path(), "claims")
        try:
            os.makedirs(claims_path, exist_ok=True)
            name = hashlib.sha256(scoped.encode("utf-8")).hexdigest()
            fd = os.open(os.path.join(claims_path, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            _claimed_cache_keys.add(scoped)
            return False
        except OSError as error:
            logger.warning(
                f"[Cache] Failed to persist a one-time cache claim: {error}. "
                "Using a process-local claim instead."
            )
    _claimed_cache_keys.add(scoped)
    return True


async def clear_cache() -> None:
    """Clear all cached results, in-flight fetch coalescing and one-time claims."""
    _inflight_fetch_responses.clear()
    await _get_instance().clear()
    _claimed_cache_keys.clear()
    if _cache_type() == "disk":
        shutil.rmtree(os.path.join(_cache_path(), "claims"), ignore_errors=True)


@contextmanager
def with_cache_enabled(enabled: bool | None) -> Iterator[None]:
    if enabled is None:
        yield
        return
    token = _enabled_override.set(enabled)
    try:
        yield
    finally:
        _enabled_override.reset(token)


@contextmanager
def with_cache_namespace(namespace: str | None) -> Iterator[None]:
    parent = _namespace.get()
    if not namespace or parent == namespace:
        yield
        return
    token = _namespace.set(f"{parent}:{namespace}" if parent else namespace)
    try:
        yield
    finally:
        _namespace.reset(token)


# ---- fetchWithCache ----------------------------------------------------------------------------


@dataclass
class FetchWithCacheResult:
    data: Any
    cached: bool
    status: int
    status_text: str
    headers: dict[str, str] = field(default_factory=dict)
    latency_ms: int | None = None
    delete_from_cache: Callable[[], Awaitable[None]] | None = None
    # Another concurrent caller owns the upstream request that produced this response.
    coalesced: bool | None = None
    # (data, status, status_text, headers=None) -> replaces the cached entry.
    update_cache: Callable[..., Awaitable[None]] | None = None


SanitizeResponse = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class _PreparedFetchResponse:
    response: str
    cacheable: bool


@dataclass
class _InflightFetchResponse:
    response: asyncio.Task[_PreparedFetchResponse]
    publication: asyncio.Task[None]
    observers: set[Callable[[dict[str, Any]], None]] = field(default_factory=set)
    latest: dict[str, Any] | None = None


_inflight_fetch_responses: dict[str, _InflightFetchResponse] = {}
_IGNORED_FETCH_CACHE_OPTION_KEYS = frozenset({"method", "signal"})
_IGNORED_FETCH_CACHE_HEADERS = frozenset({"traceparent", "tracestate"})
_FETCH_CACHE_SECRET_HMAC_CONTEXT = b"promptfoo:fetch-cache-secret-key"
# A fixed, compiled-in salt (NOT a secret): it keeps secret-bearing cache keys stable across
# processes while never writing the raw secret into the key.
_FETCH_CACHE_SECRET_HMAC_SALT = b"promptfoo:fetch-cache-secret-hmac-salt:v1"
_abort_signal_ids: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()
_next_abort_signal_id = 0


def _fingerprint_fetch_cache_secret(value: str) -> dict[str, str]:
    digest = hmac.new(
        _FETCH_CACHE_SECRET_HMAC_SALT,
        _FETCH_CACHE_SECRET_HMAC_CONTEXT + b"\0" + value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {"__promptfooSecretFingerprint": digest}


def _is_sensitive_fetch_cache_string(value: str, field_name: str | None = None) -> bool:
    from promptphoo.sanitizer import is_secret_field, looks_like_secret

    return bool(field_name and is_secret_field(field_name)) or looks_like_secret(value)


def _get_string_for_fetch_cache_key(value: str, field_name: str | None = None) -> Any:
    if _is_sensitive_fetch_cache_string(value, field_name):
        return _fingerprint_fetch_cache_secret(value)
    return value


def _has_sensitive_json_value(value: Any, field_name: str | None = None) -> bool:
    if isinstance(value, str):
        return _is_sensitive_fetch_cache_string(value, field_name)
    if isinstance(value, list):
        return any(_has_sensitive_json_value(item, field_name) for item in value)
    if isinstance(value, dict):
        return any(_has_sensitive_json_value(v, k) for k, v in value.items())
    return False


def _get_json_value_for_fetch_cache_key(value: Any, field_name: str | None = None) -> Any:
    if isinstance(value, str):
        return _get_string_for_fetch_cache_key(value, field_name)
    if isinstance(value, list):
        return [_get_json_value_for_fetch_cache_key(item, field_name) for item in value]
    if isinstance(value, dict):
        return {k: _get_json_value_for_fetch_cache_key(v, k) for k, v in value.items()}
    return value


def _json_parse(text: str) -> Any:
    """JSON.parse: rejects NaN/Infinity, which Python's json accepts."""

    def reject(name: str) -> Any:
        raise ValueError(f"Unexpected token {name}")

    return json.loads(text, parse_constant=reject)


def _get_body_string_for_fetch_cache_key(value: str) -> Any:
    try:
        parsed = _json_parse(value)
    except ValueError:
        return _get_string_for_fetch_cache_key(value)
    if _has_sensitive_json_value(parsed):
        return {"encoding": "json", "value": _get_json_value_for_fetch_cache_key(parsed)}
    return value


def _has_sensitive_search_param(entries: list[tuple[str, str]]) -> bool:
    return any(_is_sensitive_fetch_cache_string(value, name) for name, value in entries)


def _get_search_params_for_fetch_cache_key(entries: list[tuple[str, str]]) -> Any:
    from promptphoo.whatwg_url import form_urlencoded_serialize

    if not _has_sensitive_search_param(entries):
        return form_urlencoded_serialize([[k, v] for k, v in entries])
    return [[name, _get_string_for_fetch_cache_key(value, name)] for name, value in entries]


def _get_url_for_fetch_cache_key(url: str) -> Any:
    from promptphoo.whatwg_url import URL

    try:
        parsed = URL(url)
    except Exception:
        return _get_string_for_fetch_cache_key(url)
    entries = parsed.search_params_entries()
    if not _has_sensitive_search_param(entries):
        return url
    parsed.search = ""
    return {"href": parsed.href, "searchParams": _get_search_params_for_fetch_cache_key(entries)}


def get_headers_for_cache_key(url: str, options: dict[str, Any]) -> list[list[Any]]:
    """Request headers as sent (lowercased, version header included), minus trace context,
    sorted, with credential-looking values replaced by HMAC fingerprints."""
    from promptphoo.fetch import get_fetch_with_proxy_headers

    headers: dict[str, str] = {}
    for name, value in get_fetch_with_proxy_headers(url, options).items():
        lower = name.lower()
        headers[lower] = f"{headers[lower]}, {value}" if lower in headers else str(value)
    entries = sorted(
        (name, value) for name, value in headers.items() if name not in _IGNORED_FETCH_CACHE_HEADERS
    )
    return [[name, _get_string_for_fetch_cache_key(value, name)] for name, value in entries]


def _hash_bytes_for_cache_key(data: bytes) -> dict[str, Any]:
    digest = hmac.new(
        _FETCH_CACHE_SECRET_HMAC_SALT,
        _FETCH_CACHE_SECRET_HMAC_CONTEXT + b":bytes\0" + data,
        hashlib.sha256,
    ).hexdigest()
    return {"byteLength": len(data), "hmacSha256": digest}


def _get_body_for_fetch_cache_key(body: Any) -> tuple[bool, Any]:
    """(cacheable, identity). A missing body has no identity; opaque bodies are uncacheable."""
    if body is None:
        return True, None
    if isinstance(body, str):
        return True, {"type": "string", "value": _get_body_string_for_fetch_cache_key(body)}
    if isinstance(body, bytes | bytearray | memoryview):
        return True, {"type": "Uint8Array", **_hash_bytes_for_cache_key(bytes(body))}
    return False, None


def _get_options_for_fetch_cache_key(
    options: dict[str, Any], body_identity: Any
) -> tuple[bool, dict[str, Any] | None]:
    identity: dict[str, Any] = {}
    for key, value in sorted(options.items()):
        if key == "headers" or key in _IGNORED_FETCH_CACHE_OPTION_KEYS:
            continue
        if key == "body":
            if body_identity is not None:
                identity["body"] = body_identity
            continue
        if value is None or isinstance(value, bool | int | float | str):
            identity[key] = value
            continue
        return False, None
    if "body" not in options and body_identity is not None:
        identity["body"] = body_identity
    return True, identity


def _get_fetch_cache_key(
    url: str,
    options: dict[str, Any],
    method: str,
    format: str,
    repeat_index: Any = None,
) -> str | None:
    body_cacheable, body_identity = _get_body_for_fetch_cache_key(options.get("body"))
    if not body_cacheable:
        return None
    options_cacheable, options_identity = _get_options_for_fetch_cache_key(options, body_identity)
    if not options_cacheable:
        return None
    identity = {
        "format": format,
        "headers": get_headers_for_cache_key(url, options),
        "method": method,
        "options": options_identity,
        "url": _get_url_for_fetch_cache_key(url),
    }
    digest = hashlib.sha256(json_stringify(identity).encode("utf-8")).hexdigest()
    return get_scoped_cache_key(f"fetch:v3:{digest}{_repeat_suffix(repeat_index)}")


def get_abort_signal_scoped_key(key: str, signal: Any = None) -> str:
    """Key in-flight requests by cancellation scope so only callers cancelled together share."""
    global _next_abort_signal_id
    if signal is None:
        return json_stringify([key, None])
    signal_id = _abort_signal_ids.get(signal)
    if signal_id is None:
        _next_abort_signal_id += 1
        signal_id = _next_abort_signal_id
        _abort_signal_ids[signal] = signal_id
    return json_stringify([key, signal_id])


def _get_sanitized_response(
    data: Any,
    status_text: str,
    headers: dict[str, str] | None,
    sanitize_response: SanitizeResponse | None,
) -> dict[str, Any]:
    if sanitize_response:
        return sanitize_response(
            {"data": data, "statusText": status_text, "headers": dict(headers or {})}
        )
    return {"data": data, "statusText": status_text, "headers": headers}


def _serialize_fetch_response(
    data: Any,
    status: int,
    status_text: str,
    headers: dict[str, str] | None,
    latency_ms: int | None,
    sanitize_response: SanitizeResponse | None = None,
) -> str:
    sanitized = _get_sanitized_response(data, status_text, headers, sanitize_response)
    return json_stringify(
        {
            "data": sanitized.get("data"),
            "status": status,
            "statusText": sanitized.get("statusText"),
            "headers": sanitized.get("headers"),
            "latencyMs": latency_ms,
        }
    )


def _deserialize_fetch_response(
    response: Any,
    cached: bool,
    cache: Cache,
    cache_key: str,
    sanitize_response: SanitizeResponse | None,
) -> FetchWithCacheResult:
    parsed = json.loads(response) if isinstance(response, str) else response
    sanitized = _get_sanitized_response(
        parsed.get("data"), parsed.get("statusText"), parsed.get("headers"), sanitize_response
    )

    async def delete_from_cache() -> None:
        await cache.delete(cache_key)
        logger.debug(f"Evicted from cache: {cache_key}")

    async def update_cache(
        data: Any, status: int, status_text: str, headers: dict[str, str] | None = None
    ) -> None:
        await cache.set(
            cache_key,
            _serialize_fetch_response(
                data,
                status,
                status_text,
                headers if headers is not None else {},
                parsed.get("latencyMs"),
                sanitize_response,
            ),
        )
        logger.debug(f"Updated cached response: {cache_key}")

    return FetchWithCacheResult(
        data=sanitized.get("data"),
        cached=cached,
        status=parsed.get("status"),
        status_text=sanitized["statusText"],
        headers=sanitized.get("headers") or {},
        latency_ms=parsed.get("latencyMs"),
        delete_from_cache=delete_from_cache,
        update_cache=update_cache,
    )


def _response_headers(resp: Any) -> dict[str, str]:
    return {str(k).lower(): v for k, v in dict(resp.headers or {}).items()}


async def _fetch_and_read_body(
    url: str,
    options: dict[str, Any],
    timeout: float,
    max_retries: int | None,
    on_rate_limit_backoff: Callable[[dict[str, Any]], None] | None,
) -> tuple[str, Any, int]:
    """(text, response, latency ms). The transport reads bodies eagerly, so a body-stream failure
    surfaces (and is retried) inside fetch_with_retries rather than here."""
    from promptphoo import fetch as fetch_mod

    start = time.monotonic()
    kwargs: dict[str, Any] = {}
    if on_rate_limit_backoff is not None:
        kwargs["on_rate_limit_backoff"] = on_rate_limit_backoff
    resp = await fetch_mod.fetch_with_retries(url, options, timeout, max_retries, **kwargs)
    latency_ms = int((time.monotonic() - start) * 1000)
    return resp.text(), resp, latency_ms


def _url_for_log(url: str) -> str:
    from promptphoo.sanitizer import sanitize_url_for_logging

    return sanitize_url_for_logging(url)


def _parse_fetch_response(
    url: str,
    response: Any,
    response_text: str,
    format: str,
    sanitize_response: SanitizeResponse | None,
) -> Any:
    if format != "json":
        return response_text
    try:
        return _json_parse(response_text)
    except ValueError as err:
        message = f"Error parsing response from {_url_for_log(url)}:"
        if sanitize_response:
            # Malformed JSON cannot pass through the sanitizer; omit body and status text.
            raise ValueError(f"{message} Invalid JSON. HTTP {response.status}.") from None
        raise ValueError(
            f"{message} {err}. HTTP {response.status} {response.status_text}. "
            f"Received text: {response_text}"
        ) from None


async def _prepare_fetch_response(
    url: str,
    options: dict[str, Any],
    timeout: float,
    max_retries: int | None,
    format: str,
    log_enabled: bool,
    sanitize_response: SanitizeResponse | None,
    on_rate_limit_backoff: Callable[[dict[str, Any]], None] | None,
) -> _PreparedFetchResponse:
    from promptphoo.jsvalues import is_truthy

    response_text, response, latency_ms = await _fetch_and_read_body(
        url, options, timeout, max_retries, on_rate_limit_backoff
    )
    parsed = _parse_fetch_response(url, response, response_text, format, sanitize_response)
    # Capture cacheability before a sanitizer can remove or change an upstream error.
    response_error = parsed.get("error") if format == "json" and isinstance(parsed, dict) else None
    serialized = _serialize_fetch_response(
        f"Empty Response: {response.status}: {response.status_text}"
        if not response.ok and response_text == ""
        else parsed,
        response.status,
        response.status_text,
        _response_headers(response),
        latency_ms,
        sanitize_response,
    )

    if not response.ok:
        return _PreparedFetchResponse(serialized, False)

    if is_truthy(response_error):
        if log_enabled:
            shown = serialized if sanitize_response else json_stringify(response_error)
            logger.debug(
                f"Not caching {_url_for_log(url)} because it contains an 'error' key: {shown}"
            )
        return _PreparedFetchResponse(serialized, False)

    if log_enabled:
        logger.debug(
            f"Storing {_url_for_log(url)} response in cache with latencyMs={latency_ms}: "
            f"{serialized}"
        )
    return _PreparedFetchResponse(serialized, True)


async def _race_observer(
    awaitable: asyncio.Future[Any], observer_failure: asyncio.Future[Any] | None
) -> Any:
    """Await `awaitable` (shielded), failing early if this caller's rate-limit observer raised."""
    if observer_failure is None:
        return await asyncio.shield(awaitable)
    done, _ = await asyncio.wait({awaitable, observer_failure}, return_when=asyncio.FIRST_COMPLETED)
    if awaitable in done:
        return awaitable.result()
    raise observer_failure.exception()  # type: ignore[misc]


def _swallow(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


async def fetch_with_cache(
    url: str,
    options: dict[str, Any] | None = None,
    timeout: float | str | None = None,
    format: Literal["json", "text"] = "json",
    bust: bool | dict[str, Any] | None = False,
    max_retries: int | None = None,
    *,
    on_response_prepared: Callable[[FetchWithCacheResult], None] | None = None,
    on_rate_limit_backoff: Callable[[dict[str, Any]], None] | None = None,
) -> FetchWithCacheResult:
    """Fetch a URL, caching successful responses.

    `bust` is a bool or CacheOptions: `{bust, repeatIndex, cacheKey, sanitizeResponse}`.
    Concurrent identical requests share one upstream call (`coalesced=True` on the joiners).
    Non-OK responses and JSON bodies with a truthy `error` key are never cached.
    """
    from promptphoo import fetch as fetch_mod

    options = options or {}
    timeout_ms = float(fetch_mod.get_request_timeout_ms() if timeout is None else timeout)
    cache_options: dict[str, Any] = {"bust": bust} if isinstance(bust, bool) else dict(bust or {})
    bust_flag = bool(cache_options.get("bust", False))
    repeat_index = cache_options.get("repeatIndex")
    provided_cache_key = cache_options.get("cacheKey")
    sanitize_response: SanitizeResponse | None = cache_options.get("sanitizeResponse")
    log_enabled = fetch_mod._is_log_enabled(fetch_mod.get_fetch_with_proxy_headers(url, options))

    method_value = options.get("method")
    method = ("GET" if method_value is None else str(method_value)).upper()

    cache_enabled = is_cache_enabled()
    if cache_enabled and not bust_flag and options.get("getAuthHeaders") and not provided_cache_key:
        raise ValueError(
            "Request-time authentication requires cache bypass or an explicit "
            "principal-scoped cache key."
        )
    if cache_enabled and not bust_flag and sanitize_response and not provided_cache_key:
        raise ValueError(
            "Response sanitization requires cache bypass or an explicit cache key identifying "
            "the sanitizer policy."
        )
    cache_key: str | None = None
    if cache_enabled and not bust_flag:
        cache_key = (
            get_scoped_cache_key(f"fetch:v3:{provided_cache_key}{_repeat_suffix(repeat_index)}")
            if provided_cache_key
            else _get_fetch_cache_key(url, options, method, format, repeat_index)
        )

    loop = asyncio.get_running_loop()
    notify: Callable[[dict[str, Any]], None] | None = None
    observer_failure: asyncio.Future[Any] | None = None
    if on_rate_limit_backoff is not None:
        failure: asyncio.Future[Any] = loop.create_future()
        observer_failure = failure
        failure.add_done_callback(_swallow)
        observer = on_rate_limit_backoff

        def _notify(observation: dict[str, Any]) -> None:
            if failure.done():
                return
            try:
                observer(observation)
            except Exception as error:
                failure.set_exception(error)

        notify = _notify

    if cache_key is None:
        task = asyncio.ensure_future(
            _fetch_and_read_body(url, options, timeout_ms, max_retries, notify)
        )
        try:
            response_text, resp, latency_ms = await _race_observer(task, observer_failure)
        except BaseException:
            task.add_done_callback(_swallow)
            if not task.done():
                task.cancel()
            raise
        parsed = _parse_fetch_response(url, resp, response_text, format, sanitize_response)
        sanitized = _get_sanitized_response(
            parsed, resp.status_text, _response_headers(resp), sanitize_response
        )

        async def no_op() -> None:
            return None

        result = FetchWithCacheResult(
            data=sanitized.get("data"),
            cached=False,
            status=resp.status,
            status_text=sanitized["statusText"],
            headers=sanitized.get("headers") or {},
            latency_ms=latency_ms,
            delete_from_cache=no_op,
        )
        if on_response_prepared:
            on_response_prepared(result)
        return result

    cache = _get_instance()
    cached_response = await cache.get(cache_key)
    if cached_response is not None:
        result = _deserialize_fetch_response(
            cached_response, True, cache, cache_key, sanitize_response
        )
        if log_enabled:
            logged = cached_response
            if sanitize_response or not isinstance(cached_response, str):
                logged = _serialize_fetch_response(
                    result.data,
                    result.status,
                    result.status_text,
                    result.headers,
                    result.latency_ms,
                )
            logger.debug(f"Returning cached response for {_url_for_log(url)}: {logged}")
        return result

    # Only share an active transport when its deadline works for both callers.
    inflight_key = f"{number_to_string(timeout_ms)}:{get_abort_signal_scoped_key(cache_key)}"
    inflight = _inflight_fetch_responses.get(inflight_key)
    coalesced = inflight is not None
    if inflight is None:
        holder: dict[str, _InflightFetchResponse] = {}

        def fan_out(observation: dict[str, Any]) -> None:
            entry = holder["entry"]
            entry.latest = observation
            for obs in list(entry.observers):
                obs(observation)

        response_task = asyncio.ensure_future(
            _prepare_fetch_response(
                url,
                options,
                timeout_ms,
                max_retries,
                format,
                log_enabled,
                sanitize_response,
                # Python deviation: only hook the transport when the creating caller observes
                # backoffs, so fetch_with_retries stand-ins with the 4-argument signature work.
                # A late-joining observer then misses backoffs of a transport it did not start.
                fan_out if notify is not None else None,
            )
        )
        key_for_set = cache_key

        async def publish() -> None:
            try:
                prepared = await response_task
                if prepared.cacheable:
                    await cache.set(key_for_set, prepared.response)
            finally:
                if _inflight_fetch_responses.get(inflight_key) is holder.get("entry"):
                    _inflight_fetch_responses.pop(inflight_key, None)

        publication = asyncio.ensure_future(publish())
        # Publication may fail after a caller stops waiting or the transport fails.
        publication.add_done_callback(_swallow)
        inflight = _InflightFetchResponse(response=response_task, publication=publication)
        holder["entry"] = inflight
        _inflight_fetch_responses[inflight_key] = inflight

    if notify is not None:
        inflight.observers.add(notify)
        latest = inflight.latest
        # A late joiner learns the original selected deadline.
        if latest and latest.get("resetAt", 0) > time.time() * 1000:
            notify(latest)

    try:
        prepared = await _race_observer(inflight.response, observer_failure)
        result = _deserialize_fetch_response(
            prepared.response, False, cache, cache_key, sanitize_response
        )
        if coalesced:
            result.coalesced = True
        if on_response_prepared:
            on_response_prepared(result)
        # Keep publication alive for other callers while this caller can stop waiting.
        await asyncio.shield(inflight.publication)
        return result
    finally:
        if notify is not None:
            inflight.observers.discard(notify)
