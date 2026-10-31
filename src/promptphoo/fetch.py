"""HTTP transport for providers. Ports the public surface of src/util/fetch/index.ts and
fetchWithCache from src/cache.ts.

Providers call `fetch_with_cache` (cached JSON/text responses) or `fetch_with_retries`
(a raw response). Both go through one shared `httpx.AsyncClient`, so tests can mock it with respx.

Options are a RequestInit-like dict: `method`, `headers`, `body` (str or bytes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from promptphoo import cache, envars
from promptphoo.jsvalues import json_stringify

logger = logging.getLogger(__name__)

FetchOptions = dict[str, Any]


def get_request_timeout_ms() -> int:
    return envars.get_env_int("REQUEST_TIMEOUT_MS", 300_000)


@dataclass
class FetchResponse:
    """The parts of a WHATWG Response that callers use, with the body already read."""

    status: int
    status_text: str
    headers: dict[str, str]
    content: bytes

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text())


@dataclass
class FetchWithCacheResult:
    data: Any
    cached: bool
    status: int
    status_text: str
    headers: dict[str, str] = field(default_factory=dict)
    latency_ms: int | None = None
    delete_from_cache: Callable[[], Awaitable[None]] | None = None


_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(follow_redirects=True)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def fetch_with_proxy(url: str, options: FetchOptions | None = None) -> FetchResponse:
    options = options or {}
    response = await get_client().request(
        options.get("method") or "GET",
        url,
        headers=options.get("headers"),
        content=options.get("body"),
        timeout=options.get("timeout"),
    )
    return FetchResponse(
        status=response.status_code,
        status_text=response.reason_phrase,
        headers={k.lower(): v for k, v in response.headers.items()},
        content=response.content,
    )


async def fetch_with_timeout(url: str, options: FetchOptions, timeout_ms: float) -> FetchResponse:
    try:
        return await asyncio.wait_for(fetch_with_proxy(url, options), timeout_ms / 1000)
    except TimeoutError as e:
        raise TimeoutError(f"Request timed out after {int(timeout_ms)} ms") from e


def is_rate_limited(response: FetchResponse) -> bool:
    return response.status == 429 or response.headers.get("x-ratelimit-remaining") == "0"


def is_transient_error(response: FetchResponse) -> bool:
    text = (response.status_text or "").lower()
    if not text:
        return False
    return {
        502: "bad gateway",
        503: "service unavailable",
        504: "gateway timeout",
        524: "timeout",
    }.get(response.status, "\0") in text


async def fetch_with_retries(
    url: str,
    options: FetchOptions | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> FetchResponse:
    options = options or {}
    timeout = get_request_timeout_ms() if timeout is None else timeout
    max_retries = max(0, 4 if max_retries is None else max_retries)
    backoff = envars.get_env_int("PROMPTFOO_REQUEST_BACKOFF_MS", 5000)
    last_error: str | None = None
    for i in range(max_retries + 1):
        try:
            response = await fetch_with_timeout(url, options, timeout)
            if envars.get_env_bool("PROMPTFOO_RETRY_5XX") and 500 <= response.status < 600:
                raise RuntimeError(
                    f"Internal Server Error: {response.status} {response.status_text}"
                )
            if is_rate_limited(response) and i < max_retries:
                await asyncio.sleep(_rate_limit_wait_ms(response, i, backoff) / 1000)
                continue
            return response
        except (httpx.HTTPError, TimeoutError, RuntimeError) as e:
            last_error = str(e) or type(e).__name__
            logger.debug("Request to %s failed (attempt #%d), retrying: %s", url, i + 1, last_error)
            if i < max_retries:
                await asyncio.sleep(2**i * (backoff + 1000 * random.random()) / 1000)
    raise RuntimeError(f"Request failed after {max_retries} retries: {last_error}")


def _rate_limit_wait_ms(response: FetchResponse, attempt: int, backoff: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after) * 1000
        except ValueError:
            pass
    return 2**attempt * (backoff + 1000 * random.random())


def _cache_key(url: str, options: FetchOptions, format: str, repeat_index: Any) -> str:
    headers = {k.lower(): v for k, v in (options.get("headers") or {}).items()}
    body = options.get("body")
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    key = json_stringify(
        {"url": url, "method": options.get("method") or "GET", "headers": headers, "body": body}
    )
    suffix = f":repeat{repeat_index}" if isinstance(repeat_index, int) and repeat_index > 0 else ""
    return f"fetch:v3:{format}:{key}{suffix}"


async def fetch_with_cache(
    url: str,
    options: FetchOptions | None = None,
    timeout: float | None = None,
    format: Literal["json", "text"] = "json",
    bust: bool | dict[str, Any] = False,
    max_retries: int | None = None,
) -> FetchWithCacheResult:
    """Fetch, caching successful responses. `bust` is a bool or CacheOptions
    ({bust, repeatIndex, cacheKey})."""
    options = options or {}
    cache_options = {"bust": bust} if isinstance(bust, bool) else dict(bust or {})
    enabled = cache.is_cache_enabled() and not cache_options.get("bust")
    key = None
    if enabled:
        key = (
            cache.get_scoped_cache_key(f"fetch:v3:{cache_options['cacheKey']}")
            if cache_options.get("cacheKey")
            else cache.get_scoped_cache_key(
                _cache_key(url, options, format, cache_options.get("repeatIndex"))
            )
        )
        hit = await cache._get_instance().get(key)
        if hit is not None:
            stored = json.loads(hit) if isinstance(hit, str) else hit
            store, k = cache._get_instance(), key

            async def delete_hit() -> None:
                await store.delete(k)

            return FetchWithCacheResult(
                data=stored["data"],
                cached=True,
                status=stored.get("status", 200),
                status_text=stored.get("statusText", "OK"),
                headers=stored.get("headers") or {},
                latency_ms=stored.get("latencyMs"),
                delete_from_cache=delete_hit,
            )

    start = time.monotonic()
    response = await fetch_with_retries(url, options, timeout, max_retries)
    latency_ms = int((time.monotonic() - start) * 1000)
    text = response.text()
    if format == "json":
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError as e:
            if response.ok:
                raise ValueError(f"Error parsing response as JSON: {text}") from e
            data = text
    else:
        data = text

    async def delete_entry() -> None:
        if key is not None:
            await cache._get_instance().delete(key)

    if key is not None and response.ok:
        entry = {
            "data": data,
            "status": response.status,
            "statusText": response.status_text,
            "headers": response.headers,
            "latencyMs": latency_ms,
        }
        await cache._get_instance().set(key, json_stringify(entry))
    return FetchWithCacheResult(
        data=data,
        cached=False,
        status=response.status,
        status_text=response.status_text,
        headers=response.headers,
        latency_ms=latency_ms,
        delete_from_cache=delete_entry,
    )
