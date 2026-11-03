"""HTTP transport for providers. Ports src/util/fetch/index.ts (plus retryContext.ts and the
compression/decompression parts of monkeyPatchFetch.ts and decompress.ts).

Providers call `fetch_with_cache` (cached JSON/text responses, ported from src/cache.ts and
re-exported here) or `fetch_with_retries` (a raw response). Requests go through shared
`httpx.AsyncClient`s, one per proxy / TLS setting / event loop, so tests can mock them with respx.

Options are a RequestInit-like dict: `method`, `headers`, `body` (str or bytes), plus the
FetchOptions extras `compress`, `disableTransientRetries` and `getAuthHeaders`. Two Python
additions: `timeout` (seconds, forwarded to httpx) and `dispatcher` (a caller-owned
`httpx.AsyncClient`, the analogue of an undici dispatcher, e.g. for mTLS). A third,
`sdkTransport`, marks a request that upstream sends through a vendor SDK's own HTTP client
rather than fetchWithProxy, so it carries no `x-promptfoo-version` header.

Differences from Node:
- Response bodies are read eagerly inside the request deadline, so a body-stream failure is
  retried by `fetch_with_retries` like any transport error.
- There is no AbortSignal: cancel the awaiting task instead.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import importlib.util
import inspect
import itertools
import json
import logging
import math
import os
import random
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import httpx

# httpcore runs `import sniffio` on every pool lock operation, and recent anyio no longer
# installs sniffio. A failed import is not cached, so each one searched sys.path again: about
# 46k path scans in a 2,560-result eval at -j 256. Recording the module as absent makes the
# import fail at once, with the same result.
if "sniffio" not in sys.modules and importlib.util.find_spec("sniffio") is None:
    sys.modules["sniffio"] = None  # type: ignore[assignment]

from promptphoo import UPSTREAM_VERSION, cli_state, envars
from promptphoo.fetch import header_parser
from promptphoo.fetch.errors import (
    HttpRateLimitError,
    JsError,
    error_message,
    error_name,
    error_to_string,
    extract_rate_limit_error_code,
    extract_rate_limit_error_type,
)
from promptphoo.jsonutil import json_loads
from promptphoo.jsvalues import number_to_string

logger = logging.getLogger(__name__)

FetchOptions = dict[str, Any]
FetchRateLimitObservation = dict[str, Any]

DEFAULT_MAX_CONCURRENCY = 4
VERSION = UPSTREAM_VERSION


def get_request_timeout_ms() -> int:
    return envars.get_env_int("REQUEST_TIMEOUT_MS", 300_000)


async def sleep(ms: float) -> None:
    """`util/time.sleep`. Tests patch this to skip waits."""
    await asyncio.sleep(max(ms, 0) / 1000)


@dataclass
class FetchResponse:
    """The parts of a WHATWG Response that callers use, with the body already read.

    Header names are lowercased, like `Headers.entries()`.
    """

    status: int
    status_text: str
    headers: dict[str, str]
    content: bytes
    url: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def status_code(self) -> int:
        return self.status

    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json_loads(self.text())

    def header(self, name: str) -> str | None:
        """`response.headers.get(name)`, case-insensitively."""
        lower = name.lower()
        value = self.headers.get(lower)
        if value is not None:
            return value
        for key, v in self.headers.items():
            if key.lower() == lower:
                return v
        return None

    def header_entries(self) -> dict[str, str]:
        """`Object.fromEntries(response.headers.entries())`."""
        return {k.lower(): v for k, v in self.headers.items()}


# ---- retryContext.ts --------------------------------------------------------------------------

_fetch_retry_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "fetch_retry_context", default=None
)


@contextmanager
def with_fetch_retry_context(max_retries: int | None) -> Iterator[None]:
    """Nested `fetch_with_retries` / `fetch_with_proxy` calls inherit the provider's `maxRetries`.

    `None` deliberately shadows any outer context so providers without an override fall back to
    defaults.
    """
    token = _fetch_retry_context.set({"maxRetries": max_retries})
    try:
        yield
    finally:
        _fetch_retry_context.reset(token)


def get_fetch_retry_context_max_retries() -> int | None:
    store = _fetch_retry_context.get()
    return None if store is None else store.get("maxRetries")


# ---- request authentication --------------------------------------------------------------------


class RequestAuthenticationError(JsError):
    """Credential failures are not transient HTTP failures and must not be retried."""


async def _resolve_authentication_headers(
    get_auth_headers: Callable[[], Any], explicit_headers: dict[str, str] | None
) -> dict[str, str]:
    try:
        resolved = get_auth_headers()
        if inspect.isawaitable(resolved):
            resolved = await resolved
    except Exception as error:
        raise RequestAuthenticationError(
            error_message(error) or "Request authentication failed", cause=error
        ) from error
    headers = {str(k).lower(): str(v) for k, v in dict(resolved or {}).items()}
    # Never mutate the original headers: a retry must not inherit the previous attempt's token.
    for name, value in (explicit_headers or {}).items():
        headers[name.lower()] = value
    return headers


# ---- clients (undici agents) -------------------------------------------------------------------

# Each key maps to a group of clients that share the pool size between them (see
# `_get_or_create_client`), the event loop they belong to, and a round-robin over the group.
_clients: dict[
    tuple[Any, ...],
    tuple[tuple[httpx.AsyncClient, ...], asyncio.AbstractEventLoop | None, Iterator[int]],
] = {}

# httpcore's pool does work quadratic in its size on every request event: for each idle
# connection it recounts every connection. At -j 256 that was half the CPU of an eval, and
# throughput fell as -j rose. A pool of N connections is therefore split into clients of at
# most this many, which keeps the total number of connections what Node would open.
CONNECTIONS_PER_CLIENT = 16


def get_connection_pool_size() -> int:
    """PROMPTFOO_FETCH_CONNECTIONS > CLI -j > DEFAULT_MAX_CONCURRENCY (4)."""
    env_connections = envars.get_env_string("PROMPTFOO_FETCH_CONNECTIONS")
    if env_connections is not None:
        normalized = env_connections.strip()
        if re.fullmatch(r"\d+", normalized) and 0 < int(normalized) <= 2**53 - 1:
            return int(normalized)
    return cli_state.max_concurrency() or DEFAULT_MAX_CONCURRENCY


@dataclass(frozen=True)
class TlsOptions:
    reject_unauthorized: bool
    ca: str | None = None


def get_tls_options(log_enabled: bool = True) -> TlsOptions:
    """`rejectUnauthorized` from PROMPTFOO_INSECURE_SSL and `ca` from PROMPTFOO_CA_CERT_PATH."""
    # Departs from upstream, which defaults PROMPTFOO_INSECURE_SSL to true and so skips
    # certificate checks: promptphoo verifies unless PROMPTFOO_INSECURE_SSL=true (D61).
    reject_unauthorized = not envars.get_env_bool("PROMPTFOO_INSECURE_SSL", False)
    ca: str | None = None
    ca_cert_path = envars.get_env_string("PROMPTFOO_CA_CERT_PATH")
    if ca_cert_path:
        try:
            resolved_path = os.path.abspath(os.path.join(cli_state.base_path() or "", ca_cert_path))
            with open(resolved_path, encoding="utf-8") as f:
                ca = f.read()
            if log_enabled:
                logger.debug(f"Using custom CA certificate from {resolved_path}")
        except OSError as e:
            if log_enabled:
                logger.warning(f"Failed to read CA certificate from {ca_cert_path}: {e}")
    return TlsOptions(reject_unauthorized=reject_unauthorized, ca=ca)


def _verify_for(tls: TlsOptions) -> Any:
    if not tls.reject_unauthorized:
        return False
    if tls.ca:
        import ssl

        return ssl.create_default_context(cadata=tls.ca)
    return True


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _get_or_create_client(proxy_url: str | None, tls: TlsOptions) -> httpx.AsyncClient:
    loop = _running_loop()
    # Drop clients whose event loop is gone: their pooled connections cannot be reused.
    for key, (group, client_loop, _) in list(_clients.items()):
        if group[0].is_closed or (client_loop is not None and client_loop.is_closed()):
            del _clients[key]
    concurrency = get_connection_pool_size()
    key = (proxy_url or "", tls, concurrency, id(loop) if loop else None)
    existing = _clients.get(key)
    if existing is not None:
        group, _, turn = existing
        return group[next(turn) % len(group)]
    timeout_s = get_request_timeout_ms() / 1000
    shards = max(1, math.ceil(concurrency / CONNECTIONS_PER_CLIENT))
    keepalive = math.ceil(concurrency / shards)
    verify = _verify_for(tls)
    if verify is True:
        # What httpx builds for verify=True with trust_env=False, made once for the group:
        # loading the CA bundle takes ~30 ms, and -j 1024 makes 64 clients.
        verify = httpx.create_ssl_context(trust_env=False)
    group = tuple(
        httpx.AsyncClient(
            follow_redirects=True,
            trust_env=False,
            proxy=proxy_url or None,
            verify=verify,
            # undici's headersTimeout; the overall deadline comes from fetch_with_timeout.
            timeout=httpx.Timeout(timeout_s, write=None, pool=None),
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=keepalive,
                keepalive_expiry=30,
            ),
        )
        for _ in range(shards)
    )
    turn = itertools.count(1)
    _clients[key] = (group, loop, turn)
    return group[0]


def get_client() -> httpx.AsyncClient:
    """The shared direct (non-proxy) client for the current TLS settings and event loop."""
    return _get_or_create_client(None, get_tls_options(log_enabled=False))


async def close_client() -> None:
    """Close every shared client (clearAgentCache)."""
    groups = list(_clients.values())
    _clients.clear()
    for group, loop, _ in groups:
        for client in group:
            if client.is_closed or (loop is not None and loop.is_closed()):
                continue
            if loop is None or loop is _running_loop():
                await client.aclose()


def clear_agent_cache() -> None:
    """Forget cached clients so the next request creates fresh ones. For tests."""
    _clients.clear()


# ---- proxy-from-env ----------------------------------------------------------------------------

_DEFAULT_PORTS = {"ftp": 21, "gopher": 70, "http": 80, "https": 443, "ws": 80, "wss": 443}


def _proxy_env(key: str) -> str:
    return os.environ.get(key.lower()) or os.environ.get(key.upper()) or ""


def _should_proxy(hostname: str, port: int) -> bool:
    no_proxy = (_proxy_env("npm_config_no_proxy") or _proxy_env("no_proxy")).lower()
    if not no_proxy:
        return True
    if no_proxy == "*":
        return False
    for proxy in re.split(r"[,\s]", no_proxy):
        if not proxy:
            continue
        parsed = re.fullmatch(r"(.+):(\d+)", proxy, flags=re.S)
        parsed_hostname = parsed.group(1) if parsed else proxy
        parsed_port = int(parsed.group(2)) if parsed else 0
        if parsed_port and parsed_port != port:
            continue
        if not re.match(r"[.*]", parsed_hostname):
            if hostname == parsed_hostname:
                return False
            continue
        if parsed_hostname.startswith("*"):
            parsed_hostname = parsed_hostname[1:]
        if hostname.endswith(parsed_hostname):
            return False
    return True


def get_proxy_for_url(url: str) -> str:
    """Port of proxy-from-env's getProxyForUrl: the proxy URL for `url`, or ''."""
    from urllib.parse import urlsplit

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port_value = parsed.port
    except ValueError:
        return ""
    proto = parsed.scheme
    if not hostname or not proto or not parsed.netloc:
        return ""
    port = port_value or _DEFAULT_PORTS.get(proto, 0)
    if not _should_proxy(hostname.lower(), port):
        return ""
    proxy = (
        _proxy_env(f"npm_config_{proto}_proxy")
        or _proxy_env(f"{proto}_proxy")
        or _proxy_env("npm_config_proxy")
        or _proxy_env("all_proxy")
    )
    if proxy and "://" not in proxy:
        proxy = f"{proto}://{proxy}"
    return proxy


# ---- headers -----------------------------------------------------------------------------------


def _headers_to_record(headers: Any) -> dict[str, str]:
    if not headers:
        return {}
    if isinstance(headers, httpx.Headers):
        return dict(headers.items())
    if isinstance(headers, list | tuple):
        out: dict[str, str] = {}
        for name, value in headers:
            lower = str(name).lower()
            out[lower] = f"{out[lower]}, {value}" if lower in out else str(value)
        return out
    return dict(headers)


def get_fetch_with_proxy_headers(url: Any, options: FetchOptions) -> dict[str, str]:
    headers = _headers_to_record(options.get("headers"))
    if options.get("sdkTransport"):
        return headers
    return {**headers, "x-promptfoo-version": VERSION}


def _get_header(headers: dict[str, str] | None, name: str) -> str | None:
    """`new Headers(headers).get(name)`: case-insensitive, duplicates joined with ', '."""
    values = [v for k, v in (headers or {}).items() if k.lower() == name.lower()]
    return ", ".join(values) if values else None


def _is_log_enabled(headers: dict[str, str] | None) -> bool:
    return _get_header(headers, "x-promptfoo-silent") != "true"


def _url_for_log(url: str) -> str:
    """URL with credentials and sensitive query values stripped, for diagnostics."""
    from promptphoo.sanitizer import sanitize_url_for_logging

    return sanitize_url_for_logging(url)


# ---- transport ---------------------------------------------------------------------------------


def _strip_decompression_headers(headers: httpx.Headers) -> dict[str, str]:
    """httpx decodes gzip/deflate bodies; drop the headers that describe the encoded body."""
    from httpx._decoders import SUPPORTED_DECODERS

    out = {k.lower(): v for k, v in headers.items()}
    encoding = out.get("content-encoding")
    if encoding is not None:
        encodings = [e.strip().lower() for e in encoding.split(",") if e.strip()]
        if all(e in SUPPORTED_DECODERS for e in encodings):
            out.pop("content-encoding", None)
            out.pop("content-length", None)
    return out


def _body_content(body: Any) -> bytes | None:
    if body is None:
        return None
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray | memoryview):
        return bytes(body)
    if isinstance(body, str):
        return body.encode("utf-8")
    raise TypeError(f"Unsupported request body type: {type(body).__name__}")


async def monkey_patch_fetch(url: str, options: FetchOptions) -> FetchResponse:
    """The global `fetch` as wrapped by monkeyPatchFetch: optional gzip compression, then one
    HTTP exchange on the chosen client. (Cloud auth injection and request logging to the
    Promptfoo backend are not ported.)"""
    opts = dict(options)
    headers = dict(opts.get("headers") or {})
    body = opts.get("body")
    if opts.get("compress") and body and isinstance(body, str):
        try:
            body = gzip.compress(body.encode("utf-8"))
            headers = {k: v for k, v in headers.items() if k.lower() != "content-encoding"}
            headers["content-encoding"] = "gzip"
        except Exception as e:  # pragma: no cover - gzip of a str does not fail
            logger.warning(f"Failed to compress request body: {e}")
    client = opts.get("dispatcher")
    if not isinstance(client, httpx.AsyncClient):
        client = opts.get("_client") or get_client()
    kwargs: dict[str, Any] = {}
    if opts.get("timeout") is not None:
        kwargs["timeout"] = opts["timeout"]
    response = await client.request(
        opts.get("method") or "GET",
        url,
        headers=headers,
        content=_body_content(body),
        **kwargs,
    )
    return FetchResponse(
        status=response.status_code,
        status_text=response.reason_phrase,
        headers=_strip_decompression_headers(response.headers),
        content=response.content,
        url=str(response.url),
    )


def _resolve_transient_retry_disabled(explicit: bool | None) -> bool:
    """An explicit caller flag wins; otherwise disable only when the retry context says 0."""
    if explicit is not None:
        return explicit
    return get_fetch_retry_context_max_retries() == 0


def _apply_url_credentials(
    url: str, headers: dict[str, str], log_enabled: bool
) -> tuple[str, dict[str, str]]:
    from promptphoo.sanitizer import decode_uri_component
    from promptphoo.whatwg_url import URL

    try:
        parsed = URL(url)
    except Exception as e:
        if log_enabled:
            logger.debug(f"URL parsing failed in fetchWithProxy: {e}")
        return url, headers
    if not (parsed.username or parsed.password):
        return url, headers
    # Header names are case-insensitive: matching only `Authorization` would add a second value.
    if any(name.lower() == "authorization" for name in headers):
        if log_enabled:
            logger.warning(
                "Both URL credentials and Authorization header present - "
                "URL credentials will be ignored"
            )
    else:
        # Userinfo percent-encodes reserved characters; HTTP clients decode it before Basic
        # auth. A malformed escape has no decoding and stays as written.
        def decode(part: str) -> str:
            try:
                return decode_uri_component(part)
            except ValueError:
                return part

        credentials = ":".join(decode(p) for p in (parsed.username, parsed.password))
        token = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
        headers = {**headers, "Authorization": f"Basic {token}"}
    parsed.username = ""
    parsed.password = ""
    return parsed.href, headers


async def fetch_with_proxy(url: str, options: FetchOptions | None = None) -> FetchResponse:
    options = options or {}
    if not isinstance(url, str) or not url:
        raise JsError("Invalid URL")

    request_options = {
        k: v for k, v in options.items() if k not in ("getAuthHeaders", "sdkTransport")
    }
    get_auth_headers = options.get("getAuthHeaders")
    final_options: FetchOptions = {
        **request_options,
        "headers": get_fetch_with_proxy_headers(url, options),
    }
    log_enabled = _is_log_enabled(final_options["headers"])

    final_url, final_options["headers"] = _apply_url_credentials(
        url, final_options["headers"], log_enabled
    )

    tls = get_tls_options(log_enabled)
    proxy_url = get_proxy_for_url(final_url)

    # Respect a caller-provided dispatcher (e.g. a custom mTLS client).
    if not isinstance(final_options.get("dispatcher"), httpx.AsyncClient):
        if proxy_url:
            if log_enabled:
                from promptphoo.sanitizer import sanitize_url

                logger.debug(f"Using proxy: {sanitize_url(proxy_url)}")
        final_options["_client"] = _get_or_create_client(proxy_url or None, tls)

    # Transient error retry logic (502/503/504/524 with matching status text).
    disable_transient_retries = _resolve_transient_retry_disabled(
        options.get("disableTransientRetries")
    )
    max_transient_retries = 0 if disable_transient_retries else 3

    for attempt in range(max_transient_retries + 1):
        attempt_options = final_options
        if get_auth_headers:
            attempt_options = {
                **final_options,
                "headers": await _resolve_authentication_headers(
                    get_auth_headers, final_options["headers"]
                ),
            }
        response = await monkey_patch_fetch(final_url, attempt_options)

        if (
            not disable_transient_retries
            and is_transient_error(response)
            and attempt < max_transient_retries
        ):
            backoff_ms = 2**attempt * 1000  # 1s, 2s, 4s
            if log_enabled:
                logger.debug(
                    f"Transient error ({response.status} {response.status_text}), retry "
                    f"{attempt + 1}/{max_transient_retries} after {backoff_ms}ms"
                )
            await sleep(backoff_ms)
            continue

        return response

    raise JsError("Unexpected end of transient retry loop")  # pragma: no cover


async def fetch_with_timeout(
    url: str, options: FetchOptions | None, timeout_ms: float | str
) -> FetchResponse:
    # Provider config templates can render numeric settings as strings.
    timeout = float(timeout_ms)
    try:
        async with asyncio.timeout(max(timeout, 0) / 1000):
            return await fetch_with_proxy(url, options or {})
    except TimeoutError as e:
        raise JsError(f"Request timed out after {number_to_string(timeout)} ms") from e


def is_rate_limited(response: FetchResponse) -> bool:
    """Check if a response indicates rate limiting."""
    if response.headers is None:
        raise JsError("Invariant failed: Response headers are missing")
    if not response.status:
        raise JsError("Invariant failed: Response status is missing")
    return (
        response.header("X-RateLimit-Remaining") == "0"
        or response.status == 429
        # OpenAI specific error codes
        or response.header("x-ratelimit-remaining-requests") == "0"
        or response.header("x-ratelimit-remaining-tokens") == "0"
    )


def compute_rate_limit_wait_ms(response: FetchResponse) -> float:
    """How long to wait after a rate-limited response. Reads `retry-after-ms`, `Retry-After`,
    `X-RateLimit-Reset` and OpenAI-style reset headers. Default: 60s."""
    parsed = header_parser.parse_rate_limit_headers(response.header_entries())
    rate_limit_reset = response.header("X-RateLimit-Reset")
    openai_reset = response.header("x-ratelimit-reset-requests") or response.header(
        "x-ratelimit-reset-tokens"
    )

    if openai_reset and "resetAt" in parsed:
        return max(parsed["resetAt"] - header_parser.date_now(), 0)

    if rate_limit_reset:
        seconds = header_parser.parse_int(rate_limit_reset)
        if seconds is not None and seconds >= 0:
            return max(seconds * 1000 - header_parser.date_now() + 1000, 0)

    return parsed.get("retryAfterMs", 60_000)


# Maximum jitter (in ms) added to rate-limit waits, so concurrent waiters desynchronize.
RATE_LIMIT_JITTER_MS = 1000


async def handle_rate_limit(
    response: FetchResponse,
    on_rate_limit_backoff: Callable[[FetchRateLimitObservation], None] | None = None,
) -> None:
    """Wait out a rate limit, plus a uniform random jitter."""
    wait_time = compute_rate_limit_wait_ms(response)
    if on_rate_limit_backoff is not None:
        on_rate_limit_backoff(
            {
                "headers": response.header_entries(),
                "status": response.status,
                "resetAt": header_parser.date_now() + wait_time,
            }
        )
    jitter = math.floor(random.random() * RATE_LIMIT_JITTER_MS)
    total_wait = wait_time + jitter
    logger.debug(
        f"Rate limited, waiting {number_to_string(total_wait)}ms "
        f"(base {number_to_string(wait_time)}ms + {jitter}ms jitter) before retry"
    )
    await sleep(total_wait)


# Hard cap on bytes read from a 429 response body during classification.
RATE_LIMIT_BODY_PEEK_BYTES = 64 * 1024


def _peek_rate_limit_body(response: FetchResponse, log_enabled: bool) -> tuple[Any, Any, Any]:
    """(body, code, type) of a rate-limited response: parsed JSON if possible, else raw text."""
    try:
        text = read_bounded_text(response, RATE_LIMIT_BODY_PEEK_BYTES, require_stream=True)
    except Exception as err:
        if log_enabled:
            logger.debug(f"[fetch] peekRateLimitBody: body read failed: {err}")
        return None, None, None
    if not text:
        return None, None, None
    try:
        body = json.loads(text)
    except ValueError:
        if log_enabled:
            logger.debug("[fetch] peekRateLimitBody: response body was not JSON")
        return text, None, None
    return body, extract_rate_limit_error_code(body), extract_rate_limit_error_type(body)


def read_bounded_text(response: FetchResponse, max_bytes: int, require_stream: bool = False) -> str:
    """The body as text, decoding at most `max_bytes` bytes. Bodies are read eagerly here, so
    `require_stream` (skip streamless responses) never applies."""
    return bytes(response.content[:max_bytes]).decode("utf-8", errors="replace")


def rate_limit_timing_from_headers(headers: dict[str, str]) -> dict[str, float]:
    """`{retryAfterMs, resetAt}` from rate-limit headers (undefined keys omitted)."""
    parsed = header_parser.parse_rate_limit_headers(headers)
    return {k: parsed[k] for k in ("retryAfterMs", "resetAt") if k in parsed}  # type: ignore[literal-required]


def _build_http_rate_limit_error(
    response: FetchResponse, body: Any, code: str | None, type_: str | None
) -> HttpRateLimitError:
    headers = response.header_entries()
    parsed = header_parser.parse_rate_limit_headers(headers)
    return HttpRateLimitError(
        status=response.status,
        status_text=response.status_text,
        retry_after_ms=parsed.get("retryAfterMs"),
        reset_at=parsed.get("resetAt"),
        code=code,
        type=type_,
        headers=headers,
        body=body,
    )


def is_transient_error(response: FetchResponse | None) -> bool:
    """A transient server error worth retrying: the status code AND its expected status text,
    so permanent failures (some APIs return 502 for auth errors) are not retried."""
    if response is None or not response.status_text:
        return False
    status_text = response.status_text.lower()
    if response.status == 502:
        return "bad gateway" in status_text
    if response.status == 503:
        return "service unavailable" in status_text
    if response.status == 504:
        return "gateway timeout" in status_text
    if response.status == 524:  # Cloudflare-specific timeout error
        return "timeout" in status_text
    return False


async def _handle_rate_limited_response(
    response: FetchResponse,
    url: str,
    attempt: int,
    max_retries: int,
    log_enabled: bool,
    on_rate_limit_backoff: Callable[[FetchRateLimitObservation], None] | None,
) -> None:
    # Only a 429 produces a structured error; a 200 with X-RateLimit-Remaining=0 is a soft hint.
    is_hard_rate_limit = response.status == 429
    safe_url = _url_for_log(url)

    rate_limit_error: HttpRateLimitError | None = None
    if is_hard_rate_limit:
        body, code, type_ = _peek_rate_limit_body(response, log_enabled)
        rate_limit_error = _build_http_rate_limit_error(response, body, code, type_)

    # Hard quota failures won't resolve on retry: fail fast.
    if rate_limit_error is not None and rate_limit_error.kind == "quota":
        if log_enabled:
            logger.debug(
                f"Quota exhausted on URL {safe_url}: HTTP {response.status} "
                f"(code: {rate_limit_error.code}), failing fast."
            )
        raise rate_limit_error

    if attempt >= max_retries:
        if rate_limit_error is not None:
            if log_enabled:
                logger.debug(
                    f"Rate limited on URL {safe_url}: HTTP {response.status} "
                    f"{response.status_text}, attempt {attempt + 1}/{max_retries + 1}, "
                    "no retries remain."
                )
            raise rate_limit_error
        raise JsError(
            f"Rate limited: {response.status} {response.status_text} "
            f"after {max_retries + 1} attempts"
        )

    if log_enabled:
        logger.debug(
            f"Rate limited on URL {safe_url}: HTTP {response.status} {response.status_text}, "
            f"attempt {attempt + 1}/{max_retries + 1}, waiting before retry..."
        )
    await handle_rate_limit(response, on_rate_limit_backoff)


def format_fetch_error_message(error: Any, url: str) -> str:
    if not isinstance(error, BaseException):
        return error_to_string(error)
    safe_url = _url_for_log(url)

    def redact(value: str) -> str:
        return value.replace(url, safe_url) if url else value

    message = f"{error_name(error)}: {redact(error_message(error))}"
    cause = getattr(error, "cause", None)
    if cause:
        message += f" (Cause: {redact(error_to_string(cause))})"
    code = getattr(error, "code", None)
    if code:
        message += f" (Code: {code})"
    return message


async def fetch_with_retries(
    url: str,
    options: FetchOptions | None = None,
    timeout: float | str | None = None,
    max_retries: int | None = None,
    *,
    on_rate_limit_backoff: Callable[[FetchRateLimitObservation], None] | None = None,
) -> FetchResponse:
    """Fetch with automatic retries and rate limit handling."""
    options = options or {}
    timeout = get_request_timeout_ms() if timeout is None else timeout
    if max_retries is None:
        max_retries = get_fetch_retry_context_max_retries()
    max_retries = max(0, 4 if max_retries is None else max_retries)

    last_error_message: str | None = None
    backoff = envars.get_env_int("PROMPTFOO_REQUEST_BACKOFF_MS", 5000)
    log_enabled = _is_log_enabled(get_fetch_with_proxy_headers(url, options))

    for i in range(max_retries + 1):
        try:
            # Disable transient retries in fetch_with_proxy to avoid double-retrying
            response = await fetch_with_timeout(
                url, {**options, "disableTransientRetries": True}, timeout
            )

            if envars.get_env_bool("PROMPTFOO_RETRY_5XX") and 500 <= response.status < 600:
                raise JsError(f"Internal Server Error: {response.status} {response.status_text}")

            if response and is_rate_limited(response):
                await _handle_rate_limited_response(
                    response, url, i, max_retries, log_enabled, on_rate_limit_backoff
                )
                continue

            return response
        except Exception as error:
            # Don't retry on abort - propagate immediately
            if error_name(error) == "AbortError":
                raise
            # Rate-limit and credential failures never consume another retry.
            if isinstance(error, HttpRateLimitError | RequestAuthenticationError):
                raise

            error_message_text = format_fetch_error_message(error, url)
            if log_enabled:
                logger.debug(
                    f"Request to {_url_for_log(url)} failed (attempt #{i + 1}), retrying: "
                    f"{error_message_text}"
                )
            if i < max_retries:
                wait_time = 2**i * (backoff + 1000 * random.random())
                await sleep(wait_time)
            last_error_message = error_message_text
    raise JsError(f"Request failed after {max_retries} retries: {last_error_message}")


# fetchWithCache lives in cache.py, as in upstream src/cache.ts; re-exported for providers.
from promptphoo.cache import FetchWithCacheResult, fetch_with_cache  # noqa: E402

__all__ = [
    "FetchOptions",
    "FetchResponse",
    "FetchWithCacheResult",
    "HttpRateLimitError",
    "JsError",
    "RequestAuthenticationError",
    "close_client",
    "compute_rate_limit_wait_ms",
    "fetch_with_cache",
    "fetch_with_proxy",
    "fetch_with_retries",
    "fetch_with_timeout",
    "get_client",
    "get_fetch_retry_context_max_retries",
    "get_proxy_for_url",
    "get_request_timeout_ms",
    "handle_rate_limit",
    "is_rate_limited",
    "is_transient_error",
    "rate_limit_timing_from_headers",
    "read_bounded_text",
    "with_fetch_retry_context",
]
