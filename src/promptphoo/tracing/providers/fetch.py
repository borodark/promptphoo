"""Ports src/tracing/providers/fetch.ts and hasTraceCredentialPath from
src/contracts/traceProviderEndpoint.ts.

promptphoo's `fetch_with_proxy` reads response bodies eagerly, so there is no stream to cancel:
`release_response` is a no-op and `read_limited_response` checks the byte length after the
read. `fetch_trace_response` stands in for the fetch options upstream passes
(`redirect: 'error'`, `signal: AbortSignal.any([timeout, abortSignal])`): the request runs
under an asyncio deadline, an already-aborted signal stops it before it starts, and a response
that arrived through a redirect is rejected the way undici rejects one (`fetch failed`).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from promptphoo.fetch import FetchResponse, fetch_with_proxy
from promptphoo.fetch.errors import JsError
from promptphoo.tracing.providers.types import TraceProviderError

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_TRACE_RESPONSE_BYTES",
    "TRACE_CREDENTIAL_PATH_SEGMENT",
    "fetch_trace_response",
    "fetch_with_proxy",
    "has_trace_credential_path",
    "read_limited_response",
    "release_response",
    "validate_trace_provider_endpoint",
]

MAX_TRACE_RESPONSE_BYTES = 10 * 1024 * 1024

# Identifies URL path segments that are likely to contain credentials.
TRACE_CREDENTIAL_PATH_SEGMENT = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32,}"
    r"|(?:token|key|secret|credential|auth|sk|sk-proj|sk-ant)[-_][a-z0-9._-]{8,}"
    r"|AKIA[A-Z0-9]{16}|AIza[a-zA-Z0-9_-]{35}|[a-zA-Z0-9+/=_-]{64,}"
    r"|eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)\Z",
    re.IGNORECASE,
)


def has_trace_credential_path(pathname: str) -> bool:
    from promptphoo.sanitizer import decode_uri_component

    for segment in pathname.split("/"):
        try:
            if TRACE_CREDENTIAL_PATH_SEGMENT.search(decode_uri_component(segment)):
                return True
        except ValueError:
            return True
    return False


def validate_trace_provider_endpoint(endpoint: str, provider_name: str) -> None:
    from promptphoo.whatwg_url import URL

    try:
        url = URL(endpoint)
    except Exception:
        raise ValueError(
            f"{provider_name} provider endpoint must be a valid HTTP or HTTPS URL"
        ) from None

    if (
        url.scheme not in ("http", "https")
        or url.username
        or url.password
        or url.search
        or url.hash
        or has_trace_credential_path(url.pathname)
    ):
        raise ValueError(
            f"{provider_name} provider endpoint must be an HTTP or HTTPS URL without "
            "credentials, query parameters, or fragments"
        )


async def release_response(response: FetchResponse, provider_name: str) -> None:
    """Bodies are already read, so there is nothing to cancel."""
    return None


async def read_limited_response(
    response: FetchResponse, provider_name: str, max_response_bytes: int = MAX_TRACE_RESPONSE_BYTES
) -> str:
    content = response.content or b""
    if len(content) > max_response_bytes:
        raise TraceProviderError(f"{provider_name} trace exceeds the maximum response size")
    return content.decode("utf-8", errors="replace")


def _abort_error(signal: Any) -> Exception:
    return JsError(
        "This operation was aborted", name="AbortError", cause=getattr(signal, "reason", None)
    )


async def fetch_trace_response(
    url: str,
    options: dict[str, Any],
    *,
    timeout_ms: float,
    abort_signal: Any = None,
) -> FetchResponse:
    """`fetchWithProxy(url, {...options, redirect: 'error', signal})`."""
    if abort_signal is not None and getattr(abort_signal, "aborted", False):
        raise _abort_error(abort_signal)
    try:
        async with asyncio.timeout(timeout_ms / 1000):
            response = await fetch_with_proxy(url, options)
    except TimeoutError:
        raise JsError("The operation was aborted due to timeout", name="TimeoutError") from None
    if response.url and httpx.URL(response.url) != httpx.URL(url):
        # redirect: 'error' — undici rejects the redirect response.
        raise JsError("fetch failed")
    return response
