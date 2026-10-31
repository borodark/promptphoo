"""Fetch error types and classification. Ports src/util/fetch/errors.ts.

`JsError` stands in for a JS `Error`: it carries `name`, `code` and `cause` so that messages
formatted from it (`"<name>: <message> (Cause: ...) (Code: ...)"`) read the same as Node's.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from promptphoo.fetch import header_parser


class JsError(RuntimeError):
    """A JS-style `Error` with `name`, optional `code` and optional `cause`."""

    name = "Error"

    def __init__(
        self,
        message: str = "",
        *,
        name: str | None = None,
        code: str | None = None,
        cause: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if name is not None:
            self.name = name
        self.code = code
        self.cause = cause


def error_name(error: BaseException) -> str:
    """The JS `error.name` of an exception: its `name` attribute, else the class name."""
    name = getattr(error, "name", None)
    return name if isinstance(name, str) and name else type(error).__name__


def error_message(error: BaseException) -> str:
    message = getattr(error, "message", None)
    return message if isinstance(message, str) else str(error)


def error_to_string(value: Any) -> str:
    """`String(value)`: `"<name>: <message>"` for errors."""
    if isinstance(value, BaseException):
        message = error_message(value)
        name = error_name(value)
        return f"{name}: {message}" if message else name
    from promptphoo.jsvalues import js_string

    return js_string(value)


# Body-level error codes that indicate a hard quota/billing failure rather than
# a transient per-window rate limit. Retrying these will not succeed.
HARD_QUOTA_ERROR_CODES: frozenset[str] = frozenset(
    {
        "insufficient_quota",
        "billing_hard_limit_reached",
        "billing_not_active",
        "access_terminated",
        "quota_exceeded",
        # Returned by OpenAI when a prepaid credit balance is fully consumed.
        "credit_balance_exhausted",
    }
)

# Subset of HARD_QUOTA_ERROR_CODES that unambiguously describes the account's billing state.
DEFINITIVE_BILLING_ERROR_CODES: frozenset[str] = frozenset(
    {
        "credit_balance_exhausted",
        "billing_hard_limit_reached",
        "billing_not_active",
        "access_terminated",
    }
)

# Body codes that name a per-window throttle outright.
TRANSIENT_RATE_LIMIT_ERROR_CODES: frozenset[str] = frozenset(
    {
        "rate_limit_exceeded",
        "rate_limit_error",
        "rate_limit",
        "rate_limited",
        "overloaded_error",
        "too_many_requests",
        "tokens",
        "requests",
        "tokens_per_min",
        "requests_per_min",
        "tokens_per_day",
        "requests_per_day",
    }
)

# A server that says "retry in <= 1 hour" is signalling a per-window throttle, not billing.
RATE_LIMIT_QUOTA_DOWNGRADE_THRESHOLD_MS = 60 * 60 * 1000

RateLimitKind = Literal["quota", "rate_limit"]


def _normalize_non_negative_ms(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if math.isfinite(value) and value >= 0 else None


class HttpRateLimitError(JsError):
    """A final HTTP rate limit: retries are exhausted, or the body names a hard quota.

    `kind` is `'quota'` (billing/daily exhaustion, don't retry) or `'rate_limit'`
    (per-window throttling, retry honoring `retry_after_ms`).
    """

    name = "HttpRateLimitError"

    def __init__(
        self,
        *,
        status: int,
        status_text: str | None = None,
        retry_after_ms: float | None = None,
        reset_at: float | None = None,
        code: str | None = None,
        type: str | None = None,
        headers: dict[str, str] | None = None,
        body: Any = None,
    ) -> None:
        status_text = status_text or "Too Many Requests"
        retry_after_ms = _normalize_non_negative_ms(retry_after_ms)
        reset_at = _normalize_non_negative_ms(reset_at)

        type_implies_quota = is_hard_quota_code(type) and (
            is_definitive_billing_code(type) or not is_transient_rate_limit_code(code)
        )
        kind: RateLimitKind = (
            "quota" if is_hard_quota_code(code) or type_implies_quota else "rate_limit"
        )
        if (
            kind == "quota"
            and not is_definitive_billing_code(code)
            and not is_definitive_billing_code(type)
            and _has_near_term_recovery_hint(retry_after_ms, reset_at)
        ):
            kind = "rate_limit"

        prefix = "Quota exceeded" if kind == "quota" else "Rate limit exceeded"
        code_suffix = f" (code: {code})" if code else ""
        super().__init__(f"{prefix}: HTTP {status} {status_text}{code_suffix}")
        self.status = status
        self.status_text = status_text
        self.retry_after_ms = retry_after_ms
        self.reset_at = reset_at
        self.code = code
        self.type = type
        self.kind: RateLimitKind = kind
        self.headers = dict(headers) if headers else None
        self.body = body


def is_hard_quota_code(code: str | None) -> bool:
    return code is not None and code in HARD_QUOTA_ERROR_CODES


def is_definitive_billing_code(code: str | None) -> bool:
    return code is not None and code in DEFINITIVE_BILLING_ERROR_CODES


def is_transient_rate_limit_code(code: str | None) -> bool:
    return code is not None and code in TRANSIENT_RATE_LIMIT_ERROR_CODES


def _has_near_term_recovery_hint(retry_after_ms: float | None, reset_at: float | None) -> bool:
    if retry_after_ms is not None:
        return retry_after_ms <= RATE_LIMIT_QUOTA_DOWNGRADE_THRESHOLD_MS
    if reset_at is not None:
        return reset_at - header_parser.date_now() <= RATE_LIMIT_QUOTA_DOWNGRADE_THRESHOLD_MS
    return False


def _js_round(value: float) -> int:
    return math.floor(value + 0.5)


def format_rate_limit_detail(err: HttpRateLimitError) -> str:
    """The `[retry after Xs]` / `[resets in Xs]` suffix; empty for quota errors."""
    if err.kind == "quota":
        return ""
    parts: list[str] = []
    if err.retry_after_ms is not None:
        parts.append(f"retry after {_js_round(err.retry_after_ms / 1000)}s")
    elif err.reset_at is not None:
        remaining_ms = max(err.reset_at - header_parser.date_now(), 0)
        if remaining_ms > 0:
            parts.append(f"resets in {_js_round(remaining_ms / 1000)}s")
    return f" [{', '.join(parts)}]" if parts else ""


def format_rate_limit_error_message(err: HttpRateLimitError, details: str | None = None) -> str:
    """The user-facing error string for a structured rate-limit / quota failure."""
    code_part = f" (code: {err.code})" if err.code else ""
    trailing_details = f" {details}" if details else ""
    if err.kind == "quota":
        return (
            f"Quota exceeded: HTTP {err.status} {err.status_text}{code_part}{trailing_details}. "
            "Retries will not help — check your billing or daily quota."
        )
    detail = format_rate_limit_detail(err)
    return (
        f"Rate limit exceeded: HTTP {err.status} {err.status_text}{code_part}"
        f"{trailing_details}{detail}"
    )


def is_http_rate_limit_error(err: Any) -> bool:
    return isinstance(err, HttpRateLimitError)


def _non_empty_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def extract_rate_limit_error_code(body: Any) -> str | None:
    """Best-effort extraction of an error code from a parsed response body."""
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict):
        metadata = err.get("metadata")
        provider_code = metadata.get("provider_code") if isinstance(metadata, dict) else None
        if isinstance(provider_code, str) and is_definitive_billing_code(provider_code.lower()):
            return provider_code.lower()
        if _non_empty_str(err.get("code")):
            return err["code"]
    if _non_empty_str(body.get("code")):
        return body["code"]
    return extract_rate_limit_error_type(body)


def extract_rate_limit_error_type(body: Any) -> str | None:
    """Best-effort extraction of the broad error class (`error.type` / top-level `type`)."""
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    nested = err.get("type") if isinstance(err, dict) else None
    if _non_empty_str(nested):
        return nested
    return _non_empty_str(body.get("type"))


# Non-transient HTTP statuses: the target is unavailable or misconfigured.
NON_TRANSIENT_HTTP_STATUSES = (401, 403, 404, 501)


def is_non_transient_http_status(status: int) -> bool:
    return status in NON_TRANSIENT_HTTP_STATUSES


def find_target_error_status(results: list[dict[str, Any]]) -> int | None:
    """The first non-transient HTTP error status in evaluation results."""
    for result in results:
        response = result.get("response") if isinstance(result, dict) else None
        metadata = response.get("metadata") if isinstance(response, dict) else None
        http = metadata.get("http") if isinstance(metadata, dict) else None
        status = http.get("status") if isinstance(http, dict) else None
        if (
            isinstance(status, int)
            and not isinstance(status, bool)
            and is_non_transient_http_status(status)
        ):
            return status
    return None


def is_transient_connection_error(error: BaseException | None) -> bool:
    """Transient connection errors (stale connections, mid-stream resets), not TLS misconfigs."""
    if error is None:
        return False
    code = getattr(error, "code", None)
    if code in ("ECONNRESET", "EPIPE") or isinstance(error, ConnectionResetError | BrokenPipeError):
        return True

    message = error_message(error).lower()
    if "eproto" in message and (
        "wrong version number" in message
        or "self signed" in message
        or "unable to verify" in message
        or "unknown ca" in message
        or "cert" in message
    ):
        return False
    return (
        "bad record mac" in message
        or "eproto" in message
        or "econnreset" in message
        or "socket hang up" in message
    )


def is_abort_error(error: Any) -> bool:
    """True for an aborted/cancelled operation: a JS AbortError or an asyncio cancellation."""
    import asyncio

    if isinstance(error, asyncio.CancelledError):
        return True
    return isinstance(error, BaseException) and error_name(error) in (
        "AbortError",
        "AbortException",
    )


def get_abort_error(reason: Any = None) -> JsError:
    if isinstance(reason, BaseException) and error_name(reason) == "AbortError":
        return reason  # type: ignore[return-value]
    message = error_message(reason) if isinstance(reason, BaseException) else "Request was aborted"
    return JsError(message, name="AbortError")
