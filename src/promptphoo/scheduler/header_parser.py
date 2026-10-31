"""Rate-limit header parsing. Ports src/scheduler/headerParser.ts.

`parse_rate_limit_headers` returns a ParsedRateLimitHeaders dict with Node's camelCase keys
(remainingRequests, remainingTokens, limitRequests, limitTokens, resetAt, retryAfterMs); keys
whose value is unknown are omitted, like `undefined` in Node.
"""

from __future__ import annotations

import datetime as _dt
import email.utils
import math
import re
import sys
from collections.abc import Mapping
from typing import Any, TypedDict

from promptphoo.scheduler import clock


class ParsedRateLimitHeaders(TypedDict, total=False):
    remainingRequests: int
    remainingTokens: int
    limitRequests: int
    limitTokens: int
    resetAt: float  # Absolute Unix timestamp in milliseconds
    retryAfterMs: float  # Relative duration in milliseconds


# --- Header names by provider ----------------------------------------------------------------

# OpenAI-style headers (also used by Azure, many OpenAI-compatible APIs)
OPENAI_HEADERS = {
    "remainingRequests": "x-ratelimit-remaining-requests",
    "remainingTokens": "x-ratelimit-remaining-tokens",
    "limitRequests": "x-ratelimit-limit-requests",
    "limitTokens": "x-ratelimit-limit-tokens",
    "resetRequests": "x-ratelimit-reset-requests",
    "resetTokens": "x-ratelimit-reset-tokens",
}

# Anthropic-style headers
ANTHROPIC_HEADERS = {
    "remainingRequests": "anthropic-ratelimit-requests-remaining",
    "remainingTokens": "anthropic-ratelimit-tokens-remaining",
    "limitRequests": "anthropic-ratelimit-requests-limit",
    "limitTokens": "anthropic-ratelimit-tokens-limit",
    "resetRequests": "anthropic-ratelimit-requests-reset",
    "resetTokens": "anthropic-ratelimit-tokens-reset",
}

# Standard/generic headers (RFC 6585 style)
STANDARD_HEADERS = {
    "remaining": "ratelimit-remaining",
    "limit": "ratelimit-limit",
    "reset": "ratelimit-reset",
    # Fallback variants
    "remainingAlt": "x-ratelimit-remaining",
    "limitAlt": "x-ratelimit-limit",
    "resetAlt": "x-ratelimit-reset",
}

_PARSE_INT_RE = re.compile(r"\s*([+-]?[0-9]+)")


def _parse_int(value: Any) -> int | None:
    """`Number.parseInt(value, 10)`; None stands for NaN."""
    m = _PARSE_INT_RE.match(str(value))
    if not m:
        return None
    num = int(m.group(1))
    # Past the double range Node yields +/-Infinity, which every caller rejects as non-finite.
    return None if abs(num) > sys.float_info.max else num


def parse_rate_limit_headers(headers: Mapping[str, Any]) -> ParsedRateLimitHeaders:
    """Parse rate limit headers from a response."""
    result: ParsedRateLimitHeaders = {}
    h = _lowercase_keys(headers)

    # --- Remaining counts (ordered: OpenAI, Anthropic, Standard) ---
    remaining_requests = _parse_first_match(h, [
        OPENAI_HEADERS["remainingRequests"],
        ANTHROPIC_HEADERS["remainingRequests"],
        STANDARD_HEADERS["remainingAlt"],
        STANDARD_HEADERS["remaining"],
    ])  # fmt: skip
    if remaining_requests is not None:
        result["remainingRequests"] = remaining_requests

    remaining_tokens = _parse_first_match(h, [
        OPENAI_HEADERS["remainingTokens"],
        ANTHROPIC_HEADERS["remainingTokens"],
    ])  # fmt: skip
    if remaining_tokens is not None:
        result["remainingTokens"] = remaining_tokens

    # --- Limits (ordered: OpenAI, Anthropic, Standard) ---
    limit_requests = _parse_first_match(h, [
        OPENAI_HEADERS["limitRequests"],
        ANTHROPIC_HEADERS["limitRequests"],
        STANDARD_HEADERS["limitAlt"],
        STANDARD_HEADERS["limit"],
    ])  # fmt: skip
    if limit_requests is not None:
        result["limitRequests"] = limit_requests

    limit_tokens = _parse_first_match(h, [
        OPENAI_HEADERS["limitTokens"],
        ANTHROPIC_HEADERS["limitTokens"],
    ])  # fmt: skip
    if limit_tokens is not None:
        result["limitTokens"] = limit_tokens

    # --- Reset time (ordered: OpenAI, Anthropic, Standard) ---
    for name in (
        OPENAI_HEADERS["resetRequests"],
        OPENAI_HEADERS["resetTokens"],
        ANTHROPIC_HEADERS["resetRequests"],
        # Token limits bind before request limits on eval workloads, and a token-limited 429
        # may carry only the tokens reset.
        ANTHROPIC_HEADERS["resetTokens"],
        STANDARD_HEADERS["resetAlt"],
        STANDARD_HEADERS["reset"],
    ):
        if name in h:
            parsed = _parse_reset_time(h[name])
            if parsed is not None:
                result["resetAt"] = parsed
                break

    # --- Retry-After ---
    if "retry-after-ms" in h:
        ms = _parse_int(h["retry-after-ms"])
        # Accept 0 as valid (means "retry immediately")
        if ms is not None and ms >= 0:
            result["retryAfterMs"] = ms
            if "resetAt" not in result:
                result["resetAt"] = clock.date_now() + ms

    if "retryAfterMs" not in result and "retry-after" in h:
        parsed_retry = parse_retry_after(h["retry-after"])
        if parsed_retry is not None:
            result["retryAfterMs"] = parsed_retry
            if "resetAt" not in result:
                result["resetAt"] = clock.date_now() + parsed_retry

    return result


def parse_retry_after(value: str) -> float | None:
    """Parse a Retry-After header value into a duration in milliseconds."""
    value = str(value)
    # Try as integer seconds (must be non-negative)
    seconds = _parse_int(value)
    if (
        seconds is not None
        and seconds >= 0
        and str(seconds) == value.strip()
        and seconds * 1000 <= sys.float_info.max
    ):
        return seconds * 1000

    # Try HTTP-date format
    http_date = _parse_http_date(value)
    if http_date is not None:
        return max(0, http_date - clock.date_now())

    return None


def _parse_first_match(headers: Mapping[str, Any], names: list[str]) -> int | None:
    for name in names:
        if name in headers and headers[name] is not None:
            num = _parse_int(headers[name])
            if num is not None and num >= 0:
                return num
    return None


# A bare non-negative decimal number and nothing else: "0", "120", "1.5". Without this guard
# a timestamp's leading digits ("2026-08-07T21:00:00Z" -> 2026) would be read as seconds.
BARE_NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _parse_reset_time(value: Any) -> float | None:
    """Parse a reset time in any supported format into an absolute Unix timestamp in ms."""
    value = str(value)
    # Try duration format first (e.g., "1s", "100ms", "1m30s")
    duration_ms = _parse_duration(value)
    if duration_ms is not None:
        return clock.date_now() + duration_ms

    # Try as numeric, but only when the whole value is a bare number
    if BARE_NUMBER_RE.fullmatch(value.strip()):
        num = float(value.strip())
        if math.isfinite(num) and num >= 0:
            # Disambiguate by magnitude:
            # - < 1 billion: relative seconds
            # - 1-10 billion: Unix seconds (10 digits)
            # - > 10 billion: Unix milliseconds (13 digits)
            if num < 1_000_000_000:
                return clock.date_now() + num * 1000
            if num < 10_000_000_000:
                return num * 1000
            return num

    # Try HTTP-date format, plus the RFC 3339 timestamps Anthropic returns in
    # `anthropic-ratelimit-*-reset`.
    return _parse_http_date(value)


def _date_parse(value: str) -> float | None:
    """The subset of `Date.parse` that rate-limit headers use: RFC 7231 HTTP-dates and
    ISO 8601 / RFC 3339 timestamps. Returns Unix milliseconds, or None for NaN."""
    text = value.strip()
    if not text:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        parsed = None
    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.UTC)
        return parsed.timestamp() * 1000
    try:
        iso = _dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if iso.tzinfo is None:
        # ES: date-only forms are UTC, date-time forms without an offset are local time.
        iso = (
            iso.replace(tzinfo=_dt.UTC)
            if "T" not in text and " " not in text
            else (iso.astimezone())
        )
    return iso.timestamp() * 1000


def _parse_http_date(value: str) -> float | None:
    """Parse an HTTP-date (RFC 7231) within a year of now."""
    timestamp = _date_parse(value)
    if timestamp is not None and math.isfinite(timestamp):
        # Sanity check: within reasonable range (not too far past or future)
        now = clock.date_now()
        one_year_ms = 365 * 24 * 60 * 60 * 1000
        if now - one_year_ms < timestamp < now + one_year_ms:
            return timestamp
    return None


_DURATION_RE = re.compile(r"(?:([0-9]+)h)?(?:([0-9]+)m(?!s))?(?:([0-9]+(?:\.[0-9]+)?)(ms|s))?")


def _parse_duration(value: str) -> float | None:
    """Parse duration strings like "1s", "100ms", "1m30s", "1h30s", "2h15m30s"."""
    match = _DURATION_RE.fullmatch(value)
    if not match:
        return None
    hours, minutes, seconds_value, seconds_unit = match.groups()

    # At least one component must be present
    if not hours and not minutes and not seconds_value:
        return None

    ms: float = 0
    if hours:
        ms += float(hours) * 3600_000
    if minutes:
        ms += float(minutes) * 60_000
    if seconds_value:
        num = float(seconds_value)
        ms += num if seconds_unit == "ms" else num * 1000

    return ms if math.isfinite(ms) else None


def _lowercase_keys(obj: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key).lower(): value for key, value in obj.items()}
