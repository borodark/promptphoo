"""WHATWG URL helpers used by the OpenAI-family providers (`new URL(x).hostname`)."""

from __future__ import annotations

from promptphoo.whatwg_url import URL


def url_hostname(value: str | None) -> str | None:
    """`new URL(value).hostname`, or None where Node's constructor would throw."""
    if not isinstance(value, str):
        return None
    try:
        return URL(value).host or ""
    except Exception:
        return None


def url_pathname(value: str) -> str | None:
    try:
        return URL(value).pathname
    except Exception:
        return None
