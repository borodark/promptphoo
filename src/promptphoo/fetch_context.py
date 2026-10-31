"""Fetch-layer helpers the scheduler depends on.

Ports src/util/fetch/retryContext.ts and src/util/fetch/responseHeadersObserver.ts, plus the
two predicates from src/util/fetch/errors.ts that the retry policy reads
(`isTransientConnectionError`, `isHttpRateLimitError`). They live here rather than in fetch.py
so that fetch.py can adopt them (read `get_fetch_retry_context_max_retries()` as the default
`maxRetries` and raise an error class whose `name` is "HttpRateLimitError") without the
scheduler importing the HTTP client.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

T = TypeVar("T")

# --- retryContext.ts / errors.ts ------------------------------------------------------------
# One retry-context ContextVar and one set of error predicates, owned by promptphoo.fetch, so
# the scheduler's `maxRetries` scope reaches nested fetches.

from promptphoo.fetch import (  # noqa: E402
    get_fetch_retry_context_max_retries as get_fetch_retry_context_max_retries,
)
from promptphoo.fetch import with_fetch_retry_context as fetch_retry_context  # noqa: E402
from promptphoo.fetch.errors import HttpRateLimitError  # noqa: E402
from promptphoo.fetch.errors import (  # noqa: E402
    is_transient_connection_error as is_transient_connection_error,
)


async def with_fetch_retry_context(max_retries: int | None, fn: Callable[[], Awaitable[T]]) -> T:
    with fetch_retry_context(max_retries):
        return await fn()


def is_http_rate_limit_error(err: Any) -> bool:
    """`err instanceof HttpRateLimitError`."""
    return isinstance(err, HttpRateLimitError) or (
        isinstance(err, BaseException) and getattr(err, "name", None) == "HttpRateLimitError"
    )


# --- responseHeadersObserver.ts ---------------------------------------------------------------


class _ObserverErrorResponse(dict):
    """A provider response converted from a caller-observer exception.

    Node marks the object with a private, non-enumerable symbol; the Python equivalent is this
    dict subclass, which (like the symbol) is lost when the response is copied with `{**r}`.
    """


def _mark_response_headers_observer_error_response(response: T) -> T:
    if isinstance(response, dict) and not isinstance(response, _ObserverErrorResponse):
        return _ObserverErrorResponse(response)  # type: ignore[return-value]
    return response


def is_response_headers_observer_error_response(response: Any) -> bool:
    return isinstance(response, _ObserverErrorResponse)


def preserve_response_headers_observer_error_response(source: Any, response: T) -> T:
    """Copy private caller-exception provenance when the same error response is projected."""
    if is_response_headers_observer_error_response(source):
        return _mark_response_headers_observer_error_response(response)
    return response


# observe(args, already_observed) -> bool; args is (headers, backoff)
Observe = Callable[[tuple[dict[str, str], Any], bool], bool]
ResponseHeadersObserver = Callable[..., None]


@dataclass
class _ObserverOwner:
    owner: object
    observe: Observe
    caller: ResponseHeadersObserver | None = None
    errors: list[BaseException] = field(default_factory=list)


class _OwnedObserver:
    """Callable observer that carries its owner record (Node stores it under a symbol)."""

    def __init__(self, record: _ObserverOwner | None, errors: list[BaseException]) -> None:
        self._owner_record = record
        self._caller_errors = errors

    def __call__(self, headers: dict[str, str], backoff: Any = None) -> None:
        record = self._owner_record
        assert record is not None
        args = (headers, backoff)
        observed = record.observe(args, False)
        caller = record.caller
        owners = [record]
        while caller:
            inherited = getattr(caller, "_owner_record", None)
            if inherited is None:
                try:
                    caller(*args) if backoff is not None else caller(headers)
                except BaseException as error:
                    # Keep provenance local to these calls without mutating the thrown value.
                    for owner in owners:
                        owner.errors.append(error)
                    raise
                break
            # A nested provider must not publish its quota to an unrelated ancestor.
            # Matching owners still record delivery so later result handling will not parse
            # the same relative reset again, but only one live call updates quota.
            if inherited.owner is record.owner:
                observed = inherited.observe(args, observed)
            owners.append(inherited)
            caller = inherited.caller


def _owned_observer(record: _ObserverOwner) -> ResponseHeadersObserver:
    return _OwnedObserver(record, record.errors)


def create_response_headers_observer(owner: object, observe: Observe) -> ResponseHeadersObserver:
    return _owned_observer(_ObserverOwner(owner=owner, observe=observe))


class _CombinedObserver:
    def __init__(self, observer: ResponseHeadersObserver, caller: ResponseHeadersObserver):
        self._observer = observer
        self._caller = caller
        self._caller_errors: list[BaseException] = []

    def __call__(self, headers: dict[str, str], backoff: Any = None) -> None:
        args = (headers,) if backoff is None else (headers, backoff)
        self._observer(*args)
        try:
            self._caller(*args)
        except BaseException as error:
            self._caller_errors.append(error)
            raise


def compose_response_headers_observers(
    observer: ResponseHeadersObserver, caller: ResponseHeadersObserver | None
) -> ResponseHeadersObserver:
    if not caller:
        return observer
    record = getattr(observer, "_owner_record", None)
    if record is not None:
        return _owned_observer(replace(record, caller=caller))
    # Preserve the callback contract for structurally supplied registries too.
    return _CombinedObserver(observer, caller)


def is_response_headers_observer_error(
    observer: ResponseHeadersObserver | None, error: Any
) -> bool:
    errors = getattr(observer, "_caller_errors", None) if observer is not None else None
    return bool(errors) and any(e is error for e in errors or [])


def preserve_response_headers_observer_error(
    observer: ResponseHeadersObserver | None, error: Any, response: T
) -> T:
    """Retain the provider's existing diagnostic when it converts a caller exception."""
    if is_response_headers_observer_error(observer, error):
        return _mark_response_headers_observer_error_response(response)
    return response
