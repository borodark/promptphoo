"""Fetch-layer helpers the scheduler depends on.

Ports src/util/fetch/retryContext.ts and src/util/fetch/responseHeadersObserver.ts, plus the
two predicates from src/util/fetch/errors.ts that the retry policy reads
(`isTransientConnectionError`, `isHttpRateLimitError`). They live here rather than in fetch.py
so that fetch.py can adopt them (read `get_fetch_retry_context_max_retries()` as the default
`maxRetries` and raise an error class whose `name` is "HttpRateLimitError") without the
scheduler importing the HTTP client.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

T = TypeVar("T")

# --- retryContext.ts --------------------------------------------------------------------------

_fetch_retry_context: ContextVar[dict[str, int | None] | None] = ContextVar(
    "fetch_retry_context", default=None
)


@contextmanager
def fetch_retry_context(max_retries: int | None) -> Iterator[None]:
    """Scope in which nested fetches inherit the provider's configured `maxRetries`.

    When `max_retries` is None the new scope deliberately shadows any outer provider context so
    providers without an override fall back to defaults.
    """
    token = _fetch_retry_context.set({"maxRetries": max_retries})
    try:
        yield
    finally:
        _fetch_retry_context.reset(token)


async def with_fetch_retry_context(max_retries: int | None, fn: Callable[[], Awaitable[T]]) -> T:
    with fetch_retry_context(max_retries):
        return await fn()


def get_fetch_retry_context_max_retries() -> int | None:
    """The active context's `maxRetries`, or None when none is set."""
    store = _fetch_retry_context.get()
    return store.get("maxRetries") if store else None


# --- errors.ts (predicates only) --------------------------------------------------------------


def _error_message(error: BaseException | None) -> str:
    if error is None:
        return ""
    message = getattr(error, "message", None)
    return message if isinstance(message, str) else str(error)


def is_http_rate_limit_error(err: Any) -> bool:
    """`err instanceof HttpRateLimitError`. Duck-typed on the error's name until fetch.py
    ports the class; such errors carry `kind` ('rate_limit' | 'quota')."""
    return isinstance(err, BaseException) and (
        getattr(err, "name", None) == "HttpRateLimitError"
        or type(err).__name__ == "HttpRateLimitError"
    )


def is_transient_connection_error(error: BaseException | None) -> bool:
    if not error:
        return False
    # Check error.code first; system errors always set it.
    code = getattr(error, "code", None)
    if code in ("ECONNRESET", "EPIPE") or isinstance(error, (ConnectionResetError,
                                                               BrokenPipeError)):  # fmt: skip
        return True
    message = _error_message(error).lower()
    # EPROTO can wrap permanent TLS misconfigs. Exclude when paired with known permanent
    # error phrases to avoid futile retries.
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
