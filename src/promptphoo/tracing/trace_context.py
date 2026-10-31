"""Ports src/tracing/traceContext.ts: fetchTraceContext, extractTraceIdFromTraceparent and the
TraceContextData shape.

TraceContextData is a dict: {traceId, spans: [TraceSpan], insights: [str], fetchedAt}. A
TraceSpan is {spanId, parentSpanId?, name, kind, startTime, endTime?, durationMs, attributes,
status: {code: 'unset'|'ok'|'error', message?}, depth, events: []}; JS-undefined keys are
omitted, as JSON.stringify would drop them.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from promptphoo.jsvalues import js_string, json_stringify
from promptphoo.scheduler.cancellation import AbortError
from promptphoo.tracing.tool_attributes import get_tool_name_from_attributes

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_MS = 500
DEFAULT_QUERY_DELAY_MS = 3000
EXTERNAL_SPAN_BATCH_SIZE = 500

SPAN_KIND_MAP: dict[int, str] = {
    0: "unspecified",
    1: "internal",
    2: "server",
    3: "client",
    4: "producer",
    5: "consumer",
}

# Keyed by id(providerConfig): {key: task}. Upstream keys a WeakMap by the config object, so
# only callers that share one config object share an in-flight fetch.
_in_flight_external_fetches: dict[int, tuple[Any, dict[str, asyncio.Task[Any]]]] = {}

_HEX2 = re.compile(r"[0-9a-f]{2}\Z", re.IGNORECASE)
_HEX32 = re.compile(r"[0-9a-f]{32}\Z", re.IGNORECASE)
_HEX16 = re.compile(r"[0-9a-f]{16}\Z", re.IGNORECASE)
_ZEROS = re.compile(r"0+\Z")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _resolve_span_kind(span: dict[str, Any]) -> str:
    attributes = span.get("attributes") or {}
    attribute_kind = (
        attributes.get("span.kind")
        or attributes.get("otel.span.kind")
        or attributes.get("spanKind")
        or attributes.get("kind")
    )
    if attribute_kind:
        return js_string(attribute_kind).lower()
    numeric_kind = attributes.get("otel.span.kind_code")
    if (
        isinstance(numeric_kind, int | float)
        and not isinstance(numeric_kind, bool)
        and numeric_kind in SPAN_KIND_MAP
    ):
        return SPAN_KIND_MAP[int(numeric_kind)]
    return "unspecified"


def _map_status_code(span: dict[str, Any]) -> str:
    code = span.get("statusCode")
    if code == 1 and not isinstance(code, bool):
        return "ok"
    if code == 2 and not isinstance(code, bool):
        return "error"
    return "unset"


def _compute_span_depth(
    span: dict[str, Any],
    span_map: dict[str, dict[str, Any]],
    depth_cache: dict[str, int | None],
) -> int | None:
    """The depth of a span in the span tree, or None for a cyclic parent chain."""
    ancestors: list[dict[str, Any]] = []
    visited: set[str] = set()
    current: dict[str, Any] | None = span
    depth = -1
    while current is not None:
        span_id = current.get("spanId")
        if span_id in depth_cache:
            cached = depth_cache[span_id]
            if cached is None:
                for ancestor in ancestors:
                    depth_cache[ancestor.get("spanId")] = None  # type: ignore[index]
                return None
            depth = cached
            break
        if span_id in visited:
            for ancestor in ancestors:
                depth_cache[ancestor.get("spanId")] = None  # type: ignore[index]
            return None
        visited.add(span_id)  # type: ignore[arg-type]
        ancestors.append(current)
        parent = current.get("parentSpanId")
        current = span_map.get(parent) if parent else None
    for ancestor in reversed(ancestors):
        depth += 1
        depth_cache[ancestor.get("spanId")] = depth  # type: ignore[index]
    return depth


def create_trace_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans_by_id = {span.get("spanId"): span for span in spans}
    depth_map: dict[str, int | None] = {}
    out = []
    for span in spans:
        start: Any = span.get("startTime")
        end = span.get("endTime")
        duration = max(0, (end if end is not None else start) - start)
        trace_span: dict[str, Any] = {"spanId": span.get("spanId")}
        if span.get("parentSpanId") is not None:
            trace_span["parentSpanId"] = span["parentSpanId"]
        trace_span["name"] = span.get("name")
        trace_span["kind"] = _resolve_span_kind(span)
        trace_span["startTime"] = start
        if end is not None:
            trace_span["endTime"] = end
        trace_span["durationMs"] = duration
        trace_span["attributes"] = span.get("attributes") or {}
        status: dict[str, Any] = {"code": _map_status_code(span)}
        if span.get("statusMessage") is not None:
            status["message"] = span["statusMessage"]
        trace_span["status"] = status
        depth = _compute_span_depth(span, spans_by_id, depth_map)  # type: ignore[arg-type]
        trace_span["depth"] = depth if depth is not None else 0
        trace_span["events"] = []
        out.append(trace_span)
    return out


def derive_insights(trace_spans: list[dict[str, Any]]) -> list[str]:
    if not trace_spans:
        return []
    insights: list[str] = []
    for span in trace_spans:
        if span["status"]["code"] == "error":
            message = span["status"].get("message")
            status_message = f": {message}" if message else ""
            insights.append(f'Error span "{span["name"]}" ({span["spanId"][:8]}){status_message}')
    for span in trace_spans:
        tool_name = get_tool_name_from_attributes(span["attributes"])
        if tool_name:
            duration = span.get("durationMs")
            insights.append(
                f'Tool call {tool_name} via "{span["name"]}" '
                f"(duration {js_string(duration if duration is not None else 0)}ms)"
            )
    for span in trace_spans:
        attributes = span["attributes"]
        if attributes.get("guardrail.name") or attributes.get("guardrails.decision"):
            decision = attributes.get("guardrails.decision")
            if decision is None:
                decision = attributes.get("guardrail.decision")
            name = attributes.get("guardrail.name")
            insights.append(
                f"Guardrail {js_string(name if name is not None else span['name'])} "
                f"decision: {js_string(decision if decision is not None else 'unknown')}"
            )
    return insights[:20]


def extract_trace_id_from_traceparent(traceparent: str | None) -> str | None:
    if not traceparent:
        return None
    parts = traceparent.split("-")
    padded = [*parts, "", "", "", ""]
    version, trace_id, parent_span_id, flags = padded[:4]
    valid_part_count = len(parts) == 4 if version.lower() == "00" else len(parts) >= 4
    if (
        not _HEX2.match(version)
        or version.lower() == "ff"
        or not valid_part_count
        or not _HEX32.match(trace_id)
        or _ZEROS.match(trace_id)
        or not _HEX16.match(parent_span_id)
        or _ZEROS.match(parent_span_id)
        or not _HEX2.match(flags)
    ):
        return None
    return trace_id.lower()


def _discard_cyclic_external_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    span_map = {span.get("spanId"): span for span in spans}
    depth_cache: dict[str, int | None] = {}
    cyclic: set[str] = set()
    valid = []
    for span in spans:
        if _compute_span_depth(span, span_map, depth_cache) is None:  # type: ignore[arg-type]
            cyclic.add(span.get("spanId"))  # type: ignore[arg-type]
        else:
            valid.append(span)
    if cyclic:
        logger.warning(
            "[TraceContext] Skipping %d spans with cyclic parent relationships", len(cyclic)
        )
    return valid


async def _store_external_spans(trace_id: str, spans: list[dict[str, Any]]) -> bool:
    """Store spans fetched from an external provider in the local database."""
    from promptphoo.tracing.store import get_trace_store

    try:
        trace_store = get_trace_store()
        for index in range(0, len(spans), EXTERNAL_SPAN_BATCH_SIZE):
            options: dict[str, Any] = {"warnIfMissingTrace": False}
            if index > 0:
                options["skipTraceCheck"] = True
            result = await trace_store.add_spans(
                trace_id, spans[index : index + EXTERNAL_SPAN_BATCH_SIZE], options
            )
            if not result.get("stored"):
                return False
        logger.debug("[TraceContext] Stored %d spans from external provider", len(spans))
        return True
    except Exception as e:
        logger.warning("[TraceContext] Failed to store external spans: %s", e)
        return False


def _sanitize_trace_attributes(attributes: Any, options: dict[str, Any]) -> dict[str, Any]:
    from promptphoo.tracing.sanitize_attributes import sanitize_trace_attributes

    return sanitize_trace_attributes(attributes, options)


def _redact_external_span(span: dict[str, Any], redact_attributes: list[str]) -> dict[str, Any]:
    attributes = span.get("attributes") or {}
    sanitized_attributes = _sanitize_trace_attributes(
        attributes,
        {
            "redactAttributes": redact_attributes,
            "sanitizeSensitiveAttributes": False,
            "truncateValues": False,
        },
    )
    redacted_values: list[str] = []
    pending: list[tuple[Any, Any]] = [(attributes, sanitized_attributes)]
    while pending:
        original, sanitized = pending.pop()
        if not isinstance(original, dict | list):
            if original is not None and sanitized == "[REDACTED]":
                value = js_string(original)
                if value and value not in redacted_values:
                    redacted_values.append(value)
            continue
        if isinstance(original, list):
            for index, item in enumerate(original):
                if sanitized == "[REDACTED]":
                    child: Any = sanitized
                elif isinstance(sanitized, list) and index < len(sanitized):
                    child = sanitized[index]
                else:
                    child = None
                pending.append((item, child))
            continue
        for key, value in original.items():
            if sanitized == "[REDACTED]":
                child = sanitized
            elif isinstance(sanitized, dict):
                child = sanitized.get(key)
            else:
                child = None
            pending.append((value, child))
    ordered = sorted(redacted_values, key=len, reverse=True)

    def scrub_echo(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        for redacted in ordered:
            value = value.replace(redacted, "[REDACTED]")
        return value

    out = dict(span)
    out["name"] = scrub_echo(span.get("name"))
    if "statusMessage" in span:
        out["statusMessage"] = scrub_echo(span.get("statusMessage"))
    out["attributes"] = sanitized_attributes
    return out


def _get_provider_fetch_options(
    span_options: dict[str, Any], abort_signal: Any = None
) -> dict[str, Any] | None:
    requires_post_fetch_filtering = span_options.get("includeInternalSpans") is False or bool(
        span_options.get("spanFilter")
    )
    provider_options: dict[str, Any] = {}
    if span_options.get("earliestStartTime") is not None:
        provider_options["earliestStartTime"] = span_options["earliestStartTime"]
    if span_options.get("maxSpans") is not None and not requires_post_fetch_filtering:
        provider_options["maxSpans"] = span_options["maxSpans"]
    if abort_signal:
        provider_options["abortSignal"] = abort_signal
    return provider_options or None


class TraceAbortError(AbortError):
    """createTraceAbortError: name AbortError, message "cancelled by user"."""

    def __init__(self, signal: Any = None) -> None:
        super().__init__("cancelled by user")
        self.__cause__ = getattr(signal, "reason", None)


def _create_trace_abort_error(signal: Any = None) -> TraceAbortError:
    return TraceAbortError(signal)


def _is_aborted(signal: Any) -> bool:
    return bool(signal is not None and getattr(signal, "aborted", False))


def _abort_reason(signal: Any) -> BaseException | None:
    reason = getattr(signal, "reason", None)
    return reason if isinstance(reason, BaseException) else None


async def _wait_for_retry(delay: float, signal: Any = None) -> None:
    if _is_aborted(signal):
        raise _create_trace_abort_error(signal)
    if signal is None or not hasattr(signal, "add_listener"):
        await asyncio.sleep(max(0.0, delay) / 1000)
        return
    loop = asyncio.get_running_loop()
    future: asyncio.Future[None] = loop.create_future()

    def on_abort() -> None:
        if not future.done():
            future.set_exception(_create_trace_abort_error(signal))

    def on_timer() -> None:
        if not future.done():
            future.set_result(None)

    timer = loop.call_later(max(0.0, delay) / 1000, on_timer)
    signal.add_listener(on_abort)
    try:
        await future
    finally:
        timer.cancel()
        signal.remove_listener(on_abort)


def _span_query_options(options: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in options.items()
        if k
        in (
            "earliestStartTime",
            "maxSpans",
            "maxDepth",
            "includeInternalSpans",
            "spanFilter",
            "sanitizeAttributes",
        )
    }


async def _fetch_from_external_provider(
    trace_id: str, provider_config: dict[str, Any], options: dict[str, Any]
) -> dict[str, Any] | None:
    from promptphoo.tracing.providers import TraceProviderError, create_trace_provider
    from promptphoo.tracing.store import get_trace_store

    query_delay = options["queryDelay"]
    max_retries = options["maxRetries"]
    retry_delay_ms = options["retryDelayMs"]
    redact_attributes = options.get("redactAttributes")
    abort_signal = options.get("abortSignal")
    span_options = _span_query_options(options)
    provider_fetch_options = _get_provider_fetch_options(span_options, abort_signal)

    try:
        provider = create_trace_provider(provider_config)
    except Exception as e:
        logger.warning("[TraceContext] Failed to initialize trace provider: %s", _error_text(e))
        return None

    if query_delay > 0:
        logger.debug(
            "[TraceContext] Waiting %sms for spans to arrive at external backend", query_delay
        )
        await _wait_for_retry(query_delay, abort_signal)

    for attempt in range(max_retries + 1):
        if _is_aborted(abort_signal):
            raise _create_trace_abort_error(abort_signal)
        try:
            result = await provider.fetch_trace(trace_id, provider_fetch_options)
            valid_spans = _discard_cyclic_external_spans(result["spans"]) if result else []
            if not result or not valid_spans:
                if attempt == max_retries:
                    logger.debug(
                        "[TraceContext] No spans found for trace %s from %s after %d attempts",
                        trace_id,
                        provider.id,
                        attempt + 1,
                    )
                    return None
                logger.debug(
                    "[TraceContext] No spans yet for trace %s from %s, retrying in %sms "
                    "(attempt %d/%d)",
                    trace_id,
                    provider.id,
                    retry_delay_ms,
                    attempt + 1,
                    max_retries,
                )
                await _wait_for_retry(retry_delay_ms, abort_signal)
                continue

            stored_spans = (
                [_redact_external_span(span, redact_attributes) for span in valid_spans]
                if redact_attributes
                else valid_spans
            )
            if not await _store_external_spans(trace_id, stored_spans):
                return None
            spans = await get_trace_store().get_spans(trace_id, span_options)
            if not spans:
                return None
            trace_spans = create_trace_spans(spans)
            insights = derive_insights(trace_spans)
            logger.debug(
                "[TraceContext] Resolved %d spans for trace %s from %s with %d insights",
                len(trace_spans),
                trace_id,
                provider.id,
                len(insights),
            )
            return {
                "traceId": trace_id,
                "spans": trace_spans,
                "insights": insights,
                "fetchedAt": result["fetchedAt"],
            }
        except Exception as e:
            if _is_aborted(abort_signal):
                raise _create_trace_abort_error(abort_signal) from getattr(
                    abort_signal, "reason", None
                )
            logger.error("[TraceContext] Failed to fetch from %s: %s", provider.id, _error_text(e))
            if attempt == max_retries or (isinstance(e, TraceProviderError) and not e.retryable):
                return None
            await _wait_for_retry(retry_delay_ms, abort_signal)
    return None


def _error_text(error: BaseException) -> str:
    """`${error}` for a JS Error: "Name: message"."""
    name = getattr(error, "name", None)
    if not isinstance(name, str):
        name = "TypeError" if isinstance(error, TypeError) else "Error"
    message = str(error)
    return f"{name}: {message}" if message else name


async def _fetch_from_local_store(trace_id: str, options: dict[str, Any]) -> dict[str, Any] | None:
    from promptphoo.tracing.store import get_trace_store

    max_retries = options["maxRetries"]
    retry_delay_ms = options["retryDelayMs"]
    abort_signal = options.get("abortSignal")
    redact_attributes = options.get("redactAttributes")
    span_options = _span_query_options(options)
    trace_store = get_trace_store()

    for attempt in range(max_retries + 1):
        if _is_aborted(abort_signal):
            raise _create_trace_abort_error(abort_signal)
        try:
            spans = await trace_store.get_spans(trace_id, span_options)
            if not spans:
                if attempt == max_retries:
                    logger.debug(
                        "[TraceContext] No spans found for trace %s after %d attempts",
                        trace_id,
                        attempt + 1,
                    )
                    return None
                logger.debug(
                    "[TraceContext] No spans yet for trace %s, retrying in %sms (attempt %d/%d)",
                    trace_id,
                    retry_delay_ms,
                    attempt + 1,
                    max_retries,
                )
                await _wait_for_retry(retry_delay_ms, abort_signal)
                continue
            if redact_attributes:
                spans = [
                    {
                        **span,
                        "attributes": _sanitize_trace_attributes(
                            span.get("attributes"),
                            {
                                "redactAttributes": redact_attributes,
                                "sanitizeSensitiveAttributes": span_options.get(
                                    "sanitizeAttributes"
                                ),
                            },
                        ),
                    }
                    for span in spans
                ]
            trace_spans = create_trace_spans(spans)
            insights = derive_insights(trace_spans)
            logger.debug(
                "[TraceContext] Resolved %d spans for trace %s with %d insights",
                len(trace_spans),
                trace_id,
                len(insights),
            )
            return {
                "traceId": trace_id,
                "spans": trace_spans,
                "insights": insights,
                "fetchedAt": _now_ms(),
            }
        except Exception as e:
            if _is_aborted(abort_signal):
                raise _create_trace_abort_error(abort_signal) from getattr(
                    abort_signal, "reason", None
                )
            logger.error(
                "[TraceContext] Failed to fetch spans for trace %s: %s", trace_id, _error_text(e)
            )
            if attempt == max_retries:
                return None
            await _wait_for_retry(retry_delay_ms, abort_signal)
    return None


def _is_external(provider_config: Any) -> bool:
    from promptphoo.tracing.providers import is_external_trace_provider

    return is_external_trace_provider(provider_config)


async def fetch_trace_context(
    trace_id: str, options: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Fetch trace context for a trace id: from the configured external provider (Tempo,
    Langfuse, Braintrust), otherwise from the local TraceStore. Options are camelCase, as
    upstream's FetchTraceContextOptions."""
    options = dict(options or {})

    def pop(key: str, default: Any) -> Any:
        value = options.pop(key, None)
        return default if value is None else value

    include_internal_spans = pop("includeInternalSpans", True)
    sanitize_attributes = pop("sanitizeAttributes", True)
    max_retries = pop("maxRetries", DEFAULT_MAX_RETRIES)
    retry_delay_ms = pop("retryDelayMs", DEFAULT_RETRY_DELAY_MS)
    provider_config = options.pop("providerConfig", None)
    query_delay = pop("queryDelay", DEFAULT_QUERY_DELAY_MS)
    span_options = {k: v for k, v in options.items() if v is not None}

    fetch_options: dict[str, Any] = {
        "maxRetries": max_retries,
        "retryDelayMs": retry_delay_ms,
        "includeInternalSpans": include_internal_spans,
        "sanitizeAttributes": sanitize_attributes,
        **span_options,
    }

    if _is_external(provider_config):
        request_options = {"queryDelay": query_delay, **fetch_options}
        # Calls with an abort signal retain independent cancellation ownership.
        if request_options.get("abortSignal"):
            return await _fetch_from_external_provider(trace_id, provider_config, request_options)
        key = json_stringify({"traceId": trace_id, **request_options})
        entry = _in_flight_external_fetches.get(id(provider_config))
        if entry is None or entry[0] is not provider_config:
            entry = (provider_config, {})
            _in_flight_external_fetches[id(provider_config)] = entry
        pending = entry[1]
        existing = pending.get(key)
        if existing is not None:
            return await asyncio.shield(existing)

        task = asyncio.ensure_future(
            _fetch_from_external_provider(trace_id, provider_config, request_options)
        )

        def done(_: Any) -> None:
            pending.pop(key, None)
            if not pending and _in_flight_external_fetches.get(id(provider_config)) is entry:
                _in_flight_external_fetches.pop(id(provider_config), None)

        task.add_done_callback(done)
        pending[key] = task
        return await asyncio.shield(task)

    return await _fetch_from_local_store(trace_id, fetch_options)
