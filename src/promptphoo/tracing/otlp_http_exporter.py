"""OTLP/HTTP JSON span export, standing in for `@opentelemetry/exporter-trace-otlp-http`.

The request body is the ExportTraceServiceRequest that the JS JsonTraceSerializer writes
(otlp-transformer's `createExportTraceServiceRequest` with the JSON encoder): hex ids,
nanosecond timestamps as decimal strings, `kind` offset by one, attributes as
`{key, value: {stringValue | intValue | doubleValue | boolValue | arrayValue}}`, and
undefined fields omitted. Spans are grouped by resource, then by instrumentation scope.

Transport follows otlp-exporter-base: POST with `Content-Type: application/json`, headers from
OTEL_EXPORTER_OTLP_HEADERS and OTEL_EXPORTER_OTLP_TRACES_HEADERS, a timeout from
OTEL_EXPORTER_OTLP_TRACES_TIMEOUT / OTEL_EXPORTER_OTLP_TIMEOUT (10 s), and up to five retries
with jittered exponential backoff for 429/502/503/504, honouring Retry-After. The configured
URL is used as given, as when `url` is passed to the JS exporter.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import os
import random
import time
from collections.abc import Sequence
from typing import Any
from urllib.parse import unquote

import httpx

from promptphoo import UPSTREAM_VERSION
from promptphoo.jsvalues import json_stringify
from promptphoo.tracing.otel_trace_sdk import (
    ExportResult,
    ExportResultCode,
    ReadableSpan,
    hr_time_to_nanoseconds,
)

logger = logging.getLogger("promptphoo.tracing")

MAX_ATTEMPTS = 5
INITIAL_BACKOFF = 1000
MAX_BACKOFF = 5000
BACKOFF_MULTIPLIER = 1.5
JITTER = 0.2
DEFAULT_TIMEOUT_MILLIS = 10000
USER_AGENT = f"OTel-OTLP-Exporter-Python-promptphoo/{UPSTREAM_VERSION}"

SPAN_FLAGS_CONTEXT_HAS_IS_REMOTE_MASK = 0x100
SPAN_FLAGS_CONTEXT_IS_REMOTE_MASK = 0x200


# --- serialization --------------------------------------------------------------------------


def to_any_value(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": value}
    if isinstance(value, float):
        return {"intValue": int(value)} if value.is_integer() else {"doubleValue": value}
    if isinstance(value, bytes | bytearray):
        import base64

        return {"bytesValue": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, list | tuple):
        return {"arrayValue": {"values": [to_any_value(v) for v in value]}}
    if isinstance(value, dict):
        return {
            "kvlistValue": {
                "values": [{"key": k, "value": to_any_value(v)} for k, v in value.items()]
            }
        }
    return {}


def to_attributes(attributes: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": key, "value": to_any_value(value)} for key, value in attributes.items()]


def _span_flags(trace_flags: int, is_remote: bool | None) -> int:
    flags = (trace_flags & 0xFF) | SPAN_FLAGS_CONTEXT_HAS_IS_REMOTE_MASK
    if is_remote:
        flags |= SPAN_FLAGS_CONTEXT_IS_REMOTE_MASK
    return flags


def _encode_time(hr_time: Any) -> str:
    return str(hr_time_to_nanoseconds(hr_time))


def span_to_otlp(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.span_context()
    parent = span.parent_span_context
    out: dict[str, Any] = {"traceId": ctx.trace_id, "spanId": ctx.span_id}
    if parent is not None and parent.span_id:
        out["parentSpanId"] = parent.span_id
    if ctx.trace_state:
        out["traceState"] = ctx.trace_state
    status: dict[str, Any] = {"code": int(span.status.code)}
    if span.status.message is not None:
        status["message"] = span.status.message
    out.update(
        {
            "name": span.name,
            "kind": 0 if span.kind is None else int(span.kind) + 1,
            "startTimeUnixNano": _encode_time(span.start_time),
            "endTimeUnixNano": _encode_time(span.end_time),
            "attributes": to_attributes(span.attributes),
            "droppedAttributesCount": span.dropped_attributes_count,
            "events": [
                {
                    "attributes": to_attributes(event.attributes),
                    "name": event.name,
                    "timeUnixNano": _encode_time(event.time),
                    "droppedAttributesCount": event.dropped_attributes_count or 0,
                }
                for event in span.events
            ],
            "droppedEventsCount": span.dropped_events_count,
            "status": status,
            "links": [],
            "droppedLinksCount": span.dropped_links_count,
            "flags": _span_flags(ctx.trace_flags, parent.is_remote if parent else None),
        }
    )
    return out


def create_export_trace_service_request(spans: Sequence[ReadableSpan]) -> dict[str, Any]:
    resource_map: dict[int, tuple[Any, dict[str, list[ReadableSpan]]]] = {}
    for span in spans:
        entry = resource_map.setdefault(id(span.resource), (span.resource, {}))
        scope = span.instrumentation_scope
        key = f"{scope.name}@{scope.version or ''}:{scope.schema_url or ''}"
        entry[1].setdefault(key, []).append(span)

    resource_spans = []
    for resource, scopes in resource_map.values():
        scope_spans = []
        for records in scopes.values():
            scope = records[0].instrumentation_scope
            scope_entry: dict[str, Any] = {"scope": {"name": scope.name}}
            if scope.version is not None:
                scope_entry["scope"]["version"] = scope.version
            scope_entry["spans"] = [span_to_otlp(span) for span in records]
            if scope.schema_url is not None:
                scope_entry["schemaUrl"] = scope.schema_url
            scope_spans.append(scope_entry)
        processed_resource: dict[str, Any] = {
            "attributes": to_attributes(resource.attributes),
            "droppedAttributesCount": 0,
        }
        if resource.schema_url:
            processed_resource["schemaUrl"] = resource.schema_url
        entry_out: dict[str, Any] = {"resource": processed_resource, "scopeSpans": scope_spans}
        if resource.schema_url:
            entry_out["schemaUrl"] = resource.schema_url
        resource_spans.append(entry_out)
    return {"resourceSpans": resource_spans}


# --- configuration --------------------------------------------------------------------------


def parse_key_pairs_into_record(raw: str | None) -> dict[str, str]:
    """@opentelemetry/core parseKeyPairsIntoRecord (`k1=v1,k2=v2`, values URI-decoded)."""
    out: dict[str, str] = {}
    if not raw:
        return out
    for entry in raw.split(","):
        key, sep, value = entry.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if key:
            out[key] = unquote(value)
    return out


def _headers_from_env() -> dict[str, str]:
    return {
        **parse_key_pairs_into_record(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")),
        **parse_key_pairs_into_record(os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS")),
    }


def _timeout_from_env() -> int:
    for name in ("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "OTEL_EXPORTER_OTLP_TIMEOUT"):
        raw = os.environ.get(name)
        if raw:
            try:
                value = float(raw)
            except ValueError:
                continue
            if value > 0:
                return int(value)
    return DEFAULT_TIMEOUT_MILLIS


def parse_retry_after_to_millis(retry_after: str | None) -> float | None:
    if retry_after is None:
        return None
    try:
        seconds = int(retry_after.strip())
    except ValueError:
        seconds = None
    if seconds is not None:
        return seconds * 1000 if seconds > 0 else -1
    try:
        when = email.utils.parsedate_to_datetime(retry_after)
    except (TypeError, ValueError):
        return 0
    delay = when.timestamp() * 1000 - time.time() * 1000
    return delay if delay >= 0 else 0


def is_export_http_error_retryable(status_code: int) -> bool:
    return status_code in (429, 502, 503, 504)


class OTLPTraceExporter:
    """POST spans to `url` as OTLP JSON."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout_millis: int | None = None,
    ) -> None:
        self.url = url
        self.headers = {**_headers_from_env(), **(headers or {})}
        self.timeout_millis = timeout_millis or _timeout_from_env()
        self._shutdown = False
        self._pending: set[asyncio.Task[Any]] = set()

    async def _send_once(
        self, client: httpx.AsyncClient, body: bytes, timeout_millis: float
    ) -> tuple[str, float | None, BaseException | None]:
        headers = {
            **self.headers,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        try:
            response = await client.post(
                self.url, content=body, headers=headers, timeout=max(timeout_millis, 0) / 1000
            )
        except httpx.TimeoutException as e:
            return "failure", None, e
        except httpx.HTTPError as e:
            return "retryable", None, e
        if 200 <= response.status_code < 300:
            return "success", None, None
        error = RuntimeError(f"Export failed with status code {response.status_code}")
        if is_export_http_error_retryable(response.status_code):
            return (
                "retryable",
                parse_retry_after_to_millis(response.headers.get("retry-after")),
                error,
            )
        return "failure", None, error

    async def _send(self, body: bytes) -> BaseException | None:
        deadline = time.monotonic() * 1000 + self.timeout_millis
        async with httpx.AsyncClient() as client:
            status, retry_in, error = await self._send_once(client, body, self.timeout_millis)
            attempts = MAX_ATTEMPTS
            next_backoff: float = INITIAL_BACKOFF
            while status == "retryable" and attempts > 0:
                attempts -= 1
                jitter = random.random() * (2 * JITTER) - JITTER
                backoff = max(min(next_backoff * (1 + jitter), MAX_BACKOFF), 0)
                next_backoff *= BACKOFF_MULTIPLIER
                wait = retry_in if retry_in is not None else backoff
                remaining = deadline - time.monotonic() * 1000
                if wait > remaining:
                    logger.debug(
                        "Export retry time %dms exceeds remaining timeout %dms, not retrying "
                        "further.",
                        round(wait),
                        round(remaining),
                    )
                    break
                await asyncio.sleep(max(wait, 0) / 1000)
                status, retry_in, error = await self._send_once(
                    client, body, deadline - time.monotonic() * 1000
                )
        return None if status == "success" else error

    async def export(self, spans: Sequence[ReadableSpan]) -> ExportResult:
        if self._shutdown:
            return ExportResult(ExportResultCode.FAILED, RuntimeError("Exporter has been shutdown"))
        body = json_stringify(create_export_trace_service_request(spans)).encode("utf-8")
        task = asyncio.ensure_future(self._send(body))
        self._pending.add(task)
        try:
            error = await task
        except Exception as e:
            error = e
        finally:
            self._pending.discard(task)
        if error is not None:
            logger.debug("[OTLPTraceExporter] export failed: %s", error)
            return ExportResult(ExportResultCode.FAILED, error)
        return ExportResult(ExportResultCode.SUCCESS)

    async def force_flush(self) -> None:
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    async def shutdown(self) -> None:
        self._shutdown = True
        await self.force_flush()
