"""A span exporter that writes finished spans to the local TraceStore (SQLite).

Ports src/tracing/localSpanExporter.ts. Spans are grouped by trace id and converted to the
store's SpanData dicts: HrTime start/end become milliseconds (`s * 1e3 + ns / 1e6`), resource
attributes come first and span attributes override them, status code and message are copied.
Spans whose trace record is missing (grading calls outside an evaluation trace) are skipped
after one retry.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from promptphoo.tracing.otel_trace_sdk import ExportResult, ExportResultCode, ReadableSpan

logger = logging.getLogger("promptphoo.tracing")

MISSING_TRACE_RETRY_DELAY_MS = 50


def _stored(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("stored"))
    return bool(getattr(result, "stored", False))


def _reason(result: Any) -> Any:
    if isinstance(result, dict):
        return result.get("reason")
    return getattr(result, "reason", None)


def to_span_data(span: ReadableSpan) -> dict[str, Any]:
    """Convert a finished span to the store's SpanData."""
    span_context = span.span_context()
    start_time_ms = span.start_time[0] * 1e3 + span.start_time[1] / 1e6
    end_time_ms = span.end_time[0] * 1e3 + span.end_time[1] / 1e6
    parent = span.parent_span_context
    span_data: dict[str, Any] = {"spanId": span_context.span_id}
    if parent is not None and parent.span_id:
        span_data["parentSpanId"] = parent.span_id
    span_data.update(
        {
            "name": span.name,
            "startTime": start_time_ms,
            "endTime": end_time_ms,
            "attributes": {**span.resource.attributes, **span.attributes},
            "statusCode": int(span.status.code),
        }
    )
    if span.status.message is not None:
        span_data["statusMessage"] = span.status.message
    return span_data


class LocalSpanExporter:
    async def export(self, spans: Sequence[ReadableSpan]) -> ExportResult:
        try:
            error = await self._export(spans)
        except Exception as e:
            logger.error("[LocalSpanExporter] Failed to export spans: %s", e)
            return ExportResult(ExportResultCode.FAILED, e)
        if error is not None:
            return ExportResult(ExportResultCode.FAILED, error)
        return ExportResult(ExportResultCode.SUCCESS)

    async def _export(self, spans: Sequence[ReadableSpan]) -> BaseException | None:
        """Returns the first non-FK error, or None when every trace was handled."""
        if not spans:
            return None
        from promptphoo.tracing.store import get_trace_store

        trace_store = get_trace_store()
        logger.debug("[LocalSpanExporter] Exporting %d spans", len(spans))

        spans_by_trace: dict[str, list[dict[str, Any]]] = {}
        for span in spans:
            spans_by_trace.setdefault(span.span_context().trace_id, []).append(to_span_data(span))

        first_error: BaseException | None = None
        for trace_id, span_data_list in spans_by_trace.items():
            try:
                options = {"skipTraceCheck": False, "warnIfMissingTrace": False}
                result = await trace_store.add_spans(trace_id, span_data_list, options)
                if not _stored(result):
                    await asyncio.sleep(MISSING_TRACE_RETRY_DELAY_MS / 1000)
                    result = await trace_store.add_spans(trace_id, span_data_list, options)
                if _stored(result):
                    logger.debug(
                        "[LocalSpanExporter] Added %d spans to trace %s",
                        len(span_data_list),
                        trace_id,
                    )
                else:
                    # Spans from grading calls or other internal operations that were not
                    # started through the evaluation tracing flow.
                    logger.debug(
                        "[LocalSpanExporter] Skipping %d spans for orphan trace %s: %s",
                        len(span_data_list),
                        trace_id,
                        _reason(result),
                    )
            except Exception as error:
                if "FOREIGN KEY" in str(error):
                    # The trace was deleted between the check and the insert.
                    logger.debug(
                        "[LocalSpanExporter] Skipping %d spans for orphan trace %s",
                        len(span_data_list),
                        trace_id,
                    )
                else:
                    logger.error(
                        "[LocalSpanExporter] Failed to add spans to trace %s: %s", trace_id, error
                    )
                    if first_error is None:
                        first_error = error
        return first_error

    async def shutdown(self) -> None:
        logger.debug("[LocalSpanExporter] Shutting down")

    async def force_flush(self) -> None:
        return None
