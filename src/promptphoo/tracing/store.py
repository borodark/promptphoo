# SHIM: replaced by the trace-store branch at merge
"""Ports src/tracing/store.ts (TraceStore over the drizzle `traces` and `spans` tables).

Spans and traces are camelCase dicts with the keys of upstream's SpanData and TraceData.
JS `undefined` fields are omitted.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from promptphoo import db

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _rows(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cursor = db.get_db().cursor()
    cursor.row_factory = None
    cursor.execute(sql, params)
    columns = [c[0] for c in cursor.description or ()]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _sanitize(attributes: Any) -> dict[str, Any]:
    from promptphoo.tracing.sanitize_attributes import sanitize_trace_attributes

    return sanitize_trace_attributes(attributes)


def _serialize_span(
    row: dict[str, Any], should_sanitize: bool = True, raw_attributes: Any = None
) -> dict[str, Any]:
    if raw_attributes is None:
        raw_attributes = db.from_db_json(row.get("attributes"))
    span: dict[str, Any] = {"spanId": row["span_id"]}
    if row.get("parent_span_id") is not None:
        span["parentSpanId"] = row["parent_span_id"]
    span["name"] = row["name"]
    span["startTime"] = row["start_time"]
    if row.get("end_time") is not None:
        span["endTime"] = row["end_time"]
    if raw_attributes is not None:
        span["attributes"] = _sanitize(raw_attributes) if should_sanitize else raw_attributes
    if row.get("status_code") is not None:
        span["statusCode"] = row["status_code"]
    if row.get("status_message") is not None:
        span["statusMessage"] = row["status_message"]
    return span


def _trace_dict(row: dict[str, Any], spans: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "traceId": row["trace_id"],
        "evaluationId": row["evaluation_id"],
        "testCaseId": row["test_case_id"],
    }
    metadata = db.from_db_json(row.get("metadata"))
    if metadata is not None:
        out["metadata"] = metadata
    out["spans"] = spans
    return out


def _compute_depth(
    span: dict[str, Any], span_map: dict[str, dict[str, Any]], depth_cache: dict[str, int]
) -> int:
    chain: list[str] = []
    current: dict[str, Any] | None = span
    depth = -1
    while current is not None:
        span_id = current["spanId"]
        if span_id in depth_cache:
            depth = depth_cache[span_id]
            break
        if span_id in chain:  # upstream recurses forever here; stop instead
            depth = -1
            break
        chain.append(span_id)
        parent = current.get("parentSpanId")
        current = span_map.get(parent) if parent else None
    for span_id in reversed(chain):
        depth += 1
        depth_cache[span_id] = depth
    return depth_cache[span["spanId"]]


class TraceStore:
    async def create_trace(self, trace: dict[str, Any]) -> None:
        try:
            logger.debug(
                "[TraceStore] Creating trace %s for evaluation %s",
                trace["traceId"],
                trace.get("evaluationId"),
            )
            conn = db.get_db()
            with db.transaction(conn):
                conn.execute(
                    "INSERT INTO traces (id, trace_id, evaluation_id, test_case_id, created_at, "
                    "metadata) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (trace_id) DO NOTHING",
                    (
                        str(uuid.uuid4()),
                        trace["traceId"],
                        trace.get("evaluationId"),
                        trace.get("testCaseId"),
                        _now_ms(),
                        db.to_db_json(trace.get("metadata")),
                    ),
                )
        except Exception as e:
            logger.error("[TraceStore] Failed to create trace: %s", e)
            raise

    async def add_spans(
        self, trace_id: str, spans: list[dict[str, Any]], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        options = options or {}
        try:
            if not options.get("skipTraceCheck"):
                found = _rows("SELECT 1 AS x FROM traces WHERE trace_id = ? LIMIT 1", (trace_id,))
                if not found:
                    message = (
                        f"[TraceStore] Trace {trace_id} not found, skipping {len(spans)} spans. "
                        "This may indicate spans arrived before trace was created."
                    )
                    if options.get("warnIfMissingTrace") is False:
                        logger.debug(message)
                    else:
                        logger.warning(message)
                    return {"stored": False, "reason": f"Trace {trace_id} not found"}
            if not spans:
                return {"stored": True}
            conn = db.get_db()
            with db.transaction(conn):
                conn.executemany(
                    "INSERT INTO spans (id, trace_id, span_id, parent_span_id, name, start_time, "
                    "end_time, attributes, status_code, status_message) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (trace_id, span_id) DO NOTHING",
                    [
                        (
                            str(uuid.uuid4()),
                            trace_id,
                            span.get("spanId"),
                            span.get("parentSpanId"),
                            span.get("name"),
                            span.get("startTime"),
                            span.get("endTime"),
                            db.to_db_json(span.get("attributes")),
                            span.get("statusCode"),
                            span.get("statusMessage"),
                        )
                        for span in spans
                    ],
                )
            return {"stored": True}
        except Exception as e:
            logger.error("[TraceStore] Failed to add spans: %s", e)
            raise

    async def get_traces_by_evaluation(
        self, evaluation_id: str, options: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        should_sanitize = (options or {}).get("sanitizeAttributes", True) is not False
        traces = _rows("SELECT * FROM traces WHERE evaluation_id = ?", (evaluation_id,))
        out = []
        for trace in traces:
            spans = _rows("SELECT * FROM spans WHERE trace_id = ?", (trace["trace_id"],))
            out.append(_trace_dict(trace, [_serialize_span(s, should_sanitize) for s in spans]))
        return out

    async def get_trace(
        self, trace_id: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        should_sanitize = (options or {}).get("sanitizeAttributes", True) is not False
        traces = _rows("SELECT * FROM traces WHERE trace_id = ? LIMIT 1", (trace_id,))
        if not traces:
            logger.debug("[TraceStore] Trace %s not found", trace_id)
            return None
        spans = _rows("SELECT * FROM spans WHERE trace_id = ?", (trace_id,))
        return _trace_dict(traces[0], [_serialize_span(s, should_sanitize) for s in spans])

    async def get_trace_metadata(self, trace_id: str) -> dict[str, Any] | None:
        rows = _rows("SELECT metadata FROM traces WHERE trace_id = ? LIMIT 1", (trace_id,))
        if not rows:
            return None
        return db.from_db_json(rows[0]["metadata"]) or {}

    async def delete_old_traces(self, days: float) -> None:
        cutoff = _now_ms() - days * 24 * 60 * 60 * 1000
        cutoff_text = datetime.fromtimestamp(cutoff / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S")
        condition = (
            "((typeof(created_at) in ('integer', 'real') and created_at < ?) "
            "or (typeof(created_at) = 'text' and ("
            "(cast(created_at as integer) > 1000000000000 and cast(created_at as integer) < ?) "
            "or datetime(created_at) < datetime(?))))"
        )
        params = (cutoff, cutoff, cutoff_text)
        conn = db.get_db()
        with db.transaction(conn):
            conn.execute(
                "DELETE FROM spans WHERE trace_id IN "
                f"(SELECT trace_id FROM traces WHERE {condition})",
                params,
            )
            conn.execute(f"DELETE FROM traces WHERE {condition}", params)

    async def get_spans(
        self, trace_id: str, options: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        from promptphoo.tracing.span_filter import is_relevant_span, matches_span_filter
        from promptphoo.tracing.span_roles import SPAN_ROLE_ATTRIBUTE

        options = options or {}
        earliest = options.get("earliestStartTime")
        max_spans = options.get("maxSpans")
        max_depth = options.get("maxDepth")
        include_internal = options.get("includeInternalSpans", True) is not False
        span_filter = options.get("spanFilter")
        should_sanitize = options.get("sanitizeAttributes", True) is not False

        rows = _rows(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time ASC, span_id ASC",
            (trace_id,),
        )
        parsed = {row["span_id"]: db.from_db_json(row.get("attributes")) for row in rows}
        rows_by_id = {row["span_id"]: row for row in rows}
        grader_owned: dict[str, bool] = {}
        span_map: dict[str, dict[str, Any]] = {}
        for row in rows:
            if earliest and row["start_time"] < earliest:
                continue
            raw_attributes = parsed[row["span_id"]] or {}
            if not include_internal:
                ancestor: dict[str, Any] | None = row
                visited: list[str] = []
                belongs = False
                while ancestor is not None and ancestor["span_id"] not in visited:
                    cached = grader_owned.get(ancestor["span_id"])
                    if cached is not None:
                        belongs = cached
                        break
                    visited.append(ancestor["span_id"])
                    attrs = parsed.get(ancestor["span_id"])
                    if isinstance(attrs, dict) and attrs.get(SPAN_ROLE_ATTRIBUTE) == "grader":
                        belongs = True
                        break
                    parent = ancestor.get("parent_span_id")
                    ancestor = rows_by_id.get(parent) if parent else None
                for span_id in visited:
                    grader_owned[span_id] = belongs
                if belongs:
                    continue
            span = _serialize_span(row, should_sanitize, raw_attributes)
            has_filter = bool(span_filter)
            if has_filter and not matches_span_filter(span["name"], span_filter):
                continue
            if (
                not include_internal
                and not has_filter
                and not is_relevant_span(
                    {"attributes": raw_attributes, "statusCode": span.get("statusCode")}
                )
            ):
                continue
            span_map[span["spanId"]] = span
        spans = list(span_map.values())
        if max_depth is not None:
            depth_cache: dict[str, int] = {}
            spans = [s for s in spans if _compute_depth(s, span_map, depth_cache) < max_depth]
        if max_spans is not None:
            spans = spans[: max(0, int(max_spans))]
        return spans


_trace_store: TraceStore | None = None


def get_trace_store() -> TraceStore:
    global _trace_store
    if _trace_store is None:
        _trace_store = TraceStore()
    return _trace_store
