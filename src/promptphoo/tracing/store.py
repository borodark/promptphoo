"""The trace store over the `traces`/`spans` tables. Ports src/tracing/store.ts.

Runtime data is upstream's camelCase dicts:
- SpanData: `{spanId, parentSpanId?, name, startTime, endTime?, attributes?, statusCode?,
  statusMessage?}`.
- ParsedTrace: `{traceId, span: SpanData}`.
- TraceData: `{traceId, evaluationId, testCaseId, metadata?, spans: [SpanData]}`.

Rows are written exactly as drizzle writes them (a uuid `id`, `created_at` in epoch ms, JSON
text for `metadata`/`attributes`, NULL for undefined), so Node reads them back unchanged.

The public methods are async like upstream's. SQLite is synchronous, so each one wraps a
`*_sync` twin; synchronous callers (`import`, `export`, output writers) use those directly.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from promptphoo import db
from promptphoo.jsvalues import UNDEFINED, is_truthy, js_string
from promptphoo.tracing.sanitize_attributes import sanitize_trace_attributes
from promptphoo.tracing.span_filter import is_relevant_span, matches_span_filter
from promptphoo.tracing.span_roles import SPAN_ROLE_ATTRIBUTE

logger = logging.getLogger(__name__)

_SPAN_COLUMNS = (
    "id, trace_id, span_id, parent_span_id, name, start_time, end_time, attributes, "
    "status_code, status_message"
)
_TRACE_COLUMNS = "id, trace_id, evaluation_id, test_case_id, created_at, metadata"


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _row_factory(cursor: Any, row: tuple[Any, ...]) -> dict[str, Any]:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def _select(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    cur = db.get_db().cursor()
    cur.row_factory = _row_factory
    return cur.execute(sql, params).fetchall()


def _span_row(row: dict[str, Any]) -> dict[str, Any]:
    """A spans row as drizzle selects it (camelCase, JSON columns parsed)."""
    return {
        "id": row["id"],
        "traceId": row["trace_id"],
        "spanId": row["span_id"],
        "parentSpanId": row["parent_span_id"],
        "name": row["name"],
        "startTime": row["start_time"],
        "endTime": row["end_time"],
        "attributes": db.from_db_json(row["attributes"]),
        "statusCode": row["status_code"],
        "statusMessage": row["status_message"],
    }


_RAW_DEFAULT: Any = object()


def serialize_span(
    span: dict[str, Any],
    should_sanitize_attributes: bool = True,
    raw_attributes: Any = _RAW_DEFAULT,
) -> dict[str, Any]:
    """serializeSpan: a selected row as SpanData; null columns become absent keys."""
    if raw_attributes is _RAW_DEFAULT:
        # `span.attributes || undefined`: an empty object is truthy in JS.
        raw_attributes = span.get("attributes") if is_truthy(span.get("attributes")) else None
    out: dict[str, Any] = {"spanId": span["spanId"]}
    if span.get("parentSpanId") is not None:
        out["parentSpanId"] = span["parentSpanId"]
    out["name"] = span["name"]
    out["startTime"] = span["startTime"]
    if span.get("endTime") is not None:
        out["endTime"] = span["endTime"]
    if raw_attributes is not None:
        out["attributes"] = (
            sanitize_trace_attributes(raw_attributes)
            if should_sanitize_attributes
            else raw_attributes
        )
    if span.get("statusCode") is not None:
        out["statusCode"] = span["statusCode"]
    if span.get("statusMessage") is not None:
        out["statusMessage"] = span["statusMessage"]
    return out


def _compute_depth(
    span: dict[str, Any], span_map: dict[str, dict[str, Any]], depth_cache: dict[str, int]
) -> int:
    # Iterative walk up the parent chain (upstream recurses; cycles cannot occur because a
    # span's ancestors are resolved before the span is cached).
    chain: list[str] = []
    current: dict[str, Any] | None = span
    base = 0
    seen: set[str] = set()
    while current is not None:
        span_id = current["spanId"]
        if span_id in depth_cache:
            base = depth_cache[span_id]
            break
        if span_id in seen:
            raise RecursionError("Maximum call stack size exceeded")
        seen.add(span_id)
        parent = current.get("parentSpanId")
        if not parent or parent not in span_map:
            depth_cache[span_id] = 0
            base = 0
            break
        chain.append(span_id)
        current = span_map[parent]
    for span_id in reversed(chain):
        base += 1
        depth_cache[span_id] = base
    return depth_cache[span["spanId"]]


def _trace_data(trace: dict[str, Any], spans: list[dict[str, Any]], sanitize: bool) -> dict:
    out: dict[str, Any] = {
        "traceId": trace["trace_id"],
        "evaluationId": trace["evaluation_id"],
        "testCaseId": trace["test_case_id"],
    }
    metadata = db.from_db_json(trace["metadata"])
    if metadata is not None:
        out["metadata"] = metadata
    out["spans"] = [serialize_span(_span_row(span), sanitize) for span in spans]
    return out


def _cutoff_condition(column: str) -> str:
    """deleteOldTraces' cutoff: numeric ms, numeric text, or a SQLite datetime string."""
    return f"""(
    (
      typeof({column}) in ('integer', 'real')
      and {column} < :cutoff
    )
    or (
      typeof({column}) = 'text'
      and (
        (
          cast({column} as integer) > 1000000000000
          and cast({column} as integer) < :cutoff
        )
        or datetime({column}) < datetime(:cutoff_text)
      )
    )
  )"""


class TraceStore:
    # --- writes ------------------------------------------------------------------------------

    def create_trace_sync(self, trace: dict[str, Any]) -> None:
        trace_id = trace["traceId"]
        try:
            logger.debug(
                "[TraceStore] Creating trace %s for evaluation %s", trace_id, trace["evaluationId"]
            )
            conn = db.get_db()
            conn.execute(
                f"INSERT INTO traces ({_TRACE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (trace_id) DO NOTHING",
                (
                    str(uuid.uuid4()),
                    trace_id,
                    trace["evaluationId"],
                    trace["testCaseId"],
                    _now_ms(),
                    db.to_db_json(trace.get("metadata")),
                ),
            )
            logger.debug("[TraceStore] Successfully created or found existing trace %s", trace_id)
        except Exception as error:
            logger.error("[TraceStore] Failed to create trace: %s", _error_string(error))
            raise

    def add_spans_sync(
        self, trace_id: str, spans: list[dict[str, Any]], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        options = options or {}
        try:
            logger.debug("[TraceStore] Adding %d spans to trace %s", len(spans), trace_id)
            if options.get("skipTraceCheck"):
                logger.debug("[TraceStore] Skipping trace existence check for OTLP scenario")
            else:
                logger.debug("[TraceStore] Verifying trace %s exists", trace_id)
                found = _select(
                    f"SELECT {_TRACE_COLUMNS} FROM traces WHERE trace_id = ? LIMIT 1", (trace_id,)
                )
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
                logger.debug(
                    "[TraceStore] Trace %s found, proceeding with span insertion", trace_id
                )
            records = []
            for span in spans:
                logger.debug(
                    "[TraceStore] Preparing span %s (%s) for insertion",
                    span.get("spanId"),
                    span.get("name"),
                )
                records.append((
                    str(uuid.uuid4()),
                    trace_id,
                    _column(span.get("spanId")),
                    _column(span.get("parentSpanId")),
                    _column(span.get("name")),
                    _column(span.get("startTime")),
                    _column(span.get("endTime")),
                    db.to_db_json(_column(span.get("attributes"))),
                    _column(span.get("statusCode")),
                    _column(span.get("statusMessage")),
                ))  # fmt: skip
            if not records:
                return {"stored": True}
            conn = db.get_db()
            # Drizzle sends one multi-row INSERT, so a failing row stores none of them.
            with db.transaction(conn):
                conn.executemany(
                    f"INSERT INTO spans ({_SPAN_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (trace_id, span_id) DO NOTHING",
                    records,
                )
            logger.debug(
                "[TraceStore] Successfully added %d spans to trace %s", len(records), trace_id
            )
            return {"stored": True}
        except Exception as error:
            logger.error("[TraceStore] Failed to add spans: %s", _error_string(error))
            raise

    # --- reads -------------------------------------------------------------------------------

    def get_traces_by_evaluation_sync(
        self, evaluation_id: str, options: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        sanitize = _should_sanitize(options)
        try:
            logger.debug("[TraceStore] Fetching traces for evaluation %s", evaluation_id)
            traces = _select(
                f"SELECT {_TRACE_COLUMNS} FROM traces WHERE evaluation_id = ?", (evaluation_id,)
            )
            logger.debug(
                "[TraceStore] Found %d traces for evaluation %s", len(traces), evaluation_id
            )
            out = []
            for trace in traces:
                logger.debug("[TraceStore] Fetching spans for trace %s", trace["trace_id"])
                spans = _select(
                    f"SELECT {_SPAN_COLUMNS} FROM spans WHERE trace_id = ?", (trace["trace_id"],)
                )
                logger.debug(
                    "[TraceStore] Found %d spans for trace %s", len(spans), trace["trace_id"]
                )
                out.append(_trace_data(trace, spans, sanitize))
            logger.debug("[TraceStore] Returning %d traces with spans", len(out))
            return out
        except Exception as error:
            logger.error(
                "[TraceStore] Failed to get traces for evaluation: %s", _error_string(error)
            )
            raise

    def get_trace_sync(
        self, trace_id: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        sanitize = _should_sanitize(options)
        try:
            logger.debug("[TraceStore] Fetching trace %s", trace_id)
            traces = _select(
                f"SELECT {_TRACE_COLUMNS} FROM traces WHERE trace_id = ? LIMIT 1", (trace_id,)
            )
            if not traces:
                logger.debug("[TraceStore] Trace %s not found", trace_id)
                return None
            logger.debug("[TraceStore] Found trace %s, fetching spans", trace_id)
            spans = _select(f"SELECT {_SPAN_COLUMNS} FROM spans WHERE trace_id = ?", (trace_id,))
            logger.debug("[TraceStore] Found %d spans for trace %s", len(spans), trace_id)
            return _trace_data(traces[0], spans, sanitize)
        except Exception as error:
            logger.error("[TraceStore] Failed to get trace: %s", _error_string(error))
            raise

    def get_trace_metadata_sync(self, trace_id: str) -> dict[str, Any] | None:
        """The trace's metadata (`{}` when NULL), or None when the trace does not exist."""
        try:
            logger.debug("[TraceStore] Fetching metadata for trace %s", trace_id)
            rows = _select("SELECT metadata FROM traces WHERE trace_id = ? LIMIT 1", (trace_id,))
            if not rows:
                return None
            metadata = db.from_db_json(rows[0]["metadata"])
            return {} if metadata is None else metadata
        except Exception as error:
            logger.error("[TraceStore] Failed to get trace metadata: %s", _error_string(error))
            raise

    def delete_old_traces_sync(self, retention_days: float) -> None:
        try:
            logger.debug(
                "[TraceStore] Deleting traces older than %s days", _js_number(retention_days)
            )
            cutoff = _now_ms() - retention_days * 24 * 60 * 60 * 1000
            # new Date(cutoff).toISOString().slice(0, 19).replace('T', ' ')
            cutoff_text = datetime.fromtimestamp(int(cutoff) // 1000, UTC).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            params = {"cutoff": cutoff, "cutoff_text": cutoff_text}
            condition = _cutoff_condition('"traces"."created_at"')
            conn = db.get_db()
            # `spans.trace_id` is FK-enforced without ON DELETE CASCADE.
            with db.transaction(conn):
                conn.execute(
                    'DELETE FROM spans WHERE "spans"."trace_id" in (\n'
                    '              select "traces"."trace_id"\n'
                    '              from "traces"\n'
                    f"              where {condition}\n"
                    "            )",
                    params,
                )
                conn.execute(f"DELETE FROM traces WHERE {condition}", params)
            logger.debug(
                "[TraceStore] Successfully deleted traces older than %s days",
                _js_number(retention_days),
            )
        except Exception as error:
            logger.error("[TraceStore] Failed to delete old traces: %s", _error_string(error))
            raise

    def get_spans_sync(
        self, trace_id: str, options: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """getSpans. Options use upstream's keys: `earliestStartTime`, `maxSpans`, `maxDepth`,
        `includeInternalSpans` (default true), `spanFilter`, `sanitizeAttributes`."""
        options = options or {}
        earliest_start_time = options.get("earliestStartTime")
        max_spans = options.get("maxSpans")
        max_depth = options.get("maxDepth")
        include_internal = options.get("includeInternalSpans")
        include_internal = True if include_internal is None else include_internal
        span_filter = options.get("spanFilter")
        sanitize = _should_sanitize(options)
        try:
            logger.debug("[TraceStore] Fetching spans for trace %s", trace_id)
            rows = [
                _span_row(r)
                for r in _select(
                    f"SELECT {_SPAN_COLUMNS} FROM spans WHERE trace_id = ? "
                    "ORDER BY start_time ASC, span_id ASC",
                    (trace_id,),
                )
            ]
            rows_by_span_id = {row["spanId"]: row for row in rows}
            grader_owned: dict[str, bool] = {}
            span_map: dict[str, dict[str, Any]] = {}
            depth_cache: dict[str, int] = {}

            for row in rows:
                if earliest_start_time and row["startTime"] < earliest_start_time:
                    continue
                raw_attributes = row["attributes"]
                if raw_attributes is None:
                    raw_attributes = {}

                if not include_internal:
                    ancestor: dict[str, Any] | None = row
                    visited: dict[str, None] = {}
                    belongs_to_grader = False
                    while ancestor is not None and ancestor["spanId"] not in visited:
                        cached = grader_owned.get(ancestor["spanId"])
                        if cached is not None:
                            belongs_to_grader = cached
                            break
                        visited[ancestor["spanId"]] = None
                        attributes = ancestor["attributes"]
                        if isinstance(attributes, dict) and (
                            attributes.get(SPAN_ROLE_ATTRIBUTE) == "grader"
                        ):
                            belongs_to_grader = True
                            break
                        parent = ancestor["parentSpanId"]
                        ancestor = rows_by_span_id.get(parent) if parent else None
                    for span_id in visited:
                        grader_owned[span_id] = belongs_to_grader
                    if belongs_to_grader:
                        continue

                span_data = serialize_span(row, sanitize, raw_attributes)
                has_explicit_filter = bool(span_filter)
                if has_explicit_filter and not matches_span_filter(span_data["name"], span_filter):
                    continue
                if (
                    not include_internal
                    and not has_explicit_filter
                    and not is_relevant_span(
                        {"attributes": raw_attributes, "statusCode": span_data.get("statusCode")}
                    )
                ):
                    continue
                # A Map keeps the first insertion position when a key is set again.
                span_map[span_data["spanId"]] = span_data

            spans = list(span_map.values())
            if max_depth is not None:
                spans = [s for s in spans if _compute_depth(s, span_map, depth_cache) < max_depth]
            if max_spans is not None:
                spans = spans[: int(max_spans)]
            logger.debug("[TraceStore] Returning %d spans for trace %s", len(spans), trace_id)
            return spans
        except Exception as error:
            logger.error(
                "[TraceStore] Failed to fetch spans for trace %s: %s",
                trace_id,
                _error_string(error),
            )
            raise

    # --- async API (upstream's signatures) ---------------------------------------------------

    async def create_trace(self, trace: dict[str, Any]) -> None:
        """createTrace({traceId, evaluationId, testCaseId, metadata?}); an existing trace id
        is left as it is."""
        self.create_trace_sync(trace)

    async def add_spans(
        self, trace_id: str, spans: list[dict[str, Any]], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """addSpans: returns `{stored: true}` or `{stored: false, reason}`. Options:
        `skipTraceCheck`, `warnIfMissingTrace`."""
        return self.add_spans_sync(trace_id, spans, options)

    async def get_traces_by_evaluation(
        self, evaluation_id: str, options: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self.get_traces_by_evaluation_sync(evaluation_id, options)

    async def get_trace(
        self, trace_id: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        return self.get_trace_sync(trace_id, options)

    async def get_trace_metadata(self, trace_id: str) -> dict[str, Any] | None:
        return self.get_trace_metadata_sync(trace_id)

    async def delete_old_traces(self, retention_days: float) -> None:
        self.delete_old_traces_sync(retention_days)

    async def get_spans(
        self, trace_id: str, options: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self.get_spans_sync(trace_id, options)


def _should_sanitize(options: dict[str, Any] | None) -> bool:
    value = (options or {}).get("sanitizeAttributes")
    return True if value is None else bool(value)


def _column(value: Any) -> Any:
    """A value bound the way drizzle binds it: JS undefined is NULL."""
    return None if value is UNDEFINED else value


def _js_number(value: Any) -> str:
    return js_string(value)


def _error_string(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


# Singleton instance
_trace_store: TraceStore | None = None


def get_trace_store() -> TraceStore:
    global _trace_store
    if _trace_store is None:
        logger.debug("[TraceStore] Creating new TraceStore instance")
        _trace_store = TraceStore()
    return _trace_store
