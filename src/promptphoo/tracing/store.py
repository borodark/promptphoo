# SHIM: replaced by the trace-store branch at merge
"""Minimal TraceStore (src/tracing/store.ts) so the span exporter can write to the DB.

Only the contract the span branch needs is real here: `create_trace` and `add_spans` write the
`traces` and `spans` rows as drizzle does. The read methods are thin and unsanitized.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from promptphoo import db

logger = logging.getLogger("promptphoo.tracing.store")


class TraceStore:
    async def create_trace(self, trace: dict[str, Any]) -> None:
        conn = db.get_db()
        with db.transaction(conn):
            conn.execute(
                "INSERT INTO traces (id, trace_id, evaluation_id, test_case_id, created_at, "
                "metadata) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (trace_id) DO NOTHING",
                (
                    str(uuid.uuid4()),
                    trace["traceId"],
                    trace["evaluationId"],
                    trace["testCaseId"],
                    int(time.time() * 1000),
                    db.to_db_json(trace.get("metadata")),
                ),
            )

    async def add_spans(
        self,
        trace_id: str,
        spans: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        conn = db.get_db()
        if not (options or {}).get("skipTraceCheck"):
            row = conn.execute(
                "SELECT 1 FROM traces WHERE trace_id = ? LIMIT 1", (trace_id,)
            ).fetchone()
            if row is None:
                message = (
                    f"[TraceStore] Trace {trace_id} not found, skipping {len(spans)} spans. "
                    "This may indicate spans arrived before trace was created."
                )
                if (options or {}).get("warnIfMissingTrace") is False:
                    logger.debug(message)
                else:
                    logger.warning(message)
                return {"stored": False, "reason": f"Trace {trace_id} not found"}
        if not spans:
            return {"stored": True}
        with db.transaction(conn):
            for span in spans:
                conn.execute(
                    "INSERT INTO spans (id, trace_id, span_id, parent_span_id, name, start_time, "
                    "end_time, attributes, status_code, status_message) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (trace_id, span_id) DO NOTHING",
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
                    ),
                )
        return {"stored": True}

    async def get_spans(
        self, trace_id: str, options: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        import json

        conn = db.get_db()
        rows = conn.execute(
            "SELECT span_id, parent_span_id, name, start_time, end_time, attributes, "
            "status_code, status_message FROM spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,),
        ).fetchall()
        out = []
        for r in rows:
            span: dict[str, Any] = {"spanId": r[0], "name": r[2], "startTime": r[3]}
            if r[1] is not None:
                span["parentSpanId"] = r[1]
            if r[4] is not None:
                span["endTime"] = r[4]
            if r[5] is not None:
                span["attributes"] = json.loads(r[5])
            if r[6] is not None:
                span["statusCode"] = r[6]
            if r[7] is not None:
                span["statusMessage"] = r[7]
            out.append(span)
        return out

    async def get_trace(self, trace_id: str, options: dict[str, Any] | None = None) -> Any:
        conn = db.get_db()
        row = conn.execute(
            "SELECT trace_id, evaluation_id, test_case_id FROM traces WHERE trace_id = ?",
            (trace_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "traceId": row[0],
            "evaluationId": row[1],
            "testCaseId": row[2],
            "spans": await self.get_spans(trace_id, options),
        }

    async def get_traces_by_evaluation(
        self, evaluation_id: str, options: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        conn = db.get_db()
        ids = [
            r[0]
            for r in conn.execute(
                "SELECT trace_id FROM traces WHERE evaluation_id = ?", (evaluation_id,)
            ).fetchall()
        ]
        return [t for t in [await self.get_trace(i, options) for i in ids] if t]

    async def get_trace_metadata(self, trace_id: str) -> Any:
        return None

    async def delete_old_traces(self, retention_days: int) -> None:
        return None


_store: TraceStore | None = None


def get_trace_store() -> TraceStore:
    global _store
    if _store is None:
        _store = TraceStore()
    return _store
