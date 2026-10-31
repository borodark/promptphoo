"""Ports test/tracing/store.test.ts, storePersistence.test.ts and storeRetention.test.ts against
a real SQLite database with upstream's migrations."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import pytest

from promptphoo import db
from promptphoo.tracing import store as store_module
from promptphoo.tracing.store import TraceStore, get_trace_store

DAY_MS = 24 * 60 * 60 * 1000


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("IS_TESTING", raising=False)
    db.close_db()
    db.run_db_migrations()
    conn = db.get_db()
    for eval_id in ("eval-1", "eval-2"):
        conn.execute(
            "INSERT INTO evals (id, created_at, config, results) VALUES (?, 0, '{}', '{}')",
            (eval_id,),
        )
    yield get_trace_store()
    db.close_db()


def _rows(table: str) -> list[dict]:
    cur = db.get_db().cursor()
    cur.row_factory = lambda c, r: {col[0]: r[i] for i, col in enumerate(c.description)}
    return cur.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()


async def _trace(store: TraceStore, trace_id: str, eval_id: str = "eval-1", **metadata) -> None:
    await store.create_trace(
        {"traceId": trace_id, "evaluationId": eval_id, "testCaseId": f"{trace_id}-test",
         **({"metadata": metadata} if metadata else {})}
    )  # fmt: skip


def test_get_trace_store_is_a_singleton(store):
    assert get_trace_store() is store
    assert isinstance(store, TraceStore)


# --- createTrace ---------------------------------------------------------------------------


async def test_create_trace_writes_the_row_as_drizzle_does(store):
    before = int(time.time() * 1000)
    await store.create_trace(
        {"traceId": "t1", "evaluationId": "eval-1", "testCaseId": "test-1",
         "metadata": {"testIdx": 0, "vars": {"a": 1.0}, "skip": None}}
    )  # fmt: skip
    [row] = _rows("traces")
    assert len(row["id"]) == 36 and row["id"].count("-") == 4
    assert row["trace_id"] == "t1"
    assert row["evaluation_id"] == "eval-1"
    assert row["test_case_id"] == "test-1"
    assert isinstance(row["created_at"], int) and row["created_at"] >= before
    assert row["metadata"] == '{"testIdx":0,"vars":{"a":1},"skip":null}'


async def test_create_trace_ignores_an_existing_trace_id(store):
    await _trace(store, "t1", first=True)
    await store.create_trace({"traceId": "t1", "evaluationId": "eval-2", "testCaseId": "other"})
    [row] = _rows("traces")
    assert row["evaluation_id"] == "eval-1"
    assert row["metadata"] == '{"first":true}'
    assert row["test_case_id"] == "t1-test"


async def test_create_trace_without_metadata_stores_null(store):
    await store.create_trace({"traceId": "t1", "evaluationId": "eval-1", "testCaseId": "x"})
    assert _rows("traces")[0]["metadata"] is None


async def test_create_trace_raises_for_an_unknown_evaluation(store, caplog):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        await store.create_trace({"traceId": "t1", "evaluationId": "nope", "testCaseId": "x"})
    assert "[TraceStore] Failed to create trace" in caplog.text


# --- getTraceMetadata ----------------------------------------------------------------------


async def test_get_trace_metadata(store):
    await _trace(store, "with", commandToolNames=["bash"])
    await store.create_trace({"traceId": "without", "evaluationId": "eval-1", "testCaseId": "x"})
    assert await store.get_trace_metadata("with") == {"commandToolNames": ["bash"]}
    assert await store.get_trace_metadata("without") == {}
    assert await store.get_trace_metadata("missing") is None


# --- addSpans ------------------------------------------------------------------------------


async def test_add_spans_writes_rows_as_drizzle_does(store):
    await _trace(store, "t1")
    result = await store.add_spans(
        "t1",
        [
            {"spanId": "s1", "name": "root", "startTime": 1700000000123.5, "endTime": 1700000000200,
             "attributes": {"k": "v", "n": 2.0}, "statusCode": 1},
            {"spanId": "s2", "parentSpanId": "s1", "name": "child", "startTime": 1700000000124,
             "statusMessage": "boom"},
        ],
    )  # fmt: skip
    assert result == {"stored": True}
    rows = _rows("spans")
    assert [(r["span_id"], r["parent_span_id"], r["name"]) for r in rows] == [
        ("s1", None, "root"), ("s2", "s1", "child"),
    ]  # fmt: skip
    assert rows[0]["start_time"] == 1700000000123.5
    assert rows[0]["end_time"] == 1700000000200
    assert rows[0]["attributes"] == '{"k":"v","n":2}'
    assert rows[0]["status_code"] == 1 and rows[0]["status_message"] is None
    assert rows[1]["end_time"] is None and rows[1]["attributes"] is None
    assert rows[1]["status_message"] == "boom"
    assert all(len(r["id"]) == 36 for r in rows)


async def test_add_spans_skips_a_missing_trace(store, caplog):
    result = await store.add_spans("missing", [{"spanId": "s", "name": "x", "startTime": 1}])
    assert result == {"stored": False, "reason": "Trace missing not found"}
    assert "Trace missing not found, skipping 1 spans" in caplog.text
    assert _rows("spans") == []


async def test_add_spans_missing_trace_warning_can_be_quiet(store, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    await store.add_spans("missing", [], {"warnIfMissingTrace": False})
    assert "not found" not in caplog.text


async def test_add_spans_skip_trace_check_still_needs_the_foreign_key(store):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        await store.add_spans(
            "missing", [{"spanId": "s", "name": "x", "startTime": 1}], {"skipTraceCheck": True}
        )


async def test_add_spans_empty_list_is_stored(store):
    await _trace(store, "t1")
    assert await store.add_spans("t1", []) == {"stored": True}


async def test_duplicate_span_ids_are_ignored(store):
    """storePersistence: duplicates in one insert, across inserts, and per trace."""
    await _trace(store, "t1")
    await _trace(store, "t2")
    span = {"spanId": "dup", "name": "first", "startTime": 1}
    await store.add_spans("t1", [span, {**span, "name": "second"}])
    await store.add_spans("t1", [{**span, "name": "third"}])
    await store.add_spans("t2", [{**span, "name": "other trace"}])
    assert [(r["trace_id"], r["name"]) for r in _rows("spans")] == [
        ("t1", "first"), ("t2", "other trace"),
    ]  # fmt: skip


async def test_add_spans_failure_stores_none_of_the_batch(store, caplog):
    import sqlite3

    await _trace(store, "t1")
    with pytest.raises(sqlite3.IntegrityError):
        await store.add_spans(
            "t1",
            [{"spanId": "ok", "name": "fine", "startTime": 1}, {"spanId": "bad", "startTime": 2}],
        )
    assert _rows("spans") == []
    assert "[TraceStore] Failed to add spans" in caplog.text


# --- getTracesByEvaluation / getTrace --------------------------------------------------------


async def _populate(store: TraceStore) -> None:
    await _trace(store, "t1", testIdx=0)
    await _trace(store, "t2")
    await _trace(store, "other", "eval-2")
    await store.add_spans("t1", [
        {"spanId": "s1", "name": "root", "startTime": 10, "endTime": 20,
         "attributes": {"authorization": "Bearer x", "gen_ai.usage.input_tokens": 5,
                        "long": "x" * 500, "nested": {"api_key": "k"}}},
        {"spanId": "s2", "parentSpanId": "s1", "name": "child", "startTime": 11,
         "statusCode": 2, "statusMessage": "bad"},
        {"spanId": "s3", "name": "empty attrs", "startTime": 12, "attributes": {}},
    ])  # fmt: skip


async def test_get_traces_by_evaluation(store):
    await _populate(store)
    traces = await store.get_traces_by_evaluation("eval-1")
    assert [t["traceId"] for t in traces] == ["t1", "t2"]
    first = traces[0]
    assert list(first) == ["traceId", "evaluationId", "testCaseId", "metadata", "spans"]
    assert first["metadata"] == {"testIdx": 0}
    assert "metadata" not in traces[1] and traces[1]["spans"] == []
    s1, s2, s3 = first["spans"]
    assert s1 == {
        "spanId": "s1", "name": "root", "startTime": 10, "endTime": 20,
        "attributes": {"authorization": "<redacted>", "gen_ai.usage.input_tokens": 5,
                       "long": "x" * 400 + "…", "nested": {"api_key": "<redacted>"}},
    }  # fmt: skip
    assert s2 == {"spanId": "s2", "parentSpanId": "s1", "name": "child", "startTime": 11,
                  "statusCode": 2, "statusMessage": "bad"}  # fmt: skip
    # `span.attributes || undefined`: an empty object is truthy, so it is kept.
    assert s3 == {"spanId": "s3", "name": "empty attrs", "startTime": 12, "attributes": {}}
    assert await store.get_traces_by_evaluation("none") == []


async def test_get_traces_by_evaluation_raw_attributes(store):
    await _populate(store)
    [first, _] = await store.get_traces_by_evaluation("eval-1", {"sanitizeAttributes": False})
    assert first["spans"][0]["attributes"]["authorization"] == "Bearer x"
    assert len(first["spans"][0]["attributes"]["long"]) == 500


async def test_get_trace(store):
    await _populate(store)
    trace = await store.get_trace("t1")
    assert trace is not None
    assert trace["evaluationId"] == "eval-1" and trace["testCaseId"] == "t1-test"
    assert [s["spanId"] for s in trace["spans"]] == ["s1", "s2", "s3"]
    assert trace["spans"][0]["attributes"]["authorization"] == "<redacted>"
    raw = await store.get_trace("t1", {"sanitizeAttributes": False})
    assert raw is not None and raw["spans"][0]["attributes"]["authorization"] == "Bearer x"
    assert await store.get_trace("missing") is None


# --- deleteOldTraces (storeRetention) ----------------------------------------------------------


def _insert_raw_trace(trace_id: str, created_at: object) -> None:
    conn = db.get_db()
    conn.execute(
        "INSERT INTO traces (id, trace_id, evaluation_id, test_case_id, created_at) "
        "VALUES (?, ?, 'eval-1', ?, ?)",
        (f"{trace_id}-id", trace_id, f"{trace_id}-test", created_at),
    )
    conn.execute(
        "INSERT INTO spans (id, trace_id, span_id, name, start_time) VALUES (?, ?, ?, ?, 1)",
        (f"{trace_id}-span-id", trace_id, f"{trace_id}-span", f"{trace_id}-operation"),
    )


def _sqlite_timestamp(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S")


async def test_delete_old_traces_prunes_numeric_and_sqlite_timestamps(store):
    now = int(time.time() * 1000)
    _insert_raw_trace("numeric-old", now - 60 * DAY_MS)
    _insert_raw_trace("numeric-new", now - DAY_MS)
    _insert_raw_trace("legacy-old", _sqlite_timestamp(now - 60 * DAY_MS))
    _insert_raw_trace("legacy-new", _sqlite_timestamp(now - DAY_MS))
    _insert_raw_trace("text-ms-old", str(now - 60 * DAY_MS))
    await TraceStore().delete_old_traces(30)
    assert sorted(r["trace_id"] for r in _rows("traces")) == ["legacy-new", "numeric-new"]
    assert sorted(r["trace_id"] for r in _rows("spans")) == ["legacy-new", "numeric-new"]


async def test_delete_old_traces_logs_and_raises(store, monkeypatch, caplog):
    def boom():
        raise RuntimeError("db gone")

    monkeypatch.setattr(store_module.db, "get_db", boom)
    with pytest.raises(RuntimeError):
        await store.delete_old_traces(1)
    assert "[TraceStore] Failed to delete old traces: RuntimeError: db gone" in caplog.text


# --- getSpans (storePersistence / storeRetention) ----------------------------------------------


async def test_get_spans_returns_spans_for_a_stored_trace(store):
    await _trace(store, "t1")
    await store.add_spans("t1", [{"spanId": "s", "name": "op", "startTime": 5,
                                  "attributes": {"password": "p"}}])  # fmt: skip
    assert await store.get_spans("t1") == [
        {"spanId": "s", "name": "op", "startTime": 5, "attributes": {"password": "<redacted>"}}
    ]
    raw = await store.get_spans("t1", {"sanitizeAttributes": False})
    assert raw[0]["attributes"] == {"password": "p"}


async def test_get_spans_null_attributes_become_an_empty_object(store):
    """getSpans passes `row.attributes ?? {}` as the raw attributes."""
    await _trace(store, "t1")
    await store.add_spans("t1", [{"spanId": "s", "name": "op", "startTime": 5}])
    assert await store.get_spans("t1") == [
        {"spanId": "s", "name": "op", "startTime": 5, "attributes": {}}
    ]


SEMANTIC_SPANS = [
    {"spanId": "http", "name": "POST /chat", "startTime": 1,
     "attributes": {"otel.span.kind": "server", "http.request.method": "POST"}},
    {"spanId": "handler", "name": "request handler - /chat", "startTime": 2,
     "attributes": {"otel.span.kind": "internal"}},
    {"spanId": "model", "parentSpanId": "handler", "name": "chat gpt-4.1-mini", "startTime": 3,
     "attributes": {"otel.span.kind": "internal", "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": "gpt-4.1-mini"}},
    {"spanId": "tool", "parentSpanId": "model", "name": "execute_tool search_knowledge_base",
     "startTime": 4, "attributes": {"otel.span.kind": "internal",
                                    "gen_ai.operation.name": "execute_tool",
                                    "gen_ai.tool.name": "search_knowledge_base"}},
    {"spanId": "guardrail", "parentSpanId": "model", "name": "policy check", "startTime": 5,
     "attributes": {"otel.span.kind": "internal", "guardrails.decision": "blocked"}},
    {"spanId": "command", "parentSpanId": "model", "name": "execute operation", "startTime": 6,
     "attributes": {"otel.span.kind": "client", "command.name": "git status"}},
    {"spanId": "search", "parentSpanId": "model", "name": "retrieve information", "startTime": 7,
     "attributes": {"otel.span.kind": "client", "search_query": "customer records"}},
    {"spanId": "error", "name": "POST /remote-api", "startTime": 8, "statusCode": 2,
     "statusMessage": "rate limited", "attributes": {"otel.span.kind": "client"}},
    {"spanId": "grader-model", "name": "chat grading-model", "startTime": 9,
     "attributes": {"gen_ai.operation.name": "chat", "gen_ai.request.model": "grading-model",
                    "promptfoo.span.role": "grader"}},
]  # fmt: skip


async def test_get_spans_keeps_meaningful_spans_without_internal_ones(store):
    await _trace(store, "t")
    await store.add_spans("t", SEMANTIC_SPANS)
    selected = await store.get_spans("t", {"includeInternalSpans": False})
    assert [s["name"] for s in selected] == [
        "chat gpt-4.1-mini", "execute_tool search_knowledge_base", "policy check",
        "execute operation", "retrieve information", "POST /remote-api",
    ]  # fmt: skip
    assert len(await store.get_spans("t")) == len(SEMANTIC_SPANS)
    assert len(await store.get_spans("t", {"includeInternalSpans": True})) == len(SEMANTIC_SPANS)


async def test_get_spans_filters_before_the_span_limit(store):
    await _trace(store, "t")
    await store.add_spans("t", [
        {"spanId": "http-1", "name": "POST", "startTime": 1,
         "attributes": {"otel.span.kind": "client"}},
        {"spanId": "http-2", "name": "GET", "startTime": 2,
         "attributes": {"otel.span.kind": "client"}},
        {"spanId": "model", "name": "chat gpt-4.1-mini", "startTime": 3,
         "attributes": {"otel.span.kind": "internal", "gen_ai.operation.name": "chat"}},
        {"spanId": "tool", "name": "execute_tool search", "startTime": 4,
         "attributes": {"otel.span.kind": "internal", "gen_ai.tool.name": "search"}},
    ])  # fmt: skip
    spans = await store.get_spans("t", {"includeInternalSpans": False, "maxSpans": 2})
    assert [s["name"] for s in spans] == ["chat gpt-4.1-mini", "execute_tool search"]


async def test_get_spans_excludes_grader_descendants(store):
    await _trace(store, "t")
    await store.add_spans("t", [
        {"spanId": "target", "name": "chat target-model", "startTime": 1,
         "attributes": {"gen_ai.operation.name": "chat", "promptfoo.span.role": "target"}},
        {"spanId": "grader", "name": "grader llm-rubric", "startTime": 2,
         "attributes": {"promptfoo.span.role": "grader"}},
        {"spanId": "grader-agent", "parentSpanId": "grader", "name": "agent grading-agent",
         "startTime": 3, "attributes": {"agent.name": "grading-agent"}},
        {"spanId": "grader-model", "parentSpanId": "grader-agent", "name": "chat grading-model",
         "startTime": 4, "attributes": {"gen_ai.operation.name": "chat"}},
        {"spanId": "grader-tool", "parentSpanId": "grader-model", "name": "execute_tool search",
         "startTime": 5, "attributes": {"gen_ai.tool.name": "search"}},
    ])  # fmt: skip
    only_target = await store.get_spans("t", {"includeInternalSpans": False})
    assert [s["spanId"] for s in only_target] == ["target"]
    filtered = await store.get_spans(
        "t", {"includeInternalSpans": False, "spanFilter": ["chat*", "*tool*"]}
    )
    assert [s["spanId"] for s in filtered] == ["target"]
    assert len(await store.get_spans("t", {"includeInternalSpans": True})) == 5


async def test_get_spans_wildcard_filters_keep_explicit_nonsemantic_spans(store):
    await _trace(store, "t")
    await store.add_spans("t", [
        {"spanId": "model", "name": "chat gpt-4.1-mini", "startTime": 1,
         "attributes": {"otel.span.kind": "internal", "gen_ai.operation.name": "chat"}},
        {"spanId": "tool", "name": "execute_tool search", "startTime": 2,
         "attributes": {"otel.span.kind": "internal", "gen_ai.tool.name": "search"}},
        {"spanId": "http", "name": "POST /chat", "startTime": 3,
         "attributes": {"otel.span.kind": "server"}},
    ])  # fmt: skip
    model_and_tool = await store.get_spans(
        "t", {"includeInternalSpans": False, "spanFilter": ["chat*", "*tool*"]}
    )
    assert [s["name"] for s in model_and_tool] == ["chat gpt-4.1-mini", "execute_tool search"]
    explicit = await store.get_spans("t", {"includeInternalSpans": False, "spanFilter": ["POST*"]})
    assert [s["name"] for s in explicit] == ["POST /chat"]


async def test_get_spans_orders_ties_by_span_id_before_max_spans(store):
    await _trace(store, "t")
    await store.add_spans("t", [
        {"spanId": "bbbbbbbbbbbbbbbb", "name": "second tied span", "startTime": 1000},
        {"spanId": "aaaaaaaaaaaaaaaa", "name": "first tied span", "startTime": 1000},
        {"spanId": "cccccccccccccccc", "name": "earliest span", "startTime": 999},
        {"spanId": "0000000000000001", "name": "latest span", "startTime": 1001},
    ])  # fmt: skip
    order = ["cccccccccccccccc", "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "0000000000000001"]
    assert [s["spanId"] for s in await store.get_spans("t")] == order
    assert [s["spanId"] for s in await store.get_spans("t", {"maxSpans": 2})] == order[:2]


async def test_get_spans_earliest_start_and_max_depth(store):
    await _trace(store, "t")
    await store.add_spans("t", [
        {"spanId": "root", "name": "root", "startTime": 1},
        {"spanId": "child", "parentSpanId": "root", "name": "child", "startTime": 2},
        {"spanId": "grand", "parentSpanId": "child", "name": "grand", "startTime": 3},
        {"spanId": "orphan", "parentSpanId": "gone", "name": "orphan", "startTime": 4},
    ])  # fmt: skip
    names = [s["name"] for s in await store.get_spans("t", {"maxDepth": 2})]
    assert names == ["root", "child", "orphan"]
    assert [s["name"] for s in await store.get_spans("t", {"maxDepth": 1})] == ["root", "orphan"]
    assert [s["name"] for s in await store.get_spans("t", {"earliestStartTime": 3})] == [
        "grand", "orphan",
    ]  # fmt: skip
    # With `root` filtered out, `child` has no parent in the map and becomes depth 0.
    assert [
        s["name"] for s in await store.get_spans("t", {"earliestStartTime": 2, "maxDepth": 1})
    ] == ["child", "orphan"]
    assert [s["name"] for s in await store.get_spans("t", {"maxSpans": -1})] == [
        "root", "child", "grand",
    ]  # fmt: skip


def test_rows_are_json_text_columns(store):
    """Attributes and metadata are stored as JSON.stringify text, never as Python repr."""
    get_trace_store().create_trace_sync(
        {"traceId": "t", "evaluationId": "eval-1", "testCaseId": "x", "metadata": {"a": [1.5]}}
    )
    get_trace_store().add_spans_sync(
        "t", [{"spanId": "s", "name": "n", "startTime": 1, "attributes": {"u": "é"}}]
    )
    assert json.loads(_rows("traces")[0]["metadata"]) == {"a": [1.5]}
    assert _rows("spans")[0]["attributes"] == '{"u":"é"}'
