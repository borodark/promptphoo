"""DB writes behind `promptfoo import` (and the `--force` replace).

Ports the parts of src/models/eval.ts (`Eval.create` with all of its options, `Eval.delete`),
src/models/evalResult.ts (`createManyFromEvaluateResult`), src/database/evalDeletion.ts
(`deleteTraceRecordsForEvals`) and src/tracing/store.ts (`getTrace` existence, `createTrace`,
`addSpans`) that import needs. Rows are written exactly as drizzle writes them, so Node reads
an eval that Python imported (and the reverse) without noticing.

Blob storage is not ported yet (see docs/PLAN_PORT.md, Phase 6), so responses are stored as
given; `promptfoo://blob/` references inside them stay as they are.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from promptphoo import db, sanitizer
from promptphoo.evaluator.run import ResultFailureReason, generate_id_from_prompt, sha256
from promptphoo.jsvalues import UNDEFINED, json_stringify
from promptphoo.models import result_sanitize as rs
from promptphoo.models.eval import Eval, EvalResult, create_eval_id, get_author

# SQLite's default SQLITE_MAX_VARIABLE_NUMBER bound used by upstream's batched deletes.
SQLITE_MAX_BOUND_PARAMETERS = 999


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _row_factory(cursor: Any, row: tuple[Any, ...]) -> dict[str, Any]:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def create_eval_record(
    config: dict[str, Any],
    rendered_prompts: list[dict[str, Any]],
    *,
    id: str | None = None,  # noqa: A002 - Node's option name
    created_at: int | None = None,
    author: Any = ...,
    vars: list[str] | None = None,  # noqa: A002
    runtime_options: dict[str, Any] | None = None,
    completed_prompts: list[dict[str, Any]] | None = None,
    duration_ms: Any = None,
    generation_duration_ms: Any = None,
    evaluation_duration_ms: Any = None,
) -> Eval:
    """Eval.create(config, renderedPrompts, opts) with every opt (`results` aside).

    `author=...` means "not passed" (resolve with getAuthor); any other value, including None,
    is stored as given, as upstream does for callers that resolved the author themselves.
    """
    created_at = created_at if created_at is not None else _now_ms()
    eval_id = id or create_eval_id(created_at)
    author = get_author() if author is ... else author
    dataset_id = sha256(json_stringify(config.get("tests") or []))
    duration_results: dict[str, Any] = {}
    for key, value in (
        ("durationMs", duration_ms),
        ("generationDurationMs", generation_duration_ms),
        ("evaluationDurationMs", evaluation_duration_ms),
    ):
        if value is not None:
            duration_results[key] = value
    runtime = sanitizer.sanitize_runtime_options(runtime_options)
    j = db.to_db_json
    conn = db.get_db()
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO evals (id, created_at, author, description, results, config, prompts, "
            "vars, runtime_options, is_redteam) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                eval_id, created_at, author, config.get("description"), j(duration_results),
                j(sanitizer.sanitize_tracing_config_for_persistence(config)),
                j(completed_prompts or []), j(vars or []), j(runtime),
                1 if "redteam" in config else 0,
            ),
        )  # fmt: skip
        for prompt in rendered_prompts:
            label = prompt.get("label") or prompt.get("display") or prompt.get("raw")
            prompt_id = generate_id_from_prompt(prompt)
            conn.execute(
                "INSERT INTO prompts (id, prompt) VALUES (?, ?) ON CONFLICT DO NOTHING",
                (prompt_id, label),
            )
            conn.execute(
                "INSERT INTO evals_to_prompts (eval_id, prompt_id) VALUES (?, ?) "
                "ON CONFLICT DO NOTHING",
                (eval_id, prompt_id),
            )
        conn.execute(
            "INSERT INTO datasets (id, tests) VALUES (?, ?) ON CONFLICT DO NOTHING",
            (dataset_id, j(config.get("tests"))),
        )
        conn.execute(
            "INSERT INTO evals_to_datasets (eval_id, dataset_id) VALUES (?, ?) "
            "ON CONFLICT DO NOTHING",
            (eval_id, dataset_id),
        )
        for key, value in (config.get("tags") or {}).items():
            tag_id = sha256(f"{key}:{value}")
            conn.execute(
                "INSERT INTO tags (id, name, value) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                (tag_id, key, value),
            )
            conn.execute(
                "INSERT INTO evals_to_tags (eval_id, tag_id) VALUES (?, ?) ON CONFLICT DO NOTHING",
                (eval_id, tag_id),
            )
    db.update_signal_file(eval_id)
    record = Eval(
        config,
        id=eval_id,
        author=author,
        created_at=created_at,
        persisted=True,
        runtime_options=runtime,
        dataset_id=dataset_id,
        duration_ms=duration_ms,
        prompts=completed_prompts or [],
        vars=vars or [],
    )
    record.generation_duration_ms = generation_duration_ms
    record.evaluation_duration_ms = evaluation_duration_ms
    return record


def _given(result: dict[str, Any], key: str) -> Any:
    value = result.get(key, UNDEFINED)
    return None if value is UNDEFINED else value


def create_many_from_evaluate_result(
    results: list[dict[str, Any]], eval_id: str
) -> list[EvalResult]:
    """EvalResult.createManyFromEvaluateResult: one transaction, rows as given.

    Unlike createFromEvaluateResult, `promptId`, `error` and the other scalar columns come
    straight from each result (drizzle drops keys that are not columns).
    """
    processed = []
    for result in results:
        response = result.get("response")
        processed.append(
            {**rs.serialize_result_provider_refs(result), "response": response or UNDEFINED}
        )
    j = db.to_db_json
    conn = db.get_db()
    rows: list[dict[str, Any]] = []
    with db.transaction(conn):
        for r in processed:
            metadata = rs.persist_trace_metadata(
                rs.persist_repeat_metadata(
                    r.get("metadata"), r.get("repeatIndex"), r.get("repeatGroupId")
                ),
                r.get("traceId"),
                r.get("evaluationId"),
            )
            redacted = rs.redact_sensitive_result_fields_for_db(
                response=rs.sanitize_for_db(_given(r, "response")),
                grading_result=rs.sanitize_for_db(_given(r, "gradingResult")),
                metadata=rs.sanitize_for_db(metadata),
            )
            provider = r.get("provider")
            failure_reason = _given(r, "failureReason")
            row = {
                "id": str(uuid.uuid4()),
                "evalId": eval_id,
                "promptIdx": _given(r, "promptIdx"),
                "testIdx": _given(r, "testIdx"),
                "testCase": rs.sanitize_for_db_with_secrets(_given(r, "testCase")),
                "prompt": rs.sanitize_for_db_with_secrets(_given(r, "prompt")),
                "promptId": _given(r, "promptId"),
                "provider": rs.sanitize_provider(provider) if provider else provider,
                "latencyMs": _given(r, "latencyMs"),
                "cost": _given(r, "cost"),
                "response": redacted["response"],
                "error": _given(r, "error"),
                "failureReason": (
                    ResultFailureReason.NONE if failure_reason is None else failure_reason
                ),
                "success": _given(r, "success"),
                "score": _given(r, "score"),
                "gradingResult": redacted["gradingResult"],
                "namedScores": rs.sanitize_for_db(_given(r, "namedScores")),
                "metadata": redacted["metadata"],
            }
            success = row["success"]
            conn.execute(
                "INSERT INTO eval_results (id, eval_id, prompt_idx, test_idx, test_case, prompt, "
                "prompt_id, provider, latency_ms, cost, response, error, failure_reason, "
                "success, score, grading_result, named_scores, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["id"], eval_id, row["promptIdx"], row["testIdx"], j(row["testCase"]),
                    j(row["prompt"]), row["promptId"], j(row["provider"]), row["latencyMs"],
                    row["cost"], j(row["response"]), row["error"], row["failureReason"],
                    None if success is None else (1 if success else 0), row["score"],
                    j(row["gradingResult"]), j(row["namedScores"]), j(row["metadata"]),
                ),
            )  # fmt: skip
            rows.append(row)
    return [EvalResult(row, persisted=True) for row in rows]


def insert_legacy_eval(
    eval_id: str,
    *,
    created_at: int,
    author: str | None,
    description: Any,
    results: Any,
    config: Any,
) -> None:
    """The bare `evalsTable` insert that imports a v2 (table-backed) export."""
    conn = db.get_db()
    config_value = (
        sanitizer.sanitize_tracing_config_for_persistence(config)
        if isinstance(config, dict)
        else config
    )
    is_redteam = isinstance(config, dict) and "redteam" in config
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO evals (id, created_at, author, description, results, config, "
            "is_redteam) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                eval_id,
                created_at,
                author,
                description,
                db.to_db_json(results),
                db.to_db_json(config_value),
                1 if is_redteam else 0,
            ),
        )
    db.update_signal_file(eval_id)


# --- delete -------------------------------------------------------------------------------


def delete_trace_records_for_evals(conn: Any, eval_ids: list[str]) -> None:
    """deleteTraceRecordsForEvals: spans first, then traces, in bound-parameter batches."""
    for i in range(0, len(eval_ids), SQLITE_MAX_BOUND_PARAMETERS):
        batch = eval_ids[i : i + SQLITE_MAX_BOUND_PARAMETERS]
        marks = ", ".join("?" for _ in batch)
        conn.execute(
            "DELETE FROM spans WHERE trace_id IN (SELECT trace_id FROM traces "
            f"WHERE evaluation_id IN ({marks}))",
            batch,
        )
        conn.execute(f"DELETE FROM traces WHERE evaluation_id IN ({marks})", batch)


def delete_eval(eval_id: str) -> None:
    """Eval.delete({ notify: false })."""
    conn = db.get_db()
    with db.transaction(conn):
        delete_trace_records_for_evals(conn, [eval_id])
        for table in ("evals_to_datasets", "evals_to_prompts", "evals_to_tags", "eval_results"):
            conn.execute(f"DELETE FROM {table} WHERE eval_id = ?", (eval_id,))
        conn.execute("DELETE FROM evals WHERE id = ?", (eval_id,))


def notify_evaluations_deleted(eval_ids: list[str]) -> None:
    """notifyEvaluationsDeleted. The Python signal file has only the update form (see db)."""
    del eval_ids
    db.update_signal_file()


# --- traces -------------------------------------------------------------------------------


def trace_exists(trace_id: str) -> bool:
    conn = db.get_db()
    row = conn.execute("SELECT 1 FROM traces WHERE trace_id = ? LIMIT 1", (trace_id,)).fetchone()
    return row is not None


def create_trace(
    trace_id: str, evaluation_id: str, test_case_id: str, metadata: Any = None
) -> None:
    """TraceStore.createTrace (insert, ignoring an existing trace id)."""
    conn = db.get_db()
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO traces (id, trace_id, evaluation_id, test_case_id, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (trace_id) DO NOTHING",
            (
                str(uuid.uuid4()),
                trace_id,
                evaluation_id,
                test_case_id,
                _now_ms(),
                db.to_db_json(metadata),
            ),
        )


def add_spans(trace_id: str, spans: list[dict[str, Any]]) -> bool:
    """TraceStore.addSpans with the trace-existence check. Returns whether spans were stored."""
    import logging

    if not trace_exists(trace_id):
        logging.getLogger("promptphoo").warning(
            "[TraceStore] Trace %s not found, skipping %d spans. This may indicate spans arrived "
            "before trace was created.",
            trace_id,
            len(spans),
        )
        return False
    if not spans:
        return True
    conn = db.get_db()
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
    return True


def find_eval_row(eval_id: str) -> dict[str, Any] | None:
    """The raw `evals` row, for tests and interop checks."""
    conn = db.get_db()
    cur = conn.cursor()
    cur.row_factory = _row_factory
    return cur.execute("SELECT * FROM evals WHERE id = ?", (eval_id,)).fetchone()
