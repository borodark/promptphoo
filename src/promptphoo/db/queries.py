"""Read and delete queries behind `list`, `show` and `delete`.

Ports the query helpers of src/util/database.ts (getPrompts, getTestCases, getPromptFromHash,
getDatasetFromHash, getEvalFromId, deleteEval, deleteEvals, deleteAllEvals), src/database/
evalDeletion.ts (deleteTraceRecordsForEvals), EvalQueries.getVarsFromEvals from src/models/
eval.ts and the delete half of src/database/signal.ts (updateSignalFileForDeletedEvals).

Rows come back as plain dicts with Node's camelCase keys. `recentEvalDate` is an ISO string
rather than a Date.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from promptphoo import db
from promptphoo.jsvalues import is_truthy, json_stringify

if TYPE_CHECKING:
    from promptphoo.models.eval import Eval

logger = logging.getLogger(__name__)

DEFAULT_QUERY_LIMIT = 100
SQLITE_MAX_BOUND_PARAMETERS = 999


def _sha256(text: str) -> str:
    from promptphoo.evaluator.run import sha256

    return sha256(text)


def locale_compare_key(value: str) -> tuple[str, str]:
    """Sort key approximating `String.prototype.localeCompare` (ICU root collation).

    Letters compare case-insensitively first, and lower case sorts ahead of upper case only
    when the strings are otherwise equal. Digits sort before letters, as in ASCII.
    """
    return (value.casefold(), value.swapcase())


def _rows(sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    cur = db.get_db().cursor()
    cur.row_factory = None  # Eval.find_by_id leaves a dict row factory on the connection
    cur.execute(sql, tuple(params))
    cols = [c[0] for c in cur.description or ()]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _iso(ms: int | float) -> str:
    from promptphoo.models.eval import to_iso

    return to_iso(ms)


def _dataset_id(config: dict[str, Any]) -> str:
    tests = config.get("tests")
    return _sha256(json_stringify(tests if is_truthy(tests) else []))


# --- EvalQueries -----------------------------------------------------------------------------


def get_vars_from_evals(evals: Sequence[Eval]) -> dict[str, list[str]]:
    """EvalQueries.getVarsFromEvals: distinct `test_case.vars` keys per eval, sorted."""
    if not evals:
        return {}
    ids = [e.id for e in evals]
    placeholders = ", ".join("?" for _ in ids)
    rows = _rows(
        "SELECT DISTINCT j.key, eval_id FROM ("
        " SELECT eval_id, json_extract(eval_results.test_case, '$.vars') as vars"
        f" FROM eval_results WHERE eval_id IN ({placeholders})"
        ") t, json_each(t.vars) j",
        ids,
    )
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["eval_id"], []).append(r["key"])
    for keys in out.values():
        keys.sort()
    return out


# --- prompts and datasets ------------------------------------------------------------------


def get_prompts_with_predicate(
    predicate: Callable[[Eval], bool], limit: int | float | None
) -> list[dict[str, Any]]:
    """getPromptsWithPredicate: prompts of the newest `limit` evals, grouped by sha256(raw)."""
    from promptphoo.models.eval import Eval

    grouped: dict[str, dict[str, Any]] = {}
    for eval_ in Eval.get_many(limit):
        created_ms = eval_.created_at
        if not predicate(eval_):
            continue
        dataset_id = _dataset_id(eval_.config)
        for prompt in eval_.get_prompts():
            prompt_id = _sha256(prompt["raw"])
            entry = {"id": eval_.id, "datasetId": dataset_id, "metrics": prompt.get("metrics")}
            if prompt_id in grouped:
                g = grouped[prompt_id]
                g["_recentMs"] = max(g["_recentMs"], created_ms)
                g["recentEvalDate"] = _iso(g["_recentMs"])
                g["count"] += 1
                g["evals"].append(entry)
            else:
                grouped[prompt_id] = {
                    "count": 1,
                    "id": prompt_id,
                    "prompt": prompt,
                    "recentEvalDate": _iso(created_ms),
                    "recentEvalId": eval_.id,
                    "evals": [entry],
                    "_recentMs": created_ms,
                }
    for g in grouped.values():
        del g["_recentMs"]
    return list(grouped.values())


def get_prompts_for_test_cases_hash(
    test_cases_sha256: str, limit: int | float | None = DEFAULT_QUERY_LIMIT
) -> list[dict[str, Any]]:
    return get_prompts_with_predicate(lambda e: _dataset_id(e.config) == test_cases_sha256, limit)


def read_test_cases(limit: int | float | None) -> list[dict[str, Any]]:
    """readTestCases: datasets of the newest `limit` evals, grouped by sha256(tests)."""
    from promptphoo import sanitizer
    from promptphoo.models import result_sanitize as rs
    from promptphoo.models.eval import Eval

    grouped: dict[str, dict[str, Any]] = {}
    for eval_ in Eval.get_many(limit):
        flags = rs.get_strip_flags(eval_.config.get("env"))
        config = sanitizer.sanitize_config_for_output(
            eval_.config,
            should_strip_prompt_text=flags["shouldStripPromptText"],
            should_strip_test_vars=flags["shouldStripTestVars"],
            should_strip_metadata=flags["shouldStripMetadata"],
            should_strip_response_output=flags["shouldStripResponseOutput"],
        )
        test_cases = config.get("tests") if isinstance(config, dict) else None
        if not is_truthy(test_cases):
            continue
        if not isinstance(test_cases, str | list):
            logger.warning("Skipping TestGeneratorConfig object in database storage")
            continue
        eval_id = eval_.id
        dataset_id = _dataset_id(eval_.config)
        if dataset_id in grouped:
            g = grouped[dataset_id]
            g["_recentMs"] = max(g["_recentMs"], eval_.created_at)
            g["recentEvalDate"] = _iso(g["_recentMs"])
            g["count"] += 1
        else:
            g = grouped[dataset_id] = {
                "id": dataset_id,
                "count": 1,
                "testCases": test_cases,
                "recentEvalDate": _iso(eval_.created_at),
                "recentEvalId": eval_id,
                "prompts": [],
                "_recentMs": eval_.created_at,
            }
        new_prompts = [
            {"id": _sha256(p["raw"]), "prompt": p, "evalId": eval_id} for p in eval_.get_prompts()
        ]
        by_id: dict[str, dict[str, Any]] = {}
        for p in [*g["prompts"], *new_prompts]:
            by_id.setdefault(p["id"], p)
        g["prompts"] = list(by_id.values())
    for g in grouped.values():
        del g["_recentMs"]
    return list(grouped.values())


def get_prompts(limit: int | float | None = DEFAULT_QUERY_LIMIT) -> list[dict[str, Any]]:
    return get_prompts_with_predicate(lambda _e: True, limit)


def get_test_cases(limit: int | float | None = DEFAULT_QUERY_LIMIT) -> list[dict[str, Any]]:
    return read_test_cases(limit)


def get_prompt_from_hash(hash_: str) -> dict[str, Any] | None:
    for prompt in get_prompts():
        if prompt["id"].startswith(hash_):
            return prompt
    return None


def get_dataset_from_hash(hash_: str) -> dict[str, Any] | None:
    for dataset in get_test_cases():
        if dataset["id"].startswith(hash_):
            return dataset
    return None


# --- evals -----------------------------------------------------------------------------------


def get_evals_with_predicate(
    predicate: Callable[[dict[str, Any]], bool], limit: int | float | None
) -> list[dict[str, Any]]:
    """getEvalsWithPredicate: `{id, date, config, results, description}`, newest first."""
    rows = _rows(
        "SELECT id, created_at, author, results, config, description FROM evals "
        "ORDER BY created_at DESC LIMIT ?",
        (_sql_limit(limit),),
    )
    out = []
    for r in rows:
        config = db.from_db_json(r["config"])
        results = db.from_db_json(r["results"])
        wrapper = {
            "version": 3,
            "createdAt": _iso(r["created_at"]),
            "author": r["author"],
            "results": results,
            "config": config,
        }
        if predicate(wrapper):
            item = {"id": r["id"], "date": _iso(r["created_at"]), "config": config}
            item["results"] = results
            if r["description"]:
                item["description"] = r["description"]
            out.append(item)
    return out


def get_evals(limit: int | float | None = DEFAULT_QUERY_LIMIT) -> list[dict[str, Any]]:
    return get_evals_with_predicate(lambda _r: True, limit)


def get_eval_from_id(hash_: str) -> dict[str, Any] | None:
    """getEvalFromId: the newest of the last 100 evals whose id starts with `hash_`."""
    for eval_ in get_evals():
        if eval_["id"].startswith(hash_):
            return eval_
    return None


def _sql_limit(limit: int | float | None) -> int | float:
    # drizzle's `.limit(undefined)` falls back to the default parameter upstream.
    if limit is None:
        return DEFAULT_QUERY_LIMIT
    return int(limit) if float(limit).is_integer() else limit


# --- deletion --------------------------------------------------------------------------------


def _batches(ids: Sequence[str]) -> Iterable[Sequence[str]]:
    for i in range(0, len(ids), SQLITE_MAX_BOUND_PARAMETERS):
        yield ids[i : i + SQLITE_MAX_BOUND_PARAMETERS]


def _delete_eval_rows(conn: sqlite3.Connection, ids: Sequence[str]) -> int:
    from promptphoo.db.import_export import delete_trace_records_for_evals

    delete_trace_records_for_evals(conn, list(ids))
    deleted = 0
    for batch in _batches(ids):
        marks = ", ".join("?" for _ in batch)
        params = tuple(batch)
        for table in ("evals_to_prompts", "evals_to_datasets", "evals_to_tags", "eval_results"):
            conn.execute(f"DELETE FROM {table} WHERE eval_id in ({marks})", params)
        # blob_references rows go with the eval through their ON DELETE CASCADE foreign key.
        deleted += conn.execute(f"DELETE FROM evals WHERE id in ({marks})", params).rowcount
    return deleted


def delete_eval(eval_id: str) -> None:
    """deleteEval: remove one eval and its rows; raises when the id does not exist."""
    conn = db.get_db()
    with db.transaction(conn):
        if _delete_eval_rows(conn, [eval_id]) == 0:
            raise LookupError(f"Eval with ID {eval_id} not found")
    update_signal_file_for_deleted_evals([eval_id])


def delete_evals(ids: Sequence[str]) -> None:
    """deleteEvals: bulk delete. An empty list is a no-op and writes no signal."""
    if not ids:
        return
    conn = db.get_db()
    with db.transaction(conn):
        _delete_eval_rows(conn, list(ids))
    update_signal_file_for_deleted_evals(list(ids))


def delete_all_evals() -> None:
    """deleteAllEvals: empty every eval-owned table, in FK order."""
    conn = db.get_db()
    with db.transaction(conn):
        for table in (
            "spans", "traces", "eval_results", "evals_to_prompts", "evals_to_datasets",
            "evals_to_tags", "evals",
        ):  # fmt: skip
            conn.execute(f"DELETE FROM {table}")
    update_signal_file_for_deleted_evals(None)


def update_signal_file_for_deleted_evals(deleted_eval_ids: list[str] | None = None) -> None:
    """updateSignalFileForDeletedEvals, without merging a pending signal: `None` is "all".

    Upstream coalesces with a signal written in the last few hundred milliseconds so that a
    running viewer sees both; promptphoo has no viewer, so only the delete itself is recorded.
    """
    payload: dict[str, Any] = {"type": "delete"}
    if deleted_eval_ids is not None:
        payload["deletedEvalIds"] = deleted_eval_ids
    payload["timestamp"] = db._iso_now()
    try:
        Path(db.get_db_signal_path()).write_text(json_stringify(payload), encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to write database signal file: %s", e)
