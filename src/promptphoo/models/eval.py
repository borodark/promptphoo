"""Eval and EvalResult records (v4 layout: one `eval_results` row per result).

Ports the write path and summary/export path of src/models/eval.ts and src/models/evalResult.ts.
"""

from __future__ import annotations

import os
import random
import uuid
from datetime import UTC, datetime
from typing import Any

from promptphoo import db, envars, sanitizer
from promptphoo.evaluator.run import ResultFailureReason, generate_id_from_prompt, sha256
from promptphoo.jsvalues import json_stringify
from promptphoo.models import result_sanitize as rs
from promptphoo.paths import get_config_directory_path
from promptphoo.tokens import (
    accumulate_grading_token_usage,
    accumulate_response_token_usage,
    accumulate_token_usage,
    create_empty_token_usage,
)
from promptphoo.yamlload import load_yaml

_ID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def random_sequence(length: int = 3) -> str:
    return "".join(random.choice(_ID_CHARS) for _ in range(length))


def to_iso(ms: int | float) -> str:
    dt = datetime.fromtimestamp(ms / 1000, UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def create_eval_id(created_at_ms: int | None = None) -> str:
    ms = created_at_ms if created_at_ms is not None else _now_ms()
    return f"eval-{random_sequence(3)}-{to_iso(ms)[:19]}"


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _read_global_config() -> dict[str, Any]:
    path = os.path.join(get_config_directory_path(), "promptfoo.yaml")
    try:
        with open(path, encoding="utf-8") as f:
            data = load_yaml(f.read())
    except OSError:
        return {}
    return data if isinstance(data, dict) else {}


def get_author(override: str | None = None) -> str | None:
    """getAuthor: cloud identity, then override, account email, PROMPTFOO_AUTHOR."""
    global_config = _read_global_config()
    email = (global_config.get("account") or {}).get("email")
    logged_in = bool((global_config.get("cloud") or {}).get("apiKey"))
    if logged_in and email:
        return email
    return override or email or envars.get_env_string("PROMPTFOO_AUTHOR") or None


class EvalResult:
    def __init__(self, row: dict[str, Any], persisted: bool = False) -> None:
        self.id: str = row["id"]
        self.eval_id: str = row["evalId"]
        self.prompt_idx: int = row["promptIdx"]
        self.test_idx: int = row["testIdx"]
        self.test_case: dict[str, Any] = row["testCase"]
        self.prompt: dict[str, Any] = row["prompt"]
        self.prompt_id: str = row.get("promptId") or generate_id_from_prompt(row["prompt"])
        self.error: str | None = row.get("error")
        self.score: float = row["score"]
        self.success: bool = bool(row["success"])
        self.response: dict[str, Any] | None = row.get("response") or None
        self.grading_result: dict[str, Any] | None = row.get("gradingResult")
        self.named_scores: dict[str, float] = row.get("namedScores") or {}
        self.provider: dict[str, Any] = row["provider"]
        self.latency_ms = row.get("latencyMs") or 0
        self.cost = row.get("cost") or 0
        self.description: str | None = row.get("description")
        trace = rs.surface_trace_metadata(row.get("metadata"))
        repeat = rs.surface_repeat_metadata(trace.get("metadata"))
        self.metadata: dict[str, Any] | None = repeat.get("metadata")
        self.trace_id = trace.get("traceId")
        self.evaluation_id = trace.get("evaluationId")
        self.repeat_index = repeat.get("repeatIndex")
        self.repeat_group_id = repeat.get("repeatGroupId")
        fr = row.get("failureReason")
        self.failure_reason = fr if rs.is_result_failure_reason(fr) else ResultFailureReason.NONE
        self.persisted = persisted

    @classmethod
    def from_db_row(cls, r: Any) -> EvalResult:
        j = db.from_db_json
        return cls(
            {
                "id": r["id"],
                "evalId": r["eval_id"],
                "promptIdx": r["prompt_idx"],
                "testIdx": r["test_idx"],
                "testCase": j(r["test_case"]),
                "prompt": j(r["prompt"]),
                "promptId": r["prompt_id"],
                "provider": j(r["provider"]),
                "latencyMs": r["latency_ms"],
                "cost": r["cost"],
                "response": j(r["response"]),
                "error": r["error"],
                "failureReason": r["failure_reason"],
                "success": r["success"],
                "score": r["score"],
                "gradingResult": j(r["grading_result"]),
                "namedScores": j(r["named_scores"]),
                "metadata": j(r["metadata"]),
            },
            persisted=True,
        )

    @classmethod
    def create_from_evaluate_result(
        cls, eval_id: str, result: dict[str, Any], persist: bool = True
    ) -> EvalResult:
        r = rs.serialize_result_provider_refs(result)
        metadata = rs.persist_trace_metadata(
            rs.persist_repeat_metadata(
                r.get("metadata"), r.get("repeatIndex"), r.get("repeatGroupId")
            ),
            r.get("traceId"),
            r.get("evaluationId"),
        )
        error = r.get("error")
        args: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "evalId": eval_id,
            "testCase": rs.sanitize_for_db_with_secrets(r["testCase"]),
            "promptIdx": r["promptIdx"],
            "testIdx": r["testIdx"],
            "prompt": rs.sanitize_for_db_with_secrets(r["prompt"]),
            "promptId": generate_id_from_prompt(r["prompt"]),
            "error": None if error is None else str(error),
            "success": r["success"],
            "score": 0 if r.get("score") is None else r["score"],
            "response": rs.sanitize_for_db(result.get("response") or None),
            "gradingResult": rs.sanitize_for_db(r.get("gradingResult") or None),
            "namedScores": rs.sanitize_for_db(r.get("namedScores")),
            "provider": rs.sanitize_provider(r["provider"]),
            "latencyMs": r.get("latencyMs"),
            "cost": r.get("cost"),
            "metadata": rs.sanitize_for_db(metadata),
            "failureReason": r.get("failureReason", ResultFailureReason.NONE),
        }
        if persist:
            redacted = rs.redact_sensitive_result_fields_for_db(
                response=args["response"],
                grading_result=args["gradingResult"],
                metadata=args["metadata"],
            )
            args.update(redacted)
            _insert_result(db.get_db(), args)
            return cls(args, persisted=True)
        return cls(args)

    def to_evaluate_result(self, strip_flags: dict[str, bool] | None = None) -> dict[str, Any]:
        flags = strip_flags or rs.get_strip_flags()
        strip_output = flags["shouldStripResponseOutput"]
        strip_meta = flags["shouldStripMetadata"]
        strip_vars = flags["shouldStripTestVars"]
        strip_prompt = flags["shouldStripPromptText"]
        response = rs.project_provider_response(
            self.response, strip_metadata=strip_meta, strip_output=strip_output
        )
        test_case = rs.project_test_case(
            self.test_case,
            strip_metadata=strip_meta,
            strip_vars=strip_vars,
            strip_output=strip_output,
            strip_prompt_text=strip_prompt,
        )
        token_usage = create_empty_token_usage()
        if self.response:
            accumulate_response_token_usage(token_usage, self.response)
        if self.grading_result:
            accumulate_grading_token_usage(
                token_usage,
                self.grading_result.get("tokensUsed"),
                cached=(self.grading_result.get("metadata") or {}).get("cachedResponse"),
            )
        out: dict[str, Any] = {"cost": self.cost}
        if self.response and self.response.get("incurredCost") is not None:
            out["incurredCost"] = self.response["incurredCost"]
        if self.description:
            out["description"] = self.description
        if self.error:
            out["error"] = self.error
        out["gradingResult"] = (
            None
            if flags["shouldStripGradingResult"]
            else rs.project_grading_result(self.grading_result, strip_prompt)
        )
        out.update(
            id=self.id,
            latencyMs=self.latency_ms,
            namedScores=self.named_scores,
            prompt=rs.project_prompt(self.prompt, strip_prompt),
            promptId=self.prompt_id,
            promptIdx=self.prompt_idx,
        )
        if self.trace_id:
            out["traceId"] = self.trace_id
        if self.evaluation_id:
            out["evaluationId"] = self.evaluation_id
        if self.repeat_group_id is not None:
            out["repeatGroupId"] = self.repeat_group_id
            out["repeatIndex"] = self.repeat_index
        provider: dict[str, Any] = {"id": self.provider.get("id")}
        if self.provider.get("label") is not None:
            provider["label"] = self.provider["label"]
        out["provider"] = provider
        if response is not None:
            out["response"] = response
        out.update(
            score=self.score,
            success=self.success,
            testCase=test_case,
            testIdx=self.test_idx,
            tokenUsage=token_usage,
            vars={} if strip_vars else (self.test_case.get("vars") or {}),
            metadata={}
            if strip_meta
            else rs.project_output_metadata(
                self.metadata,
                strip_output,
                (self.response or {}).get("metadata"),
                self.test_case.get("metadata"),
            ),
            failureReason=self.failure_reason,
        )
        return out


def _insert_result(conn: Any, a: dict[str, Any]) -> None:
    j = db.to_db_json
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO eval_results (id, eval_id, prompt_idx, test_idx, test_case, prompt, "
            "prompt_id, provider, latency_ms, cost, response, error, failure_reason, success, "
            "score, grading_result, named_scores, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                a["id"], a["evalId"], a["promptIdx"], a["testIdx"], j(a["testCase"]),
                j(a["prompt"]), a["promptId"], j(a["provider"]), a.get("latencyMs"),
                a.get("cost"), j(a.get("response")), a.get("error"), a["failureReason"],
                1 if a["success"] else 0, a["score"], j(a.get("gradingResult")),
                j(a.get("namedScores")), j(a.get("metadata")),
            ),
        )  # fmt: skip


class Eval:
    """Implements evaluator.run.EvaluationStore."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        id: str | None = None,  # noqa: A002 - Node's option name
        created_at: int | None = None,
        author: str | None = None,
        prompts: list[dict[str, Any]] | None = None,
        dataset_id: str | None = None,
        persisted: bool = False,
        vars: list[str] | None = None,  # noqa: A002
        runtime_options: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        old_results: dict[str, Any] | None = None,
    ) -> None:
        self.created_at = created_at if created_at is not None else _now_ms()
        self.id = id or create_eval_id(self.created_at)
        self.author = author
        self.config = config
        self.results: list[EvalResult] = []
        self.prompts: list[dict[str, Any]] = prompts or []
        self.dataset_id = dataset_id
        self.persisted = persisted
        self.vars: list[str] = vars or []
        self.runtime_options = runtime_options
        self.duration_ms = duration_ms
        self.generation_duration_ms: int | None = None
        self.evaluation_duration_ms: int | None = None
        self.old_results = old_results
        self._results_loaded = False

    # --- creation -------------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        config: dict[str, Any],
        rendered_prompts: list[dict[str, Any]],
        *,
        author: Any = ...,
        runtime_options: dict[str, Any] | None = None,
        created_at: int | None = None,
        eval_id: str | None = None,
    ) -> Eval:
        created_at = created_at if created_at is not None else _now_ms()
        eval_id = eval_id or create_eval_id(created_at)
        author = get_author() if author is ... else author
        conn = db.get_db()
        dataset_id = sha256(json_stringify(config.get("tests") or []))
        runtime = sanitizer.sanitize_runtime_options(runtime_options)
        j = db.to_db_json
        with db.transaction(conn):
            conn.execute(
                "INSERT INTO evals (id, created_at, author, description, results, config, "
                "prompts, vars, runtime_options, is_redteam) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    eval_id, created_at, author, config.get("description"), "{}",
                    j(sanitizer.sanitize_tracing_config_for_persistence(config)), "[]", "[]",
                    j(runtime),
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
                    "INSERT INTO evals_to_tags (eval_id, tag_id) VALUES (?, ?) "
                    "ON CONFLICT DO NOTHING",
                    (eval_id, tag_id),
                )
        db.update_signal_file(eval_id)
        return cls(
            config,
            id=eval_id,
            author=author,
            created_at=created_at,
            persisted=True,
            runtime_options=runtime,
            dataset_id=dataset_id,
        )

    @classmethod
    def find_by_id(cls, eval_id: str) -> Eval | None:
        conn = db.get_db()
        conn.row_factory = _row_factory
        row = conn.execute("SELECT * FROM evals WHERE id = ?", (eval_id,)).fetchone()
        if row is None:
            return None
        dataset = conn.execute(
            "SELECT dataset_id FROM evals_to_datasets WHERE eval_id = ? LIMIT 1", (eval_id,)
        ).fetchone()
        return cls._from_row(row, dataset["dataset_id"] if dataset else None)

    @classmethod
    def latest(cls) -> Eval | None:
        conn = db.get_db()
        conn.row_factory = _row_factory
        row = conn.execute("SELECT id FROM evals ORDER BY created_at DESC LIMIT 1").fetchone()
        return cls.find_by_id(row["id"]) if row else None

    @classmethod
    def _from_row(cls, row: Any, dataset_id: str | None) -> Eval:
        j = db.from_db_json
        results = j(row["results"]) or {}
        old = results if isinstance(results, dict) and "table" in results else None
        ev = cls(
            j(row["config"]) or {},
            id=row["id"],
            created_at=row["created_at"],
            author=row["author"],
            prompts=j(row["prompts"]) or [],
            dataset_id=dataset_id,
            persisted=True,
            vars=j(row["vars"]) or [],
            runtime_options=j(row["runtime_options"]),
            old_results=old,
        )
        if not old and isinstance(results, dict):
            ev.duration_ms = results.get("durationMs")
            ev.generation_duration_ms = results.get("generationDurationMs")
            ev.evaluation_duration_ms = results.get("evaluationDurationMs")
        return ev

    # --- EvaluationStore ------------------------------------------------------------------

    def version(self) -> int:
        return 3 if self.old_results and "table" in self.old_results else 4

    async def append_prompts(self, prompts: list[dict[str, Any]]) -> None:
        self.prompts = prompts
        if self.persisted:
            conn = db.get_db()
            with db.transaction(conn):
                conn.execute(
                    "UPDATE evals SET prompts = ? WHERE id = ?",
                    (db.to_db_json(prompts), self.id),
                )
            db.update_signal_file(self.id)

    async def append_result(self, row: dict[str, Any]) -> None:
        result = EvalResult.create_from_evaluate_result(self.id, row, persist=self.persisted)
        if not self.persisted:
            self.results.append(result)
        else:
            db.update_signal_file(self.id)

    def set_vars(self, vars_: list[str]) -> None:
        self.vars = vars_

    def set_duration_ms(self, ms: int) -> None:
        if ms < 0:
            return
        self.evaluation_duration_ms = ms
        self.duration_ms = (self.generation_duration_ms or 0) + ms

    async def save(self) -> None:
        conn = db.get_db()
        j = db.to_db_json
        params: list[Any] = [
            j(sanitizer.sanitize_tracing_config_for_persistence(self.config)),
            1 if "redteam" in self.config else 0,
            j(self.prompts),
            self.config.get("description"),
            self.author,
            j(list(self.vars)),
            j(sanitizer.sanitize_runtime_options(self.runtime_options)),
        ]
        results_sql = ""
        if self.version() < 4:
            results_sql = ", results = ?"
            params.append(j(self.old_results))
        else:
            expr = (
                "CASE WHEN json_valid(results) AND json_type(results) = 'object' "
                "THEN results ELSE '{}' END"
            )
            for key, value in (
                ("durationMs", self.duration_ms),
                ("generationDurationMs", self.generation_duration_ms),
                ("evaluationDurationMs", self.evaluation_duration_ms),
            ):
                if value is not None:
                    expr = f"json_set({expr}, '$.{key}', ?)"
                    params.append(value)
            if len(params) > 7:
                results_sql = f", results = {expr}"
        params.append(self.id)
        with db.transaction(conn):
            conn.execute(
                "UPDATE evals SET config = ?, is_redteam = ?, prompts = ?, description = ?, "
                f"author = ?, vars = ?, runtime_options = ?{results_sql} WHERE id = ?",
                params,
            )
        db.update_signal_file(self.id)
        self.persisted = True

    # --- reading --------------------------------------------------------------------------

    def load_results(self) -> list[EvalResult]:
        if self.persisted:
            conn = db.get_db()
            conn.row_factory = _row_factory
            rows = conn.execute(
                "SELECT * FROM eval_results WHERE eval_id = ?",  # Node: no ORDER BY
                (self.id,),
            ).fetchall()
            self.results = [EvalResult.from_db_row(r) for r in rows]
        self._results_loaded = True
        return self.results

    def get_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "successes": 0,
            "failures": 0,
            "errors": 0,
            "tokenUsage": create_empty_token_usage(),
        }
        for key, value in (
            ("durationMs", self.duration_ms),
            ("generationDurationMs", self.generation_duration_ms),
            ("evaluationDurationMs", self.evaluation_duration_ms),
        ):
            if value is not None:
                stats[key] = value
        for prompt in self.prompts:
            metrics = prompt.get("metrics") or {}
            stats["successes"] += metrics.get("testPassCount") or 0
            stats["failures"] += metrics.get("testFailCount") or 0
            stats["errors"] += metrics.get("testErrorCount") or 0
            if metrics.get("testCacheHitCount"):
                stats["cacheHits"] = stats.get("cacheHits", 0) + metrics["testCacheHitCount"]
            accumulate_token_usage(stats["tokenUsage"], metrics.get("tokenUsage"))
        return stats

    def to_evaluate_summary(self) -> dict[str, Any]:
        if self.version() < 4:
            old = self.old_results or {}
            return {
                "version": 2,
                "timestamp": to_iso(self.created_at),
                "results": old.get("results"),
                "table": old.get("table"),
                "stats": old.get("stats"),
            }
        if not self._results_loaded and self.persisted:
            self.load_results()
        flags = rs.get_strip_flags(self.config.get("env"))
        return {
            "version": 3,
            "timestamp": to_iso(self.created_at),
            "prompts": [rs.project_prompt(p, flags["shouldStripPromptText"]) for p in self.prompts],
            "results": [r.to_evaluate_result(flags) for r in self.results],
            "stats": self.get_stats(),
        }

    def get_prompts(self) -> list[dict[str, Any]]:
        if self.version() < 4:
            return ((self.old_results or {}).get("table") or {}).get("head", {}).get("prompts", [])
        return self.prompts

    def to_results_file(self) -> dict[str, Any]:
        flags = rs.get_strip_flags(self.config.get("env"))
        out: dict[str, Any] = {
            "version": self.version(),
            "createdAt": to_iso(self.created_at),
            "results": self.to_evaluate_summary(),
            "config": sanitizer.sanitize_config_for_output(
                self.config,
                should_strip_prompt_text=flags["shouldStripPromptText"],
                should_strip_test_vars=flags["shouldStripTestVars"],
                should_strip_metadata=flags["shouldStripMetadata"],
                should_strip_response_output=flags["shouldStripResponseOutput"],
            ),
            "author": self.author or None,
            "prompts": [
                rs.project_prompt(p, flags["shouldStripPromptText"]) for p in self.get_prompts()
            ],
        }
        if self.vars:
            out["vars"] = list(self.vars)
        out["datasetId"] = self.dataset_id or None
        return out


def _row_factory(cursor: Any, row: tuple[Any, ...]) -> dict[str, Any]:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}
