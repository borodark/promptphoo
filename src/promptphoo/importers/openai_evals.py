"""OpenAI Evals dashboard "output items" JSONL -> promptfoo v3 eval export.

Ports src/importers/openaiEvals/{convert,validation}.ts.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import UTC, datetime
from typing import Any

from promptphoo.evaluator.run import ResultFailureReason, generate_id_from_prompt, sha256
from promptphoo.jsvalues import js_string, json_stringify
from promptphoo.tokens import accumulate_response_token_usage, create_empty_token_usage

logger = logging.getLogger("promptphoo")

OPENAI_EVALS_PROVIDER_PREFIX = "openai-evals"
OPENAI_EVALS_IMPORT_FORMAT = "dashboard-output-items-jsonl"


# --- validation ---------------------------------------------------------------------------


def is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_number_record(value: Any) -> bool:
    return is_record(value) and all(
        _is_number(entry) and math.isfinite(entry) for entry in value.values()
    )


def _is_boolean_record(value: Any) -> bool:
    return is_record(value) and all(isinstance(entry, bool) for entry in value.values())


def _is_integer(value: Any) -> bool:
    return _is_number(value) and math.isfinite(value) and float(value).is_integer()


def is_openai_evals_jsonl_row(value: Any) -> bool:
    return (
        is_record(value)
        and isinstance(value.get("run_id"), str)
        and _is_integer(value.get("data_source_idx"))
        and value["data_source_idx"] >= 0
        and is_record(value.get("item"))
        and ("sample" not in value or is_record(value["sample"]))
        and _is_number_record(value.get("grades"))
        and ("grader_samples" not in value or is_record(value["grader_samples"]))
        and ("passes" not in value or _is_boolean_record(value["passes"]))
    )


# --- conversion ---------------------------------------------------------------------------


def _iso_now() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _create_prompt(run_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    has_string_input = all(isinstance(row["item"].get("input"), str) for row in rows)
    return {
        "raw": "{{item.input}}" if has_string_input else "Imported OpenAI eval item",
        "label": f"OpenAI eval item input ({run_id})"
        if has_string_input
        else f"Imported OpenAI eval item ({run_id})",
        "provider": f"{OPENAI_EVALS_PROVIDER_PREFIX}:{run_id}",
    }


def _create_test_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "description": f"Imported OpenAI eval item {js_string(row['data_source_idx'])}",
        "vars": {"item": row["item"]},
        "metadata": {
            "openaiRunId": row["run_id"],
            "openaiDataSourceIdx": row["data_source_idx"],
        },
    }


def _failure_reason(error: str | None, success: bool) -> int:
    if error is not None:
        return ResultFailureReason.ERROR
    return ResultFailureReason.NONE if success else ResultFailureReason.ASSERT


def _sample_error(sample: dict[str, Any] | None) -> str | None:
    if sample is None:
        return None
    if "error" not in sample:
        return None
    error = sample["error"]
    if isinstance(error, str):
        return error
    if is_record(error) and isinstance(error.get("message"), str):
        return error["message"]
    logger.debug(
        "[openaiEvals] Unrecognized sample.error shape; preserving raw value: %s",
        json_stringify(error),
    )
    return json_stringify(error)


def _token_usage(sample: dict[str, Any]) -> dict[str, Any] | None:
    if "token_usage" not in sample:
        return None
    usage = sample["token_usage"]
    if not is_record(usage):
        logger.debug("[openaiEvals] Dropping non-record token_usage: %s", json_stringify(usage))
        return None
    total = usage.get("total_tokens")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    cached = usage.get("cached_tokens")
    if (
        not _is_number(total)
        or not _is_number(prompt)
        or not _is_number(completion)
        or ("cached_tokens" in usage and not _is_number(cached))
    ):
        logger.debug(
            "[openaiEvals] Dropping malformed token_usage (non-numeric field): %s",
            json_stringify(usage),
        )
        return None
    return {
        "total": total,
        "prompt": prompt,
        "completion": completion,
        "cached": cached if cached is not None else 0,
        "numRequests": 1,
    }


_NO_OUTPUT = object()


def _output(sample: dict[str, Any]) -> Any:
    if "outputs" not in sample:
        return _NO_OUTPUT
    outputs = sample["outputs"]
    if not isinstance(outputs, list):
        logger.debug("[openaiEvals] Dropping non-array outputs: %s", json_stringify(outputs))
        return _NO_OUTPUT
    if len(outputs) == 1 and is_record(outputs[0]):
        content = outputs[0].get("content")
        return content if content is not None else outputs[0]
    return outputs


def _response(sample: dict[str, Any] | None) -> dict[str, Any] | None:
    if sample is None:
        return None
    output = _output(sample)
    error = _sample_error(sample)
    token_usage = _token_usage(sample)
    response: dict[str, Any] = {}
    if output is not _NO_OUTPUT:
        response["output"] = output
    if error is not None:
        response["error"] = error
    if isinstance(sample.get("finish_reason"), str):
        response["finishReason"] = sample["finish_reason"]
    if token_usage is not None:
        response["tokenUsage"] = token_usage
    response["raw"] = sample
    return response


def _component_result(
    grader_name: str, grade: Any, passed: bool, grader_sample: Any, has_sample: bool
) -> dict[str, Any]:
    verdict = "passed" if passed else "failed"
    reason = (
        f'OpenAI grader "{grader_name}" {verdict}.'
        if grade is None
        else f'OpenAI grader "{grader_name}" {verdict} with grade {js_string(grade)}.'
    )
    result: dict[str, Any] = {"pass": passed, "score": 1 if passed else 0, "reason": reason}
    if grade is not None:
        result["namedScores"] = {grader_name: grade}
    metadata: dict[str, Any] = {"openaiGraderName": grader_name}
    if grade is not None:
        metadata["openaiGrade"] = grade
    metadata["openaiPass"] = passed
    if has_sample:
        metadata["openaiGraderSample"] = grader_sample
    result["metadata"] = metadata
    return result


def _grading_result(row: dict[str, Any]) -> dict[str, Any]:
    passes = row.get("passes") or {}
    grader_samples = row.get("grader_samples") or {}
    components = [
        _component_result(
            name,
            row["grades"].get(name),
            passes[name],
            grader_samples.get(name),
            name in grader_samples,
        )
        for name in passes
    ]
    passed_count = sum(1 for c in components if c["pass"])
    total = len(components)
    passed = total == 0 or passed_count == total
    failed_names = [c["metadata"]["openaiGraderName"] for c in components if not c["pass"]]
    if total == 0:
        reason = (
            "Imported OpenAI eval item did not include grader pass results; grader scores are "
            "preserved as named scores."
        )
    elif passed:
        reason = "All imported OpenAI eval graders passed."
    else:
        reason = f"Imported OpenAI eval graders failed: {', '.join(failed_names)}."
    return {
        "pass": passed,
        "score": 1 if total == 0 else passed_count / total,
        "reason": reason,
        "namedScores": row["grades"],
        "componentResults": components,
        "metadata": {"openaiRunId": row["run_id"], "openaiDataSourceIdx": row["data_source_idx"]},
    }


def _result(
    row: dict[str, Any], test_idx: int, prompt_idx: int, imported_prompt: dict[str, Any]
) -> dict[str, Any]:
    grading_result = _grading_result(row)
    sample = row.get("sample")
    response = _response(sample)
    error = _sample_error(sample)
    success = error is None and grading_result["pass"]
    item_input = row["item"].get("input")
    prompt_raw = item_input if isinstance(item_input, str) else json_stringify(row["item"], 2)
    out: dict[str, Any] = {
        "promptIdx": prompt_idx,
        "testIdx": test_idx,
        "testCase": _create_test_case(row),
        "promptId": generate_id_from_prompt(imported_prompt),
        "provider": {
            "id": imported_prompt["provider"],
            "label": f"OpenAI Evals run {row['run_id']}",
        },
        "prompt": {"raw": prompt_raw, "label": imported_prompt["label"]},
        "vars": {"item": row["item"]},
    }
    if error is not None:
        out["error"] = error
    if response is not None:
        out["response"] = response
    openai: dict[str, Any] = {
        "runId": row["run_id"],
        "dataSourceIdx": row["data_source_idx"],
        "item": row["item"],
    }
    if sample is not None:
        openai["sample"] = sample
    openai.update(
        grades=row["grades"],
        passes=row.get("passes") or {},
        graderSamples=row.get("grader_samples") or {},
        outputItem=row,
    )
    out.update(
        failureReason=_failure_reason(error, success),
        success=success,
        score=grading_result["score"] if error is None else 0,
        latencyMs=0,
        gradingResult=grading_result,
        namedScores=row["grades"],
        metadata={"openai": openai},
    )
    return out


def _utf16_key(s: str) -> bytes:
    return s.encode("utf-16-be", "surrogatepass")


def _canonicalize(value: Any) -> Any:
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    if not is_record(value):
        return value
    return {k: _canonicalize(value[k]) for k in sorted(value, key=_utf16_key)}


def _alignment_key(row: dict[str, Any]) -> str:
    item_hash = sha256(json_stringify(_canonicalize(row["item"])))
    return f"{js_string(row['data_source_idx'])}:{item_hash}"


def _sanitize_id_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)


def _prompt_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "score": 0,
        "testPassCount": 0,
        "testFailCount": 0,
        "testErrorCount": 0,
        "assertPassCount": 0,
        "assertFailCount": 0,
        "totalLatencyMs": 0,
        "tokenUsage": create_empty_token_usage(),
        "namedScores": {},
        "namedScoresCount": {},
        "namedScoreWeights": {},
        "cost": 0,
    }
    for result in results:
        metrics["score"] += result["score"]
        if result["success"]:
            metrics["testPassCount"] += 1
        elif result["failureReason"] == ResultFailureReason.ERROR:
            metrics["testErrorCount"] += 1
        else:
            metrics["testFailCount"] += 1
        for component in (result.get("gradingResult") or {}).get("componentResults") or []:
            key = "assertPassCount" if component["pass"] else "assertFailCount"
            metrics[key] += 1
        accumulate_response_token_usage(metrics["tokenUsage"], result.get("response"))
        for name, value in result["namedScores"].items():
            metrics["namedScores"][name] = metrics["namedScores"].get(name, 0) + value
            metrics["namedScoresCount"][name] = metrics["namedScoresCount"].get(name, 0) + 1
    return metrics


def convert_openai_evals_jsonl(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """convertOpenAIEvalsJsonl(rows) -> OpenAIEvalsImportResult."""
    if not rows:
        raise ValueError("OpenAI eval items JSONL import requires at least one row.")

    run_ids: list[str] = []
    prompt_idx_by_run: dict[str, int] = {}
    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    first_row_by_key: dict[str, dict[str, Any]] = {}
    test_idx_by_key: dict[str, int] = {}
    key_by_row: list[str] = []
    for row in rows:
        run_id = row["run_id"]
        if run_id not in prompt_idx_by_run:
            prompt_idx_by_run[run_id] = len(run_ids)
            run_ids.append(run_id)
            rows_by_run[run_id] = []
        rows_by_run[run_id].append(row)
        key = _alignment_key(row)
        key_by_row.append(key)
        if key not in first_row_by_key:
            test_idx_by_key[key] = len(first_row_by_key)
            first_row_by_key[key] = row

    imported_prompts = [_create_prompt(run_id, rows_by_run[run_id]) for run_id in run_ids]
    results_by_prompt: list[list[dict[str, Any]]] = [[] for _ in run_ids]
    results = []
    for i, row in enumerate(rows):
        prompt_idx = prompt_idx_by_run[row["run_id"]]
        result = _result(
            row, test_idx_by_key[key_by_row[i]], prompt_idx, imported_prompts[prompt_idx]
        )
        results_by_prompt[prompt_idx].append(result)
        results.append(result)
    completed_prompts = [
        {**prompt, "metrics": _prompt_metrics(results_by_prompt[idx])}
        for idx, prompt in enumerate(imported_prompts)
    ]
    tests = [_create_test_case(row) for row in first_row_by_key.values()]
    timestamp = _iso_now()
    if len(run_ids) == 1:
        eval_id = f"{OPENAI_EVALS_PROVIDER_PREFIX}-{_sanitize_id_part(run_ids[0])}"
        run_description = f"run {run_ids[0]}"
    else:
        digest = sha256("\n".join(sorted(run_ids, key=_utf16_key)))[:24]
        eval_id = f"{OPENAI_EVALS_PROVIDER_PREFIX}-runs-{digest}"
        run_description = f"runs {', '.join(run_ids)}"
    return {
        "evalId": eval_id,
        "results": {
            "version": 3,
            "timestamp": timestamp,
            "prompts": completed_prompts,
            "results": results,
        },
        "config": {
            "description": f"Imported from OpenAI Evals {run_description}",
            "tests": tests,
            "metadata": {
                "openaiEvalsImport": {
                    "format": OPENAI_EVALS_IMPORT_FORMAT,
                    "rowCount": len(rows),
                    "runIds": run_ids,
                }
            },
        },
        "metadata": {"evaluationCreatedAt": timestamp, "openaiRunIds": run_ids},
    }
