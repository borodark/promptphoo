"""Table-row conversion for stored results and the streaming JSONL writer.

Ports src/util/exportToFile/index.ts (`convertEvalResultToTableCell`,
`convertTestResultsToTableRow`) and src/util/exportToFile/writeToFile.ts (`JsonlFileWriter`).

Results may be `promptphoo.models.eval.EvalResult` objects or plain dicts with Node's camelCase
keys. Objects are first projected to the own properties a Node `EvalResult` carries, because
upstream spreads the instance into the table cell.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import IO, Any

from promptphoo.jsvalues import UNDEFINED, json_stringify
from promptphoo.output.results_table import get_table_cell_media, get_table_cell_text


def eval_result_record(result: Any) -> dict[str, Any]:
    """The enumerable own properties of a Node EvalResult, in constructor order."""
    if isinstance(result, dict):
        return result
    record: dict[str, Any] = {
        "id": result.id,
        "evalId": result.eval_id,
        "promptIdx": result.prompt_idx,
        "testIdx": result.test_idx,
        "testCase": result.test_case,
        "prompt": result.prompt,
        "promptId": result.prompt_id,
        "error": result.error if result.error is not None else UNDEFINED,
        "score": result.score,
        "success": result.success,
        "response": result.response if result.response else UNDEFINED,
        "gradingResult": result.grading_result,
        "namedScores": result.named_scores,
        "provider": result.provider,
        "latencyMs": result.latency_ms,
        "cost": result.cost,
        "metadata": result.metadata if result.metadata is not None else UNDEFINED,
        "traceId": result.trace_id if result.trace_id is not None else UNDEFINED,
        "evaluationId": result.evaluation_id if result.evaluation_id is not None else UNDEFINED,
        "repeatIndex": result.repeat_index if result.repeat_index is not None else UNDEFINED,
        "repeatGroupId": (
            result.repeat_group_id if result.repeat_group_id is not None else UNDEFINED
        ),
        "failureReason": result.failure_reason,
        "persisted": result.persisted,
        "pluginId": ((result.test_case or {}).get("metadata") or {}).get("pluginId", UNDEFINED),
    }
    if getattr(result, "description", None) is not None:
        record["description"] = result.description
    return {k: v for k, v in record.items() if v is not UNDEFINED}


def convert_eval_result_to_table_cell(result: Any) -> dict[str, Any]:
    """convertEvalResultToTableCell(result: EvalResult) -> EvaluateTableOutput."""
    r = eval_result_record(result)
    view = {
        "response": r.get("response"),
        "error": r.get("error"),
        "testCase": r.get("testCase") or {},
        "success": r.get("success"),
    }
    result_text = get_table_cell_text(view)
    provider = r.get("provider") or {}
    return {
        **r,
        "id": r.get("id") or f"{r.get('testIdx')}-{r.get('promptIdx')}",
        "text": result_text or "",
        "prompt": (r.get("prompt") or {}).get("raw"),
        "provider": provider.get("label") or provider.get("id") or "unknown provider",
        "pass": r.get("success"),
        "cost": r.get("cost") or 0,
        **get_table_cell_media(r),
    }


def _string_or_json(value: Any) -> str:
    return value if isinstance(value, str) else (json_stringify(value) or "")


def _session_id_for_row(first: dict[str, Any]) -> str:
    from_vars = ((first.get("testCase") or {}).get("vars") or {}).get("sessionId")
    if from_vars is not None and from_vars != "":
        return _string_or_json(from_vars)
    metadata = first.get("metadata") or {}
    session_ids = metadata.get("sessionIds")
    if isinstance(session_ids, list) and session_ids:
        from promptphoo.jsvalues import js_string

        return "\n".join(js_string(i) for i in session_ids if i is not None and i != "")
    session_id = metadata.get("sessionId")
    if session_id is not None:
        return _string_or_json(session_id)
    return ""


def convert_test_results_to_table_row(
    results: Sequence[Any], vars_for_header: Sequence[str]
) -> dict[str, Any]:
    """convertTestResultsToTableRow(results, varsForHeader) -> EvaluateTableRow."""
    records = [eval_result_record(r) for r in results]
    first = records[0]
    out_vars: list[str] = []
    for var_name in vars_for_header:
        if var_name == "sessionId":
            out_vars.append(_session_id_for_row(first))
            continue
        value = ((first.get("testCase") or {}).get("vars") or {}).get(var_name)
        if value is None:
            value = ""
        out_vars.append(_string_or_json(value))
    row: dict[str, Any] = {}
    if first.get("description"):
        row["description"] = first["description"]
    outputs: list[Any] = []
    row.update(outputs=outputs, vars=out_vars, test=first.get("testCase"), testIdx=first["testIdx"])
    for record in records:
        prompt_idx = record["promptIdx"]
        while len(outputs) <= prompt_idx:
            outputs.append(None)
        outputs[prompt_idx] = convert_eval_result_to_table_cell(record)
    return row


class JsonlFileWriter:
    """JsonlFileWriter: one JSON.stringify line per write. Opens lazily, so an eval that writes
    nothing never truncates an existing file; the first write truncates unless `append`."""

    def __init__(self, file_path: str, *, append: bool = False) -> None:
        self.file_path = file_path
        self.flags = "a" if append else "w"
        self._stream: IO[str] | None = None

    def _wrap(self, action: str, error: BaseException) -> OSError:
        wrapped = OSError(f"Failed to {action} JSONL output {self.file_path}: {error}")
        wrapped.__cause__ = error
        return wrapped

    def write(self, data: Any) -> None:
        json_line = (json_stringify(data) or "undefined") + "\n"
        try:
            if self._stream is None:
                self._stream = open(self.file_path, self.flags, encoding="utf-8", newline="")  # noqa: SIM115
            self._stream.write(json_line)
            self._stream.flush()
        except OSError as error:
            raise self._wrap("write", error) from error

    def close(self) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            stream.close()
        except OSError as error:
            raise self._wrap("close", error) from error
