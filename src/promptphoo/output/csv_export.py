"""CSV export of an eval (`promptfoo eval -o out.csv`).

Ports the CLI path of src/util/eval/evalTableUtils.ts (`streamEvalCsv`, `buildCsvHeaders`,
`tableRowToCsvValues`, formula-injection escaping) and the default behaviour of
csv-stringify 6's `stringify(rows)` (comma delimiter, `"` quoting only when a field contains a
comma, quote or line break, booleans as `1`/``, `\\n` after every record).
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

from promptphoo.jsvalues import js_string, json_stringify, to_fixed

REDTEAM_METADATA_KEYS_TO_CSV_COLUMN_NAMES = {
    "messages": "Messages",
    "redteamHistory": "RedteamHistory",
    "redteamTreeHistory": "RedteamTreeHistory",
    "pluginId": "pluginId",
    "strategyId": "strategyId",
    "sessionId": "sessionId",
    "sessionIds": "sessionIds",
}
REDTEAM_METADATA_COLUMNS = list(REDTEAM_METADATA_KEYS_TO_CSV_COLUMN_NAMES.values())

_ASSERT = 1  # ResultFailureReason.ASSERT


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


# --- csv-stringify ------------------------------------------------------------------------


def _cast(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else ""
    if isinstance(value, str):
        return value
    if _is_number(value):
        return js_string(value)
    return json_stringify(value)


def csv_stringify(rows: Iterable[Sequence[Any]]) -> str:
    """csv-stringify/sync `stringify(rows)` with default options."""
    out = []
    for row in rows:
        fields = []
        for raw in row:
            value = _cast(raw)
            if not value:
                fields.append("")
                continue
            if '"' in value or "," in value or "\n" in value or "\r" in value:
                value = '"' + value.replace('"', '""') + '"'
            fields.append(value)
        out.append(",".join(fields) + "\n")
    return "".join(out)


# --- formula escaping ---------------------------------------------------------------------


_JS_WS = "\t\n\x0b\x0c\r \xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_DECIMAL = re.compile(r"[+-]?(?:Infinity|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
_RADIX = re.compile(r"0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+")


def _is_numeric_cell(value: str) -> bool:
    """`value.trim() !== '' && Number.isFinite(Number(value))`."""
    text = re.sub(f"^[{_JS_WS}]+|[{_JS_WS}]+$", "", value)
    if text == "":
        return False
    if _RADIX.fullmatch(text):
        return True
    return bool(_DECIMAL.fullmatch(text)) and math.isfinite(float(text))


def escape_csv_formula(value: str) -> str:
    """escapeCsvFormula: prefix spreadsheet formula triggers with `'` (CWE-1236)."""
    if len(value) == 0:
        return value
    unpadded = re.sub(r"^[\s​‌‍]+", "", value)
    if not unpadded:
        return value
    first = unpadded[0]
    trigger = first in ("=", "@") or (first in ("-", "+") and not _is_numeric_cell(unpadded))
    return f"'{value}" if trigger else value


def _escape_rows(rows: list[list[Any]], env: dict[str, Any] | None) -> list[list[Any]]:
    configured = (env or {}).get("PROMPTFOO_DISABLE_CSV_FORMULA_ESCAPING")
    if configured is None:
        configured = os.environ.get("PROMPTFOO_DISABLE_CSV_FORMULA_ESCAPING")
    if js_string(configured).lower() in ("1", "true", "yes", "yup", "yeppers") and (
        configured is not None
    ):
        return rows
    return [[escape_csv_formula(c) if isinstance(c, str) else c for c in row] for row in rows]


# --- headers and rows ---------------------------------------------------------------------


def _output_status(output: dict[str, Any]) -> str:
    if output.get("pass"):
        return "PASS"
    return "FAIL" if output.get("failureReason") == _ASSERT else "ERROR"


def _format_named_scores(named_scores: dict[str, Any] | None) -> str:
    if not named_scores:
        return ""
    rounded: dict[str, Any] = {}
    for key, value in named_scores.items():
        if _is_number(value) and not _is_nan(value):
            rounded[key] = float(to_fixed(value, 2))
    if not rounded:
        return ""
    return json_stringify(rounded)


def _format_named_score_value(value: Any) -> str:
    return to_fixed(value, 2) if _is_number(value) and not _is_nan(value) else ""


def _locale_sorted(names: Iterable[str]) -> list[str]:
    # localeCompare (ICU root collation) is approximated by a case-insensitive sort that keeps
    # lower case ahead of upper case for otherwise equal names.
    return sorted(set(names), key=lambda n: (n.casefold(), n.swapcase()))


def collect_named_score_names_from_aggregate(prompts: Sequence[dict[str, Any]]) -> list[list]:
    out = []
    for prompt in prompts:
        counts = (prompt.get("metrics") or {}).get("namedScoresCount")
        if not counts:
            out.append([])
            continue
        out.append(_locale_sorted(n for n in counts if (counts.get(n) or 0) > 0))
    return out


def prompt_needs_discovery_fallback(prompt: dict[str, Any]) -> bool:
    metrics = prompt.get("metrics")
    return not metrics or not metrics.get("namedScoresCount")


def build_csv_headers(
    vars_: Sequence[str],
    prompts: Sequence[dict[str, Any]],
    *,
    has_descriptions: bool = False,
    is_redteam: bool = False,
    named_score_names_by_prompt: Sequence[Sequence[str]] | None = None,
) -> list[str]:
    headers: list[str] = ["Description"] if has_descriptions else []
    headers.extend(vars_)
    for idx, prompt in enumerate(prompts):
        provider = prompt.get("provider") or ""
        label = js_string(prompt.get("label"))
        names = (named_score_names_by_prompt or [])[idx] if named_score_names_by_prompt else []
        headers.extend(
            [
                f"[{provider}] {label}" if provider else label,
                "Status",
                "Score",
                "Named Scores",
                *(f"Metric: {n}" for n in names or []),
                "Grader Reason",
                "Comment",
            ]
        )
    if is_redteam:
        headers.extend(REDTEAM_METADATA_COLUMNS)
    return headers


def table_row_to_csv_values(
    row: dict[str, Any],
    *,
    has_descriptions: bool = False,
    is_redteam: bool = False,
    named_score_names_by_prompt: Sequence[Sequence[str]] | None = None,
) -> list[Any]:
    values: list[Any] = (
        [(row.get("test") or {}).get("description") or ""] if has_descriptions else []
    )
    values.extend(row["vars"])
    for idx, output in enumerate(row["outputs"]):
        names: Sequence[str] = []
        if named_score_names_by_prompt and idx < len(named_score_names_by_prompt):
            names = named_score_names_by_prompt[idx] or []
        if not output:
            values.extend(["", "", "", "", *("" for _ in names), "", ""])
            continue
        score = output.get("score")
        named = output.get("namedScores") or {}
        grading = output.get("gradingResult") or {}
        values.extend(
            [
                output.get("text") or "",
                _output_status(output),
                to_fixed(score, 2) if score is not None else "",
                _format_named_scores(output.get("namedScores")),
                *(_format_named_score_value(named.get(n)) for n in names),
                grading.get("reason") or "",
                grading.get("comment") or "",
            ]
        )
    if is_redteam:
        outputs = row["outputs"]
        first_meta = (outputs[0] or {}).get("metadata") if outputs else None
        for key in REDTEAM_METADATA_KEYS_TO_CSV_COLUMN_NAMES:
            value = (first_meta or {}).get(key)
            if key == "strategyId" and value is None:
                value = "basic"
            if value is None:
                values.append("")
            elif isinstance(value, str | int | float | bool):
                values.append(js_string(value))
            else:
                values.append(json_stringify(value))
    return values


# --- streamEvalCsv ------------------------------------------------------------------------


def fetch_results_batched(eval_record: Any, batch_size: int = 100) -> Iterator[list[Any]]:
    """Eval.fetchResultsBatched: in-memory slices, or DB rows grouped in testIdx windows."""
    if not getattr(eval_record, "persisted", False):
        results = list(eval_record.results)
        for offset in range(0, len(results), batch_size):
            yield results[offset : offset + batch_size]
        return
    results = sorted(eval_record.load_results(), key=lambda r: r.test_idx)
    i = 0
    while i < len(results):
        start = results[i].test_idx
        batch = []
        while i < len(results) and results[i].test_idx < start + batch_size:
            batch.append(results[i])
            i += 1
        yield batch


def _batch_to_stream_rows(
    batch: Iterable[Any], var_names: Sequence[str], num_prompts: int
) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for result in batch:
        row = rows.get(result.test_idx)
        test_case = result.test_case or {}
        if row is None:
            tc_vars = test_case.get("vars") or {}
            row = {
                "vars": ["" if n not in tc_vars else js_string(tc_vars[n]) for n in var_names],
                "outputs": [None] * num_prompts,
                "test": {"description": test_case.get("description")},
            }
            rows[result.test_idx] = row
        while len(row["outputs"]) <= result.prompt_idx:
            row["outputs"].append(None)
        output = (result.response or {}).get("output")
        row["outputs"][result.prompt_idx] = {
            "text": "" if output is None else output,
            "pass": result.success,
            "score": result.score,
            "namedScores": result.named_scores,
            "failureReason": result.failure_reason,
            "gradingResult": result.grading_result,
            "metadata": result.metadata,
        }
    return list(rows.values())


def stream_eval_csv(
    eval_record: Any, write: Callable[[str], Any], *, is_redteam: bool = False
) -> None:
    """streamEvalCsv: header row, then one CSV chunk per result batch."""
    env = (eval_record.config or {}).get("env")
    var_names = list(eval_record.vars or [])
    prompts = list(eval_record.prompts or [])
    names_by_prompt = collect_named_score_names_from_aggregate(prompts)
    fallback = {i for i, p in enumerate(prompts) if prompt_needs_discovery_fallback(p)}
    metric_names: dict[int, set[str]] = {}
    has_descriptions = False
    for batch in fetch_results_batched(eval_record):
        for result in batch:
            if not has_descriptions and (result.test_case or {}).get("description"):
                has_descriptions = True
            if result.prompt_idx not in fallback or not result.named_scores:
                continue
            names = metric_names.setdefault(result.prompt_idx, set())
            for name, value in result.named_scores.items():
                if _is_number(value) and not _is_nan(value):
                    names.add(name)
    for idx in sorted(fallback):
        names_by_prompt[idx] = _locale_sorted(metric_names.get(idx, set()))

    opts = {
        "has_descriptions": has_descriptions,
        "is_redteam": is_redteam,
        "named_score_names_by_prompt": names_by_prompt,
    }
    headers = build_csv_headers(var_names, prompts, **opts)
    write(csv_stringify(_escape_rows([headers], env)))
    for batch in fetch_results_batched(eval_record):
        rows = _batch_to_stream_rows(batch, var_names, len(prompts))
        csv_rows = [table_row_to_csv_values(r, **opts) for r in rows]
        if csv_rows:
            write(csv_stringify(_escape_rows(csv_rows, env)))
