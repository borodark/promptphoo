"""Convert a v4+ results file into the display table (head + rows).

Ports src/presentation/evalResults.ts (`convertResultsToTable`, re-exported upstream from
src/util/convertEvalResultsToTable.ts), src/presentation/evalTableCells.ts and `getActualPrompt`
from src/util/providerResponse.ts.
"""

from __future__ import annotations

import logging
from typing import Any

from promptphoo.jsvalues import is_truthy, js_string, json_stringify

logger = logging.getLogger("promptphoo")


def get_actual_prompt(response: dict[str, Any] | None, formatted: bool = False) -> str | None:
    """getActualPrompt: the provider-reported prompt, else the legacy redteamFinalPrompt."""
    if not response:
        return None
    if "prompt" in response:
        prompt = response["prompt"]
        if isinstance(prompt, str):
            return prompt or None
        if isinstance(prompt, list) and prompt:
            return json_stringify(prompt, 2) if formatted else json_stringify(prompt)
        return None
    return (response.get("metadata") or {}).get("redteamFinalPrompt")


def get_table_cell_text(result: dict[str, Any]) -> str:
    """getTableCellText."""
    response = result.get("response") or {}
    raw_output = response.get("output")
    if isinstance(raw_output, dict | list):
        display = json_stringify(raw_output)
    elif raw_output is None or raw_output == "":
        display = result.get("error") or ""
    else:
        display = js_string(raw_output)
    if (result.get("testCase") or {}).get("assert"):
        if result.get("success"):
            return display or result.get("error") or ""
        return display
    if result.get("error"):
        return js_string(result["error"])
    return display


def get_table_cell_media(result: dict[str, Any]) -> dict[str, Any]:
    """getTableCellMedia (keys that are undefined in JS are omitted)."""
    response = result.get("response") or {}
    out: dict[str, Any] = {}
    audio = response.get("audio")
    if audio:
        out["audio"] = {
            k: audio[k]
            for k in (
                "id", "expiresAt", "data", "blobRef", "transcript", "format", "sampleRate",
                "channels", "duration",
            )
            if k in audio
        }  # fmt: skip
    video = response.get("video")
    if video:
        out["video"] = {
            k: video[k]
            for k in (
                "id", "blobRef", "storageRef", "url", "format", "size", "duration", "thumbnail",
                "spritesheet", "model", "aspectRatio", "resolution",
            )
            if k in video
        }  # fmt: skip
    images = response.get("images")
    if images is not None:
        out["images"] = [
            {k: img[k] for k in ("data", "blobRef", "mimeType") if k in img} for img in images
        ]
    return out


def convert_results_to_table(results_file: dict[str, Any]) -> dict[str, Any]:
    """convertResultsToTable(eval_: ResultsFile) -> EvaluateTable."""
    if not is_truthy(results_file.get("prompts")):
        raise ValueError(
            "Invariant failed: Prompts are required in this version of the results file, this "
            "needs to be results file version >= 4, version: "
            f"{js_string(results_file.get('version'))}"
        )
    results = results_file["results"]
    raw_persisted = results_file.get("vars")
    if raw_persisted is not None and not isinstance(raw_persisted, list):
        logger.warning(
            "[convertResultsToTable] eval.vars is %s, not string[]; falling back to alphabetical "
            "ordering",
            "object" if isinstance(raw_persisted, dict) else type(raw_persisted).__name__,
        )
    persisted_vars = list(dict.fromkeys(raw_persisted if isinstance(raw_persisted, list) else []))
    persisted_set = set(persisted_vars)
    vars_for_header: dict[str, None] = dict.fromkeys(persisted_vars)
    var_values_for_row: dict[int, dict[str, Any]] = {}
    row_map: dict[int, dict[str, Any]] = {}

    for result in results.get("results") or []:
        for var_name in result.get("vars") or {}:
            vars_for_header[var_name] = None

        test_idx = result["testIdx"]
        row = row_map.get(test_idx)
        if row is None:
            row = {"outputs": [], "vars": [], "test": result.get("testCase")}
            if result.get("description"):
                row = {"description": result["description"], **row}

        actual_prompt = get_actual_prompt(result.get("response")) or (
            result.get("metadata") or {}
        ).get("redteamFinalPrompt")
        if result.get("vars") and actual_prompt:
            var_keys = list(result["vars"].keys())
            if len(var_keys) == 1 and var_keys[0] != "harmCategory":
                result["vars"][var_keys[0]] = actual_prompt
            elif len(var_keys) > 1:
                key = next(
                    (k for k in ("prompt", "query", "question") if result["vars"].get(k)), None
                )
                if key:
                    result["vars"][key] = actual_prompt

        metadata = result.get("metadata") or {}
        if not (result.get("vars") or {}).get("sessionId"):
            session_ids = metadata.get("sessionIds")
            if isinstance(session_ids, list) and session_ids:
                result["vars"] = result.get("vars") or {}
                result["vars"]["sessionId"] = "\n".join(
                    js_string(i) for i in session_ids if i is not None and i != ""
                )
                vars_for_header["sessionId"] = None
            elif metadata.get("sessionId"):
                result["vars"] = result.get("vars") or {}
                result["vars"]["sessionId"] = metadata["sessionId"]
                vars_for_header["sessionId"] = None

        transform_display_vars = ((result.get("response") or {}).get("metadata") or {}).get(
            "transformDisplayVars"
        )
        if transform_display_vars:
            result["vars"] = result.get("vars") or {}
            for key, value in transform_display_vars.items():
                if key not in result["vars"]:
                    result["vars"][key] = value
                    vars_for_header[key] = None
                elif result["vars"][key] != value:
                    logger.debug(
                        "[convertResultsToTable] transformDisplayVars key '%s' collides with "
                        "result.vars; preserving original value",
                        key,
                    )

        var_values_for_row[test_idx] = result.get("vars")  # type: ignore[assignment]
        row_map[test_idx] = row

        provider = result.get("provider") or {}
        output = {
            "id": result.get("id") or f"{test_idx}-{result['promptIdx']}",
            **result,
            "text": get_table_cell_text(result) or "",
            "prompt": (result.get("prompt") or {}).get("raw"),
            "provider": provider.get("label") or provider.get("id") or "unknown provider",
            "pass": result.get("success"),
            "failureReason": result.get("failureReason"),
            "cost": result.get("cost") or 0,
            "tokenUsage": result.get("tokenUsage"),
            **get_table_cell_media(result),
        }
        outputs = row["outputs"]
        prompt_idx = result["promptIdx"]
        while len(outputs) <= prompt_idx:
            outputs.append(None)
        outputs[prompt_idx] = output
        if not result.get("promptId"):
            raise ValueError("Invariant failed: Prompt ID is required")
        row["testIdx"] = test_idx

    # Object.values on integer keys iterates in ascending numeric order.
    rows = [row_map[k] for k in sorted(row_map)]
    additional = sorted(name for name in vars_for_header if name not in persisted_set)
    ordered_vars = [*persisted_vars, *additional] if persisted_vars else additional
    for row in rows:
        values = var_values_for_row.get(row["testIdx"]) or {}
        out_vars = []
        for name in ordered_vars:
            value = values.get(name)
            if value is None:
                value = ""
            out_vars.append(value if isinstance(value, str) else json_stringify(value, 2))
        row["vars"] = out_vars

    return {"head": {"prompts": results_file["prompts"], "vars": ordered_vars}, "body": rows}
