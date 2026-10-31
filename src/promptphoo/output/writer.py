"""Write eval results to `--output` files.

Ports src/util/output.ts (`writeOutput`, `writeMultipleOutputs`, `createOutputData`,
`createOutputMetadata`) and src/util/outputFormats.ts. JSON is `JSON.stringify(data, null, 2)`,
YAML/TXT is js-yaml `dump`, HTML renders the bundled `tableOutput.html` with the Nunjucks-compatible
engine, XML uses fast-xml-parser's builder, CSV is the streaming WebUI-format export, JSONL has one
sanitized EvaluateResult per line and `.junit.xml` is JUnit.

`eval_record` is a `promptphoo.models.eval.Eval` (or anything exposing the same surface: `id`,
`config`, `created_at`, `author`, `vars`, `prompts`, `runtime_options`, `persisted`, `results`,
`load_results()`, `version()`, `old_results`, `to_evaluate_summary()`, `to_results_file()`).
"""

from __future__ import annotations

import copy
import logging
import math
import os
import platform
import shutil
import sys
import tempfile
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import resources
from typing import Any, TypeGuard

from promptphoo import UPSTREAM_VERSION
from promptphoo.jsvalues import UNDEFINED, is_truthy, js_string, json_stringify, to_fixed
from promptphoo.output import jsyaml
from promptphoo.output.csv_export import fetch_results_batched, stream_eval_csv
from promptphoo.output.junit import write_junit_xml_output
from promptphoo.output.results_table import convert_results_to_table
from promptphoo.output.xmlbuilder import XMLBuilder

logger = logging.getLogger("promptphoo")

JUNIT_XML_OUTPUT_SUFFIX = ".junit.xml"
OUTPUT_FILE_EXTENSIONS = ("csv", "html", "json", "jsonl", "txt", "xml", "yaml", "yml")
SUPPORTED_OUTPUT_FILE_FORMATS = (*OUTPUT_FILE_EXTENSIONS, "junit.xml")

_ERROR = 2  # ResultFailureReason.ERROR
_ASSERT = 1


class OutputWriteError(Exception):
    """`writeMultipleOutputs` rejection: carries every individual failure in `errors`."""

    def __init__(self, message: str, errors: list[BaseException]) -> None:
        super().__init__(message)
        self.errors = errors


def _extname(output_path: str) -> str:
    """path.extname(p).slice(1).toLowerCase()."""
    base = os.path.basename(output_path.rstrip("/"))
    idx = base.rfind(".")
    if idx <= 0:
        return ""
    return base[idx + 1 :].lower()


def get_output_file_format(output_path: str) -> str | None:
    """getOutputFileFormat: `junit.xml` for `*.junit.xml`, else a supported extension or None."""
    if output_path.lower().endswith(JUNIT_XML_OUTPUT_SUFFIX):
        return "junit.xml"
    ext = _extname(output_path)
    return ext if ext in OUTPUT_FILE_EXTENSIONS else None


# --- helpers --------------------------------------------------------------------------------


def _rs() -> Any:
    from promptphoo.models import result_sanitize

    return result_sanitize


def _strip_flags(eval_record: Any) -> dict[str, bool]:
    return _rs().get_strip_flags((eval_record.config or {}).get("env"))


def sanitize_config_for_output(config: Any, strip_flags: dict[str, bool]) -> Any:
    """sanitizeConfigForOutput(config, stripFlags)."""
    from promptphoo import sanitizer

    return sanitizer.sanitize_config_for_output(
        config,
        should_strip_prompt_text=strip_flags["shouldStripPromptText"],
        should_strip_test_vars=strip_flags["shouldStripTestVars"],
        should_strip_metadata=strip_flags["shouldStripMetadata"],
        should_strip_response_output=strip_flags["shouldStripResponseOutput"],
    )


def _iso_now() -> str:
    dt = datetime.now(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _iso_from_ms(ms: Any) -> str | None:
    try:
        value = float(ms)
        if not math.isfinite(value):
            return None
        dt = datetime.fromtimestamp(value / 1000, UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _node_arch() -> str:
    machine = platform.machine().lower()
    return {
        "x86_64": "x64",
        "amd64": "x64",
        "aarch64": "arm64",
        "arm64": "arm64",
        "i386": "ia32",
        "i686": "ia32",
        "x86": "ia32",
        "armv7l": "arm",
        "armv6l": "arm",
        "ppc64le": "ppc64",
        "s390x": "s390x",
        "riscv64": "riscv64",
    }.get(machine, machine)


def _node_platform() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("freebsd"):
        return "freebsd"
    return sys.platform  # darwin, win32, ...


def to_locale_string(value: float | int) -> str:
    """Number.prototype.toLocaleString('en-US'): grouping, at most 3 fraction digits."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if value != value else ("∞" if value > 0 else "-∞")
    text = to_fixed(value, 3)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    sign = ""
    if text.startswith("-"):
        sign, text = "-", text[1:]
    whole, _, frac = text.partition(".")
    grouped = f"{int(whole):,}"
    if sign and whole == "0" and not frac:
        sign = "-"
    return sign + grouped + (f".{frac}" if frac else "")


def _is_finite_number(value: Any) -> TypeGuard[float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def get_table(eval_record: Any) -> dict[str, Any]:
    """Eval.getTable."""
    version = eval_record.version() if hasattr(eval_record, "version") else 4
    if version < 4:
        old = getattr(eval_record, "old_results", None) or {}
        return old.get("table") or {"head": {"prompts": [], "vars": []}, "body": []}
    return convert_results_to_table(copy.deepcopy(eval_record.to_results_file()))


# --- createOutputData -----------------------------------------------------------------------


def create_output_summary(eval_record: Any, strip_flags: dict[str, bool]) -> dict[str, Any]:
    summary = eval_record.to_evaluate_summary()
    if "prompts" in summary:
        source = summary["prompts"]
    else:
        source = ((summary.get("table") or {}).get("head") or {}).get("prompts") or []
    prompts = [
        {**p, "config": sanitize_config_for_output(p["config"], strip_flags)}
        if p.get("config")
        else p
        for p in source
    ]
    if "prompts" in summary:
        return {**summary, "prompts": prompts}
    table = summary.get("table") or {}
    return {
        **summary,
        "table": {**table, "head": {**(table.get("head") or {}), "prompts": prompts}},
    }


def create_output_metadata(eval_record: Any) -> dict[str, Any]:
    """createOutputMetadata. `nodeVersion` reports the Python runtime instead of Node's."""
    metadata: dict[str, Any] = {
        "promptfooVersion": UPSTREAM_VERSION,
        "nodeVersion": f"python-{platform.python_version()}",
        "platform": _node_platform(),
        "arch": _node_arch(),
        "exportedAt": _iso_now(),
    }
    created_at = getattr(eval_record, "created_at", None)
    if created_at:
        evaluation_created_at = _iso_from_ms(created_at)
        if evaluation_created_at is not None:
            metadata["evaluationCreatedAt"] = evaluation_created_at
    author = getattr(eval_record, "author", None)
    if author is not None:
        metadata["author"] = author
    return metadata


def create_output_data(
    eval_record: Any, shareable_url: str | None, include_media: bool = False
) -> dict[str, Any]:
    """createOutputData: the OutputFile written for json/yaml/txt outputs."""
    strip_flags = _strip_flags(eval_record)
    summary = create_output_summary(eval_record, strip_flags)
    traces = None
    try:
        # TraceStore redacts sensitive attribute keys on reads by default.
        from promptphoo.tracing.store import get_trace_store

        traces = get_trace_store().get_traces_by_evaluation_sync(eval_record.id)
    except Exception as error:
        logger.warning(
            "Failed to fetch traces for output %s; traces omitted from export: %s",
            eval_record.id,
            error,
        )
    output: dict[str, Any] = {
        "evalId": eval_record.id,
        "results": summary,
        "config": sanitize_config_for_output(eval_record.config, strip_flags),
        "shareableUrl": shareable_url,
        "metadata": create_output_metadata(eval_record),
    }
    vars_ = getattr(eval_record, "vars", None)
    if vars_:
        output["vars"] = list(vars_)
    runtime_options = getattr(eval_record, "runtime_options", None)
    if runtime_options:
        from promptphoo import sanitizer

        output["runtimeOptions"] = sanitizer.sanitize_runtime_options(runtime_options)
    if traces:
        output["traces"] = project_traces_for_output(traces, strip_flags)
    if include_media:
        blob_assets = export_blob_assets(
            eval_record.id,
            summary,
            strip_flags["shouldStripResponseOutput"],
            output.get("traces"),
        )
        if blob_assets:
            output["blobAssets"] = blob_assets
    return output


REQUEST_BODY_ATTRIBUTE = "promptfoo.request.body"
RESPONSE_BODY_ATTRIBUTE = "promptfoo.response.body"


def project_traces_for_output(
    traces: list[dict[str, Any]], strip_flags: dict[str, bool]
) -> list[dict[str, Any]]:
    """projectTracesForOutput (src/models/evalResult.ts): PROMPTFOO_STRIP_* applied to traces.
    Metadata (or just its `vars`) is dropped, and so are the request/response body span
    attributes."""
    strip_metadata = strip_flags["shouldStripMetadata"]
    strip_prompt = strip_flags["shouldStripPromptText"]
    strip_output = strip_flags["shouldStripResponseOutput"]
    strip_vars = strip_flags["shouldStripTestVars"]
    if not (strip_metadata or strip_prompt or strip_output or strip_vars):
        return traces
    projected_traces = []
    for trace in traces:
        projected = trace
        if strip_metadata:
            projected = {k: v for k, v in trace.items() if k != "metadata"}
        elif strip_vars and isinstance(trace.get("metadata"), dict) and "vars" in trace["metadata"]:
            metadata = {k: v for k, v in trace["metadata"].items() if k != "vars"}
            projected = {k: v for k, v in trace.items() if k != "metadata"}
            if metadata:
                projected["metadata"] = metadata
        if not strip_prompt and not strip_output:
            projected_traces.append(projected)
            continue
        spans = []
        for span in projected["spans"]:
            if span.get("attributes") is None:  # `{}` is truthy in JS
                spans.append(span)
                continue
            attributes = dict(span["attributes"])
            if strip_prompt:
                attributes.pop(REQUEST_BODY_ATTRIBUTE, None)
            if strip_output:
                attributes.pop(RESPONSE_BODY_ATTRIBUTE, None)
            out = {k: v for k, v in span.items() if k != "attributes"}
            if attributes:
                out["attributes"] = attributes
            spans.append(out)
        projected_traces.append({**projected, "spans": spans})
    return projected_traces


def results_for_media_export_scan(results: Any, should_strip_response_output: bool) -> Any:
    """resultsForMediaExportScan: with response output stripping on, `metadata.blobUris` must
    not pull the stripped media back into the export."""
    if not should_strip_response_output:
        return results

    def project(result: Any) -> Any:
        response = result.get("response") if isinstance(result, dict) else None
        metadata = response.get("metadata") if isinstance(response, dict) else None
        if not isinstance(metadata, dict) or "blobUris" not in metadata:
            return result
        assert isinstance(response, dict)
        rest = {k: v for k, v in metadata.items() if k != "blobUris"}
        projected = {k: v for k, v in response.items() if k != "metadata"}
        if rest:
            projected["metadata"] = rest
        return {**result, "response": projected}

    return {**results, "results": [project(r) for r in results.get("results") or []]}


def export_blob_assets(
    eval_id: str, results: Any, should_strip_response_output: bool, traces: Any = None
) -> list[dict[str, Any]]:
    """exportBlobAssets (src/util/output.ts): the referenced blobs the eval has trusted
    provenance for, base64-embedded in discovery order."""
    import base64

    from promptphoo.blobs import BLOB_MAX_SIZE, get_share_authorized_blob
    from promptphoo.blobs.blob_refs import collect_blob_hashes

    scan = {
        "results": results_for_media_export_scan(results, should_strip_response_output),
        "traces": traces,
    }
    assets: list[dict[str, Any]] = []
    for hash_ in collect_blob_hashes(scan):
        try:
            blob = get_share_authorized_blob(hash_, eval_id)
            if not blob:
                continue
            if len(blob.data) > BLOB_MAX_SIZE:
                logger.warning(
                    "[Output] Skipping oversized blob in eval export (hash=%s, sizeBytes=%d)",
                    hash_,
                    len(blob.data),
                )
                continue
            assets.append(
                {
                    "hash": hash_,
                    "mimeType": blob.metadata.get("mimeType"),
                    "sizeBytes": len(blob.data),
                    "data": base64.b64encode(blob.data).decode("ascii"),
                }
            )
        except Exception as error:
            logger.warning(
                "[Output] Skipping missing blob in eval export (hash=%s): %s", hash_, error
            )
    return assets


# --- per-format writers ---------------------------------------------------------------------


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def _write_json(
    output_path: str, eval_record: Any, shareable_url: str | None, include_media: bool = False
) -> None:
    try:
        data = create_output_data(eval_record, shareable_url, include_media)
        text = json_stringify(data, 2)
    except MemoryError as exc:
        count = len(getattr(eval_record, "results", None) or [])
        logger.error("Dataset too large for JSON export (%s results).", count)
        raise RuntimeError(
            f"Dataset too large for JSON export. The evaluation has {count} results which exceeds "
            "memory limits. Consider using JSONL format instead: --output output.jsonl"
        ) from exc
    _write_text(output_path, text)


def _output_to_html_cell(output: dict[str, Any]) -> dict[str, Any]:
    if output.get("pass"):
        status = "pass"
    elif output.get("failureReason") == _ERROR:
        status = "error"
    else:
        status = "fail"
    named_scores = []
    for name, value in (output.get("namedScores") or {}).items():
        entry: dict[str, Any] = {"name": name}
        if value is not None:
            entry["value"] = to_fixed(value, 2)
        named_scores.append(entry)
    cell: dict[str, Any] = {"status": status, "statusLabel": status.upper()}
    if output.get("score") is not None:
        cell["score"] = to_fixed(output["score"], 2)
    token_usage = output.get("tokenUsage") or {}

    def tokens(key: str) -> str:
        value = token_usage.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return to_locale_string(value)
        return ""

    latency = output.get("latencyMs")
    cost = output.get("cost")
    cell.update(
        namedScores=named_scores,
        text=output.get("text"),
        reason=(output.get("gradingResult") or {}).get("reason") or "",
        prompt=output.get("prompt"),
        provider=output.get("provider") or "",
        error=output.get("error") or "",
        failureReason=output.get("failureReason"),
        latencyDisplay=f"{to_fixed(latency, 0)} ms" if _is_finite_number(latency) else "",
        costDisplay=f"${to_fixed(cost, 6)}" if _is_finite_number(cost) and cost > 0 else "",
        totalTokensDisplay=tokens("total"),
        promptTokensDisplay=tokens("prompt"),
        completionTokensDisplay=tokens("completion"),
    )
    return cell


def _render_html(eval_record: Any) -> str:
    from promptphoo.templating import get_engine

    table = get_table(eval_record)
    strip_flags = _strip_flags(eval_record)
    summary = create_output_summary(eval_record, strip_flags)
    config = sanitize_config_for_output(eval_record.config, strip_flags)
    metadata = create_output_metadata(eval_record)
    template = (
        resources.files("promptphoo").joinpath("_data", "tableOutput.html").read_text("utf-8")
    )
    head = table["head"]
    header_row: list[Any] = [
        *head["vars"],
        *(f"[{js_string(p.get('provider'))}] {js_string(p.get('label'))}" for p in head["prompts"]),
    ]
    html_table: list[list[Any]] = [header_row]
    for row_index, row in enumerate(table["body"]):
        cells: list[Any] = [
            {"kind": "variable", "name": head["vars"][i] if i < len(head["vars"]) else None,
             "text": value}
            for i, value in enumerate(row["vars"])
        ]  # fmt: skip
        description = row.get("description") or (row.get("test") or {}).get("description") or ""
        for output_index, output in enumerate(row["outputs"]):
            if output is None:
                cells.append(None)
                continue
            cells.append(
                {
                    "kind": "output",
                    "detailId": f"result-detail-{row_index}-{output_index}",
                    "detailTitle": (
                        f"Result detail - row {row_index + 1}, prompt {output_index + 1}"
                    ),
                    "description": description,
                    **_output_to_html_cell(output),
                }
            )
        html_table.append(cells)
    outputs = [o for row in table["body"] for o in row["outputs"] if o is not None]
    total = len(outputs)
    successes = sum(1 for o in outputs if o.get("pass"))
    errors = sum(1 for o in outputs if not o.get("pass") and o.get("failureReason") == _ERROR)
    failures = total - successes - errors
    pass_rate = (successes / total) * 100 if total > 0 else 0
    return get_engine().render_string(
        template,
        {
            "config": config,
            "table": html_table,
            "results": summary,
            "metadata": metadata,
            "report": {
                "totalResults": total,
                "totalRows": len(table["body"]),
                "promptCount": len(head["prompts"]),
                "variableCount": len(head["vars"]),
                "successes": successes,
                "failures": failures,
                "errors": errors,
                "passRateDisplay": f"{to_fixed(pass_rate, 1)}%",
            },
        },
    )


def _sanitize_for_xml(obj: Any) -> Any:
    if obj is None or obj is UNDEFINED:
        return ""
    if isinstance(obj, list | tuple):
        return [_sanitize_for_xml(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_for_xml(v) for k, v in obj.items()}
    return js_string(obj)


def _render_xml(eval_record: Any, shareable_url: str | None) -> str:
    strip_flags = _strip_flags(eval_record)
    summary = create_output_summary(eval_record, strip_flags)
    config = sanitize_config_for_output(eval_record.config, strip_flags)
    builder = XMLBuilder(ignore_attributes=False, format=True, indent_by="  ")
    return builder.build(
        {
            "promptfoo": {
                "evalId": eval_record.id,
                "results": _sanitize_for_xml(summary),
                "config": _sanitize_for_xml(config),
                "shareableUrl": shareable_url or "",
            }
        }
    )


# --- JSONL ----------------------------------------------------------------------------------


def _jsonl_text(eval_record: Any) -> str:
    flags = _strip_flags(eval_record)
    chunks = []
    for batch in fetch_results_batched(eval_record):
        lines = [
            json_stringify(
                _rs().sanitize_result_for_jsonl_artifact(r.to_evaluate_result(flags), flags)
            )
            for r in batch
        ]
        if lines:
            chunks.append(os.linesep.join(lines) + os.linesep)
    return "".join(chunks)


def _resolve_jsonl_output_path(output_path: str) -> str:
    if not os.path.islink(output_path):
        return output_path
    target = os.readlink(output_path)
    resolved = os.path.realpath(output_path)
    if os.path.exists(resolved):
        return resolved
    return os.path.abspath(os.path.join(os.path.dirname(output_path), target))


def _write_jsonl(output_path: str, eval_record: Any) -> None:
    jsonl_path = _resolve_jsonl_output_path(output_path)
    if jsonl_path != output_path:
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    try:
        mode: int | None = os.stat(jsonl_path).st_mode & 0o7777
    except FileNotFoundError:
        mode = None
    text = _jsonl_text(eval_record)
    temp_path = os.path.join(os.path.dirname(jsonl_path) or ".", f".promptfoo-{uuid.uuid4()}.tmp")
    try:
        try:
            _write_text(temp_path, text)
        except PermissionError:
            logger.warning(
                "[Output] Falling back to JSONL rewrite with external backup because the output "
                "directory is not writable"
            )
            _rewrite_with_external_backup(jsonl_path, mode, text)
            return
        if mode is not None:
            os.chmod(temp_path, mode)
        try:
            os.replace(temp_path, jsonl_path)
        except PermissionError:
            logger.warning(
                "[Output] Falling back to JSONL rewrite with external backup because replacing "
                "the output file is not permitted"
            )
            _rewrite_with_external_backup(jsonl_path, mode, text)
            _remove_quietly(temp_path)
    except BaseException:
        _remove_quietly(temp_path)
        raise


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("[Output] Failed to remove temporary JSONL output: %s (%s)", path, exc)


def _rewrite_with_external_backup(output_path: str, mode: int | None, text: str) -> None:
    temp_dir = tempfile.mkdtemp(prefix="promptfoo-jsonl-")
    backup = os.path.join(temp_dir, "backup.jsonl")
    preserve = False
    has_backup = False
    attempted = False
    try:
        try:
            shutil.copyfile(output_path, backup)
            has_backup = True
        except FileNotFoundError:
            pass
        attempted = True
        _write_text(output_path, text)
        if mode is not None:
            os.chmod(output_path, mode)
    except BaseException as exc:
        if attempted and has_backup:
            try:
                shutil.copyfile(backup, output_path)
            except OSError as restore_error:
                preserve = True
                raise RuntimeError(
                    f"Failed to rewrite JSONL output ({exc}) and restore backup ({restore_error}). "
                    f"Backup retained at {backup}"
                ) from exc
        raise
    finally:
        if not preserve:
            shutil.rmtree(temp_dir, ignore_errors=True)


# --- public API -----------------------------------------------------------------------------


def write_output(
    output_path: str, eval_record: Any, share_url: str | None = None, include_media: bool = False
) -> None:
    """writeOutput(outputPath, evalRecord, shareableUrl, {includeMedia}). Only the json and
    yaml/yml/txt formats embed media, as upstream."""
    if output_path.startswith("https://docs.google.com/spreadsheets/"):
        raise RuntimeError(
            "Writing results to Google Sheets is not supported by promptphoo; "
            "write a .csv file instead."
        )
    fmt = get_output_file_format(output_path)
    if not fmt:
        raise ValueError(
            f"Invariant failed: Unsupported output file format {_extname(output_path)}. "
            f"Please use one of: {', '.join(SUPPORTED_OUTPUT_FILE_FORMATS)}."
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if fmt == "junit.xml":
        write_junit_xml_output(output_path, eval_record)
    elif fmt == "csv":
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            stream_eval_csv(
                eval_record,
                f.write,
                is_redteam=is_truthy((eval_record.config or {}).get("redteam")),
            )
    elif fmt == "json":
        _write_json(output_path, eval_record, share_url, include_media)
    elif fmt in ("yaml", "yml", "txt"):
        data = create_output_data(eval_record, share_url, include_media)
        _write_text(output_path, jsyaml.dump(data))
    elif fmt == "html":
        _write_text(output_path, _render_html(eval_record))
    elif fmt == "jsonl":
        _write_jsonl(output_path, eval_record)
    elif fmt == "xml":
        _write_text(output_path, _render_xml(eval_record, share_url))


def write_multiple_outputs(
    paths: Sequence[str], eval_record: Any, share_url: str | None = None
) -> None:
    """writeMultipleOutputs: attempt every path, then raise once if any failed."""
    errors: list[BaseException] = []
    for path in paths:
        try:
            write_output(path, eval_record, share_url)
        except Exception as exc:  # collected like Promise.allSettled
            errors.append(exc)
    if errors:
        raise OutputWriteError("One or more output writes failed", errors)
