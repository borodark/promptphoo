"""`promptfoo import <file>`. Ports src/commands/import.ts.

Reads a promptfoo eval export (a v3 `results` summary written for a v4 eval, or a legacy
table-backed v2 summary) or an OpenAI Evals output-items JSONL file and writes it to the shared
SQLite DB. Traces in the export are restored into the `traces`/`spans` tables. Embedded blob
assets (`--include-media` exports) are validated exactly as upstream validates them, stored in
the blob store before any `--force` delete, and referenced from the new eval with
`location='import'` (see D49).
"""

from __future__ import annotations

import email.utils
import errno
import hashlib
import logging
import math
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import click

from promptphoo import db
from promptphoo.blobs import (
    BLOB_MAX_SIZE,
    is_safe_inline_blob_mime_type,
    record_blob_reference,
    store_blob,
)
from promptphoo.blobs.blob_refs import BLOB_HASH_REGEX, collect_blob_hashes
from promptphoo.blobs.extractor import buffer_from_base64
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.models import result_sanitize as rs

logger = logging.getLogger("promptphoo")

MAX_EXPORTED_BLOB_BASE64_LENGTH = math.ceil(BLOB_MAX_SIZE / 3) * 4 + 4
# Portable exports are untrusted input; imported blobs must not become active same-origin content.
IMPORTED_BLOB_MIME_TYPE_FALLBACK = "application/octet-stream"

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MAX_DATE_MS = 8.64e15


# --- small JS shims ---------------------------------------------------------------------------


def _get(obj: Any, key: str) -> Any:
    """`obj?.[key]` on parsed JSON: None for anything that is not an object."""
    return obj.get(key) if isinstance(obj, dict) else None


def _js_or(*values: Any) -> Any:
    """`a || b || ...`: the first truthy value, else the last one."""
    for value in values[:-1]:
        if is_truthy(value):
            return value
    return values[-1]


_ISO_RE = re.compile(
    r"^([+-]\d{6}|\d{4})(?:-(\d\d)(?:-(\d\d))?)?"
    r"(?:T(\d\d):(\d\d)(?::(\d\d)(?:\.(\d+))?)?(Z|[+-]\d\d:\d\d)?)?$"
)


def _parse_js_date_string(value: str) -> int | None:
    """Date.parse for the ISO-8601 forms (date-only is UTC, date-time without offset is local)
    and the RFC 2822 form that Date#toUTCString writes. Other strings are unparseable."""
    s = value.strip()
    m = _ISO_RE.match(s)
    if m:
        year, month, day, hh, mm, ss, frac, tz = m.groups()
        try:
            base = datetime(int(year), int(month or 1), int(day or 1))
        except ValueError:
            return None
        if hh is not None:
            if int(hh) > 24 or int(mm) > 59 or int(ss or 0) > 59:
                return None
            if int(hh) == 24 and (int(mm) or int(ss or 0) or int(frac or 0)):
                return None
        ms = int((frac or "0")[:3].ljust(3, "0"))
        offset = timedelta(hours=int(hh or 0), minutes=int(mm or 0), seconds=int(ss or 0))
        naive = base + offset + timedelta(milliseconds=ms)
        if hh is None or tz == "Z":
            aware = naive.replace(tzinfo=UTC)
        elif tz:
            sign = 1 if tz[0] == "+" else -1
            tz_offset = timedelta(hours=int(tz[1:3]), minutes=int(tz[4:6]))
            aware = naive.replace(tzinfo=UTC) - sign * tz_offset
        else:
            aware = naive.astimezone()
        return (aware - _EPOCH) // timedelta(milliseconds=1)
    try:
        parsed = email.utils.parsedate_to_datetime(s)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return (parsed - _EPOCH) // timedelta(milliseconds=1)


def parse_imported_date(value: Any, field: str) -> int | None:
    """parseImportedDate -> epoch ms, warning and returning None for an invalid date."""
    if value is None:
        return None
    ms: float | int | None
    if isinstance(value, bool):
        ms = int(value)
    elif isinstance(value, int | float):
        ms = value if math.isfinite(value) else None
    elif isinstance(value, str):
        ms = _parse_js_date_string(value)
    else:
        ms = None
    if ms is not None and abs(ms) <= _MAX_DATE_MS:
        return int(ms)
    logger.warning(
        "Imported eval has an unparseable %s (%s); falling back.", field, json_stringify(value)
    )
    return None


def _now_ms() -> int:
    return (datetime.now(UTC) - _EPOCH) // timedelta(milliseconds=1)


# --- extraction -------------------------------------------------------------------------------


def extract_eval_id(eval_data: Any) -> Any:
    if eval_data is None:
        raise TypeError("Cannot read properties of null (reading 'evalId')")
    return _js_or(_get(eval_data, "evalId"), _get(eval_data, "id"))


def extract_created_at(eval_data: Any) -> int:
    for value, field in (
        (_get(_get(eval_data, "metadata"), "evaluationCreatedAt"), "metadata.evaluationCreatedAt"),
        (_get(_get(eval_data, "results"), "timestamp"), "results.timestamp"),
        (_get(eval_data, "createdAt"), "createdAt"),
    ):
        parsed = parse_imported_date(value, field)
        if parsed is not None:
            return parsed
    return _now_ms()


def extract_author(eval_data: Any) -> Any:
    author = _get(_get(eval_data, "metadata"), "author")
    if is_truthy(author):
        return author
    return _get(eval_data, "author")


def derive_vars_from_results(results: list[Any]) -> list[str]:
    seen: dict[str, None] = {}
    for result in results:
        vars_ = _get(result, "vars")
        if isinstance(vars_, dict):
            seen.update(dict.fromkeys(vars_))
    return list(seen)


def extract_vars(eval_data: Any) -> list[str]:
    vars_ = _get(eval_data, "vars")
    if isinstance(vars_, list):
        return [v for v in vars_ if isinstance(v, str)]
    results = _get(_get(eval_data, "results"), "results")
    return derive_vars_from_results(results if isinstance(results, list) else [])


def _extract_duration(value: Any) -> Any:
    ok = isinstance(value, int | float) and not isinstance(value, bool)
    return value if ok and math.isfinite(value) and value >= 0 else None


def extract_durations(eval_data: Any) -> dict[str, Any]:
    stats = _get(_get(eval_data, "results"), "stats")
    return {
        "duration_ms": _extract_duration(_get(stats, "durationMs")),
        "generation_duration_ms": _extract_duration(_get(stats, "generationDurationMs")),
        "evaluation_duration_ms": _extract_duration(_get(stats, "evaluationDurationMs")),
    }


def is_importable_v3_results(results: Any) -> bool:
    return (
        isinstance(results, dict)
        and results.get("version") == 3
        and not isinstance(results.get("version"), bool)
        and isinstance(results.get("prompts"), list)
        and isinstance(results.get("results"), list)
    )


def is_importable_legacy_results(results: Any) -> bool:
    if not isinstance(results, dict):
        return False
    version = results.get("version")
    return (
        not (version == 3 and not isinstance(version, bool))
        and isinstance(results.get("results"), list)
        and isinstance(results.get("table"), dict | list)
    )


# --- blob assets ------------------------------------------------------------------------------


def _is_importable_blob_asset(asset: Any) -> bool:
    if not isinstance(asset, dict):
        return False
    size = asset.get("sizeBytes")
    return (
        isinstance(asset.get("hash"), str)
        and bool(BLOB_HASH_REGEX.match(asset["hash"]))
        and isinstance(asset.get("mimeType"), str)
        and len(asset["mimeType"]) > 0
        and isinstance(size, int | float)
        and not isinstance(size, bool)
        and math.isfinite(size)
        and float(size).is_integer()
        and size >= 0
        and isinstance(asset.get("data"), str)
    )


def _sanitize_imported_blob_mime_type(mime_type: str) -> str:
    normalized = mime_type.strip().lower()
    if is_safe_inline_blob_mime_type(normalized):
        return normalized
    return IMPORTED_BLOB_MIME_TYPE_FALLBACK


def _validate_blob_asset_data(asset: dict[str, Any]) -> bytes:
    if asset["sizeBytes"] > BLOB_MAX_SIZE or len(asset["data"]) > MAX_EXPORTED_BLOB_BASE64_LENGTH:
        raise ValueError(
            f"Embedded blob {asset['hash']} exceeds the {BLOB_MAX_SIZE} byte import limit"
        )
    data = buffer_from_base64(asset["data"])
    if len(data) != asset["sizeBytes"]:
        raise ValueError(f"Embedded blob {asset['hash']} size does not match its exported metadata")
    if hashlib.sha256(data).hexdigest() != asset["hash"].lower():
        raise ValueError(f"Embedded blob hash mismatch for {asset['hash']}")
    return data


def prepare_blob_assets(
    blob_assets: Any, referenced: dict[str, None]
) -> list[tuple[dict[str, Any], bytes]]:
    if not isinstance(blob_assets, list):
        return []
    prepared = []
    for asset in blob_assets:
        if not _is_importable_blob_asset(asset):
            logger.warning("Skipping malformed embedded blob during import")
            continue
        if asset["hash"].lower() not in referenced:
            logger.debug("Skipping unreferenced embedded blob during import")
            continue
        prepared.append((asset, _validate_blob_asset_data(asset)))
    return prepared


def import_blob_assets(blob_assets: list[tuple[dict[str, Any], bytes]]) -> None:
    """importBlobAssets: store each validated asset (with a sanitized MIME type) before any
    destructive replace, so a corrupt artifact cannot delete the existing eval."""
    for asset, data in blob_assets:
        result = store_blob(data, _sanitize_imported_blob_mime_type(asset["mimeType"]))
        if result.ref["hash"] != asset["hash"].lower():
            raise ValueError(f"Embedded blob hash mismatch for {asset['hash']}")


def record_imported_blob_references(
    blob_assets: list[tuple[dict[str, Any], bytes]], eval_id: str
) -> None:
    for asset, _data in blob_assets:
        record_blob_reference(asset["hash"].lower(), {"evalId": eval_id, "location": "import"})


# --- traces -----------------------------------------------------------------------------------


def _is_importable_trace(trace: Any) -> bool:
    return (
        isinstance(trace, dict)
        and isinstance(trace.get("traceId"), str)
        and isinstance(trace.get("testCaseId"), str)
        and isinstance(trace.get("spans"), list)
    )


def _is_importable_span(span: Any) -> bool:
    start = _get(span, "startTime")
    return (
        isinstance(span, dict)
        and isinstance(span.get("spanId"), str)
        and isinstance(span.get("name"), str)
        and isinstance(start, int | float)
        and not isinstance(start, bool)
        and math.isfinite(start)
    )


def prepare_traces(traces: Any) -> list[dict[str, Any]]:
    if not isinstance(traces, list):
        return []
    prepared = []
    for trace in traces:
        if not _is_importable_trace(trace):
            logger.warning("Skipping malformed trace during import")
            continue
        spans = []
        for span in trace["spans"]:
            if _is_importable_span(span):
                spans.append(span)
            else:
                logger.warning("Skipping malformed trace span during import")
        prepared.append({**trace, "spans": spans})
    return prepared


def import_traces(
    traces: list[dict[str, Any]], eval_id: str, generate_new_trace_ids: bool
) -> dict[str, str]:
    from promptphoo.tracing.store import get_trace_store

    trace_store = get_trace_store()
    used: set[str] = set()
    imported: dict[str, str] = {}
    for trace in traces:
        # Trace ids are globally unique in the store; duplicates and ids already stored (for
        # another eval) get a fresh id.
        trace_id = trace["traceId"]
        if (
            generate_new_trace_ids
            or trace_id in used
            or trace_store.get_trace_sync(trace_id) is not None
        ):
            trace_id = uuid.uuid4().hex
        used.add(trace_id)
        trace_store.create_trace_sync({
            "traceId": trace_id,
            "evaluationId": eval_id,
            "testCaseId": trace["testCaseId"],
            "metadata": trace.get("metadata"),
        })  # fmt: skip
        if trace["spans"]:
            trace_store.add_spans_sync(trace_id, trace["spans"])
        imported[trace["traceId"]] = trace_id
    return imported


def remap_imported_result_trace_linkage(
    results: list[Any], imported_trace_ids: dict[str, str], eval_id: str
) -> list[Any]:
    out = []
    for result in results:
        trace_id = _get(result, "traceId")
        imported = imported_trace_ids.get(trace_id) if isinstance(trace_id, str) else None
        if imported:
            out.append(
                {
                    **result,
                    "traceId": imported,
                    "evaluationId": eval_id,
                    "metadata": rs.strip_trace_linkage_from_metadata(result.get("metadata")),
                }
            )
            continue
        if not is_truthy(trace_id) and not is_truthy(_get(result, "evaluationId")):
            out.append(result)
            continue
        unlinked = {k: v for k, v in result.items() if k not in ("traceId", "evaluationId")}
        unlinked["metadata"] = rs.strip_trace_linkage_from_metadata(result.get("metadata"))
        out.append(unlinked)
    return out


# --- the command ------------------------------------------------------------------------------


def _prepare_import_artifacts(eval_data: Any, import_v3: bool) -> dict[str, Any]:
    traces = prepare_traces(_get(eval_data, "traces")) if import_v3 else []
    blob_assets_raw = _get(eval_data, "blobAssets")
    blob_assets: list[tuple[dict[str, Any], bytes]] = []
    if import_v3 and isinstance(blob_assets_raw, list) and blob_assets_raw:
        referenced = collect_blob_hashes(
            {"results": eval_data.get("results"), "traces": traces},
            max_depth=64,
            max_string_length=100_000,
        )
        blob_assets = prepare_blob_assets(blob_assets_raw, referenced)
    return {"traces": traces, "blobAssets": blob_assets}


def _create_imported_v3_eval(
    eval_data: dict[str, Any],
    traces: list[dict[str, Any]],
    blob_assets: list[tuple[dict[str, Any], bytes]],
    context: dict[str, Any],
) -> str:
    from promptphoo.db import import_export as ie

    logger.debug("Importing v3 eval")
    results = eval_data["results"]
    record = ie.create_eval_record(
        eval_data.get("config"),  # type: ignore[arg-type]
        results["prompts"],
        id=None if context["newId"] else context["importId"],
        created_at=context["importCreatedAt"],
        author=context["importAuthor"],
        completed_prompts=results["prompts"],
        vars=extract_vars(eval_data),
        runtime_options=eval_data.get("runtimeOptions"),
        **extract_durations(eval_data),
    )
    imported_trace_ids = import_traces(traces, record.id, context["newId"])
    imported_results = remap_imported_result_trace_linkage(
        results["results"], imported_trace_ids, record.id
    )
    ie.create_many_from_evaluate_result(imported_results, record.id)
    record_imported_blob_references(blob_assets, record.id)
    return record.id


def _create_imported_v2_eval(eval_data: dict[str, Any], context: dict[str, Any]) -> str:
    from promptphoo.db import import_export as ie
    from promptphoo.models.eval import create_eval_id

    logger.debug("Importing v2 eval")
    created_at = context["importCreatedAt"]
    if context["newId"]:
        eval_id = create_eval_id(created_at)
    else:
        eval_id = context["importId"] if is_truthy(context["importId"]) else None
        eval_id = eval_id or create_eval_id(created_at)
    config = eval_data.get("config")
    ie.insert_legacy_eval(
        eval_id,
        created_at=created_at,
        author=context["importAuthor"],
        description=_js_or(eval_data.get("description"), _get(config, "description")),
        results=eval_data["results"],
        config=config,
    )
    return eval_id


_NODE_ERRNO_TEXT = {
    errno.ENOENT: ("ENOENT", "no such file or directory", "open"),
    errno.EACCES: ("EACCES", "permission denied", "open"),
    errno.EISDIR: ("EISDIR", "illegal operation on a directory", "read"),
}


class NodeFsError(Exception):
    """An fs error worded like Node's (`ENOENT: no such file or directory, open 'x'`)."""


def _read_file(path: str) -> str:
    """fs.readFile(path, 'utf-8'), with Node's error text for the common failures."""
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="replace")
    except OSError as error:
        known = _NODE_ERRNO_TEXT.get(error.errno or 0)
        if known is None:
            raise
        code, text, syscall = known
        suffix = f", open '{path}'" if syscall == "open" else ", read"
        raise NodeFsError(f"{code}: {text}{suffix}") from error


def _error_message(error: BaseException) -> str:
    return str(error)


def import_eval(file: str, *, new_id: bool = False, force: bool = False) -> int:
    """The `import` action. Returns the process exit code."""
    from promptphoo.db import import_export as ie
    from promptphoo.importers.parse import parse_import_file
    from promptphoo.models.eval import Eval

    removed_existing_eval_id: Any = None
    try:
        file_content = _read_file(file)
        parsed = parse_import_file(file_content)
        eval_data = parsed["evalData"]
        source = parsed["source"]

        import_id = extract_eval_id(eval_data)
        import_created_at = extract_created_at(eval_data)
        import_author = extract_author(eval_data)
        existing_eval = None
        state: dict[str, Any] = {"importV3": False, "artifacts": None}

        def prepare_artifacts() -> dict[str, Any]:
            if state["artifacts"] is None:
                results = _get(eval_data, "results")
                state["importV3"] = is_importable_v3_results(results)
                if not state["importV3"] and not is_importable_legacy_results(results):
                    raise ValueError("Unsupported eval export results format")
                state["artifacts"] = _prepare_import_artifacts(eval_data, state["importV3"])
            return state["artifacts"]

        # Validate replacement artifacts before consulting the database (upstream's preflight).
        if force:
            prepare_artifacts()

        if is_truthy(import_id) and not new_id:
            existing = Eval.find_by_id(import_id) if isinstance(import_id, str) else None
            if existing:
                if force:
                    existing_eval = existing
                else:
                    logger.error(
                        "Eval %s already exists. Use --new-id to import with a new ID, or "
                        "--force to replace it.",
                        import_id,
                    )
                    return 1

        artifacts = prepare_artifacts()
        import_blob_assets(artifacts["blobAssets"])
        if existing_eval is not None:
            logger.info("Replacing existing eval %s", import_id)
            ie.delete_eval(existing_eval.id)
            removed_existing_eval_id = import_id

        context = {
            "newId": bool(new_id),
            "importId": import_id,
            "importCreatedAt": import_created_at,
            "importAuthor": import_author,
        }
        if state["importV3"]:
            eval_id = _create_imported_v3_eval(
                eval_data, artifacts["traces"], artifacts["blobAssets"], context
            )
        else:
            eval_id = _create_imported_v2_eval(eval_data, context)
        db.update_signal_file(eval_id)
        logger.info("Eval with ID %s has been successfully imported.", eval_id)
        logger.debug("import source: %s", source)
        return 0
    except Exception as error:
        logger.error("Failed to import eval: %s", _error_message(error))
        if removed_existing_eval_id:
            ie.notify_evaluations_deleted([removed_existing_eval_id])
            logger.error(
                "The existing eval %s was deleted for a --force replacement that did not "
                "complete. Re-import it from a backup.",
                removed_existing_eval_id,
            )
        return 1


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    del default_config, default_config_path

    @cli.command(
        "import",
        help="Import a Promptfoo eval JSON export or an OpenAI Evals JSONL export",
    )
    @click.argument("file")
    @click.option(
        "--new-id", "new_id", is_flag=True, help="Generate a new eval ID instead of preserving "
        "the original"
    )  # fmt: skip
    @click.option("--force", is_flag=True, help="Replace existing eval with the same ID")
    def import_cmd(file: str, new_id: bool, force: bool) -> None:
        db.run_db_migrations()
        code = import_eval(file, new_id=new_id, force=force)
        if code:
            raise SystemExit(code)
