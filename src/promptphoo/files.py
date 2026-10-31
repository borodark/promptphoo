"""file:// reference loading. Ports src/util/file.ts, fileReference.ts and fileExtensions.ts."""

from __future__ import annotations

import csv
import glob as _glob
import io
import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from promptphoo import cli_state, envars, python_hooks
from promptphoo.yamlload import load_yaml

logger = logging.getLogger(__name__)

LoadContext = Literal["assertion", "general", "vars"] | None

JAVASCRIPT_EXTENSIONS = ("js", "cjs", "mjs", "ts", "cts", "mts")
_JS_RE = re.compile(r"\.(" + "|".join(JAVASCRIPT_EXTENSIONS) + r")$", re.I)
IMAGE_EXTENSIONS = {
    "jpg", "jpeg", "png", "gif", "bmp", "webp", "svg", "heic", "heif", "avif", "tif", "tiff",
}  # fmt: skip
VIDEO_EXTENSIONS = {
    "mp4", "mpeg", "mpg", "webm", "ogg", "mov", "avi", "flv", "wmv", "mkv", "m4v", "3gp", "3gpp",
}  # fmt: skip
AUDIO_EXTENSIONS = {"wav", "mp3", "ogg", "aac", "m4a", "flac", "wma", "aif", "aiff", "opus"}


def node_join(*parts: str) -> str:
    """Node path.join: join then normalize (collapses `..`); joining only empties gives '.'."""
    return os.path.normpath(os.path.join(*parts)) if any(parts) else "."


def is_javascript_file(path: str) -> bool:
    return bool(_JS_RE.search(path))


def _ext(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


def is_image_file(path: str) -> bool:
    return _ext(path) in IMAGE_EXTENSIONS


def is_video_file(path: str) -> bool:
    return _ext(path) in VIDEO_EXTENSIONS


def is_audio_file(path: str) -> bool:
    return _ext(path) in AUDIO_EXTENSIONS


def is_script_file(path: str) -> bool:
    return path.endswith(".py") or is_javascript_file(path)


def parse_file_url(file_url: str) -> tuple[str, str | None]:
    """Split file://path[:function]. Only .py / JS paths take a function suffix."""
    if not file_url.startswith("file://"):
        raise ValueError("URL must start with file://")
    rest = file_url[len("file://") :]
    idx = rest.rfind(":")
    if idx > 1:
        candidate = rest[:idx]
        if is_script_file(candidate):
            return os.path.normpath(candidate), rest[idx + 1 :]
    return os.path.normpath(rest) if rest else rest, None


# --- globbing (node-glob compatible enough for config references) ------------------------------

_MAGIC = re.compile(r"[*?\[]|\{[^{}]*,[^{}]*\}|[!@+]\(")


def has_magic(pattern: str) -> bool:
    return bool(_MAGIC.search(pattern))


def expand_braces(pattern: str) -> list[str]:
    m = re.search(r"\{([^{}]*,[^{}]*)\}", pattern)
    if not m:
        return [pattern]
    head, tail = pattern[: m.start()], pattern[m.end() :]
    out: list[str] = []
    for option in m.group(1).split(","):
        out.extend(expand_braces(head + option + tail))
    return out


def glob_sync(pattern: str) -> list[str]:
    """Expand braces and `**`; results are deduplicated and sorted for determinism."""
    seen: dict[str, None] = {}
    for p in expand_braces(pattern):
        for match in _glob.glob(p, recursive=True):
            seen[os.path.abspath(match) if os.path.isabs(p) else match] = None
    return sorted(seen)


def path_exists(path: str | Path) -> bool:
    try:
        os.stat(path)
        return True
    except (FileNotFoundError, NotADirectoryError):
        return False


# --- parsers ----------------------------------------------------------------------------------


def parse_csv_records(contents: str, delimiter: str = ",") -> list[dict[str, str]]:
    """csv-parse `{columns: true}`: header row keys, strict record length, BOM-free."""
    reader = csv.reader(io.StringIO(contents), delimiter=delimiter)
    rows = [row for row in reader if row]
    if not rows:
        return []
    header, *body = rows
    records = []
    for lineno, row in enumerate(body, start=2):
        if len(row) != len(header):
            raise ValueError(
                f"Invalid Record Length: columns length is {len(header)}, "
                f"got {len(row)} on line {lineno}"
            )
        records.append(dict(zip(header, row, strict=True)))
    return records


def _csv_value(contents: str) -> list[Any]:
    records = parse_csv_records(contents)
    if records and len(records[0]) == 1:
        return [next(iter(r.values())) for r in records]
    return records


def _render_file_path(file_path: str) -> str:
    if envars.get_env_bool("PROMPTFOO_DISABLE_TEMPLATING") or "{" not in file_path:
        return file_path
    from promptphoo import templating

    return templating.get_engine().render_string(file_path, {})


def maybe_load_from_external_file(file_path: Any, context: LoadContext = None) -> Any:
    """Load file:// references (json/yaml/csv parsed, others as text); pass other values through.

    'assertion' context keeps .py/JS references as strings for the assertion runner; 'vars'
    context keeps every reference so the evaluator can expand globs and run scripts later.
    """
    if isinstance(file_path, list):
        return [maybe_load_from_external_file(p, context) for p in file_path]
    if not isinstance(file_path, str) or not file_path.startswith("file://"):
        return file_path

    rendered = _render_file_path(file_path)
    clean_path, function_name = parse_file_url(rendered)

    if context == "assertion" and is_script_file(clean_path):
        logger.debug("Preserving script file reference in assertion context: %s", rendered)
        return rendered
    if context == "vars":
        return rendered
    if function_name and is_script_file(clean_path):
        return rendered

    path_to_use = rendered[len("file://") :] if function_name else clean_path
    base = cli_state.base_path() or ""
    resolved = os.path.abspath(os.path.join(base, path_to_use))

    if not os.path.exists(resolved) and has_magic(path_to_use):
        matched = glob_sync(resolved)
        if not matched:
            raise FileNotFoundError(f"No files found matching pattern: {resolved}")
        contents: list[Any] = []
        for match in matched:
            try:
                text = Path(match).read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            if match.endswith(".json"):
                parsed = json.loads(text)
                contents.extend(parsed if isinstance(parsed, list) else [parsed])
            elif match.endswith((".yaml", ".yml")):
                parsed = load_yaml(text)
                if parsed is None:
                    continue
                contents.extend(parsed if isinstance(parsed, list) else [parsed])
            elif match.endswith(".csv"):
                contents.extend(_csv_value(text))
            else:
                contents.append(text)
        return contents

    try:
        text = Path(resolved).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"File does not exist: {resolved}") from None
    except OSError as e:
        raise OSError(f"Failed to read file {resolved}: {e}") from e
    if resolved.endswith(".json"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file {resolved}: {e}") from e
    if resolved.endswith((".yaml", ".yml")):
        try:
            return load_yaml(text)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML file {resolved}: {e}") from e
    if resolved.endswith(".csv"):
        return _csv_value(text)
    return text


def maybe_load_config_from_external_file(
    config: Any, context: LoadContext = None, base_path: str | None = None
) -> Any:
    """Recursively apply maybe_load_from_external_file, switching context for script values/vars."""
    if base_path is not None:
        with cli_state.with_base_path(base_path):
            return maybe_load_config_from_external_file(config, context)
    if isinstance(config, list):
        return [maybe_load_config_from_external_file(item, context) for item in config]
    if isinstance(config, dict):
        result: dict[str, Any] = {}
        for key, value in config.items():
            is_assertion_value = key == "value" and config.get("type") in ("python", "javascript")
            child: LoadContext = (
                "assertion" if is_assertion_value else "vars" if key == "vars" else context
            )
            result[key] = maybe_load_config_from_external_file(value, child)
        return result
    return maybe_load_from_external_file(config, context)


def maybe_load_response_format_from_external_file(
    response_format: Any, vars: Mapping[str, Any] | None = None
) -> Any:
    """Render and load a response_format, including a nested json_schema file reference.
    Ports maybeLoadResponseFormatFromExternalFile from src/util/file.ts."""
    from promptphoo.jsvalues import is_truthy
    from promptphoo.templating import render_vars_in_object

    if response_format is None:
        return response_format
    loaded = maybe_load_from_external_file(render_vars_in_object(response_format, vars))
    if not is_truthy(loaded) or not isinstance(loaded, dict):
        return loaded
    if loaded.get("type") == "json_schema":
        json_schema = loaded.get("json_schema")
        nested = loaded.get("schema") or (
            json_schema.get("schema") if isinstance(json_schema, dict) else None
        )
        if is_truthy(nested):
            loaded_schema = maybe_load_from_external_file(render_vars_in_object(nested, vars))
            if "schema" in loaded:
                return {**loaded, "schema": loaded_schema}
            if isinstance(json_schema, dict) and "schema" in json_schema:
                return {**loaded, "json_schema": {**json_schema, "schema": loaded_schema}}
    return loaded


def parse_path_or_glob(base_path: str, prompt_path: str) -> dict[str, Any]:
    """Return {extension, function_name, is_path_pattern, file_path} for a prompt/test path."""
    prompt_path = prompt_path.removeprefix("file://")
    file_path = os.path.abspath(os.path.join(base_path, prompt_path))
    filename = os.path.relpath(file_path, base_path or ".")
    function_name: str | None = None
    idx = filename.rfind(":")
    if idx > 1:
        without = filename[:idx]
        if is_script_file(without) or without.endswith((".go", ".rb")):
            function_name = filename[idx + 1 :]
            filename = without
    stats_is_dir = False
    try:
        stats_is_dir = os.path.isdir(os.path.join(base_path, filename))
        os.stat(os.path.join(base_path, filename))
    except OSError:
        if envars.get_env_bool("PROMPTFOO_STRICT_FILES"):
            raise
    is_pattern = stats_is_dir or has_magic(prompt_path) or has_magic(file_path.replace("\\", "/"))
    target = filename if os.path.isabs(filename) else os.path.join(base_path, filename)
    safe_filename = os.path.relpath(os.path.abspath(target), base_path or ".")
    return {
        "extension": None if is_pattern else os.path.splitext(safe_filename)[1],
        "file_path": node_join(base_path, safe_filename),
        "function_name": function_name,
        "is_path_pattern": is_pattern,
    }


def load_file_reference(file_ref: str, base_path: str = "") -> Any:
    """Load file:// json/yaml/txt/md, or call a Python file's function (default get_config)."""
    file_path, function_name = parse_file_url(file_ref)
    resolved = os.path.abspath(os.path.join(base_path, file_path))
    ext = os.path.splitext(resolved)[1].lower()
    if ext == ".json":
        return json.loads(Path(resolved).read_text(encoding="utf-8"))
    if ext in (".yaml", ".yml"):
        return load_yaml(Path(resolved).read_text(encoding="utf-8"))
    if ext == ".py":
        fn = python_hooks.load_function(resolved, function_name or "get_config")
        return python_hooks.call_sync(fn)
    if is_javascript_file(resolved):
        raise ValueError(f"JavaScript file references are not supported: {file_ref}")
    if ext in (".txt", ".md", ""):
        return Path(resolved).read_text(encoding="utf-8")
    raise ValueError(f"Unsupported file extension: {ext}")


def process_config_file_references(config: Any, base_path: str = "") -> Any:
    if isinstance(config, str) and config.startswith("file://"):
        return load_file_reference(config, base_path)
    if isinstance(config, list):
        return [process_config_file_references(v, base_path) for v in config]
    if isinstance(config, Mapping):
        return {k: process_config_file_references(v, base_path) for k, v in config.items()}
    return config
