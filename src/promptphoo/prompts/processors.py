"""Per-format prompt file readers. Ports src/prompts/processors/*.ts (JS processor dropped).

A prompt is a dict: {raw, label, config?, id?, function?}. `function` is an async callable
taking {vars, provider, config} and returning a string, chat messages or {prompt, config}.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

from promptphoo import envars, python_hooks
from promptphoo.files import maybe_load_config_from_external_file, parse_csv_records
from promptphoo.jsvalues import json_stringify
from promptphoo.yamlload import load_yaml

logger = logging.getLogger(__name__)

Prompt = dict[str, Any]

_BOM = "﻿"


def defined(**fields: Any) -> dict[str, Any]:
    """Build a dict omitting None values, like JS object literals with undefined members."""
    return {k: v for k, v in fields.items() if v is not None}


def process_string(prompt: Prompt) -> list[Prompt]:
    raw = prompt.get("raw")
    if not isinstance(raw, str):
        raise TypeError(f"prompt.raw must be a string, but got {json_stringify(raw)}")
    label = prompt.get("label")
    return [defined(id=prompt.get("id"), raw=raw, label=raw if label is None else label,
                    config=prompt.get("config"))]  # fmt: skip


def process_txt_file(path: str, prompt: Prompt, label_path: str | None = None) -> list[Prompt]:
    label_path = label_path or path
    content = Path(path).read_text(encoding="utf-8")
    label = prompt.get("label")
    delimiter = envars.get_env_string("PROMPTFOO_PROMPT_SEPARATOR") or "---"
    prompts: list[Prompt] = []
    buffer: list[str] = []

    def flush() -> None:
        raw = "\n".join(buffer).strip()
        if raw:
            text = f"{label}: {label_path}: {raw}" if label else f"{label_path}: {raw}"
            prompts.append(defined(raw=raw, label=text, config=prompt.get("config")))
        buffer.clear()

    for line in re.split(r"\r?\n", content):
        if line.strip() == delimiter:
            flush()
        else:
            buffer.append(line)
    flush()
    return prompts


def process_template_file(path: str, prompt: Prompt, label_path: str | None = None) -> list[Prompt]:
    label_path = label_path or path
    content = Path(path).read_text(encoding="utf-8")
    label = prompt.get("label") or f"{label_path}: {content[:50]}..."
    return [defined(raw=content, label=label, config=prompt.get("config"))]


def process_json_file(path: str, prompt: Prompt, label_path: str | None = None) -> list[Prompt]:
    label_path = label_path or path
    contents = Path(path).read_text(encoding="utf-8").removeprefix(_BOM)
    try:
        contents = json_stringify(maybe_load_config_from_external_file(json.loads(contents)))
    except (ValueError, OSError):
        pass
    label = prompt.get("label") or f"{label_path}: {contents}"
    return [defined(raw=contents, label=label, config=prompt.get("config"))]


def process_jsonl_file(path: str, prompt: Prompt, label_path: str | None = None) -> list[Prompt]:
    label_path = label_path or path
    content = Path(path).read_text(encoding="utf-8").removeprefix(_BOM)
    lines = [line for line in re.split(r"\r?\n", content) if line]
    multiple = len(lines) > 1
    base_label = prompt.get("label")
    out = []
    for line in lines:
        if multiple:
            label = f"{base_label}: {line}" if base_label else f"{label_path}: {line}"
        else:
            label = base_label or label_path
        out.append(defined(raw=line, label=label, config=prompt.get("config")))
    return out


def process_yaml_file(path: str, prompt: Prompt, label_path: str | None = None) -> list[Prompt]:
    label_path = label_path or path
    contents = Path(path).read_text(encoding="utf-8")
    try:
        contents = json_stringify(maybe_load_config_from_external_file(load_yaml(contents)))
    except Exception as e:  # noqa: BLE001 - keep raw text on any parse failure, as Node does
        logger.debug("Error parsing YAML file %s: %s", path, e)
    label = prompt.get("label") or f"{label_path}: {contents[:80]}"
    return [defined(raw=contents, label=label, config=prompt.get("config"))]


# --- CSV ----------------------------------------------------------------------------------------


def _csv_row_label(base: str | None, content: str, index: int, multiple: bool) -> str:
    if base:
        return f"{base}: {content}" if multiple else base
    return f"Prompt {index + 1} - {content}"


def _build_csv_prompts(rows: list[dict[str, Any]], base: Prompt) -> list[Prompt]:
    multiple = len(rows) > 1
    used = {row["label"] for row in rows if row.get("label")}
    out = []
    for index, row in enumerate(rows):
        label = row.get("label") or _csv_row_label(base.get("label"), row["raw"], index, multiple)
        if not row.get("label"):
            initial, suffix = label, 1
            while label in used:
                label = f"{initial} (row {index + 1}{f'.{suffix}' if suffix > 1 else ''})"
                suffix += 1
            used.add(label)
        prompt_id = f"{base['id']}:{index + 1}" if multiple and base.get("id") else base.get("id")
        out.append({**base, "raw": row["raw"], "label": label, **defined(id=prompt_id)})
    return out


def _csv_lines(content: str, base: Prompt) -> list[Prompt]:
    lines = [line for line in re.split(r"\r?\n", content) if line.strip()]
    start = 1 if lines and lines[0].lower().strip() == "prompt" else 0
    return _build_csv_prompts([{"raw": line} for line in lines[start:]], base)


def process_csv_prompts(path: str, base: Prompt) -> list[Prompt]:
    content = Path(path).read_text(encoding="utf-8")
    delimiter = envars.get_env_string("PROMPTFOO_CSV_DELIMITER", ",")
    if delimiter not in content:
        return _csv_lines(content, base)
    try:
        records = parse_csv_records(content.removeprefix(_BOM), delimiter)
        records = [{k.strip(): v.strip() for k, v in r.items()} for r in records]
        rows = [{"raw": r["prompt"], "label": r.get("label")} for r in records if r.get("prompt")]
        return _build_csv_prompts(rows, base)
    except ValueError:
        return _csv_lines(content, base)


# --- Python -----------------------------------------------------------------------------------


def _function_context(context: dict[str, Any], config: Any) -> dict[str, Any]:
    provider = context.get("provider")
    if provider is None:
        raise ValueError("provider.id is required")
    provider_id = provider.id() if callable(getattr(provider, "id", None)) else provider["id"]
    label = (
        getattr(provider, "label", None)
        if not isinstance(provider, dict)
        else provider.get("label")
    )
    return {
        "vars": context.get("vars", {}),
        "provider": defined(id=provider_id, label=label),
        "config": config if config is not None else {},
    }


def python_interpreter() -> str:
    return envars.get_env_string("PROMPTFOO_PYTHON") or sys.executable


def process_python_file(
    path: str, prompt: Prompt, function_name: str | None, label_path: str | None = None
) -> list[Prompt]:
    label_path = label_path or path
    content = Path(path).read_text(encoding="utf-8")
    label = prompt.get("label")
    if label is None:
        label = f"{label_path}:{function_name}" if function_name else f"{label_path}: {content}"
    config = prompt.get("config")

    if function_name:

        async def fn(context: dict[str, Any]) -> Any:
            func = python_hooks.load_function(path, function_name)
            return await python_hooks.call_async(func, _function_context(context, config))
    else:

        async def fn(context: dict[str, Any]) -> Any:
            # Legacy mode: run the whole file with the context JSON as argv[1].
            arg = json_stringify(_function_context(context, config))
            proc = await asyncio.create_subprocess_exec(
                python_interpreter(), path, arg,
                stdout=asyncio.subprocess.PIPE,
                env=envars.get_process_env(),
            )  # fmt: skip
            stdout, _ = await proc.communicate()
            if proc.returncode:
                raise RuntimeError(f"Python prompt script {path} exited with {proc.returncode}")
            return "\n".join(stdout.decode("utf-8").splitlines())

    return [defined(raw=content, label=label, function=fn, config=config)]


# --- executables --------------------------------------------------------------------------------

_SCRIPT_PARTS = re.compile(r"""[^\s"']+|"([^"]*)"|'([^']*)'""")
_CONTROL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def parse_script_parts(script_path: str) -> list[str]:
    return [m.group(1) or m.group(2) or m.group(0) for m in _SCRIPT_PARTS.finditer(script_path)]


def get_file_hashes(parts: list[str]) -> list[str]:
    hashes = []
    for part in parts:
        clean = re.sub(r"^['\"]|['\"]$", "", part)
        if os.path.isfile(clean):
            hashes.append(hashlib.sha256(Path(clean).read_bytes()).hexdigest())
    return hashes


async def executable_prompt_function(script_path: str, context: dict[str, Any]) -> str:
    config = context.get("config") or {}
    ctx = _function_context(context, config)
    parts = parse_script_parts(script_path)
    if not parts:
        raise ValueError("No command found in script path")
    proc = await asyncio.create_subprocess_exec(
        *parts, json_stringify(ctx),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=config.get("basePath"),
        env=envars.get_process_env(),
    )  # fmt: skip
    timeout = (config.get("timeout") or 60000) / 1000
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        raise
    if proc.returncode:
        raise RuntimeError(
            f"Command failed: {script_path}\n{stderr_b.decode('utf-8', 'replace').strip()}"
        )
    stdout = stdout_b.decode("utf-8", "replace").strip()
    stderr = stderr_b.decode("utf-8", "replace").strip()
    if stderr and not stdout:
        raise RuntimeError(stderr)
    return stdout


def process_executable_file(
    path: str, prompt: Prompt, function_name: str | None = None, display_path: str | None = None
) -> list[Prompt]:
    display_path = display_path or path
    raw = display_path
    parts = parse_script_parts(path)
    if parts:
        try:
            st = os.stat(parts[0])
            if stat.S_ISREG(st.st_mode) and st.st_size < 100 * 1024:
                content = Path(parts[0]).read_text(encoding="utf-8")
                if not _CONTROL.search(content[:1000]):
                    raw = content
        except (OSError, UnicodeDecodeError):
            pass
    config = prompt.get("config")

    async def fn(context: dict[str, Any]) -> str:
        return await executable_prompt_function(path, {**context, "config": config})

    label = prompt.get("label")
    return [defined(raw=raw, label=display_path if label is None else label, function=fn,
                    config=config)]  # fmt: skip
