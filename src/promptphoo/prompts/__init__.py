"""Prompt loading. Ports src/prompts/index.ts and src/prompts/utils.ts."""

from __future__ import annotations

import logging
import os
import stat
from typing import Any

from promptphoo.files import glob_sync, is_javascript_file, node_join, parse_path_or_glob
from promptphoo.jsvalues import json_stringify
from promptphoo.prompts.processors import (
    Prompt,
    process_csv_prompts,
    process_executable_file,
    process_json_file,
    process_jsonl_file,
    process_python_file,
    process_string,
    process_template_file,
    process_txt_file,
    process_yaml_file,
)

logger = logging.getLogger(__name__)

VALID_FILE_EXTENSIONS = (
    ".cjs", ".cts", ".j2", ".js", ".json", ".jsonl", ".md", ".mjs", ".mts", ".py", ".ts", ".txt",
    ".yml", ".yaml",
)  # fmt: skip
_EXECUTABLE_EXTENSIONS = (".sh", ".bash", ".exe", ".bat", ".cmd", ".ps1", ".rb", ".pl")


def maybe_file_path(value: str) -> bool:
    if not isinstance(value, str):
        raise TypeError(f"Invalid input: {json_stringify(value)}")
    if any(s in value for s in ("\n", "portkey://", "langfuse://", "helicone://")):
        return False
    tokens = value.split(":")
    last = tokens[-1]
    second_last = tokens[-2] if len(tokens) > 1 else None
    return (
        value.startswith("file://")
        or any(
            last.endswith(ext) or (second_last is not None and second_last.endswith(ext))
            for ext in VALID_FILE_EXTENSIONS
        )
        or (len(value) >= 3 and value[-3] == ".")
        or (len(value) >= 4 and value[-4] == ".")
        or "*" in value
        or "/" in value
        or "\\" in value
    )


def normalize_input(prompts: Any) -> list[Prompt]:
    if not prompts or (isinstance(prompts, str | list) and len(prompts) == 0):
        raise ValueError(f"Invalid input prompt: {json_stringify(prompts)}")
    if isinstance(prompts, str):
        return [{"raw": prompts}]
    if isinstance(prompts, list):
        out = []
        for p in prompts:
            if isinstance(p, str):
                out.append({"raw": p})
            else:
                out.append({"raw": p.get("raw") or p.get("id"), **p})
        return out
    if isinstance(prompts, dict) and prompts:
        # Legacy {path: label} map format.
        return [{"label": label, "raw": raw} for raw, label in prompts.items()]
    raise ValueError(f"Invalid input prompt: {json_stringify(prompts)}")


async def _process_prompt_file(
    file_path: str, prompt: Prompt, ext: str | None, fn: str | None, label_path: str
) -> list[Prompt]:
    if ext == ".csv":
        return process_csv_prompts(file_path, prompt)
    if ext in (".j2", ".md"):
        return process_template_file(file_path, prompt, label_path)
    if ext == ".json":
        return process_json_file(file_path, prompt, label_path)
    if ext == ".jsonl":
        return process_jsonl_file(file_path, prompt, label_path)
    if ext and is_javascript_file(ext):
        raise ValueError(
            f"JavaScript prompt files are not supported by promptphoo: {file_path}. "
            "Use a Python prompt function (file://prompt.py:fn) instead."
        )
    if ext == ".py":
        return process_python_file(file_path, prompt, fn, label_path)
    if ext == ".txt":
        return process_txt_file(file_path, prompt, label_path)
    if ext in (".yml", ".yaml"):
        return process_yaml_file(file_path, prompt, label_path)
    if ext in _EXECUTABLE_EXTENSIONS:
        return process_executable_file(file_path, prompt, fn, label_path)
    try:
        st = os.stat(file_path)
        if stat.S_ISREG(st.st_mode) and st.st_mode & 0o111:
            return process_executable_file(file_path, prompt, fn, label_path)
    except OSError:
        pass
    return []


async def _process_prompt(
    prompt: Prompt, base_path: str = "", max_depth: int = 1, label_base: str | None = None
) -> list[Prompt]:
    def label_path(file_path: str) -> str:
        if label_base is None:
            return file_path
        return node_join(label_base, os.path.relpath(file_path, base_path or "."))

    raw = prompt.get("raw")
    if not isinstance(raw, str):
        raise TypeError(f"prompt.raw must be a string, but got {json_stringify(raw)}")
    if prompt.get("function"):
        return [prompt]
    if raw.startswith("exec:"):
        parsed = parse_path_or_glob(base_path, raw[5:])
        fp = parsed["file_path"]
        return process_executable_file(fp, prompt, parsed["function_name"], label_path(fp))
    if not maybe_file_path(raw):
        return process_string(prompt)

    parsed = parse_path_or_glob(base_path, raw)
    file_path, fn = parsed["file_path"], parsed["function_name"]
    if parsed["is_path_pattern"] and max_depth > 0:
        prompts: list[Prompt] = []
        for match in glob_sync(file_path.replace("\\", "/")):
            matched = os.path.abspath(match)
            rel = os.path.relpath(matched, base_path or ".").replace("\\", "/")
            expanded_id = f"{prompt['id']}:{rel}" if prompt.get("id") else prompt.get("id")
            label = prompt.get("label")
            if label:
                label = label if os.path.splitext(matched)[1] == ".txt" else f"{label}: {rel}"
            sub = {**prompt, "raw": f"{matched}:{fn}" if fn else matched}
            for key, value in (("id", expanded_id), ("label", label)):
                if value is None:
                    sub.pop(key, None)
                else:
                    sub[key] = value
            processed = await _process_prompt(sub, base_path, max_depth - 1, label_base)
            for i, p in enumerate(processed):
                if expanded_id and not p.get("id"):
                    p = {**p, "id": f"{expanded_id}:{i + 1}" if len(processed) > 1 else expanded_id}
                prompts.append(p)
        if not prompts:
            logger.debug(
                'Attempted to load file at "%s", but no file found. Using raw string.', raw
            )
            prompts.extend(process_string(prompt))
        return prompts

    return await _process_prompt_file(file_path, prompt, parsed["extension"], fn,
                                      label_path(file_path))  # fmt: skip


async def read_prompts(
    prompt_path_or_globs: Any,
    base_path: str = "",
    label_base_path: str | None = None,
    source_base_paths: list[str] | None = None,
) -> list[Prompt]:
    partials = normalize_input(prompt_path_or_globs)
    prompts: list[Prompt] = []
    for index, prompt in enumerate(partials):
        source = source_base_paths[index] if source_base_paths else None
        batch = await _process_prompt(
            prompt,
            source if source is not None else base_path,
            1,
            label_base_path if source is None else os.path.relpath(source, os.getcwd()),
        )
        if not batch:
            raise ValueError(f"There are no prompts in {json_stringify(prompt.get('raw'))}")
        prompts.extend(batch)
    return prompts
