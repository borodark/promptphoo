"""`eval --watch`: re-run the eval when its config, prompt, provider or test files change.

Ports the watch block of src/node/doEval.ts and `resolveTestsWatchPaths` (with its helpers)
from src/util/testCaseReader.ts. Node watches with chokidar; promptphoo polls each file's
mtime and size, because no file-watching package is a dependency.
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from promptphoo.files import has_magic, is_javascript_file
from promptphoo.templating import render_env_only_in_object
from promptphoo.testcases import is_remote_tests_reference, strip_function_suffix

logger = logging.getLogger("promptphoo")

DEFAULT_POLL_INTERVAL = 0.5
# chokidar's `ignored: /^\./`, tested against each watched path string.
_IGNORED = re.compile(r"^\.")


def print_border() -> None:
    """printBorder from src/util/runtime.ts."""
    from promptphoo.output.table import terminal_max_width

    logger.info("%s", "=" * terminal_max_width())


def _resolve(base: str, ref: str) -> str:
    return os.path.abspath(os.path.join(base, ref))


def _strip_sheet_selector(resolved: str) -> str:
    base = os.path.basename(resolved)
    hash_index = base.find("#")
    if hash_index == -1:
        return resolved
    without_sheet = base[:hash_index]
    ext = os.path.splitext(without_sheet)[1][1:].lower()
    if ext not in ("xlsx", "xls"):
        return resolved
    return os.path.join(os.path.dirname(resolved), without_sheet)


def _is_script(path: str) -> bool:
    return is_javascript_file(path) or path.endswith(".py")


def _resolve_tests_file_reference(reference: str, base_path: str) -> list[str]:
    reference = render_env_only_in_object(reference)
    without_scheme = re.sub(r"^file://", "", reference)
    if is_remote_tests_reference(without_scheme):
        return []
    resolved = _resolve(base_path, without_scheme)
    if has_magic(without_scheme):
        base = os.path.abspath(base_path)
        pattern = os.path.join(_glob.escape(base), os.path.relpath(resolved, base))
        from promptphoo.files import glob_sync

        matches = glob_sync(pattern)
        if matches:
            return [_strip_sheet_selector(m) for m in matches]
    without_sheet = _strip_sheet_selector(resolved)
    without_function = strip_function_suffix(without_sheet)
    return [without_function if _is_script(without_function) else without_sheet]


def _resolve_vars_file_references(value: Any, base_path: str) -> Any:
    from promptphoo.testcases import _resolve_vars_file_references as resolve

    return resolve(value, base_path)


def _collect_config_file_references(value: Any, base_path: str, seen: set[int]) -> list[str]:
    if isinstance(value, str):
        return (
            _resolve_tests_file_reference(value, base_path) if value.startswith("file://") else []
        )
    if not isinstance(value, dict | list):
        return []
    if id(value) in seen:
        return []
    seen.add(id(value))
    children = value if isinstance(value, list) else list(value.values())
    return [p for item in children for p in _collect_config_file_references(item, base_path, seen)]


def _collect_nested_file_references(tests_file: str, base_path: str, vars_base: str) -> list[str]:
    ext = os.path.splitext(tests_file)[1][1:].lower()
    if ext not in ("yaml", "yml", "json", "jsonl"):
        return []
    try:
        from promptphoo.yamlload import load_yaml

        with open(tests_file, encoding="utf-8") as fh:
            raw = fh.read()
        if ext == "json":
            parsed = json.loads(raw)
        elif ext == "jsonl":
            parsed = [json.loads(line) for line in raw.split("\n") if line.strip()]
        else:
            parsed = load_yaml(raw)
        rows = parsed if isinstance(parsed, list) else [parsed]
        vars_files = [
            p
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("vars"), str | list)
            for p in resolve_tests_watch_paths([{"vars": row["vars"]}], vars_base)
        ]
        scoped = []
        for row in rows:
            provider = row.get("provider") if isinstance(row, dict) else None
            if isinstance(provider, str):
                pid: Any = provider
            else:
                pid = provider.get("id") if isinstance(provider, dict) else None
            if not isinstance(pid, str) or not pid.startswith("file://"):
                scoped.append(row)
                continue
            script = strip_function_suffix(render_env_only_in_object(pid)[len("file://") :])
            if not _is_script(script):
                scoped.append(row)
                continue
            resolved_id = _resolve_vars_file_references(pid, vars_base)
            new_provider = (
                {**provider, "id": resolved_id} if isinstance(provider, dict) else resolved_id
            )
            scoped.append({**row, "provider": new_provider})
        return [*_collect_config_file_references(scoped, base_path, set()), *vars_files]
    except Exception:
        return []


def resolve_tests_watch_paths(tests: Any, base_path: str = "") -> list[str]:
    """The files a `tests` config value reads (files, globs, generator configs, vars files)."""
    if tests is None:
        return []
    entries = tests if isinstance(tests, list) else [tests]
    paths: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            source = re.sub(r"^file://", "", render_env_only_in_object(entry))
            use_source_dir = (
                isinstance(tests, list)
                or source.endswith("yaml")
                or source.endswith("yml")
                or (has_magic(source) and not os.path.exists(_resolve(base_path, source)))
            )
            for file in _resolve_tests_file_reference(entry, base_path):
                paths.append(file)
                paths.extend(
                    _collect_nested_file_references(
                        file, base_path, os.path.dirname(file) if use_source_dir else base_path
                    )
                )
            continue
        if not entry or not isinstance(entry, dict):
            continue
        if isinstance(entry.get("path"), str):
            paths.extend(_resolve_tests_file_reference(entry["path"], base_path))
            paths.extend(_collect_config_file_references(entry.get("config"), base_path, set()))
            continue
        vars_ = entry.get("vars")
        if "vars" in entry and vars_:
            if isinstance(vars_, str | list):
                refs = vars_ if isinstance(vars_, list) else [vars_]
                for value in refs:
                    if isinstance(value, str):
                        paths.extend(_resolve_tests_file_reference(value, base_path))
            elif isinstance(vars_, dict):
                for value in vars_.values():
                    if isinstance(value, str) and value.startswith("file://"):
                        paths.extend(_resolve_tests_file_reference(value, base_path))
    return list(dict.fromkeys(paths))


def get_watch_paths(
    cmd: dict[str, Any],
    config: dict[str, Any],
    config_paths: list[str],
    test_sources: list[dict[str, Any]] | None,
) -> list[str]:
    """The watch list doEval builds after the first run."""
    base_path = config.get("basePath")
    if base_path is None:
        base_path = os.path.dirname(config_paths[0])
    prompt_paths: list[str] = []
    if isinstance(config.get("prompts"), list):
        for p in config["prompts"]:
            if isinstance(p, str) and p.startswith("file://"):
                prompt_paths.append(_resolve(base_path, p[len("file://") :]))
            elif (
                isinstance(p, dict)
                and isinstance(p.get("id"), str)
                and p["id"].startswith("file://")
            ):
                prompt_paths.append(_resolve(base_path, p["id"][len("file://") :]))
    provider_paths: list[str] = []
    if isinstance(config.get("providers"), list):
        provider_paths = [
            _resolve(base_path, p[len("file://") :])
            for p in config["providers"]
            if isinstance(p, str) and p.startswith("file://")
        ]
    cli_tests = cmd.get("tests") or cmd.get("vars")
    var_paths: list[str] = []
    if cli_tests:
        # --tests resolves against the CWD, its --vars alias against the config directory.
        var_paths.extend(
            resolve_tests_watch_paths(cli_tests, os.getcwd() if cmd.get("tests") else base_path)
        )
    else:
        var_paths.extend(resolve_tests_watch_paths(config.get("tests"), base_path))
        for source in test_sources or []:
            var_paths.extend(
                resolve_tests_watch_paths(source.get("tests"), source.get("base_path") or "")
            )
    return list(dict.fromkeys([*config_paths, *prompt_paths, *provider_paths, *var_paths]))


def _stamp(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def poll_for_changes(
    paths: Iterable[str],
    *,
    interval: float = DEFAULT_POLL_INTERVAL,
    should_stop: Callable[[], bool] = lambda: False,
) -> Iterator[str]:
    """Yield each path whose mtime or size changes. A file replaced by an atomic save counts
    as changed once it exists again (chokidar's `atomic: true`); a deleted file does not."""
    watched = [p for p in paths if not _IGNORED.search(p)]
    stamps = {p: _stamp(p) for p in watched}
    while not should_stop():
        time.sleep(interval)
        for path in watched:
            stamp = _stamp(path)
            if stamp is not None and stamp != stamps[path]:
                stamps[path] = stamp
                yield path
            elif stamp is None:
                stamps[path] = None


def watch_and_rerun(
    paths: list[str],
    rerun: Callable[[], None],
    *,
    interval: float = DEFAULT_POLL_INTERVAL,
    should_stop: Callable[[], bool] = lambda: False,
) -> None:
    """Log the watched files, then call `rerun` after each change until Ctrl-C."""
    for path in paths:
        logger.info("Watching for file changes on %s ...", path)
    try:
        for path in poll_for_changes(paths, interval=interval, should_stop=should_stop):
            print_border()
            logger.info("File change detected: %s", path)
            print_border()
            rerun()
    except KeyboardInterrupt:
        logger.info("Stopping watch mode...")
