"""Test-case loading. Ports src/util/testCaseReader.ts.

Supported sources: inline tests, YAML/JSON/JSONL/CSV files and globs, Python generators
(`file://tests.py:generate_tests`), and vars files. Remote sources (Hugging Face, Azure Blob,
SharePoint, Google Sheets) and xlsx files are not ported and raise a clear error.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from promptphoo import cli_state, envars, python_hooks
from promptphoo.csvtests import test_case_from_csv_row
from promptphoo.files import (
    glob_sync,
    has_magic,
    is_javascript_file,
    maybe_load_config_from_external_file,
)
from promptphoo.jsvalues import json_stringify
from promptphoo.refs import dereference
from promptphoo.safe_mode import check_test_data
from promptphoo.templating import render_env_only_in_object
from promptphoo.yamlload import load_yaml

logger = logging.getLogger(__name__)

TestCase = dict[str, Any]
EnvOverrides = Mapping[str, Any] | None
_UNSET: Any = object()


def is_remote_tests_reference(reference: str) -> bool:
    return "://" in reference and not reference.startswith("file://")


def strip_function_suffix(path: str) -> str:
    idx = path.rfind(":")
    return path[:idx] if idx > 1 else path


def _is_remote_test_case(test: Any) -> bool:
    return (
        isinstance(test, dict)
        and (test.get("metadata") or {}).get("__promptfoo", {}).get("remote") is True
    )


def _unsupported_remote(source: str) -> list[TestCase]:
    raise NotImplementedError(
        f"Remote test sources are not supported by promptphoo: {source}. "
        "Download the file and reference it locally."
    )


# --- standalone files -------------------------------------------------------------------------


def _parse_csv_rows(content: str) -> list[dict[str, str]]:
    from promptphoo.files import parse_csv_records

    delimiter = envars.get_env_string("PROMPTFOO_CSV_DELIMITER", ",")
    return parse_csv_records(content.removeprefix("﻿"), delimiter)


def _csv_rows_to_test_cases(rows: list[dict[str, str]]) -> list[TestCase]:
    out = []
    for idx, row in enumerate(rows):
        test = test_case_from_csv_row(row)
        if not test.get("description"):
            test["description"] = f"Row #{idx + 1}"
        out.append(test)
    return out


def _with_default_description(test: TestCase, index: int) -> TestCase:
    return {**test, "description": test.get("description") or f"Row #{index + 1}"}


def _parse_jsonl_lines(content: str, path: str) -> list[TestCase]:
    rows = []
    for i, line in enumerate(content.split("\n")):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSONL test file {path} on line {i + 1}: {e}") from e
    return rows


def _standalone_metadata(vars_path: str, base_path: str) -> tuple[str, str, str | None, str]:
    resolved = os.path.abspath(os.path.join(base_path, re.sub(r"^file://", "", vars_path)))
    colon_count = resolved.count(":")
    if re.match(r"^[A-Za-z]:", resolved):
        colon_count -= 1
    if colon_count > 1:
        raise ValueError(f"Too many colons. Invalid test file script path: {vars_path}")
    without_fn = strip_function_suffix(resolved)
    last = resolved.rfind(":")
    fn = resolved[last + 1 :] if last > 1 else None
    name = os.path.basename(without_fn).split("#")[0]
    sheet_ext = os.path.splitext(name)[1][1:]
    ext = sheet_ext if sheet_ext in ("xlsx", "xls") else os.path.splitext(without_fn)[1][1:]
    return resolved, without_fn, fn, ext


async def read_standalone_tests_file(
    vars_path: str, base_path: str | None = None, config: dict[str, Any] | None = None
) -> list[TestCase]:
    base_path = (cli_state.base_path() or "") if base_path is None else base_path
    vars_path = render_env_only_in_object(vars_path)
    final_config = maybe_load_config_from_external_file(config) if config else config
    if is_remote_tests_reference(vars_path):
        return _unsupported_remote(vars_path)

    resolved, without_fn, fn, ext = _standalone_metadata(vars_path, base_path)
    if is_javascript_file(without_fn):
        raise ValueError(
            f"JavaScript test generators are not supported by promptphoo: {without_fn}. "
            "Use a Python generator (file://tests.py:generate_tests) instead."
        )
    if ext == "py":
        func = python_hooks.load_function(without_fn, fn or "generate_tests")
        args = [] if final_config is None else [final_config]
        result = await python_hooks.call_async(func, *args)
        if not isinstance(result, list):
            kind = type(result).__name__
            raise TypeError(f"Python test function must return a list of test cases, got {kind}")
        return result
    if ext == "csv":
        cases = _csv_rows_to_test_cases(_parse_csv_rows(Path(resolved).read_text(encoding="utf-8")))
        check_test_data(cases, resolved, dataset=True)
        return cases
    if ext in ("xlsx", "xls"):
        raise NotImplementedError(f"Excel test files are not supported by promptphoo: {resolved}")
    if ext == "json":
        content = Path(resolved).read_text(encoding="utf-8")
        try:
            data = load_yaml(content)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON test file {resolved}: {e}") from e
        cases = data if isinstance(data, list) else [data]
        check_test_data(cases, resolved)
        return [_with_default_description(t, i) for i, t in enumerate(cases)]
    if ext == "jsonl":
        content = Path(resolved).read_text(encoding="utf-8")
        cases = _parse_jsonl_lines(content, resolved)
        check_test_data(cases, resolved, dataset=True)
        return [_with_default_description(t, i) for i, t in enumerate(cases)]
    if ext in ("yaml", "yml"):
        raw_rows = load_yaml(Path(resolved).read_text(encoding="utf-8"))
        check_test_data(raw_rows, resolved)
        rows = maybe_load_config_from_external_file(raw_rows)
        cases = _csv_rows_to_test_cases(rows)
        check_test_data(cases, resolved)
        return cases
    return []


# --- single test ------------------------------------------------------------------------------


async def read_test_files(path_or_globs: str | list[str], base_path: str | None = None) -> dict:
    base_path = (cli_state.base_path() or "") if base_path is None else base_path
    if isinstance(path_or_globs, str):
        path_or_globs = [path_or_globs]
    ret: dict[str, Any] = {}
    for pattern in path_or_globs:
        for p in glob_sync(os.path.abspath(os.path.join(base_path, pattern))):
            data = maybe_load_config_from_external_file(load_yaml(Path(p).read_text("utf-8")))
            if isinstance(data, dict):
                ret.update(data)
    return ret


async def _load_test_with_vars(test: TestCase, base_path: str) -> TestCase:
    ret = {**test}
    vars_ = test.get("vars")
    if isinstance(vars_, str | list):
        ret["vars"] = await read_test_files(vars_, base_path)
    return ret


def _resolve_vars_file_references(value: Any, base_path: str) -> Any:
    if isinstance(value, str):
        if not value.startswith("file://"):
            return value
        reference = render_env_only_in_object(value)
        return "file://" + os.path.abspath(os.path.join(base_path, reference[7:]))
    if isinstance(value, list):
        return [_resolve_vars_file_references(v, base_path) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_vars_file_references(v, base_path) for k, v in value.items()}
    return value


def validate_test_case(test: TestCase) -> None:
    description_only = test.get("description") and all(
        k == "description" or v is None for k, v in test.items()
    )
    if (
        not any(
            test.get(k)
            for k in ("assert", "vars", "options", "metadata", "provider", "providerOutput")
        )
        and not description_only
        and not isinstance(test.get("threshold"), int | float)
    ):
        raise ValueError(
            "Test case must contain assert, vars, options, metadata, provider, providerOutput, "
            "threshold, or only a description.\n\nInstead got:\n" + json_stringify(test, 2)
        )


async def _read_test_with_env(
    test: str | TestCase,
    base_path: str,
    is_default_test: bool,
    env: EnvOverrides,
    load_providers: bool = True,
    source_base_path: str | None = None,
) -> TestCase:
    source_base_path = base_path if source_base_path is None else source_base_path
    if isinstance(test, dict) and _is_remote_test_case(test):
        return test
    effective = source_base_path
    if isinstance(test, str):
        test_file = os.path.abspath(os.path.join(source_base_path, test))
        effective = os.path.dirname(test_file)
        raw = maybe_load_config_from_external_file(load_yaml(Path(test_file).read_text("utf-8")))
        test_case = await _load_test_with_vars(raw, effective)
    else:
        test_case = await _load_test_with_vars(test, source_base_path)

    provider = test_case.get("provider")
    provider_id = provider if isinstance(provider, str) else (
        provider.get("id") if isinstance(provider, dict) else None
    )  # fmt: skip
    if effective != base_path and isinstance(provider_id, str) and isinstance(provider, str | dict):
        meta = dict(test_case.get("metadata") or {})
        meta["__promptfoo"] = {
            **(meta.get("__promptfoo") or {}),
            "providerBasePath": os.path.abspath(effective),
        }
        test_case["metadata"] = meta
    pbp = ((test_case.get("metadata") or {}).get("__promptfoo") or {}).get("providerBasePath")
    if isinstance(pbp, str):
        effective = pbp

    if not load_providers:
        if "vars" in test_case:
            test_case["vars"] = _resolve_vars_file_references(test_case["vars"], base_path)
        if isinstance(provider, str) and provider.startswith("file://"):
            test_case["provider"] = _resolve_vars_file_references(provider, effective)
        elif (
            isinstance(provider, dict)
            and isinstance(provider.get("id"), str)
            and provider["id"].startswith("file://")
        ):
            test_case["provider"] = {
                **provider,
                "id": _resolve_vars_file_references(provider["id"], effective),
            }

    if load_providers and isinstance(provider, str | dict) and provider:
        from promptphoo.providers import load_api_provider

        with cli_state.with_env(env):
            if isinstance(provider, str):
                test_case["provider"] = await load_api_provider(provider, base_path=effective)
            elif isinstance(provider.get("id"), str):
                test_case["provider"] = await load_api_provider(
                    provider["id"], options=provider, base_path=effective
                )

    if not is_default_test:
        validate_test_case(test_case)
    return test_case


async def read_test(
    test: str | TestCase,
    base_path: str | None = None,
    is_default_test: bool = False,
    env: EnvOverrides = _UNSET,
    load_providers: bool = True,
) -> TestCase:
    base_path = (cli_state.base_path() or "") if base_path is None else base_path
    env = cli_state.env() if env is _UNSET else env
    with cli_state.with_base_path(base_path), cli_state.with_env(env):
        return await _read_test_with_env(test, base_path, is_default_test, env, load_providers)


async def read_test_config(
    test: str | TestCase,
    base_path: str | None = None,
    is_default_test: bool = False,
    env: EnvOverrides = _UNSET,
) -> TestCase:
    """Read a replayable test row without constructing its provider."""
    return await read_test(test, base_path, is_default_test, env, load_providers=False)


# --- globs and suites -------------------------------------------------------------------------


async def _load_tests_from_glob(
    pattern: str, base_path: str, env: EnvOverrides, load_providers: bool = True
) -> list[TestCase]:
    pattern = render_env_only_in_object(pattern)
    if is_remote_tests_reference(pattern):
        return _unsupported_remote(pattern)
    pattern = pattern.removeprefix("file://")
    resolved = os.path.abspath(os.path.join(base_path, pattern))
    test_files = [resolved] if os.path.exists(resolved) else glob_sync(resolved)
    without_fn = strip_function_suffix(resolved)
    if (is_javascript_file(without_fn) or without_fn.endswith(".py")) and not any(
        f in (resolved, without_fn) for f in test_files
    ):
        test_files.append(resolved)

    ret: list[TestCase] = []
    if not test_files:
        message = f"No test files found for path: {resolved}"
        if not has_magic(os.path.relpath(resolved, os.path.abspath(base_path or "."))):
            raise FileNotFoundError(message)
        logger.warning(message)
        return ret

    for test_file in test_files:
        file_without_fn = strip_function_suffix(test_file)
        if (
            test_file.endswith(".csv")
            or is_javascript_file(file_without_fn)
            or file_without_fn.endswith(".py")
            or test_file.split("#")[0].endswith((".xlsx", ".xls"))
        ):
            cases: Any = await read_standalone_tests_file(test_file, base_path)
        elif test_file.endswith((".yaml", ".yml")):
            raw = load_yaml(Path(test_file).read_text("utf-8"))
            check_test_data(raw, test_file)
            cases = dereference(maybe_load_config_from_external_file(raw))
        elif test_file.endswith(".jsonl"):
            raw_cases = _parse_jsonl_lines(Path(test_file).read_text("utf-8"), test_file)
            check_test_data(raw_cases, test_file, dataset=True)
            cases = dereference(maybe_load_config_from_external_file(raw_cases))
        elif test_file.endswith(".json"):
            try:
                raw = json.loads(Path(test_file).read_text("utf-8"))
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON test file {test_file}: {e}") from e
            check_test_data(raw, test_file)
            cases = dereference(maybe_load_config_from_external_file(raw))
        else:
            raise ValueError(f"Unsupported file type for test file: {test_file}")

        if cases:
            if isinstance(cases, dict):
                cases = [cases]
            for case in cases:
                ret.append(
                    await _read_test_with_env(
                        case, base_path, False, env, load_providers, os.path.dirname(test_file)
                    )
                )
    return ret


async def load_tests_from_glob(
    pattern: str, base_path: str | None = None, env: EnvOverrides = _UNSET
) -> list[TestCase]:
    base_path = (cli_state.base_path() or "") if base_path is None else base_path
    env = cli_state.env() if env is _UNSET else env
    with cli_state.with_base_path(base_path), cli_state.with_env(env):
        return await _load_tests_from_glob(pattern, base_path, env)


async def _read_tests_with_env(
    tests: Any, base_path: str, env: EnvOverrides, load_providers: bool = True
) -> list[TestCase]:
    async def load_standalone(source: str, config: dict | None = None) -> list[TestCase]:
        source = render_env_only_in_object(source)
        cases = await read_standalone_tests_file(source, base_path, config)
        if is_remote_tests_reference(source):
            return cases
        return [await _read_test_with_env(t, base_path, False, env, load_providers) for t in cases]

    if isinstance(tests, str):
        source = render_env_only_in_object(tests)
        if source.endswith(("yaml", "yml")):
            return await _load_tests_from_glob(source, base_path, env, load_providers)
        without_scheme = re.sub(r"^file://", "", source)
        if not has_magic(without_scheme) or os.path.exists(os.path.join(base_path, without_scheme)):
            return await load_standalone(source)
        tests = [tests]
    elif isinstance(tests, dict) and isinstance(tests.get("path"), str):
        return await load_standalone(tests["path"], tests.get("config"))

    if not isinstance(tests, list):
        if tests is not None:
            logger.warning(
                "Warning: Unsupported 'tests' format in promptfooconfig.yaml.\n"
                "Expected: string, string[], or TestCase[], but received: %s",
                type(tests).__name__,
            )
        return []

    ret: list[TestCase] = []
    for entry in tests:
        item = render_env_only_in_object(entry) if isinstance(entry, str) else entry
        if isinstance(item, str):
            without_scheme = re.sub(r"^file://", "", item)
            without_fn = strip_function_suffix(without_scheme)
            if (
                is_remote_tests_reference(item)
                or is_javascript_file(without_fn)
                or without_fn.endswith(".py")
                or item.split("#")[0].endswith((".xlsx", ".xls"))
                or without_fn != without_scheme
            ):
                ret.extend(await load_standalone(item))
            else:
                ret.extend(await _load_tests_from_glob(item, base_path, env, load_providers))
        elif _is_remote_test_case(item):
            ret.append(item)
        elif isinstance(item, dict) and "path" in item:
            ret.extend(await load_standalone(item["path"], item.get("config")))
        else:
            ret.append(await _read_test_with_env(item, base_path, False, env, load_providers))

    if any(isinstance(t.get("vars"), dict) and t["vars"].get("assert") for t in ret) and (
        not envars.get_env_bool("PROMPTFOO_NO_TESTCASE_ASSERT_WARNING")
    ):
        logger.warning(
            "Warning: Found 'assert' key in vars. This is likely a mistake in your configuration."
            "\n\n'assert' should be *unindented* so it is under the test itself, not vars."
        )
    return ret


async def read_tests(
    tests: Any, base_path: str | None = None, env: EnvOverrides = _UNSET
) -> list[TestCase]:
    base_path = (cli_state.base_path() or "") if base_path is None else base_path
    env = cli_state.env() if env is _UNSET else env
    with cli_state.with_base_path(base_path), cli_state.with_env(env):
        return await _read_tests_with_env(tests, base_path, env)


async def read_test_configs(
    tests: Any, base_path: str | None = None, env: EnvOverrides = _UNSET
) -> list[TestCase]:
    """Parse source files once and keep declarative rows for persistence and replay."""
    base_path = (cli_state.base_path() or "") if base_path is None else base_path
    env = cli_state.env() if env is _UNSET else env
    with cli_state.with_base_path(base_path), cli_state.with_env(env):
        return await _read_tests_with_env(tests, base_path, env, load_providers=False)
