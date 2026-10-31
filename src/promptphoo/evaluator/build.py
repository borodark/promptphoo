"""Test expansion before evaluation. Ports buildTestsFromSuite / generateVarCombinations and the
scenario merge helpers from src/evaluator.ts."""

from __future__ import annotations

import glob as _glob
import os
from typing import Any

from promptphoo import cli_state
from promptphoo.files import expand_braces

TestCase = dict[str, Any]


def _glob_relative(pattern: str, root: str) -> list[str]:
    """node-glob with `cwd`: relative patterns yield paths relative to `root`."""
    seen: dict[str, None] = {}
    for p in expand_braces(pattern):
        for match in _glob.glob(p, root_dir=root, recursive=True):
            seen[match] = None
    return sorted(seen)


def generate_var_combinations(vars_: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product over array-valued vars; `file://` globs expand to one value per file.

    Arrays whose first element is not a string are kept whole (lists of objects/numbers).
    """
    combinations: list[dict[str, Any]] = [{}]
    for key, value in vars_.items():
        if isinstance(value, str) and value.startswith("file://"):
            pattern = value[len("file://") :]
            base = cli_state.base_path() or ""
            root = base or os.getcwd()
            values: list[Any] = [f"file://{p}" for p in _glob_relative(pattern, root)]
            if not values:
                raise FileNotFoundError(
                    f"No files found for variable {key} at path {pattern} in directory {root}"
                )
        else:
            values = value if isinstance(value, list) else [value]
        if isinstance(value, list) and (not value or not isinstance(value[0], str)):
            values = [value]
        combinations = [{**combo, key: v} for combo in combinations for v in values]
    return combinations


def get_default_test(test_suite: dict[str, Any]) -> dict[str, Any] | None:
    default = test_suite.get("defaultTest")
    return default if isinstance(default, dict) else None


def merge_scenario_test(
    test_suite: dict[str, Any], data: TestCase, test: TestCase, scenario_index: int
) -> TestCase:
    default = get_default_test(test_suite) or {}
    metadata = {**(default.get("metadata") or {}), **(data.get("metadata") or {}),
                **(test.get("metadata") or {})}  # fmt: skip
    if metadata.get("conversationId") is None:
        metadata["conversationId"] = f"__scenario_{scenario_index}__"
    return {
        **default,
        **data,
        **test,
        "vars": {**(default.get("vars") or {}), **(data.get("vars") or {}),
                 **(test.get("vars") or {})},
        "options": {**(default.get("options") or {}), **(data.get("options") or {}),
                    **(test.get("options") or {})},
        "assert": [*(data.get("assert") or []), *(test.get("assert") or [])],
        "metadata": metadata,
    }  # fmt: skip


def build_tests_from_suite(test_suite: dict[str, Any]) -> list[TestCase]:
    tests = test_suite.get("tests") or []
    scenarios = test_suite.get("scenarios") or []
    if not tests:
        tests = [] if test_suite.get("scenarios") is not None else [{}]
    tests = list(tests)
    scenario_index = 0
    for scenario in scenarios:
        for data in scenario.get("config") or []:
            for test in scenario.get("tests") or [{}]:
                tests.append(merge_scenario_test(test_suite, data, test, scenario_index))
            scenario_index += 1
    return tests
