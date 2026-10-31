"""Prompt and test filters for `eval`. Ports src/util/eval/filterPrompts.ts,
src/util/eval/filterTests.ts and src/util/filterRange.ts.

Option dicts use the camelCase FilterOptions keys; None means "not given".
"""

from __future__ import annotations

import logging
import math
import random as _random
import re
from collections.abc import Callable
from typing import Any

from promptphoo.jsregex import compile_js_regex
from promptphoo.jsvalues import js_string

logger = logging.getLogger(__name__)


def _imul(a: int, b: int) -> int:
    """JS Math.imul: 32-bit signed multiply."""
    r = (a * b) & 0xFFFFFFFF
    return r - 0x100000000 if r & 0x80000000 else r


def _i32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def _fnv1a(text: str) -> int:
    state = 2166136261
    data = text.encode("utf-16-le")  # charCodeAt iterates UTF-16 code units
    for k in range(0, len(data), 2):
        state = _imul(state ^ int.from_bytes(data[k : k + 2], "little"), 16777619)
    return state


def derive_scenario_sample_seed(seed: int, scenario_index: int) -> int:
    return _fnv1a(f"{js_string(seed)}:{scenario_index}") & 0xFFFFFFFF


def create_seeded_random(seed: int) -> Callable[[], float]:
    """mulberry32 seeded from FNV-1a of String(seed), bit-for-bit with Node."""
    state = _fnv1a(js_string(seed))

    def next_value() -> float:
        nonlocal state
        state = _i32(state + 0x6D2B79F5)
        v = state & 0xFFFFFFFF
        v = _imul(_i32(v ^ (v >> 15)), _i32(v | 1)) & 0xFFFFFFFF
        v = _i32(v ^ (v + _imul(_i32(v ^ (v >> 7)), _i32(v | 61)))) & 0xFFFFFFFF
        return ((v ^ (v >> 14)) & 0xFFFFFFFF) / 4294967296

    return next_value


def js_parse_int(value: Any) -> float:
    """Number.parseInt: leading integer of the string, NaN if none."""
    if isinstance(value, bool):
        return math.nan
    if isinstance(value, int | float):
        return math.trunc(value) if math.isfinite(value) else math.nan
    m = re.match(r"\s*([+-]?\d+)", str(value))
    return int(m.group(1)) if m else math.nan


def filter_prompts(prompts: list[dict[str, Any]], option: str | None) -> list[dict[str, Any]]:
    if not option:
        return prompts
    try:
        rx = compile_js_regex(option)
    except ValueError as e:
        raise ValueError(f'Invalid regex pattern for --filter-prompts: "{option}". {e}') from e
    return [
        p for p in prompts
        if (p.get("id") and rx.search(p["id"])) or (p.get("label") and rx.search(p["label"]))
    ]  # fmt: skip


# --- range --------------------------------------------------------------------------------------


def parse_filter_range(option: str) -> tuple[int, int | None]:
    m = re.fullmatch(r"(\d*)\s*:\s*(\d*)", option.strip())
    if not m or (m.group(1) == "" and m.group(2) == ""):
        raise ValueError(
            "--filter-range must be specified in start:end format using zero-based indices, "
            f"got: {option}"
        )
    for raw in m.groups():
        if raw and int(raw) > 2**53 - 1:
            raise ValueError(f"--filter-range bounds must be safe integers, got: {option}")
    start = int(m.group(1)) if m.group(1) else 0
    end = int(m.group(2)) if m.group(2) else None
    if end is not None and start > end:
        raise ValueError(f"--filter-range start must be less than or equal to end, got: {option}")
    return start, end


def filter_by_range(items: list[Any], option: str | None) -> list[Any]:
    if option is None:
        return items
    start, end = parse_filter_range(option)
    sliced = items[start:end]
    if items and not sliced:
        logger.warning(
            "--filter-range %s selected 0 tests (had %d before slicing). Check your range bounds.",
            option,
            len(items),
        )
    return sliced


# --- tests --------------------------------------------------------------------------------------


def _split_metadata_value(value: str) -> list[str]:
    alternatives = [""]
    i, n = 0, len(value)
    while i < n:
        end = i
        while end < n and value[end] == "\\":
            end += 1
        slashes = end - i
        if end < n and value[end] == ",":
            alternatives[-1] += "\\" * (slashes // 2)
            if slashes % 2:
                alternatives[-1] += ","
            else:
                alternatives.append("")
        else:
            alternatives[-1] += value[i : end + 1]
        i = end + 1
    return alternatives


def _metadata_filter(filters: str | list[str]) -> Callable[[dict[str, Any]], bool]:
    parsed: list[tuple[str, list[str]]] = []
    for f in filters if isinstance(filters, list) else [filters]:
        key, _, value = f.partition("=")
        if not key or value == "":
            raise ValueError("--filter-metadata must be specified in key=value format")
        values = _split_metadata_value(value)
        if "" in values:
            raise ValueError(f'--filter-metadata has an empty value in "{f}"')
        parsed.append((key, values))

    def matches(test: dict[str, Any]) -> bool:
        metadata = test.get("metadata")
        if not metadata:
            return False
        for key, values in parsed:
            actual = metadata.get(key)
            if isinstance(actual, list):
                ok = any(v in js_string(a) for v in values for a in actual)
            else:
                ok = actual is not None and any(v in js_string(actual) for v in values)
            if not ok:
                return False
        return True

    return matches


# --- tests from earlier results (src/util/comparison.ts, src/util/eval/filterTestsUtil.ts) -----

EXPLICIT_RUNTIME_VAR_KEYS = ("sessionId",)


def is_runtime_var(key: str) -> bool:
    return key.startswith("_") or key in EXPLICIT_RUNTIME_VAR_KEYS


def filter_runtime_vars(vars_: Any) -> Any:
    """Drop the variables an eval adds at run time (`_conversation`, `sessionId`, ...)."""
    if not isinstance(vars_, dict):
        return vars_
    return {k: v for k, v in vars_.items() if not is_runtime_var(k)}


def extract_runtime_vars(vars_: Any) -> dict[str, Any] | None:
    if not isinstance(vars_, dict):
        return None
    extracted = {k: v for k, v in vars_.items() if is_runtime_var(k)}
    return extracted or None


def vars_match(a: Any, b: Any) -> bool:
    from promptphoo.jsonutil import js_deep_equal

    return js_deep_equal(a, b)


def get_test_case_deduplication_key(test: dict[str, Any]) -> str:
    from promptphoo.jsvalues import json_stringify

    strategy_id = (test.get("metadata") or {}).get("strategyId") or "none"
    return json_stringify(
        {"vars": filter_runtime_vars(test.get("vars")), "strategyId": strategy_id}
    )


def deduplicate_test_cases(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for test in tests:
        key = get_test_case_deduplication_key(test)
        if key not in seen:
            seen.add(key)
            out.append(test)
    return out


def provider_to_identifier(provider: Any) -> str | None:
    """util/provider.ts providerToIdentifier: a label wins, otherwise the canonical id."""
    from promptphoo.providers import canonicalize_provider_id, normalize_provider_ref
    from promptphoo.providers.base import is_api_provider

    if not provider:
        return None
    if isinstance(provider, str):
        return canonicalize_provider_id(provider)
    label = normalize_provider_ref(provider).get("label")
    if label:
        return label
    if is_api_provider(provider):
        return canonicalize_provider_id(provider.id())
    if isinstance(provider, dict) and isinstance(provider.get("id"), str):
        return canonicalize_provider_id(provider["id"])
    return None


def result_is_for_test_case(result: dict[str, Any], test: dict[str, Any]) -> bool:
    test_provider = provider_to_identifier(test.get("provider")) if test.get("provider") else None
    result_provider = provider_to_identifier(result.get("provider"))
    # A test without a provider matches any result; so does a result without one.
    providers_match = not test_provider or not result_provider or test_provider == result_provider
    vars_ok = vars_match(
        filter_runtime_vars(test.get("vars")), filter_runtime_vars(result.get("vars"))
    )
    if not (vars_ok and providers_match):
        logger.debug(
            "[resultIsForTestCase] No match: vars=%s, providers=%s",
            "true" if vars_ok else "false",
            "true" if providers_match else "false",
        )
    return vars_ok and providers_match


def _merge_default_vars(test: dict[str, Any], default_test: Any) -> dict[str, Any]:
    if not default_test or isinstance(default_test, str):
        return test
    return {**test, "vars": {**(default_test.get("vars") or {}), **(test.get("vars") or {})}}


def _is_error(result: dict[str, Any]) -> bool:
    return result.get("failureReason") == 2  # ResultFailureReason.ERROR


def _is_assert_failure(result: dict[str, Any]) -> bool:
    return not result.get("success") and result.get("failureReason") != 2


def _log_no_tests_warning(filter_type: str, path_or_id: str, reason: str) -> None:
    logger.warning(
        '--%s returned no tests. The evaluation "%s" may have %s, or the test suite may have '
        "changed since the evaluation was run.",
        filter_type,
        path_or_id,
        reason,
    )


def read_output(output_path: str) -> dict[str, Any]:
    """util/file.ts readOutput"""
    import json
    import os

    ext = os.path.splitext(output_path)[1][1:]
    if ext == "json":
        with open(output_path, encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported output file format: {ext} currently only supports json")


def _load_results(path_or_id: str) -> list[dict[str, Any]] | None:
    if path_or_id.endswith(".json"):
        return list((read_output(path_or_id).get("results") or {}).get("results") or [])
    from promptphoo.models.eval import Eval

    record = Eval.find_by_id(path_or_id)
    if record is None:
        logger.warning("[filterTestsByResults] Evaluation not found: %s", path_or_id)
        return None
    summary = record.to_evaluate_summary()
    if "results" not in summary or summary["results"] is None:
        logger.debug("[filterTestsByResults] No results in evaluation summary")
        return None
    return list(summary["results"])


async def filter_tests_by_results(
    test_suite: dict[str, Any],
    path_or_id: str,
    filter_fn: Callable[[dict[str, Any]], bool],
    extracted_test_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """filterTestsByResults: the suite's tests whose earlier results match `filter_fn` (by vars
    and provider), plus runtime-generated tests found only in the results."""
    from promptphoo.jsvalues import json_stringify

    tests = test_suite.get("tests")
    if not tests:
        logger.debug("[filterTestsByResults] No tests in test suite")
        return []
    logger.debug("[filterTestsByResults] Loading results from: %s", path_or_id)
    try:
        results = _load_results(path_or_id)
    except Exception as error:
        logger.warning("[filterTestsByResults] Error loading results: %s", error)
        return []
    if results is None:
        return []

    filtered = [r for r in results if filter_fn(r)]
    logger.debug(
        "[filterTestsByResults] Found %d matching results out of %d total",
        len(filtered), len(results),
    )  # fmt: skip
    if not filtered:
        return []
    unique_vars = {json_stringify(filter_runtime_vars(r.get("vars"))) for r in filtered}
    logger.debug(
        "[filterTestsByResults] %d unique test cases (by vars) in filtered results",
        len(unique_vars),
    )

    default_test = test_suite.get("defaultTest")
    has_default_vars = bool(
        default_test and not isinstance(default_test, str) and default_test.get("vars")
    )

    def find_match(candidate: dict[str, Any]) -> dict[str, Any] | None:
        # Prefer results that carry runtime vars, so they are restored when available.
        return next(
            (r for r in filtered
             if result_is_for_test_case(r, candidate)
             and extract_runtime_vars(r.get("vars")) is not None),
            None,
        ) or next((r for r in filtered if result_is_for_test_case(r, candidate)), None)  # fmt: skip

    matched: list[dict[str, Any]] = []
    for test in tests:
        # New results have defaultTest.vars merged in; older ones may not.
        match = find_match(_merge_default_vars(test, default_test))
        if match is None and has_default_vars:
            match = find_match(test)
        if match is None:
            continue
        runtime_vars = extract_runtime_vars(match.get("vars"))
        if runtime_vars:
            logger.debug("[filterTestsByResults] Restored runtime vars for test")
            matched.append({**test, "vars": {**(test.get("vars") or {}), **runtime_vars}})
        else:
            matched.append(test)

    # Tests in the results that match no config test were generated at run time; keep them,
    # without their provider (it may hold stale credentials).
    matched_with_defaults = [_merge_default_vars(t, default_test) for t in matched]
    matched_keys: set[str] = set()
    for result in filtered:
        if any(result_is_for_test_case(result, t) for t in matched_with_defaults):
            matched_keys.add(json_stringify(filter_runtime_vars(result.get("vars"))))
    extracted: list[dict[str, Any]] = []
    extracted_keys: set[str] = set()
    for result in filtered:
        key = json_stringify(filter_runtime_vars(result.get("vars")))
        if key in matched_keys:
            continue
        test_case = result.get("testCase")
        if not test_case:
            logger.debug(
                "[filterTestsByResults] Skipping result without testCase data for extraction"
            )
            continue
        if extracted_test_filter and not extracted_test_filter(test_case):
            continue
        if key in extracted_keys:
            continue
        new_test = {
            "description": test_case.get("description"),
            "vars": filter_runtime_vars(test_case.get("vars")) or {},
            "assert": test_case.get("assert"),
            "metadata": test_case.get("metadata"),
            "options": test_case.get("options"),
        }
        extracted.append({k: v for k, v in new_test.items() if v is not None})
        extracted_keys.add(json_stringify(filter_runtime_vars(new_test["vars"])))
    if extracted:
        logger.info(
            "[filterTestsByResults] Extracted %d runtime-generated test(s) from results",
            len(extracted),
        )
    if not matched and not extracted and filtered:
        logger.warning(
            "[filterTestsByResults] No tests matched %d filtered results. This may indicate a "
            "vars or provider mismatch between stored results and current test suite. Use "
            "LOG_LEVEL=debug for detailed matching info.",
            len(filtered),
        )
    elif len(matched) + len(extracted) < len(unique_vars):
        logger.debug(
            "[filterTestsByResults] Note: %d unique test cases in results did not match any "
            "test in the current test suite and could not be extracted. This may indicate "
            "results without testCase data.",
            len(unique_vars) - len(matched) - len(extracted),
        )
    return deduplicate_test_cases([*matched, *extracted])


async def filter_tests(test_suite: dict[str, Any], options: dict[str, Any]) -> list[Any]:
    tests = list(test_suite.get("tests") or [])
    if not options:
        return tests

    metadata_filter = _metadata_filter(options["metadata"]) if options.get("metadata") else None
    if metadata_filter:
        tests = [t for t in tests if metadata_filter(t)]
    result_suite = {**test_suite, "tests": tests} if metadata_filter else test_suite

    # failing: all non-successful results (failures + errors); failingOnly: assertion failures
    # only; errorsOnly: errors only. failingOnly together with errorsOnly is their union.
    if options.get("failingOnly") and options.get("errorsOnly"):
        logger.debug(
            "Using both --filter-failing-only and --filter-errors-only together "
            "(equivalent to --filter-failing)"
        )
        failing_only = await filter_tests_by_results(
            result_suite, options["failingOnly"], _is_assert_failure, metadata_filter
        )
        errors = await filter_tests_by_results(
            result_suite, options["errorsOnly"], _is_error, metadata_filter
        )
        tests = deduplicate_test_cases([*failing_only, *errors])
        if not tests:
            logger.warning(
                "Combined --filter-failing-only and --filter-errors-only returned no tests. "
                "The specified evaluations may have no failures or errors, or the test suite "
                "may have changed."
            )
    elif options.get("failing"):
        tests = await filter_tests_by_results(
            result_suite, options["failing"], lambda r: not r.get("success"), metadata_filter
        )
        if not tests:
            _log_no_tests_warning("filter-failing", options["failing"], "no failures/errors")
    elif options.get("failingOnly"):
        tests = await filter_tests_by_results(
            result_suite, options["failingOnly"], _is_assert_failure, metadata_filter
        )
        if not tests:
            _log_no_tests_warning(
                "filter-failing-only", options["failingOnly"], "no assertion failures (only errors)"
            )
    elif options.get("errorsOnly"):
        tests = await filter_tests_by_results(
            result_suite, options["errorsOnly"], _is_error, metadata_filter
        )
        if not tests:
            _log_no_tests_warning("filter-errors-only", options["errorsOnly"], "no errors")

    if options.get("pattern"):
        try:
            rx = compile_js_regex(options["pattern"])
        except ValueError as e:
            raise ValueError(f'Invalid regex pattern "{options["pattern"]}": {e}') from e
        tests = [t for t in tests if t.get("description") and rx.search(t["description"])]

    if options.get("range") is not None:
        tests = filter_by_range(tests, options["range"])

    if options.get("firstN") is not None:
        count = js_parse_int(options["firstN"])
        if isinstance(count, float) and math.isnan(count):
            raise ValueError(f"firstN must be a number, got: {options['firstN']}")
        tests = tests[: int(count)]

    if options.get("sample") is not None:
        count = js_parse_int(options["sample"])
        if isinstance(count, float) and math.isnan(count):
            raise ValueError(f"sample must be a number, got: {options['sample']}")
        seed = options.get("sampleSeed")
        rand = _random.random if seed is None else create_seeded_random(seed)
        shuffled = list(tests)
        for i in range(len(shuffled) - 1, 0, -1):
            j = math.floor(rand() * (i + 1))
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        tests = shuffled[: int(count)]

    return tests
