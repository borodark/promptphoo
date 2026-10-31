"""--filter-failing / --filter-failing-only / --filter-errors-only. Ports the result-based cases
of test/util/eval/filterTestsUtil.test.ts and test/util/eval/filterTests.test.ts, and the
runtime-var helpers of src/util/comparison.ts.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from promptphoo import db
from promptphoo.filters import (
    deduplicate_test_cases,
    extract_runtime_vars,
    filter_runtime_vars,
    filter_tests,
    filter_tests_by_results,
    provider_to_identifier,
    result_is_for_test_case,
)

PROMPT = {"raw": "test prompt", "display": "test prompt", "label": "Test Prompt"}
TESTS = [
    {"vars": {"var1": "test1"}, "assert": []},
    {"vars": {"var1": "test2"}, "assert": []},
    {"vars": {"var1": "test3"}, "assert": []},
]


def _result(test_idx, success, failure_reason, vars_=None, **extra):
    vars_ = vars_ if vars_ is not None else TESTS[test_idx]["vars"]
    return {
        "vars": vars_,
        "success": success,
        "failureReason": failure_reason,
        "provider": {"id": "test-provider", "label": "Test Provider"},
        "prompt": PROMPT,
        "response": {"output": "response"},
        "promptIdx": extra.pop("promptIdx", 0),
        "testIdx": test_idx,
        "testCase": extra.pop("testCase", {"vars": vars_, "assert": []}),
        "score": 1 if success else 0,
        "namedScores": {},
        **extra,
    }


RESULTS = [_result(0, True, 0), _result(1, False, 1), _result(2, False, 2)]


def _write(tmp_path, results, name="results.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"results": {"version": 3, "results": results}}))
    return str(path)


def _run(coro):
    return asyncio.run(coro)


def _suite(tests=None, **extra):
    return {"prompts": [], "providers": [], "tests": TESTS if tests is None else tests, **extra}


# --- comparison.ts helpers --------------------------------------------------------------------


def test_runtime_var_helpers():
    vars_ = {"a": 1, "_conversation": [], "sessionId": "s", "_x": 2}
    assert filter_runtime_vars(vars_) == {"a": 1}
    assert extract_runtime_vars(vars_) == {"_conversation": [], "sessionId": "s", "_x": 2}
    assert extract_runtime_vars({"a": 1}) is None
    assert filter_runtime_vars(None) is None
    assert extract_runtime_vars(["a"]) is None


def test_deduplicate_by_vars_and_strategy():
    tests = [
        {"vars": {"a": 1}},
        {"vars": {"a": 1, "_conversation": []}},
        {"vars": {"a": 1}, "metadata": {"strategyId": "jailbreak"}},
    ]
    assert deduplicate_test_cases(tests) == [tests[0], tests[2]]


def test_provider_to_identifier():
    assert provider_to_identifier("openai:gpt-4") == "openai:gpt-4"
    assert provider_to_identifier({"id": "echo", "label": "Echo"}) == "Echo"
    assert provider_to_identifier({"id": "echo"}) == "echo"
    assert provider_to_identifier(None) is None


def test_result_is_for_test_case_matches_vars_and_provider():
    result = {"vars": {"a": 1, "sessionId": "x"}, "provider": {"id": "echo"}}
    assert result_is_for_test_case(result, {"vars": {"a": 1}})
    assert result_is_for_test_case(result, {"vars": {"a": 1}, "provider": "echo"})
    assert not result_is_for_test_case(result, {"vars": {"a": 1}, "provider": "openai:gpt-4"})
    assert not result_is_for_test_case(result, {"vars": {"a": 2}})
    # A result without provider info still matches a test with a provider.
    assert result_is_for_test_case({"vars": {"a": 1}}, {"vars": {"a": 1}, "provider": "echo"})


# --- filterTestsByResults, from a JSON file ---------------------------------------------------


def test_no_tests_in_suite(tmp_path):
    assert (
        _run(filter_tests_by_results(_suite([]), _write(tmp_path, RESULTS), lambda r: True)) == []
    )


def test_filters_on_success_and_failure_reason(tmp_path):
    path = _write(tmp_path, RESULTS)
    failing = _run(filter_tests_by_results(_suite(), path, lambda r: not r["success"]))
    assert failing == [TESTS[1], TESTS[2]]
    errors = _run(filter_tests_by_results(_suite(), path, lambda r: r["failureReason"] == 2))
    assert errors == [TESTS[2]]


def test_no_matching_results(tmp_path):
    path = _write(tmp_path, [_result(0, True, 0)])
    assert _run(filter_tests_by_results(_suite(), path, lambda r: not r["success"])) == []


def test_unreadable_file_returns_empty(tmp_path, caplog):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with caplog.at_level(logging.WARNING):
        assert _run(filter_tests_by_results(_suite(), str(path), lambda r: True)) == []
    assert "[filterTestsByResults] Error loading results" in caplog.text


def test_extracts_runtime_generated_tests(tmp_path):
    generated = _result(
        5, False, 1, vars_={"var1": "generated", "_conversation": ["x"]},
        testCase={"description": "gen", "vars": {"var1": "generated", "_conversation": ["x"]},
                  "assert": [{"type": "equals", "value": "y"}], "provider": {"id": "secret"}},
    )  # fmt: skip
    path = _write(tmp_path, [RESULTS[1], generated, {**generated, "promptIdx": 1}])
    tests = _run(filter_tests_by_results(_suite(), path, lambda r: not r["success"]))
    assert tests[0] == TESTS[1]
    assert tests[1] == {
        "description": "gen",
        "vars": {"var1": "generated"},
        "assert": [{"type": "equals", "value": "y"}],
    }
    assert len(tests) == 2  # deduplicated; provider is never copied


def test_result_runtime_vars_restored_and_preferred(tmp_path):
    suite = _suite([{"vars": {"prompt": "hello"}, "assert": []}])
    results = [
        _result(0, False, 1, vars_={"prompt": "hello"}),
        _result(
            0, False, 1, promptIdx=1,
            vars_={"prompt": "hello", "_conversation": [{"role": "user", "content": "ctx"}],
                   "sessionId": "session-abc"},
        ),
    ]  # fmt: skip
    tests = _run(filter_tests_by_results(suite, _write(tmp_path, results), lambda r: True))
    assert tests == [
        {
            "vars": {
                "prompt": "hello",
                "_conversation": [{"role": "user", "content": "ctx"}],
                "sessionId": "session-abc",
            },
            "assert": [],
        }
    ]


def test_default_test_vars_merged_then_fallback(tmp_path):
    suite = _suite([{"vars": {"q": "x"}}], defaultTest={"vars": {"lang": "en"}})
    merged = _write(tmp_path, [_result(0, False, 1, vars_={"q": "x", "lang": "en"})], "m.json")
    assert _run(filter_tests_by_results(suite, merged, lambda r: True)) == [{"vars": {"q": "x"}}]
    old = _write(tmp_path, [_result(0, False, 1, vars_={"q": "x"})], "o.json")
    assert _run(filter_tests_by_results(suite, old, lambda r: True)) == [{"vars": {"q": "x"}}]


# --- filterTests --------------------------------------------------------------------------------


def test_filter_tests_failing_variants(tmp_path):
    path = _write(tmp_path, RESULTS)
    suite = _suite()
    assert _run(filter_tests(suite, {"failing": path})) == [TESTS[1], TESTS[2]]
    assert _run(filter_tests(suite, {"failingOnly": path})) == [TESTS[1]]
    assert _run(filter_tests(suite, {"errorsOnly": path})) == [TESTS[2]]
    assert _run(filter_tests(suite, {"failingOnly": path, "errorsOnly": path})) == [
        TESTS[1],
        TESTS[2],
    ]


def test_filter_tests_failing_then_first_n(tmp_path):
    path = _write(tmp_path, RESULTS)
    assert _run(filter_tests(_suite(), {"failing": path, "firstN": 1})) == [TESTS[1]]


def test_filter_failing_warns_when_empty(tmp_path, caplog):
    path = _write(tmp_path, [_result(0, True, 0)])
    with caplog.at_level(logging.WARNING):
        assert _run(filter_tests(_suite(), {"failing": path})) == []
    assert (
        f'--filter-failing returned no tests. The evaluation "{path}" may have no '
        "failures/errors, or the test suite may have changed since the evaluation was run."
    ) in caplog.text


def test_metadata_filter_also_limits_extracted_tests(tmp_path):
    suite = _suite([{"vars": {"var1": "test2"}, "metadata": {"team": "a"}}])
    extracted = _result(
        7, False, 1, vars_={"var1": "other"},
        testCase={"vars": {"var1": "other"}, "metadata": {"team": "b"}},
    )  # fmt: skip
    path = _write(tmp_path, [RESULTS[1], extracted])
    tests = _run(filter_tests(suite, {"metadata": "team=a", "failing": path}))
    assert tests == [{"vars": {"var1": "test2"}, "metadata": {"team": "a"}}]


# --- from the database --------------------------------------------------------------------------


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    db.close_db()
    db.run_db_migrations()
    yield
    db.close_db()


def test_filter_by_eval_id(config_dir):
    from promptphoo.models.eval import Eval

    record = Eval.create({"description": "x"}, [])
    for row in RESULTS:
        _run(record.append_result({**row, "testCase": {"vars": row["vars"], "assert": []}}))
    suite = _suite()
    assert _run(filter_tests(suite, {"failing": record.id})) == [TESTS[1], TESTS[2]]
    assert _run(filter_tests(suite, {"errorsOnly": record.id})) == [TESTS[2]]
    assert _run(filter_tests(suite, {"failingOnly": record.id})) == [TESTS[1]]


def test_unknown_eval_id(config_dir, caplog):
    with caplog.at_level(logging.WARNING):
        assert _run(filter_tests_by_results(_suite(), "eval-nope", lambda r: True)) == []
    assert "[filterTestsByResults] Evaluation not found: eval-nope" in caplog.text
