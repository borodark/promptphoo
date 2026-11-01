"""Load-time assertion checks. Cases from test/assertions/validateAssertions.test.ts and the
malformed-assertion cases of test/util/config/envReload.test.ts."""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from promptphoo.assertions.validate import AssertValidationError, validate_assertions
from promptphoo.config.load import resolve_configs


def _raises(tests, default_test=None, match: str | None = None) -> None:
    with pytest.raises(AssertValidationError, match=match):
        validate_assertions(tests, default_test)


def test_missing_type():
    tests = [{"vars": {}, "assert": [{"value": "file://test.py"}]}]
    _raises(tests, match=re.compile("type", re.I))


def test_unknown_types_pass_through():
    # z.custom<RedteamAssertionTypes>() accepts any type string; unknown types fail at runtime.
    validate_assertions([{"vars": {}, "assert": [{"type": "not-a-real-type", "value": "foo"}]}])


def test_message_has_context_hint_and_received_value():
    tests = [{"vars": {}, "assert": [{"value": "my-specific-value"}]}]
    with pytest.raises(AssertValidationError) as err:
        validate_assertions(tests)
    assert str(err.value) == (
        "Invalid assertion at tests[0].assert[0]:\n"
        "Missing required 'type' property\n\n"
        'Received: {\n  "value": "my-specific-value"\n}\n\n'
        "Hint: In YAML, ensure all assertion properties are under the same list item:\n"
        "  assert:\n"
        "    - type: python\n"
        "      value: file://script.py   # No '-' before 'value'"
    )


@pytest.mark.parametrize(
    ("tests", "where"),
    [
        (
            [
                {"vars": {}, "assert": [{"type": "equals", "value": "valid"}]},
                {"vars": {}, "assert": [{"value": "missing-type"}]},
            ],
            r"tests\[1\]\.assert\[0\]",
        ),
        (
            [{"vars": {}, "assert": [{"type": "equals", "value": "v"}, {"value": "x"}]}],
            r"tests\[0\]\.assert\[1\]",
        ),
    ],
)
def test_context_points_at_the_bad_assertion(tests, where):
    _raises(tests, match=where)


def test_default_test():
    _raises([], {"assert": [{"value": "test"}]}, match=r"defaultTest\.assert\[0\]")
    validate_assertions([], {"assert": [{"type": "equals", "value": "expected"}]})
    _raises(
        [],
        {"assert": [{"type": "equals", "value": "valid"}, {"value": "missing-type"}]},
        match=r"defaultTest\.assert\[1\]",
    )


@pytest.mark.parametrize(
    "assertion",
    [
        {"type": "assert-set"},
        {"type": "assert-set", "assert": {}},
        {"type": "assert-set", "assert": [{"value": "missing-type"}]},
    ],
)
def test_bad_assert_sets(assertion):
    _raises([{"vars": {}, "assert": [assertion]}])


def test_schema_error_is_prettified():
    tests = [{"assert": [{"type": "contains", "value": "hi", "weight": "heavy"}]}]
    with pytest.raises(AssertValidationError) as err:
        validate_assertions(tests)
    first, issue = str(err.value).split("\n")[:2]
    assert first == "Invalid assertion at tests[0].assert[0]:"
    assert issue.startswith("✖ ")


@pytest.mark.parametrize(
    "asserts",
    [
        [
            {
                "type": "assert-set",
                "assert": [
                    {"type": "equals", "value": "expected"},
                    {"type": "contains", "value": "substring"},
                ],
            }
        ],
        [
            {"type": "equals", "value": "expected"},
            {"type": "contains", "value": "substring"},
            {"type": "python", "value": "return True"},
        ],
        [],
        [{"type": "is-json"}],
        [{"type": "similar", "value": "expected", "threshold": 0.8}],
        [{"type": "equals", "value": "expected", "weight": 2}],
        [{"type": "equals", "value": "expected", "transform": "output.toLowerCase()"}],
        [{"type": "not-equals", "value": "unexpected"}],
        [{"type": "select-best"}, {"type": "human"}],
        [{"type": "equals", "value": None}],
    ],
)
def test_valid_assertions(asserts):
    validate_assertions([{"vars": {}, "assert": asserts}])


def test_nothing_to_check():
    validate_assertions([{"vars": {}}])
    validate_assertions([])
    validate_assertions([], None)
    validate_assertions([], {"vars": {}})


@pytest.mark.parametrize(
    ("tests", "default_test", "message"),
    [
        ("not an array", None, "tests must be an array"),
        ({"length": 1000000, "0": {"vars": {}}}, None, "tests must be an array"),
        ([{"vars": {}, "assert": "not an array"}], None, r"tests\[0\]\.assert must be an array"),
        (
            [{"vars": {}, "assert": {"length": 100, "0": {"type": "equals"}}}],
            None,
            r"tests\[0\]\.assert must be an array",
        ),
        ([], {"assert": "not an array"}, r"defaultTest\.assert must be an array"),
        (
            [],
            {"assert": {"length": 100, "0": {"type": "equals"}}},
            r"defaultTest\.assert must be an array",
        ),
    ],
)
def test_input_shape(tests, default_test, message):
    _raises(tests, default_test, match=message)


def test_too_many_assertions():
    asserts = [{"type": "equals", "value": "x"}] * 10001
    _raises([{"assert": asserts}], match=r"tests\[0\]\.assert has 10001 assertions, exceeding")


@pytest.mark.parametrize(
    "assertion",
    [
        None,
        7,
        {"type": "assert-set"},
        {"type": "assert-set", "assert": None},
        {"type": "assert-set", "assert": {}},
    ],
)
def test_resolve_configs_rejects_malformed_assertions(tmp_path, assertion):
    (tmp_path / "tests.json").write_text(json.dumps([{"assert": [assertion]}]))
    config = tmp_path / "promptfooconfig.yaml"
    config.write_text("prompts: ['hi']\nproviders: [echo]\ntests: tests.json\n")
    with pytest.raises(AssertValidationError):
        asyncio.run(resolve_configs({"config": [str(config)]}, {}))
