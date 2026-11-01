"""Load-time assertion checks. Ports src/assertions/validateAssertions.ts.

Upstream parses every assertion with zod's AssertionOrSetSchema and prints `z.prettifyError`;
promptphoo validates against the same schema in the vendored `_data/config-schema.json` and
renders the issues through `config/zod_errors.py`, with the approximations noted there.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from promptphoo.jsvalues import is_truthy, json_stringify

# Maximum number of assertions per test case to prevent DoS
MAX_ASSERTIONS_PER_TEST = 10000

_HINT = (
    "Hint: In YAML, ensure all assertion properties are under the same list item:\n"
    "  assert:\n"
    "    - type: python\n"
    "      value: file://script.py   # No '-' before 'value'"
)


class AssertValidationError(Exception):
    pass


@cache
def _assertion_validator() -> Any:
    """AssertionOrSetSchema: the item schema of a test's `assert` array."""
    import jsonschema

    from promptphoo.config.load import _config_schema

    root = _config_schema()
    schema = {**root["definitions"]["__schema23"]["items"], "definitions": root["definitions"]}
    return jsonschema.Draft7Validator(schema)


def _typeof(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "object"


def parse_assertion(assertion: Any, context: str) -> Any:
    received = f"Received: {json_stringify(assertion, 2)}"
    # A JS array is an object, so a list falls through to the missing-'type' message.
    if not isinstance(assertion, dict | list):
        got = "null" if assertion is None else _typeof(assertion)
        raise AssertValidationError(
            f"Invalid assertion at {context}:\nExpected an object, but got {got}\n\n{received}"
        )
    if not isinstance(assertion, dict) or "type" not in assertion:
        raise AssertValidationError(
            f"Invalid assertion at {context}:\nMissing required 'type' property\n\n"
            f"{received}\n\n{_HINT}"
        )

    from promptphoo.config.zod_errors import collect_issues, prettify_issues

    validator = _assertion_validator()
    errors = list(validator.iter_errors(assertion))
    if errors:
        issues = prettify_issues(collect_issues(errors, validator.schema))
        raise AssertValidationError(f"Invalid assertion at {context}:\n{issues}\n\n{received}")

    if assertion.get("type") == "assert-set":
        nested = assertion.get("assert")
        if not isinstance(nested, list):  # `!assert || !Array.isArray(assert)`; [] is truthy
            raise AssertValidationError(
                f"Invalid assertion at {context}:\n"
                f"assert-set must have an 'assert' property that is an array\n\n{received}"
            )
        for i, child in enumerate(nested):
            parse_assertion(child, f"{context}.assert[{i}]")
    return assertion


def validate_assertions(tests: Any, default_test: dict[str, Any] | None = None) -> None:
    if default_test and is_truthy(default_test.get("assert")):
        asserts = default_test["assert"]
        if not isinstance(asserts, list):
            raise AssertValidationError("defaultTest.assert must be an array")
        if len(asserts) > MAX_ASSERTIONS_PER_TEST:
            raise AssertValidationError(
                f"defaultTest.assert has {len(asserts)} assertions, exceeding maximum of "
                f"{MAX_ASSERTIONS_PER_TEST}"
            )
        for i, assertion in enumerate(asserts):
            parse_assertion(assertion, f"defaultTest.assert[{i}]")

    if not isinstance(tests, list):
        raise AssertValidationError("tests must be an array")

    for test_idx, test in enumerate(tests):
        asserts = test.get("assert") if isinstance(test, dict) else None
        if is_truthy(asserts):
            if not isinstance(asserts, list):
                raise AssertValidationError(f"tests[{test_idx}].assert must be an array")
            if len(asserts) > MAX_ASSERTIONS_PER_TEST:
                raise AssertValidationError(
                    f"tests[{test_idx}].assert has {len(asserts)} assertions, exceeding "
                    f"maximum of {MAX_ASSERTIONS_PER_TEST}"
                )
            for i, assertion in enumerate(asserts):
                parse_assertion(assertion, f"tests[{test_idx}].assert[{i}]")
