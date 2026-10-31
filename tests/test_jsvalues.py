"""JS value semantics: JSON.stringify number formatting."""

import pytest

from promptphoo.jsvalues import json_stringify, number_to_string


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1e-7, 1.5e-10, 2.5e-7, -1e-7], "[1e-7,1.5e-10,2.5e-7,-1e-7]"),
        ([0.000001, 0.1, 1.0, 1e21], "[0.000001,0.1,1,1e+21]"),
        ({"a": 1e-7, "s": "\x00jsnum0\x00"}, '{"a":1e-7,"s":"\\u0000jsnum0\\u0000"}'),
    ],
)
def test_json_stringify_uses_js_number_formatting(value, expected):
    # Expected strings are Node's JSON.stringify output.
    assert json_stringify(value) == expected


def test_json_stringify_indented_small_float():
    assert json_stringify({"a": 1e-7}, 2) == '{\n  "a": 1e-7\n}'


def test_small_decimals_keep_all_digits():
    # Node: String(v) and JSON.stringify([v]) for each value.
    values = [1.4112000000000001e-05, 1.4112e-05, -2.5e-05, 1e-06, 9.99e-07, 0.00012345]
    expected = "0.000014112000000000001 0.000014112 -0.000025 0.000001 9.99e-7 0.00012345"
    assert " ".join(number_to_string(v) for v in values) == expected
    assert json_stringify(values) == "[" + expected.replace(" ", ",") + "]"
