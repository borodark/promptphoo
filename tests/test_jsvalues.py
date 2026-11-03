"""JS value semantics: JSON.stringify number formatting, key order, indent; toFixed."""

import hashlib
import math

import pytest

from promptphoo.fetch.errors import JsError, error_to_string
from promptphoo.jsonutil import js_json_parse, json_loads
from promptphoo.jsvalues import (
    js_string,
    json_stringify,
    number_to_string,
    to_fixed,
    to_precision,
)
from promptphoo.yamlload import load_yaml


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


# --- V8 semantics the Node golden (tests/test_js_golden.py) exposed ---------------------------
# Every expected string below is Node 22's answer, worked out by hand from V8's rules.


def test_stringify_orders_array_index_keys_first():
    # V8 enumerates canonical array indices ("0".."4294967294") first, ascending, then the
    # other keys in insertion order. "4294967295", "01" and "-1" are not indices.
    assert json_stringify({"b": 1, "1": 2, "0": 3, "a": 4}) == '{"0":3,"1":2,"b":1,"a":4}'
    assert (
        json_stringify({"4294967295": 1, "01": 2, "-1": 3, "7": 4})
        == '{"7":4,"4294967295":1,"01":2,"-1":3}'
    )
    assert (
        json_stringify({"x": {"10": 1, "9": 2}}, 2)
        == '{\n  "x": {\n    "9": 2,\n    "10": 1\n  }\n}'
    )


def test_parse_builds_objects_in_v8_order():
    parsed = js_json_parse('{"b":1,"2":2,"1":3,"a":4,"1":5}')
    assert list(parsed.items()) == [("1", 5), ("2", 2), ("b", 1), ("a", 4)]
    assert json_stringify(parsed) == '{"1":5,"2":2,"b":1,"a":4}'
    assert list(load_yaml("b: 1\n2: x\n1: y\na: 4\n")) == ["1", "2", "b", "a"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12345678901234567890", "12345678901234567000"),
        ("9007199254740993", "9007199254740992"),
        ("-9007199254740993", "-9007199254740992"),
        ("123456789012345678901234567890", "1.2345678901234568e+29"),
    ],
)
def test_integers_past_2_53_are_doubles(text, expected):
    assert json_stringify(js_json_parse(text)) == expected
    assert js_string(js_json_parse(text)) == expected
    assert json_stringify(json_loads(text)) == expected
    assert json_stringify(load_yaml(f"n: {text}")["n"]) == expected
    # A Python int computed in code is a JS number too.
    assert json_stringify([int(text)]) == f"[{expected}]"
    assert number_to_string(int(text)) == expected


def test_parsed_big_integer_compares_as_the_double():
    assert js_json_parse("9007199254740993") == js_json_parse("9007199254740992")


def test_stringify_escapes_lone_surrogates():
    assert json_stringify("\ud800") == '"\\ud800"'
    assert json_stringify(["a\udc00b", "\udc00\ud800"]) == '["a\\udc00b","\\udc00\\ud800"]'
    assert json_stringify({"\ud83d": 1}) == '{"\\ud83d":1}'
    # A pair is one character to JS, even when Python holds it as two code points.
    assert json_stringify("😀") == '"😀"'


@pytest.mark.parametrize(
    ("indent", "expected"),
    [
        (11, "[\n" + " " * 10 + "1\n]"),
        (10.9, "[\n" + " " * 10 + "1\n]"),
        ("abcdefghijkl", "[\nabcdefghij1\n]"),
        (-1, "[1]"),
        (0.5, "[1]"),
        ("", "[1]"),
        (True, "[1]"),
    ],
)
def test_stringify_clamps_the_indent(indent, expected):
    assert json_stringify([1], indent) == expected


def test_to_fixed_and_to_precision_print_every_digit():
    assert to_fixed(0.1, 30) == "0.100000000000000005551115123126"
    assert to_fixed(1, 100) == "1." + "0" * 100
    assert to_precision(0.1, 30) == "0.100000000000000005551115123126"
    assert to_precision(1e-7, 100).startswith(
        "9.99999999999999954748111825886258685613938723690807819366455078125"
    )
    # An int past 2**53 is the double it rounds to: (999999999999999900000).toFixed(0).
    assert to_fixed(999999999999999900000, 0) == "999999999999999868928"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: to_fixed(1.5, -1), "toFixed() digits argument must be between 0 and 100"),
        (lambda: to_fixed(1.5, 101), "toFixed() digits argument must be between 0 and 100"),
        (lambda: to_fixed(math.nan, 101), "toFixed() digits argument must be between 0 and 100"),
        (lambda: to_precision(1.5, 0), "toPrecision() argument must be between 1 and 100"),
        (lambda: to_precision(1.5, 101), "toPrecision() argument must be between 1 and 100"),
    ],
)
def test_out_of_range_digits_raise_range_error(call, message):
    with pytest.raises(JsError) as info:
        call()
    assert error_to_string(info.value) == f"RangeError: {message}"


def test_to_precision_of_non_finite_ignores_the_range():
    # toPrecision returns String(x) for NaN/Infinity before it checks the precision.
    assert to_precision(math.nan, 0) == "NaN"
    assert to_precision(-math.inf, 101) == "-Infinity"


_V8_CONFIG = """
prompts: ["{{q}}"]
tests:
  - vars:
      b: 1
      2: two
      1: one
      seed: 12345678901234567890
"""


def test_dataset_id_of_integer_keys_and_big_integer_matches_node():
    from promptphoo.db.queries import _dataset_id

    config = load_yaml(_V8_CONFIG)
    node_tests = '[{"vars":{"1":"one","2":"two","b":1,"seed":12345678901234567000}}]'
    assert json_stringify(config["tests"]) == node_tests
    assert _dataset_id(config) == hashlib.sha256(node_tests.encode()).hexdigest()


def test_prompt_id_of_a_parsed_object_matches_node():
    from promptphoo.evaluator.run import generate_id_from_prompt

    raw = js_json_parse('{"role":"user","10":"x","9":"y","n":9007199254740993}')
    node_raw = '{"9":"y","10":"x","role":"user","n":9007199254740992}'
    assert generate_id_from_prompt({"raw": raw}) == hashlib.sha256(node_raw.encode()).hexdigest()


def test_jsonl_dataset_rows_parse_like_json_parse():
    from promptphoo.testcases import _parse_jsonl_lines

    [row] = _parse_jsonl_lines('{"vars":{"b":1,"0":"zero","id":18446744073709551616}}', "t.jsonl")
    assert json_stringify(row["vars"]) == '{"0":"zero","b":1,"id":18446744073709552000}'
