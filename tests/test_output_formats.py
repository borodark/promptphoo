"""Byte-for-byte checks against Node golden output (tools/output_golden.js) for the serializers
behind `-o` (js-yaml dump, fast-xml-parser XMLBuilder, csv-stringify) and the console table
(cli-table3 via src/table.ts). Also ports test/util/outputFormats.test.ts and test/table.test.ts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from promptphoo.output import (
    SUPPORTED_OUTPUT_FILE_FORMATS,
    generate_table,
    get_output_file_format,
    jsyaml,
    table,
    wrap_table,
)
from promptphoo.output.csv_export import csv_stringify, escape_csv_formula
from promptphoo.output.xmlbuilder import XMLBuilder

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = {
    mode: json.loads((FIXTURES / f"output_golden_{mode}.json").read_text("utf-8"))
    for mode in ("plain", "color")
}


@pytest.fixture(autouse=True)
def _fixed_width(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(table, "terminal_max_width", lambda: 120)


def _set_color(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1" if mode == "color" else "0")


# --- outputFormats.test.ts ------------------------------------------------------------------


def test_detects_junit_by_full_suffix() -> None:
    assert get_output_file_format("results.junit.xml") == "junit.xml"
    assert get_output_file_format("reports/ci.RESULTS.JUNIT.XML") == "junit.xml"


def test_keeps_promptfoo_xml_separate_from_junit() -> None:
    assert get_output_file_format("results.xml") == "xml"
    assert get_output_file_format("results.junit") is None


def test_advertises_supported_formats() -> None:
    assert "junit.xml" in SUPPORTED_OUTPUT_FILE_FORMATS
    assert list(SUPPORTED_OUTPUT_FILE_FORMATS) == [
        "csv", "html", "json", "jsonl", "txt", "xml", "yaml", "yml", "junit.xml",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("path", "expected"),
    [("out.JSON", "json"), ("dir.v2/out", None), (".yaml", None), ("a/b.c/out.yml", "yml")],
)
def test_extension_detection(path: str, expected: str | None) -> None:
    assert get_output_file_format(path) == expected


# --- js-yaml / XMLBuilder / csv-stringify goldens --------------------------------------------


@pytest.mark.parametrize("index", range(len(GOLDEN["plain"]["yaml"])))
def test_yaml_dump_matches_js_yaml(index: int) -> None:
    case = GOLDEN["plain"]["yaml"][index]
    value: Any = case["input"]
    if isinstance(value, dict) and "ref1" in value:
        shared = {"a": 1}
        value = {"ref1": shared, "ref2": shared, "list": [shared]}
    assert jsyaml.dump(value) == case["output"]


def test_yaml_dump_uses_v8_key_order_and_drops_undefined() -> None:
    from promptphoo.jsvalues import UNDEFINED

    assert jsyaml.dump({"b": 1, "10": 2, "2": 3, "01": 4, "u": UNDEFINED}) == (
        "'2': 3\n'10': 2\nb: 1\n'01': 4\n"
    )
    assert jsyaml.dump([UNDEFINED, 1.0, -0.0, float("nan")]) == "- null\n- 1\n- -0.0\n- .nan\n"


def test_yaml_dump_rejects_functions() -> None:
    with pytest.raises(jsyaml.YAMLException):
        jsyaml.dump({"f": lambda: 1})


@pytest.mark.parametrize("index", range(len(GOLDEN["plain"]["xml"])))
def test_xml_builder_matches_fast_xml_parser(index: int) -> None:
    case = GOLDEN["plain"]["xml"][index]
    builder = XMLBuilder(ignore_attributes=False, format=True, indent_by="  ")
    assert builder.build(case["input"]) == case["output"]


@pytest.mark.parametrize("index", range(len(GOLDEN["plain"]["csv"])))
def test_csv_stringify_matches_csv_stringify(index: int) -> None:
    case = GOLDEN["plain"]["csv"][index]
    assert csv_stringify(case["input"]) == case["output"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("=SUM(A1)", "'=SUM(A1)"),
        ("@x", "'@x"),
        ("+1", "+1"),
        ("-1.5", "-1.5"),
        ("-x", "'-x"),
        ("+ 1e400", "'+ 1e400"),
        ("  =x", "'  =x"),
        ("​=x", "'​=x"),
        ("-0x1F", "'-0x1F"),
        ("plain", "plain"),
        ("", ""),
    ],
)
def test_escape_csv_formula(value: str, expected: str) -> None:
    assert escape_csv_formula(value) == expected


# --- table.test.ts + cli-table3 goldens -----------------------------------------------------


@pytest.mark.parametrize("mode", ["plain", "color"])
def test_generate_table_matches_cli_table3(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    _set_color(monkeypatch, mode)
    for case in GOLDEN[mode]["table"]:
        assert generate_table(case["input"]) == case["output"]
    small = GOLDEN[mode]["tableSmall"][0]
    assert generate_table(small["input"], 5, 1) == small["output"]


@pytest.mark.parametrize("mode", ["plain", "color"])
def test_wrap_table_matches_cli_table3(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    _set_color(monkeypatch, mode)
    for case in GOLDEN[mode]["wrapTable"]:
        assert wrap_table(case["input"]["rows"], case["input"]["widths"]) == case["output"]


def _table() -> dict[str, Any]:
    return {
        "head": {
            "vars": ["var1", "var2"],
            "prompts": [{"provider": "test-provider", "label": "test-label", "raw": "x"}],
        },
        "body": [
            {"vars": ["value1", "value2"], "outputs": [{"pass": True, "text": "passing test"}]},
            {
                "vars": ["value3", "value4"],
                "outputs": [
                    {"pass": False, "score": 0, "text": "failing test", "failureReason": 1}
                ],
            },
        ],
    }


def test_generate_table_headers_and_cell_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_color(monkeypatch, "plain")
    out = generate_table(_table())
    assert "[test-provider] test-label" in out
    assert "[PASS] passing test" in out
    assert "[FAIL] failing test" in out


def test_generate_table_colors_pass_and_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_color(monkeypatch, "color")
    out = generate_table(_table())
    assert "\x1b[32m[PASS] \x1b[39mpassing test" in out
    assert "\x1b[31m[FAIL] \x1b[39m\x1b[31m\x1b[1mfailing test\x1b[22m\x1b[39m" in out


def test_generate_table_respects_max_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_color(monkeypatch, "plain")
    out = generate_table(_table(), 250, 1)
    assert "passing test" in out
    assert "failing test" not in out


def test_generate_table_respects_cell_max_length(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_color(monkeypatch, "plain")
    long_text = "a" * 300
    out = generate_table(
        {
            "head": {"vars": [long_text], "prompts": []},
            "body": [{"vars": [long_text], "outputs": [{"pass": True, "text": "test"}]}],
        },
        10,
    )
    assert "aaaaaaa..." in out
    assert "a" * 8 not in out


def test_wrap_table_empty() -> None:
    assert wrap_table([]) == "No data to display"


def test_wrap_table_default_widths(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_color(monkeypatch, "plain")
    out = wrap_table([{"col1": "value1", "col2": 2}], {"col1": 10})
    first = out.split("\n")[0]
    assert first == "┌" + "─" * 10 + "┬" + "─" * 60 + "┐"
    assert "│ 2 " in out


def test_ellipsize() -> None:
    assert table.ellipsize("abcdef", 5) == "ab..."
    assert table.ellipsize("abcdef", 2) == "ab"
    assert table.ellipsize("abc", 5) == "abc"
