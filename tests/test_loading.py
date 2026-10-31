"""Phase 1 loaders: YAML, $ref, CSV assertions, prompts, tests, config combination."""

import json
import os
from pathlib import Path

import pytest

from promptphoo import cli_state
from promptphoo.config.load import prepare_combined_config, read_config
from promptphoo.csvtests import (
    assertion_from_string,
    parse_comma_separated_values,
)
from promptphoo.csvtests import (
    test_case_from_csv_row as csv_row_to_test,
)
from promptphoo.jsvalues import json_stringify
from promptphoo.prompts import maybe_file_path, read_prompts
from promptphoo.refs import RefError, dereference
from promptphoo.testcases import read_test_configs
from promptphoo.yamlload import load_yaml

UPSTREAM = Path(__file__).resolve().parents[2] / "promptfoo"
SMOKE = UPSTREAM / "test" / "smoke" / "fixtures"
needs_upstream = pytest.mark.skipif(not SMOKE.exists(), reason="upstream checkout not found")


@pytest.fixture
def upstream_cwd(monkeypatch):
    monkeypatch.chdir(UPSTREAM)


# --- yaml / json ------------------------------------------------------------------------------


def test_yaml_core_schema():
    data = load_yaml("a: yes\nb: off\nc: 012\nd: 0o12\ne: 0x1F\nf: ~\ng: 1.5\nh: true\n1: x\n")
    assert data == {"a": "yes", "b": "off", "c": 12, "d": 10, "e": 31, "f": None, "g": 1.5,
                    "h": True, "1": "x"}  # fmt: skip


def test_yaml_merge_and_duplicates():
    assert load_yaml("b: &b {x: 1}\nn:\n  <<: *b\n  x: 2\n")["n"] == {"x": 2}
    with pytest.raises(Exception, match="duplicated"):
        load_yaml("a: 1\na: 2\n")
    assert load_yaml("# nothing\n") is None


def test_json_stringify_matches_js():
    assert json_stringify({"a": 1.0, "b": [True, None], "c": "é", "f": len}) == (
        '{"a":1,"b":[true,null],"c":"é"}'
    )
    assert json_stringify({"a": 1}, 2) == '{\n  "a": 1\n}'


def test_dereference(tmp_path):
    (tmp_path / "defs.yaml").write_text("thing:\n  x: 1\n")
    doc = {
        "definitions": {"a": {"v": 2}},
        "use": {"$ref": "#/definitions/a"},
        "ext": {"$ref": "defs.yaml#/thing"},
    }
    out = dereference(doc, str(tmp_path))
    assert out["use"] == {"v": 2} and out["ext"] == {"x": 1}
    with pytest.raises(RefError):
        dereference({"a": {"$ref": "#/a"}})


# --- csv --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "assertion"),
    [
        ("contains: Alice", {"type": "contains", "value": "Alice"}),
        ("not-icontains:x", {"type": "not-icontains", "value": "x"}),
        ("similar:cosine(0.9):hi", {"type": "similar:cosine", "value": "hi", "threshold": 0.9}),
        ("similar(0.7):hi", {"type": "similar", "value": "hi", "threshold": 0.7}),
        ("levenshtein:abc", {"type": "levenshtein", "value": "abc", "threshold": 0.75}),
        ('contains-any:a, "b,c"', {"type": "contains-any", "value": ["a", "b,c"]}),
        ("is-json", {"type": "is-json"}),
        ("grade:be nice", {"type": "llm-rubric", "value": "be nice"}),
        ("file://check.py:fn", {"type": "python", "value": "check.py:fn"}),
        ("plain text", {"type": "equals", "value": "plain text"}),
    ],
)
def test_assertion_from_string(expected, assertion):
    assert assertion_from_string(expected) == assertion


def test_comma_separated_values():
    assert parse_comma_separated_values(' a , "b \\"q\\"", ,c') == ["a", 'b "q"', "c"]


def test_csv_row_special_columns():
    row = {
        "name": "A",
        "__expected1": "contains: A",
        "__expected2": "equals: B",
        "__metric": "m",
        "__threshold": "0",
        "__metadata:tags[]": "x, y\\,z",
        "__config:__expected2:threshold": "0.5",
        "__description": "desc",
    }
    assert csv_row_to_test(row) == {
        "vars": {"name": "A"},
        "assert": [
            {"type": "contains", "value": "A", "metric": "m"},
            {"type": "equals", "value": "B", "metric": "m", "threshold": 0.5},
        ],
        "options": {},
        "description": "desc",
        "threshold": 0,
        "metadata": {"tags": ["x", "y,z"], "threshold": 0.5},
    }


# --- prompts ----------------------------------------------------------------------------------


def test_maybe_file_path():
    assert maybe_file_path("file://x") and maybe_file_path("prompts/*.txt")
    assert maybe_file_path("prompt.py:fn") and not maybe_file_path("Hello {{name}}")
    assert not maybe_file_path("line\nbreak.txt")


@needs_upstream
async def test_prompt_files(upstream_cwd):
    base = "test/smoke/fixtures/configs"
    prompts = await read_prompts(
        ["file://../prompts/greeting.txt", "file://../prompts/chat.json", "inline {{x}}"],
        base,
        base,
    )
    assert [p["label"] for p in prompts] == [
        "test/smoke/fixtures/prompts/greeting.txt: Hello {{name}}, welcome!",
        "test/smoke/fixtures/prompts/chat.json: " + prompts[1]["raw"],
        "inline {{x}}",
    ]
    assert json.loads(prompts[1]["raw"])[1] == {"role": "user", "content": "Hello {{name}}"}


async def test_txt_separator_and_python_prompt(tmp_path):
    (tmp_path / "p.txt").write_text("one\n---\ntwo\n")
    (tmp_path / "p.py").write_text("def make(ctx):\n    return 'hi ' + ctx['vars']['n']\n")
    prompts = await read_prompts(["p.txt", "file://p.py:make"], str(tmp_path))
    assert [p["raw"] for p in prompts[:2]] == ["one", "two"]
    out = await prompts[2]["function"]({"vars": {"n": "bo"}, "provider": {"id": "echo"}})
    assert out == "hi bo"


# --- tests ------------------------------------------------------------------------------------


@needs_upstream
@pytest.mark.parametrize(
    ("ref", "names"),
    [
        ("file://tests.csv", ["Alice", "Bob"]),
        ("file://tests-external.yaml", ["Henry", "Ivy"]),
        ("file://tests.json", ["Charlie", "Diana"]),
        ("file://tests.jsonl", ["Eve", "Frank", "Grace"]),
    ],
)
async def test_test_files(upstream_cwd, ref, names):
    tests = await read_test_configs(ref, "test/smoke/fixtures/configs", None)
    assert [t["vars"]["name"] for t in tests] == names


async def test_python_test_generator(tmp_path):
    (tmp_path / "gen.py").write_text(
        "def generate_tests(config=None):\n"
        "    n = (config or {}).get('n', 1)\n"
        "    return [{'vars': {'i': i}} for i in range(n)]\n"
    )
    tests = await read_test_configs(
        [{"path": "file://gen.py:generate_tests", "config": {"n": 2}}], str(tmp_path), None
    )
    assert [t["vars"] for t in tests] == [{"i": 0}, {"i": 1}]


async def test_vars_file_references_are_preserved(tmp_path):
    (tmp_path / "v.txt").write_text("hello")
    tests = await read_test_configs([{"vars": {"a": "file://v.txt"}}], str(tmp_path), None)
    assert tests[0]["vars"]["a"] == "file://" + os.path.join(str(tmp_path), "v.txt")


# --- config -----------------------------------------------------------------------------------


@needs_upstream
def test_read_config_env_two_pass(upstream_cwd, monkeypatch):
    monkeypatch.setenv("PF_SMOKE_MODEL", "echo")
    path = SMOKE / "configs" / "basic.yaml"
    config = read_config(str(path))
    assert config["providers"] == ["echo"] and config["prompts"] == ["Hello {{name}}"]


def test_combine_configs(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "description: A\nproviders: [echo]\nprompts: ['p1', 'file://x.txt']\n"
        "defaultTest:\n  vars: {a: 1}\nenv: {K: '{{ env.HOME_TEST_PF }}'}\n"
    )
    (tmp_path / "b.yaml").write_text(
        "description: B\nproviders: [echo, other]\nprompts: ['p1', 'p2']\n"
        "defaultTest:\n  vars: {b: 2}\n  assert: [{type: contains, value: x}]\n"
    )
    with cli_state.with_env(None):
        prepared = prepare_combined_config([str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")])
    config = prepared["config"]
    assert config["description"] == "A, B"
    assert config["providers"] == ["echo", "other"]
    assert config["prompts"] == ["p1", "file://" + str(tmp_path / "x.txt"), "p2"]
    assert config["defaultTest"]["vars"] == {"a": 1, "b": 2}
    assert config["defaultTest"]["assert"] == [{"type": "contains", "value": "x"}]
    assert len(prepared["test_sources"]) == 2
