"""Extension hooks. Ports the runExtensionHook / getExtensionHookName cases of
test/evaluatorHelpers.test.ts, plus end-to-end runs of Python hooks through `eval -x`."""

from __future__ import annotations

import json
import logging
import math

import pytest
from click.testing import CliRunner

from promptphoo import db
from promptphoo.cli.main import build_cli
from promptphoo.evaluator.extensions import (
    ExtensionHookError,
    check_extensions_supported,
    get_extension_hook_name,
    run_before_all_extensions,
    run_extension_hook,
    synchronize_legacy_transport_headers,
)


class MockProvider:
    label = None
    prompts = None

    def id(self):
        return "test-provider"

    def to_json(self):
        return {"id": "test-provider"}


@pytest.fixture
def transform_mock(monkeypatch):
    """Replace transform with a recorder whose return values come from `returns`."""
    calls: list[tuple] = []
    returns: list = []

    async def fake(ext, a, b, validate=True):
        calls.append((ext, a, b, validate))
        value = returns.pop(0) if returns else None
        return value(ext, a, b) if callable(value) else value

    monkeypatch.setattr("promptphoo.transform.transform", fake)
    return calls, returns


def _suite():
    return {
        "providers": [MockProvider()],
        "prompts": [{"raw": "Test prompt {{ var1 }} {{ var2 }}", "label": "Test prompt"}],
        "tests": [{"vars": {"var1": "value1", "var2": "value2"}}],
    }


@pytest.mark.parametrize("extensions", [[], None])
async def test_no_extensions(transform_mock, extensions):
    calls, _ = transform_mock
    context = {"suite": _suite()}
    assert await run_extension_hook(extensions, "beforeAll", context) is context
    assert calls == []


async def test_legacy_convention_for_each_extension(transform_mock):
    calls, _ = transform_mock
    context = {"suite": _suite()}
    out = await run_extension_hook(["ext1", "ext2", "ext3"], "beforeAll", context)
    assert [c[0] for c in calls] == ["ext1", "ext2", "ext3"]
    assert all(c[1] == "beforeAll" and c[2] == context and c[3] is False for c in calls)
    assert out == context


async def test_before_all_only_mutable_properties(transform_mock):
    _, returns = transform_mock
    returns.append({"suite": {"foo": "bar", "tests": [{"vars": {"newVar": "newValue"}}]}})
    out = await run_extension_hook(["ext1"], "beforeAll", {"suite": _suite()})
    assert out["suite"]["providers"][0].id() == "test-provider"
    assert out["suite"]["tests"] == [{"vars": {"newVar": "newValue"}}]
    assert "foo" not in out["suite"]
    # Mutable keys the hook did not return become undefined, as in JS.
    assert "prompts" not in out["suite"]


def _fn_suite(prompts):
    return {"suite": {"providers": [MockProvider()], "prompts": prompts, "tests": [{"vars": {}}]}}


async def test_restores_prompt_function_dropped_by_json(transform_mock):
    _, returns = transform_mock

    def fn(context):
        return "x"

    returns.append({"suite": {"prompts": [{"raw": "def create_prompt(): ...", "label": "p:fn"}]}})
    out = await run_extension_hook(
        ["ext1"], "beforeAll", _fn_suite([{"raw": "def create_prompt(): ...", "label": "p:fn",
                                           "function": fn}])
    )  # fmt: skip
    assert out["suite"]["prompts"][0]["function"] is fn


async def test_does_not_restore_rewritten_prompt(transform_mock):
    _, returns = transform_mock
    returns.append({"suite": {"prompts": [{"raw": "rewritten source", "label": "p:fn"}]}})
    out = await run_extension_hook(
        ["ext1"], "beforeAll",
        _fn_suite([{"raw": "original source", "label": "p:fn", "function": lambda c: ""}]),
    )  # fmt: skip
    assert out["suite"]["prompts"][0] == {"raw": "rewritten source", "label": "p:fn"}


async def test_shared_label_is_not_restored_and_warns(transform_mock, caplog):
    _, returns = transform_mock
    returns.append(
        {"suite": {"prompts": [{"raw": "A", "label": "shared-label"},
                               {"raw": "B", "label": "shared-label"}]}}
    )  # fmt: skip
    with caplog.at_level(logging.WARNING, logger="promptphoo"):
        out = await run_extension_hook(
            ["ext1"], "beforeAll",
            _fn_suite([{"raw": "A", "label": "shared-label", "function": lambda c: ""},
                       {"raw": "B", "label": "shared-label", "function": lambda c: ""}]),
        )  # fmt: skip
    assert all("function" not in p for p in out["suite"]["prompts"])
    assert "shared-label" in caplog.text


async def test_keeps_hook_function_and_chains_restoration(transform_mock):
    _, returns = transform_mock

    def original(c):
        return ""

    def hook_fn(c):
        return ""

    returns.append({"suite": {"prompts": [{"raw": "s", "label": "p", "function": hook_fn}]}})
    out = await run_extension_hook(
        ["ext1"], "beforeAll", _fn_suite([{"raw": "s", "label": "p", "function": original}])
    )
    assert out["suite"]["prompts"][0]["function"] is hook_fn

    serialized = {"suite": {"prompts": [{"raw": "s", "label": "p"}]}}
    returns.extend([serialized, serialized])
    out = await run_extension_hook(
        ["ext1", "ext2"], "beforeAll", _fn_suite([{"raw": "s", "label": "p", "function": original}])
    )
    assert out["suite"]["prompts"][0]["function"] is original


async def test_relabelled_function_prompt_warns(transform_mock, caplog):
    _, returns = transform_mock
    returns.append({"suite": {"prompts": [{"raw": "src", "label": "new-label"}]}})
    with caplog.at_level(logging.WARNING, logger="promptphoo"):
        out = await run_extension_hook(
            ["ext1"], "beforeAll",
            _fn_suite([{"raw": "src", "label": "old-label", "function": lambda c: ""}]),
        )  # fmt: skip
    assert "function" not in out["suite"]["prompts"][0]
    assert "old-label" in caplog.text


async def test_malformed_prompt_entries_pass_through(transform_mock):
    _, returns = transform_mock

    def fn(c):
        return ""

    returns.append(
        {"suite": {"prompts": [{"raw": "s", "label": "p"}, None, "just a string",
                               {"raw": "no label here"}]}}
    )  # fmt: skip
    out = await run_extension_hook(
        ["ext1"], "beforeAll", _fn_suite([{"raw": "s", "label": "p", "function": fn}])
    )
    prompts = out["suite"]["prompts"]
    assert prompts[0]["function"] is fn
    assert prompts[1:] == [None, "just a string", {"raw": "no label here"}]


async def test_before_each(transform_mock):
    calls, returns = transform_mock
    context = {"test": {"vars": {"var1": "v"}, "assert": [{"type": "equals", "value": "e"}]}}
    assert await run_extension_hook(["ext1"], "beforeEach", context) == context
    returns.append({"test": {"vars": {"var1": "modified"}}})
    out = await run_extension_hook(["ext1"], "beforeEach", context)
    assert out == {"test": {"vars": {"var1": "modified"}}}
    returns.append({"test": "invalid_test_value"})
    assert (await run_extension_hook(["ext1"], "beforeEach", context))["test"] == (
        "invalid_test_value"
    )
    assert calls[0] == ("ext1", "beforeEach", context, False)


async def test_hook_filtering_by_function_name(transform_mock):
    calls, _ = transform_mock
    context = {"suite": _suite()}
    await run_extension_hook(["file://path/to/hooks.py:afterAll"], "beforeAll", context)
    assert calls == []
    await run_extension_hook(["file://path/to/hooks.py:beforeAll"], "beforeAll", context)
    assert calls[-1] == ("file://path/to/hooks.py:beforeAll", context, {"hookName": "beforeAll"},
                         False)  # fmt: skip
    for hook in ("beforeAll", "beforeEach", "afterEach", "afterAll"):
        await run_extension_hook(["file://hooks.py:myHandler"], hook, {"x": 1})
        assert calls[-1] == ("file://hooks.py:myHandler", hook, {"x": 1}, False)
        await run_extension_hook(["file://hooks.py"], hook, {"x": 1})
        assert calls[-1] == ("file://hooks.py", hook, {"x": 1}, False)


async def test_mixed_extensions(transform_mock):
    calls, _ = transform_mock
    context = {"suite": _suite()}
    await run_extension_hook(
        ["file://hooks.py:beforeAll", "file://hooks.py:myHandler", "file://hooks.py:afterAll"],
        "beforeAll",
        context,
    )
    assert calls == [
        ("file://hooks.py:beforeAll", context, {"hookName": "beforeAll"}, False),
        ("file://hooks.py:myHandler", "beforeAll", context, False),
    ]


async def test_before_all_chains(transform_mock):
    _, returns = transform_mock
    base = [{"vars": {"original": "yes"}}]

    def second(ext, ctx, _):
        assert ctx["suite"]["tests"][1] == {"vars": {"added_by_ext1": "yes"}}
        return {"suite": {**ctx["suite"], "tests": [*ctx["suite"]["tests"], {"vars": {"e2": 1}}]}}

    returns.extend(
        [{"suite": {"prompts": [], "tests": [*base, {"vars": {"added_by_ext1": "yes"}}]}}, second]
    )
    out = await run_extension_hook(
        ["file://ext1.py:beforeAll", "file://ext2.py:beforeAll"],
        "beforeAll",
        {"suite": {**_suite(), "tests": base}},
    )
    assert len(out["suite"]["tests"]) == 3


BASE_RESULT = {
    "vars": {},
    "response": {"output": "test output"},
    "success": True,
    "score": 1,
    "namedScores": {"existing_metric": 0.5},
    "metadata": {"existing_key": "value"},
}


def _after_each():
    return {"test": {}, "result": {**BASE_RESULT}}


async def test_after_each_merges(transform_mock):
    _, returns = transform_mock
    returns.append(
        {
            "result": {
                "namedScores": {"num_turns": 3, "bad": "x", "nan": math.nan, "inf": math.inf},
                "metadata": {"session_url": "https://example.com"},
                "success": False,
                "score": 0,
                "response": {"output": "hacked", "metadata": {"injected": True}},
            }
        }
    )
    out = await run_extension_hook(["file://hooks.py:afterEach"], "afterEach", _after_each())
    result = out["result"]
    assert result["namedScores"] == {"existing_metric": 0.5, "num_turns": 3}
    assert result["metadata"] == {"existing_key": "value", "session_url": "https://example.com"}
    assert result["success"] is True and result["score"] == 1
    assert result["response"] == {"output": "test output", "metadata": {"injected": True}}


async def test_after_each_chains_and_later_wins(transform_mock):
    _, returns = transform_mock

    def second(ext, ctx, _):
        assert ctx["result"]["namedScores"] == {"existing_metric": 0.5, "shared": 1}
        return {"result": {"namedScores": {"shared": 99}, "metadata": {"b": 2}}}

    returns.extend([{"result": {"namedScores": {"shared": 1}, "metadata": {"a": 1}}}, second])
    out = await run_extension_hook(
        ["file://h1.py:afterEach", "file://h2.py:afterEach"], "afterEach", _after_each()
    )
    assert out["result"]["namedScores"] == {"existing_metric": 0.5, "shared": 99}
    assert out["result"]["metadata"] == {"existing_key": "value", "a": 1, "b": 2}


async def test_after_each_none_keeps_result(transform_mock):
    out = await run_extension_hook(["file://hooks.py:afterEach"], "afterEach", _after_each())
    assert out["result"]["namedScores"] == {"existing_metric": 0.5}


@pytest.mark.parametrize(
    ("extension", "expected"),
    [
        ("file://path/to/hooks.js:beforeAll", "beforeAll"),
        ("file://path/to/hooks.py:afterEach", "afterEach"),
        ("file://hooks.js:myHandler", "myHandler"),
        ("file://path/to/hooks.js", None),
        ("file://hooks.py", None),
        ("path/to/hooks.js:beforeAll", None),
        ("hooks.js:beforeAll", None),
        ("file://C:/path/to/hooks.js", None),
        ("file://C:/path/to/hooks.js:beforeAll", "beforeAll"),
        ("file://D:/hooks.js:myHandler", "myHandler"),
        ("file://path:with:colons/hooks.js:functionName", "functionName"),
        ("file://hooks.js:", None),
        ("file://path/to/hooks.py:", None),
    ],
)
def test_get_extension_hook_name(extension, expected):
    assert get_extension_hook_name(extension) == expected


async def test_errors_are_wrapped(monkeypatch):
    original = RuntimeError("Python script failed")

    async def boom(*args):
        raise original

    monkeypatch.setattr("promptphoo.transform.transform", boom)
    with pytest.raises(ExtensionHookError) as info:
        await run_extension_hook(["file://broken.py:handler"], "beforeAll", {"suite": _suite()})
    assert str(info.value).startswith(
        'Extension hook "beforeAll" failed for file://broken.py:handler'
    )
    assert info.value.__cause__ is original


def test_javascript_extensions_rejected():
    check_extensions_supported(["file://hooks.py:beforeAll", "inline"])
    with pytest.raises(ExtensionHookError, match="JavaScript extension hooks are not supported"):
        check_extensions_supported(["file://hooks.mjs:beforeAll"])


async def test_before_all_seeds_provider_prompt_map(transform_mock):
    _, returns = transform_mock

    class Labelled(MockProvider):
        label = "lab"

    suite = {**_suite(), "providers": [Labelled()], "extensions": ["file://h.py:beforeAll"]}

    def change(ext, ctx, _):
        seeded = ctx["suite"]["providerPromptMap"]
        assert seeded == {"test-provider": ["Test prompt"], "lab": ["Test prompt"]}
        return {"suite": {**ctx["suite"], "providerPromptMap": {"lab": ["other"]}}}

    returns.append(change)
    out = await run_before_all_extensions(suite)
    # Unchanged seeds are dropped, removed ones become '*', changed ones override.
    assert out["providerPromptMap"] == {"lab": ["other"], "test-provider": ["*"]}
    assert out["defaultTest"] == {"assert": []}


def test_synchronize_legacy_transport_headers():
    headers = {"a": "1"}
    assert synchronize_legacy_transport_headers(
        {"headers": headers, "k": 1}, {"headers": headers}, {"headers": headers, "k": 1},
        {"headers": {"b": "2"}},
    ) == {"k": 1, "headers": {"b": "2"}}  # fmt: skip
    assert (
        synchronize_legacy_transport_headers(
            {"headers": headers}, {"headers": headers}, {"headers": headers}, {}
        )
        == {}
    )
    assert synchronize_legacy_transport_headers(
        {"headers": headers}, {"headers": {"x": 1}}, {"headers": headers}, {}
    ) == {"headers": headers}


# --- end to end ------------------------------------------------------------------------------

HOOKS = """\
import json, os

LOG = os.path.join(os.path.dirname(__file__), "hooks.log")


def log(entry):
    with open(LOG, "a") as fh:
        fh.write(json.dumps(entry) + "\\n")


def beforeAll(context, extra):
    log({"hook": extra["hookName"], "tests": len(context["suite"]["tests"])})
    context["suite"]["tests"].append({"vars": {"name": "added"}})
    return context


def beforeEach(context, extra):
    context["test"]["vars"]["name"] = context["test"]["vars"]["name"].upper()
    return context


def afterEach(context, extra):
    seen = context["result"]["vars"]["name"]
    return {"result": {"namedScores": {"hooked": 1}, "metadata": {"seen": seen}}}


def afterAll(context, extra):
    log({"hook": "afterAll", "results": len(context["results"]), "evalId": context["evalId"],
         "prompts": len(context["prompts"]), "description": context["config"].get("description")})


def legacy(hook_name, context):
    log({"legacy": hook_name})
"""

CONFIG = """\
description: hooks e2e
prompts:
  - "Hello {{name}}"
providers:
  - echo
tests:
  - vars: {name: world}
"""


def _run_eval(tmp_path, monkeypatch, *extra):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hooks.py").write_text(HOOKS)
    (tmp_path / "promptfooconfig.yaml").write_text(CONFIG)
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-table", *extra]
    try:
        result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)
    finally:
        db.close_db()
    return result


def _log(tmp_path):
    return [json.loads(line) for line in (tmp_path / "hooks.log").read_text().splitlines()]


@pytest.mark.parametrize("write", [[], ["--no-write"]])
def test_python_hooks_end_to_end(tmp_path, monkeypatch, write):
    result = _run_eval(
        tmp_path, monkeypatch, *write,
        "-x", "file://hooks.py:beforeAll", "file://hooks.py:beforeEach",
        "-x", "file://hooks.py:afterEach", "file://hooks.py:afterAll", "file://hooks.py:legacy",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    log = _log(tmp_path)
    assert log[0] == {"hook": "beforeAll", "tests": 1}
    assert {"legacy": "beforeAll"} in log and {"legacy": "afterEach"} in log
    after_all = next(e for e in log if e.get("hook") == "afterAll")
    assert after_all["results"] == 2 and after_all["prompts"] == 1
    assert after_all["description"] == "hooks e2e"
    rows = json.loads((tmp_path / "out.json").read_text())["results"]["results"]
    outputs = sorted(r["response"]["output"] for r in rows)
    assert outputs == ["Hello ADDED", "Hello WORLD"]
    for row in rows:
        assert row["namedScores"] == {"hooked": 1}
        assert row["metadata"]["seen"] == row["vars"]["name"]


def test_javascript_extension_fails_clearly(tmp_path, monkeypatch, caplog):
    result = _run_eval(tmp_path, monkeypatch, "-x", "file://hooks.js:beforeAll")
    assert result.exit_code == 1
    assert "JavaScript extension hooks are not supported by promptphoo" in caplog.text


def test_failing_before_all_hook_fails_the_eval(tmp_path, monkeypatch, caplog):
    result = _run_eval(tmp_path, monkeypatch, "-x", "file://hooks.py:missing")
    assert result.exit_code == 1
    assert 'Extension hook "beforeAll" failed for file://hooks.py:missing' in caplog.text
