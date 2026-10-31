"""`eval --watch`. Ports test/util/testCaseWatchPaths.test.ts and the watch-mode cases of
test/commands/eval.test.ts (watch list, recovery after a failing re-run)."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import pytest

from promptphoo import db
from promptphoo.cli import watch
from promptphoo.cli.eval import run_eval_command
from promptphoo.cli.watch import get_watch_paths, poll_for_changes, resolve_tests_watch_paths


@pytest.fixture
def base(tmp_path):
    for d in ("tests", "nested", "fixtures"):
        (tmp_path / d).mkdir()
    for rel in ("cases.yaml", "tests/a.yaml", "tests/b.yaml", "gen.py", "dataset.yaml",
                "vars.csv", "book.xlsx", "fixtures/a:b.txt"):  # fmt: skip
        (tmp_path / rel).write_text("")
    return str(tmp_path)


def j(base, rel):
    return os.path.join(base, rel)


def test_scalar_reference_and_glob(base):
    assert resolve_tests_watch_paths("file://cases.yaml", base) == [j(base, "cases.yaml")]
    assert sorted(resolve_tests_watch_paths("file://tests/*.yaml", base)) == [
        j(base, "tests/a.yaml"),
        j(base, "tests/b.yaml"),
    ]
    # Never the glob's parent directory.
    assert j(base, "tests") not in resolve_tests_watch_paths("file://tests/*.yaml", base)
    assert base not in resolve_tests_watch_paths("file://*.yaml", base)
    assert base not in resolve_tests_watch_paths("file://**/*.yaml", base)


def test_glob_beneath_directory_with_brackets(base):
    root = os.path.join(base, "suite[blue]")
    os.mkdir(root)
    Path(root, "cases.yaml").write_text("- description: case")
    assert resolve_tests_watch_paths("file://*.yaml", root) == [os.path.join(root, "cases.yaml")]


def test_suffixes_and_selectors(base):
    assert resolve_tests_watch_paths("file://gen.py:make_tests", base) == [j(base, "gen.py")]
    assert resolve_tests_watch_paths([{"vars": {"body": "file://fixtures/a:b.txt"}}], base) == [
        j(base, "fixtures/a:b.txt")
    ]
    assert resolve_tests_watch_paths("file://book.xlsx#DataSheet", base) == [j(base, "book.xlsx")]
    assert resolve_tests_watch_paths("file://cases.yaml#frag", base) == [j(base, "cases.yaml#frag")]


def test_generator_config_and_mixed_array(base):
    watched = resolve_tests_watch_paths(
        {"path": "file://gen.py:make", "config": {"data": "file://dataset.yaml"}}, base
    )
    assert j(base, "gen.py") in watched and j(base, "dataset.yaml") in watched
    watched = resolve_tests_watch_paths(
        ["file://cases.yaml", {"path": "file://gen.py:make"}, {"vars": {"data": "file://vars.csv"}},
         {"vars": {"inline": "not a file"}}],
        base,
    )  # fmt: skip
    assert {j(base, "cases.yaml"), j(base, "gen.py"), j(base, "vars.csv")} <= set(watched)
    assert j(base, "not a file") not in watched
    cyclic: dict = {"data": "file://dataset.yaml"}
    cyclic["self"] = cyclic
    assert j(base, "dataset.yaml") in resolve_tests_watch_paths(
        {"path": "file://gen.py:make", "config": cyclic}, base
    )


def test_literal_fallbacks(base):
    assert resolve_tests_watch_paths("file://not-created-yet.yaml", base) == [
        j(base, "not-created-yet.yaml")
    ]
    Path(base, "report[1].csv").write_text("")
    assert resolve_tests_watch_paths("file://report[1].csv", base) == [j(base, "report[1].csv")]


@pytest.mark.parametrize("ext", ["yaml", "jsonl"])
def test_nested_references(base, ext):
    content = (
        "- vars:\n    data: file://vars.csv\n" if ext == "yaml"
        else '{"vars":{"data":"file://vars.csv"}}\n'
    )  # fmt: skip
    Path(base, f"nested/cases.{ext}").write_text(content)
    watched = resolve_tests_watch_paths(f"file://nested/cases.{ext}", base)
    assert j(base, f"nested/cases.{ext}") in watched and j(base, "vars.csv") in watched


@pytest.mark.parametrize("ext", ["yaml", "json", "jsonl"])
def test_bare_vars_files_from_nested_rows(base, ext):
    file = j(base, f"nested/vars-cases.{ext}")
    row = {"vars": ["one.yaml", "two.yaml"], "provider": "file://provider.yaml"}
    Path(file).write_text(
        "- vars: [one.yaml, two.yaml]\n  provider: file://provider.yaml\n"
        if ext == "yaml"
        else json.dumps([row] if ext == "json" else row)
    )
    source = f"nested/vars-cases.{ext}"
    paths = resolve_tests_watch_paths([source], base)
    assert {file, j(base, "nested/one.yaml"), j(base, "nested/two.yaml"),
            j(base, "provider.yaml")} <= set(paths)  # fmt: skip
    assert j(base, "nested/provider.yaml") not in paths
    scalar_base = j(base, "nested") if ext == "yaml" else base
    assert os.path.join(scalar_base, "one.yaml") in resolve_tests_watch_paths(source, base)
    assert j(base, "nested/one.yaml") in resolve_tests_watch_paths(
        f"nested/vars-cases*.{ext}", base
    )


@pytest.mark.parametrize("ext", ["yaml", "json", "jsonl"])
def test_row_provider_scripts(base, ext):
    file = j(base, f"nested/provider-cases.{ext}")
    rows = [
        {"vars": {"doc": "file://provider.py:call_api"}, "provider": "file://provider.py:call_api"},
        {"vars": {}, "provider": {"id": "file://other.py:call_api"}},
        {"vars": {}, "provider": "file://provider.yaml"},
    ]
    Path(file).write_text(
        "\n".join(json.dumps(r) for r in rows) if ext == "jsonl" else json.dumps(rows)
    )
    source = f"nested/provider-cases.{ext}"
    watched = resolve_tests_watch_paths([source], base)
    assert {file, j(base, "provider.py"), j(base, "nested/provider.py"),
            j(base, "nested/other.py"), j(base, "provider.yaml")} <= set(watched)  # fmt: skip
    assert j(base, "other.py") not in watched
    assert j(base, "nested/provider.yaml") not in watched
    scalar_base = j(base, "nested") if ext == "yaml" else base
    assert os.path.join(scalar_base, "other.py") in resolve_tests_watch_paths(source, base)


def test_vars_forms_remote_and_dedup(base):
    os.mkdir(j(base, "varsdir"))
    Path(base, "varsdir/one.yaml").write_text("")
    assert j(base, "varsdir/one.yaml") in resolve_tests_watch_paths(
        [{"vars": "varsdir/*.yaml"}], base
    )
    assert resolve_tests_watch_paths([{"vars": ["va/common.yaml", "va/case.yaml"]}], base) == [
        j(base, "va/common.yaml"),
        j(base, "va/case.yaml"),
    ]
    assert resolve_tests_watch_paths([{"vars": {"data": "file://vars.csv"}}], base) == [
        j(base, "vars.csv")
    ]
    assert resolve_tests_watch_paths("https://docs.google.com/spreadsheets/d/abc", base) == []
    assert resolve_tests_watch_paths("az://container/tests.csv", base) == []
    assert resolve_tests_watch_paths(None, base) == []
    assert resolve_tests_watch_paths(["file://cases.yaml", "file://cases.yaml"], base) == [
        j(base, "cases.yaml")
    ]
    Path(base, "broken.yaml").write_text("this: [unclosed\n")
    assert resolve_tests_watch_paths("file://broken.yaml", base) == [j(base, "broken.yaml")]


def test_get_watch_paths():
    config = {
        "prompts": ["file://prompts/main.txt", {"id": "file://prompts/object.txt"}, "inline"],
        "providers": ["file://providers/provider.yaml", "echo"],
        "tests": ["file://vars/scenario.yaml", {"vars": {"body": "file://vars/body.txt"}}],
    }
    paths = get_watch_paths({}, config, ["/suite/promptfooconfig.yaml"], [])
    assert paths == [
        "/suite/promptfooconfig.yaml",
        "/suite/prompts/main.txt",
        "/suite/prompts/object.txt",
        "/suite/providers/provider.yaml",
        "/suite/vars/scenario.yaml",
        "/suite/vars/body.txt",
    ]


@pytest.mark.parametrize(
    ("flags", "base_path", "expected"),
    [
        ({"vars": "cases.csv"}, None, "/suite/cases.csv"),
        ({"vars": "cases.csv"}, "/assets", "/assets/cases.csv"),
        ({"tests": "cases.csv"}, None, "/working/cases.csv"),
        ({"tests": "cases.csv", "vars": "ignored.csv"}, None, "/working/cases.csv"),
    ],
)
def test_cli_test_file_base(monkeypatch, flags, base_path, expected):
    monkeypatch.setattr(os, "getcwd", lambda: "/working")
    config = {"prompts": [], "providers": [], "tests": [], "basePath": base_path}
    if base_path is None:
        del config["basePath"]
    paths = get_watch_paths(flags, config, ["/suite/promptfooconfig.yaml"], None)
    assert paths == ["/suite/promptfooconfig.yaml", expected]


def test_poll_for_changes(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("1")
    ignored = ".hidden"
    ticks = {"n": 0}

    def should_stop():
        ticks["n"] += 1
        if ticks["n"] == 2:
            target.write_text("22")
        if ticks["n"] == 4:
            target.unlink()
        if ticks["n"] == 6:
            target.write_text("333")
        return ticks["n"] > 7

    changes = list(
        poll_for_changes([str(target), ignored], interval=0.001, should_stop=should_stop)
    )
    assert changes == [str(target), str(target)]


CONFIG = """\
description: watch e2e
prompts:
  - "{prompt} {{{{name}}}}"
providers:
  - echo
tests:
  - vars: {{name: world}}
"""


def test_watch_reruns_on_change_and_survives_errors(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "promptfooconfig.yaml"
    config.write_text(CONFIG.format(prompt="Hello"))
    edits = [
        CONFIG.format(prompt="Goodbye"),
        "prompts: [hi]\nproviders: []\n",  # fails to resolve; watching continues
        CONFIG.format(prompt="Back again"),
    ]
    outputs: list[str] = []
    ticks = {"n": 0}

    def should_stop():
        out = tmp_path / "out.json"
        if out.exists():
            text = json.loads(out.read_text())["results"]["results"][0]["response"]["output"]
            if not outputs or outputs[-1] != text:
                outputs.append(text)
        ticks["n"] += 1
        if edits and ticks["n"] % 3 == 0:
            config.write_text(edits.pop(0))
        return not edits and ticks["n"] > 12

    cmd = {"config": ["promptfooconfig.yaml"], "watch": True, "write": False, "table": False,
           "output": ["out.json"]}  # fmt: skip
    try:
        with caplog.at_level(logging.INFO, logger="promptphoo"):
            code = run_eval_command(cmd, {}, None, {}, poll_interval=0.01, should_stop=should_stop)
    finally:
        db.close_db()
    assert code == 0
    assert outputs == ["Hello world", "Goodbye world", "Back again world"]
    assert caplog.text.count("File change detected: promptfooconfig.yaml") == 3
    assert "Watching for file changes on promptfooconfig.yaml ..." in caplog.text
    assert "You must specify at least 1 provider" in caplog.text


def test_watch_needs_a_config_path(monkeypatch, caplog):
    def fake_do_eval(cmd, default_config, default_config_path, options, state):
        state.update(config={}, config_paths=[], test_sources=None)

        class Rec:
            pass

        return Rec(), 0

    async def run(*args):
        return fake_do_eval(*args)

    monkeypatch.setattr("promptphoo.cli.eval.do_eval", run)
    assert run_eval_command({"watch": True}, {}, None, {}) == 1
    assert "Could not locate config file(s) to watch" in caplog.text


def test_watch_stops_on_keyboard_interrupt(caplog):
    def poll(*args, **kwargs):
        raise KeyboardInterrupt
        yield  # pragma: no cover

    original = watch.poll_for_changes
    watch.poll_for_changes = poll  # type: ignore[assignment]
    try:
        with caplog.at_level(logging.INFO, logger="promptphoo"):
            watch.watch_and_rerun(["a"], lambda: None)
    finally:
        watch.poll_for_changes = original
    assert "Stopping watch mode..." in caplog.text


def test_poll_runs_in_background_thread(tmp_path):
    """A rerun triggered from a real background edit (no should_stop side effects)."""
    target = tmp_path / "cfg.yaml"
    target.write_text("a")
    stop = threading.Event()
    seen: list[str] = []

    def edit():
        stop.wait(0.05)
        target.write_text("bb")

    threading.Thread(target=edit).start()
    for path in poll_for_changes([str(target)], interval=0.01, should_stop=lambda: bool(seen)):
        seen.append(path)
    assert seen == [str(target)]
