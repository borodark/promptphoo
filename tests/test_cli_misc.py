"""`cache clear`, `debug`, `feedback` and `share` (ports test/feedback.test.ts and the share
command's registration), plus the zod-style schema error rendering."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import jsonschema
import pytest
import respx
from click.testing import CliRunner

from promptphoo import cache, fetch
from promptphoo.cli import feedback
from promptphoo.cli.main import build_cli
from promptphoo.config import zod_errors
from promptphoo.config.load import config_schema_issues


@pytest.fixture(autouse=True)
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.setattr(fetch, "_clients", {})
    caplog.set_level("INFO", logger="promptphoo")
    return tmp_path / "home"


def run(args: list[str], input_: str | None = None):
    return CliRunner().invoke(build_cli(args), args, input=input_, catch_exceptions=False)


def msgs(caplog, level: str | None = None) -> list[str]:
    return [r.getMessage() for r in caplog.records if level is None or r.levelname == level]


# --- cache clear -------


def test_cache_clear(monkeypatch: pytest.MonkeyPatch, caplog):
    cleared = []

    async def fake_clear():
        cleared.append(True)

    monkeypatch.setattr(cache, "clear_cache", fake_clear)
    assert run(["cache", "clear"]).exit_code == 0
    assert cleared == [True]
    assert msgs(caplog)[0] == "Clearing cache..."


def test_cache_clear_prints_progress_messages(monkeypatch: pytest.MonkeyPatch, caplog):
    from promptphoo.cli import cache as cache_cmd

    async def slow_clear():
        await asyncio.sleep(0.05)

    monkeypatch.setattr(cache, "clear_cache", slow_clear)
    monkeypatch.setattr(cache_cmd, "MESSAGE_INTERVAL_S", 0.01)
    asyncio.run(cache_cmd.clear_with_messages())
    assert msgs(caplog)[:2] == ["Scrubbing bits...", "Sweeping stale data..."]


# --- debug -------


def test_debug_without_config(caplog):
    assert run(["debug"]).exit_code == 0
    out = msgs(caplog)
    assert out[1] == "Promptfoo Debug Information"
    info = json.loads(out[3])
    assert info["configInfo"] == {"configExists": False, "configContent": None}
    assert set(info["platform"]) == {"os", "release", "arch", "pythonVersion"}
    assert out[-1].startswith("Please include this output when reporting issues on GitHub:")


def test_debug_with_config(tmp_path: Path, caplog):
    (tmp_path / "promptfooconfig.yaml").write_text('prompts: ["hi"]\nproviders: [echo]\n')
    assert run(["debug"]).exit_code == 0
    info = json.loads(msgs(caplog)[3])
    content = info["configInfo"]["configContent"]
    assert info["configInfo"]["configExists"] is True
    assert content["config"]["providers"] == ["echo"]
    assert content["testSuite"]["providers"][0]["id"] == "echo"
    assert content["testSuite"]["prompts"] == [{"raw": "hi", "label": "hi"}]


def test_debug_config_error(tmp_path: Path, caplog):
    (tmp_path / "bad.yaml").write_text("prompts: []\n")
    assert run(["debug", "-c", "bad.yaml"]).exit_code == 0
    report = next(m for m in msgs(caplog, "INFO") if m.startswith("{"))
    content = json.loads(report)["configInfo"]["configContent"]
    assert content.startswith("Error loading config: Error: ")


# --- feedback -------


def test_send_feedback(caplog):
    with respx.mock() as mock:
        route = mock.post(feedback.FEEDBACK_URL).mock(return_value=httpx.Response(200))
        asyncio.run(feedback.send_feedback("Great tool!"))
    assert json.loads(route.calls[0].request.content) == {"message": "Great tool!"}
    assert msgs(caplog) == ["Feedback sent. Thank you!"]


def test_send_feedback_api_failure(caplog):
    with respx.mock() as mock:
        mock.post(feedback.FEEDBACK_URL).mock(return_value=httpx.Response(400))
        asyncio.run(feedback.send_feedback("x"))
    assert msgs(caplog) == ["Failed to send feedback. Please try again or open an issue on GitHub."]


def test_send_feedback_network_error(caplog):
    with respx.mock() as mock:
        mock.post(feedback.FEEDBACK_URL).mock(side_effect=httpx.ConnectError("down"))
        asyncio.run(feedback.send_feedback("x"))
    assert msgs(caplog, "ERROR") == ["Network error while sending feedback"]


def test_empty_feedback_is_not_sent():
    with respx.mock() as mock:
        asyncio.run(feedback.send_feedback("   "))
        assert not mock.calls


def test_feedback_command_with_message():
    with respx.mock() as mock:
        route = mock.post(feedback.FEEDBACK_URL).mock(return_value=httpx.Response(200))
        assert run(["feedback", "hello"]).exit_code == 0
    assert json.loads(route.calls[0].request.content) == {"message": "hello"}


def test_feedback_interactive_adds_contact():
    with respx.mock() as mock:
        route = mock.post(feedback.FEEDBACK_URL).mock(return_value=httpx.Response(200))
        assert run(["feedback"], input_="it works\nme@x.io\n").exit_code == 0
    assert json.loads(route.calls[0].request.content) == {"message": "it works\n\nContact: me@x.io"}


def test_feedback_interactive_empty(caplog):
    with respx.mock() as mock:
        run(["feedback"], input_="\n")
        assert not mock.calls
    assert "No feedback provided." in msgs(caplog)


# --- share -------


@pytest.mark.parametrize("args", [["share"], ["share", "eval-abc", "--show-auth", "-y"]])
def test_share_is_not_supported(args: list[str], caplog):
    assert run(args).exit_code == 1
    [error] = msgs(caplog, "ERROR")
    assert error.startswith("Sharing is not supported by promptphoo: results stay local")


# --- zod-style errors ----------------------------------------------------------------------------


def render(schema: dict, value) -> str:
    errors = list(jsonschema.Draft7Validator(schema).iter_errors(value))
    return zod_errors.prettify_issues(zod_errors.collect_issues(errors, schema))


def test_zod_messages():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "array", "items": {"type": "number"}, "minItems": 2},
            "c": {"enum": ["x", "y"]},
            "d": {"const": "z"},
            "e": {"type": "integer", "minimum": 0},
            "f": {"type": "object", "additionalProperties": False, "properties": {}},
            "g": {"anyOf": [{"type": "string"}, {"type": "number"}]},
            "h": {"type": "string", "pattern": "^a"},
        },
        "required": ["a", "z"],
    }
    value = {
        "b": ["1"],
        "c": "q",
        "d": "w",
        "e": 1.5,
        "f": {"k": 1, "my key": 2},
        "g": [],
        "h": "b",
    }
    assert render(schema, value).split("\n") == [
        "✖ Too small: expected array to have >=2 items",
        "  → at b",
        '✖ Invalid option: expected one of "x"|"y"',
        "  → at c",
        '✖ Invalid input: expected "z"',
        "  → at d",
        "✖ Invalid input: expected int, received number",
        "  → at e",
        '✖ Unrecognized keys: "k", "my key"',
        "  → at f",
        "✖ Invalid input",
        "  → at g",
        "✖ Invalid string: must match pattern /^a/",
        "  → at h",
        "✖ Invalid input: expected string, received undefined",
        "  → at a",
        "✖ Invalid input",
        "  → at z",
        "✖ Invalid input: expected number, received string",
        "  → at b[0]",
    ]


def test_zod_union_with_single_non_aborted_branch():
    schema = {"anyOf": [{"type": "string", "minLength": 3}, {"type": "number"}]}
    assert render(schema, "ab") == "✖ Too small: expected string to have >=3 characters"


def test_to_dot_path():
    assert zod_errors.to_dot_path(["a", 0, "b c", "d"]) == 'a[0]["b c"].d'


def test_unified_config_refine():
    [issue] = config_schema_issues({"prompts": ["x"]})
    assert issue.message == "Exactly one of 'targets' or 'providers' must be provided, but not both"
    assert config_schema_issues({"prompts": ["x"], "providers": ["echo"]}) == []
