"""`promptfoo validate`: ports test/commands/validate-exit-codes.test.ts,
validate-provider-tests.test.ts and validate-falsy-output.test.ts, plus the zod-style error
rendering (expected strings captured from Node 0.125.1)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from promptphoo.cli import validate as v
from promptphoo.cli.main import build_cli
from promptphoo.providers.base import ApiProvider


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    caplog.set_level("DEBUG", logger="promptphoo")


def run(args: list[str]):
    return CliRunner().invoke(build_cli(args), args, catch_exceptions=False)


def write(tmp_path: Path, text: str, name: str = "promptfooconfig.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


VALID = 'prompts: ["hi {{x}}"]\nproviders: [echo]\ntests:\n  - vars: {x: 1}\n'


class FakeProvider(ApiProvider):
    def __init__(self, provider_id: str, response: Any = None, config: Any = None, exc=None):
        super().__init__({"id": provider_id, "config": config or {}})
        self.response = response
        self.exc = exc
        self.calls: list[tuple[str, Any]] = []

    def default_id(self) -> str:
        return "fake"

    async def call_api(self, prompt, context=None, options=None):
        self.calls.append((prompt, context))
        if self.exc:
            raise self.exc
        return self.response


def messages(caplog, level: str | None = None) -> list[str]:
    return [r.getMessage() for r in caplog.records if level is None or r.levelname == level]


# --- validate config -------


def test_valid_default_config_exits_0(tmp_path: Path, caplog):
    write(tmp_path, VALID)
    result = run(["validate"])
    assert result.exit_code == 0
    assert "Configuration is valid." in messages(caplog, "INFO")


def test_valid_explicit_config_via_subcommand(tmp_path: Path, caplog):
    write(tmp_path, VALID, "other.yaml")
    assert run(["validate", "config", "-c", "other.yaml"]).exit_code == 0
    assert "Configuration is valid." in messages(caplog)


def test_schema_error_exits_1_with_zod_message(tmp_path: Path, caplog):
    path = write(tmp_path, VALID + "evaluateOptions:\n  maxConcurrency: abc\n")
    assert run(["validate"]).exit_code == 1
    assert messages(caplog, "ERROR") == [
        "Configuration validation error:\n"
        f"Config file path(s): {path}\n"
        "✖ Invalid input: expected number, received string\n"
        "→ at evaluateOptions.maxConcurrency"
    ]
    # the config loader's own warning keeps prettifyError's indentation
    assert (
        f"Invalid configuration file {path}:\n"
        "✖ Invalid input: expected number, received string\n"
        "  → at evaluateOptions.maxConcurrency"
    ) in messages(caplog, "WARNING")


def test_several_schema_errors_match_node(tmp_path: Path, caplog):
    path = write(
        tmp_path,
        'description: 5\nprompts: ["hi {{x}}"]\nproviders:\n  - id: echo\n'
        "tests:\n  - vars: {x: 1}\n    assert:\n      - type: not-a-real-type\n"
        '  - description: [1]\nsharing: "maybe"\noutputPath: 7\ntags: {a: 1}\n',
    )
    assert run(["validate"]).exit_code == 1
    assert (
        f"Invalid configuration file {path}:\n"
        "✖ Invalid input: expected string, received number\n  → at description\n"
        "✖ Invalid input\n  → at tests\n"
        "✖ Invalid input\n  → at outputPath\n"
        "✖ Invalid input\n  → at sharing\n"
        "✖ Invalid input: expected string, received number\n  → at tags.a"
    ) in messages(caplog, "WARNING")
    assert messages(caplog, "ERROR") == [
        "Configuration validation error:\n"
        f"Config file path(s): {path}\n"
        "✖ Invalid input\n→ at tests\n"
        "✖ Invalid input: expected string, received number\n→ at tags.a"
    ]


def test_missing_default_config_warns_and_exits_1(caplog):
    assert run(["validate"]).exit_code == 1
    [warning] = messages(caplog, "WARNING")
    assert warning.startswith("Failed to validate configuration: ⚠️  No promptfooconfig found")
    assert "\n\npromptfoo eval -c path/to/promptfooconfig.yaml\n\n" in warning
    assert warning.endswith("Or create a config with:\n\npromptfoo init")


def test_config_resolution_error_is_prefixed(caplog):
    assert run(["validate", "-c", "missing.yaml"]).exit_code == 1
    assert messages(caplog, "ERROR") == [
        "Failed to validate configuration: No configuration file found at missing.yaml. Run "
        '"promptfoo init" to create one or pass --config path/to/promptfooconfig.yaml.'
    ]


def test_both_targets_and_providers_only_warns(tmp_path: Path, caplog):
    # As in Node: the loader warns about the file; the resolved config keeps `providers` only.
    path = write(tmp_path, VALID + "targets: [echo]\n")
    assert run(["validate"]).exit_code == 0
    assert (
        f"Invalid configuration file {path}:\n"
        "✖ Exactly one of 'targets' or 'providers' must be provided, but not both"
    ) in messages(caplog, "WARNING")


def test_js_dedent():
    assert v.js_dedent("a:\nb\n✖ x\n  → at y\n✖ z\n    deeper") == (
        "a:\nb\n✖ x\n→ at y\n✖ z\n  deeper"
    )
    assert v.js_dedent("\n\n   only\n   lines\n") == "only\nlines"


# --- validate target -------


def target(provider_id: str) -> int:
    return asyncio.run(v.do_validate_target(provider_id, None, None, {}))


@pytest.fixture
def load_provider(monkeypatch: pytest.MonkeyPatch):
    loaded: dict[str, Any] = {}

    def use(provider: ApiProvider) -> dict[str, Any]:
        async def fake(provider_path, **kwargs):
            loaded["path"] = provider_path
            loaded["kwargs"] = kwargs
            return provider

        monkeypatch.setattr("promptphoo.providers.load_api_provider", fake)
        return loaded

    return use


def test_requires_target_or_config(caplog):
    assert run(["validate", "target"]).exit_code == 1
    assert messages(caplog, "ERROR") == [
        "Please specify either -t <provider-id> or -c <config-path>"
    ]


def test_echo_target_passes(caplog):
    assert run(["validate", "target", "-t", "echo"]).exit_code == 0
    info = messages(caplog, "INFO")
    assert info[:4] == ["", "Validating Target", "═" * 50, "Target: echo"]
    assert '    Response: "Hello, world!"' in info
    assert info[-1] == "Tests: 1 passed, 1 skipped"
    assert "    connectivity: passed, session: skipped" in info


@pytest.mark.parametrize("output", [0, False, ""])
def test_falsy_outputs_are_accepted(output, load_provider, caplog):
    load_provider(FakeProvider("echo", {"output": output}))
    assert target("echo") == 0
    import json

    assert f"    Response: {json.dumps(output)}" in messages(caplog, "INFO")
    assert not any("Connectivity test" in m for m in messages(caplog, "WARNING"))


@pytest.mark.parametrize("response", [{"output": None}, {}])
def test_missing_output_is_rejected(response, load_provider, caplog):
    load_provider(FakeProvider("echo", response))
    assert target("echo") == 1
    assert "  ✗ Connectivity test" in messages(caplog, "WARNING")
    assert "    No output received from provider" in messages(caplog, "INFO")


def test_truncates_long_previews_only_past_100_chars(load_provider, caplog):
    load_provider(FakeProvider("echo", {"output": "x" * 98}))  # 100 chars with quotes
    target("echo")
    assert f'    Response: "{"x" * 98}"' in messages(caplog, "INFO")
    caplog.clear()
    load_provider(FakeProvider("echo", {"output": "x" * 99}))
    target("echo")
    assert f'    Response: "{"x" * 99}...' in messages(caplog, "INFO")


def test_provider_error_response(load_provider, caplog):
    provider = FakeProvider("echo", {"error": "boom"})
    load_provider(provider)
    assert target("echo") == 1
    assert messages(caplog, "ERROR") == ["  ✗ Connectivity test", "    boom"]
    assert provider.calls[0] == (
        "Hello, world!",
        {"debug": True, "prompt": {"raw": "Hello, world!", "label": "Hello, world!"}, "vars": {}},
    )


def test_provider_exception(load_provider, caplog):
    load_provider(FakeProvider("echo", exc=RuntimeError("kaput")))
    assert target("echo") == 1
    assert "    kaput" in messages(caplog, "ERROR")


def test_load_failure(monkeypatch: pytest.MonkeyPatch, caplog):
    async def fail(*_a, **_k):
        raise RuntimeError("Could not identify provider: x")

    monkeypatch.setattr("promptphoo.providers.load_api_provider", fail)
    assert target("x") == 1
    assert "Provider tests failed: Could not identify provider: x" in messages(caplog, "ERROR")


def test_cloud_uuid_target_fails_clearly(caplog):
    assert target("3fa85f64-5717-4562-b3fc-2c963f66afa6") == 1
    assert "promptfoo cloud is not supported" in messages(caplog, "ERROR")[-1]


def test_http_target_is_patched_and_tested(load_provider, monkeypatch, caplog):
    provider = FakeProvider("http://example.com", {"output": "ok"})
    loaded = load_provider(provider)

    async def connectivity(p):
        return {"success": True, "message": "Connectivity test passed", "sessionId": "s1"}

    async def session(p):
        return {
            "success": False,
            "message": "Session test failed",
            "reason": "The target did not preserve context",
        }

    monkeypatch.setattr(v, "check_provider_connectivity", connectivity)
    monkeypatch.setattr(v, "check_provider_session", session)
    assert target("http://example.com") == 1
    assert loaded["kwargs"]["options"] == {
        "config": {"maxRetries": 1, "headers": {"x-promptfoo-silent": "true"}}
    }
    info = messages(caplog, "INFO")
    assert "  ◌ Connectivity test (running...)" in info
    assert "  ✓ Connectivity test" in info
    assert "    Session ID: s1" in info
    assert "    Reason: The target did not preserve context" in info
    assert "    connectivity: passed, session: failed" in info
    assert "Tests: 1 passed, 1 failed" in info


def test_http_suggestions_and_request_details(load_provider, monkeypatch, caplog):
    load_provider(FakeProvider("http://example.com", config={"stateful": False}))

    async def connectivity(p):
        return {
            "success": False,
            "message": "Configuration needs changes",
            "transformedRequest": {"url": "http://example.com/api", "method": "POST"},
            "analysis": {
                "changes_needed": True,
                "changes_needed_reason": "Response format needs changes",
                "changes_needed_suggestions": ["Add transformResponse"],
            },
        }

    monkeypatch.setattr(v, "check_provider_connectivity", connectivity)
    assert target("http://example.com") == 0
    assert "  ⚠ Connectivity test" in messages(caplog, "WARNING")
    info = messages(caplog, "INFO")
    assert "    Response format needs changes" in info
    assert "    1. Add transformResponse" in info
    assert "  ○ Session test (skipped - target is stateless)" in info
    assert "\n1 provider(s) have configuration suggestions - see above for details" in info
    debug = messages(caplog, "DEBUG")
    assert "      URL: http://example.com/api" in debug
    assert "      Method: POST" in debug


def test_http_malformed_suggestions_are_ignored(load_provider, monkeypatch):
    load_provider(FakeProvider("http://example.com", config={"stateful": False}))

    async def connectivity(p):
        return {
            "success": False,
            "message": "x",
            "analysis": {"changes_needed": True, "changes_needed_suggestions": "nope"},
        }

    monkeypatch.setattr(v, "check_provider_connectivity", connectivity)
    assert target("http://example.com") == 0


def test_http_connectivity_failure_skips_session(load_provider, monkeypatch, caplog):
    load_provider(FakeProvider("https://example.com"))

    async def connectivity(p):
        return {"success": False, "message": "Provider returned error: x", "error": "x"}

    monkeypatch.setattr(v, "check_provider_connectivity", connectivity)
    assert target("https://example.com") == 1
    assert messages(caplog, "ERROR")[:3] == [
        "  ✗ Connectivity test",
        "    Provider returned error: x",
        "    x",
    ]
    assert "  ○ Session test (skipped - connectivity failed)" in messages(caplog, "INFO")


def test_connectivity_runs_an_eval_with_session_id():
    provider = FakeProvider("http://example.com", {"output": "pong"})
    result = asyncio.run(v.check_provider_connectivity(provider))
    assert result["success"] is True
    assert result["message"].startswith("Provider test completed. Remote grading disabled")
    prompt, context = provider.calls[0]
    assert prompt == "Hello World!"
    assert result["sessionId"] == context["vars"]["sessionId"]


def test_session_check_never_calls_remote_grader():
    provider = FakeProvider("http://example.com", {"output": "pong", "sessionId": "abc"})
    result = asyncio.run(v.check_provider_session(provider))
    assert result == {
        "success": False,
        "message": "Session test completed. Remote grading is disabled - please examine the "
        "results yourself.",
        "reason": "Manual review required - remote grading is disabled",
    }
    assert [c[0] for c in provider.calls] == [
        "What can you help me with?",
        "What was the last thing I asked you?",
    ]
    assert provider.calls[1][1]["vars"]["sessionId"] == "abc"


def test_server_session_without_parsed_id_fails():
    provider = FakeProvider("http://x", {"output": "pong"}, config={"sessionParser": "x"})
    result = asyncio.run(v.check_provider_session(provider))
    assert result["message"].startswith("Session extraction failed")


def test_config_mode_tests_every_provider(tmp_path: Path, caplog):
    write(tmp_path, 'prompts: ["hi"]\nproviders: [echo, {id: echo, label: e2}]\n', "c.yaml")
    assert run(["validate", "target", "-c", "c.yaml"]).exit_code == 0
    info = messages(caplog, "INFO")
    assert "Configuration: c.yaml" in info
    assert info[-1] == "Tests: 2 passed, 2 skipped"


def test_config_mode_without_providers(monkeypatch, caplog):
    assert asyncio.run(v.run_provider_tests(None, {"providers": []})) == 0
    assert messages(caplog, "INFO")[-1].startswith("No providers found in configuration to test.")


def test_config_mode_load_failure(caplog):
    assert run(["validate", "target", "-c", "missing.yaml"]).exit_code == 1
    assert messages(caplog, "ERROR")[-1].startswith(
        "Failed to load configuration: No configuration file found at missing.yaml."
    )


def test_is_http_provider_and_patch():
    assert v.is_http_provider({"id": "https://x"})
    assert v.is_http_provider({"id": "http"})
    assert v.is_http_provider({"id": "custom", "config": {"url": "http://x"}})
    assert not v.is_http_provider({"id": "custom", "config": {"url": "ftp://x"}})
    assert not v.is_http_provider(FakeProvider("echo"))
    assert v.patch_http_config_for_validation(
        {"id": "https://x", "config": {"headers": {"a": "b"}, "maxRetries": 5}}
    ) == {
        "id": "https://x",
        "config": {"headers": {"a": "b", "x-promptfoo-silent": "true"}, "maxRetries": 1},
    }


def test_is_uuid():
    assert v.is_uuid("3fa85f64-5717-4562-b3fc-2c963f66afa6")
    assert not v.is_uuid("echo")
    assert not v.is_uuid("3fa85f64-5717-4562-b3fc-2c963f66afa6x")
