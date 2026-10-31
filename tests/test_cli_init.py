"""`promptphoo init`: ports test/commands/init.test.ts and test/onboarding.test.ts.

The non-interactive files are compared byte for byte with what Node's `init --no-interactive`
wrote (tests/fixtures/init_golden), and the config template with real Nunjucks renders
(tests/fixtures/init_config_golden.json, from tools/init_config_golden.js).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from click.testing import CliRunner
from naming import as_promptphoo

from promptphoo import fetch, onboarding
from promptphoo.cli import init as init_mod
from promptphoo.cli.main import build_cli
from promptphoo.templating import render_string
from promptphoo.yamlload import load_yaml

FIXTURES = Path(__file__).parent / "fixtures"
GITHUB = "https://api.github.com/repos/promptfoo/promptfoo"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    for name in ("npm_execpath", "npm_lifecycle_script", "npm_config_user_agent"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(fetch, "_clients", {})
    monkeypatch.setenv("PROMPTFOO_DISABLE_TRANSIENT_RETRIES", "true")


def run(args: list[str], input_: str | None = None):
    return CliRunner().invoke(build_cli(args), args, input=input_, catch_exceptions=False)


# --- non-interactive -----------------------------------------------------------------------------


def test_no_interactive_writes_node_files_byte_for_byte(tmp_path: Path, caplog):
    caplog.set_level("INFO", logger="promptphoo")
    result = run(["init", "--no-interactive"])
    assert result.exit_code == 0
    for name in ("README.md", "promptfooconfig.yaml"):
        golden = (FIXTURES / "init_golden" / name).read_text("utf-8")
        assert (tmp_path / name).read_text("utf-8") == as_promptphoo(golden)
    messages = [r.getMessage() for r in caplog.records]
    assert messages[:2] == ["📝 Wrote README.md", "📝 Wrote promptfooconfig.yaml"]
    assert "✅ Setup complete! Next steps:\n" in messages
    assert "  1. Run promptphoo eval to evaluate your prompts" in messages


def test_no_interactive_into_subdirectory(tmp_path: Path, caplog):
    caplog.set_level("INFO", logger="promptphoo")
    assert run(["init", "sub/dir", "--no-interactive"]).exit_code == 0
    assert (tmp_path / "sub/dir/promptfooconfig.yaml").exists()
    messages = [r.getMessage() for r in caplog.records]
    assert "📝 Wrote sub/dir/README.md" in messages
    assert "✅ Wrote promptfooconfig.yaml to ./sub/dir\n" in messages
    assert "  1. Run cd sub/dir" in messages


def test_readme_names_promptphoo_even_under_npx(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("npm_config_user_agent", "npx/10.0.0 node/v22")
    readme = onboarding.get_default_readme("rag")
    assert readme.startswith("# Promptfoo RAG evaluation\n")
    assert "promptphoo eval\n" in readme
    assert "npx promptfoo@latest view\n" in readme


def test_generated_config_is_valid_yaml():
    text = (FIXTURES / "init_golden" / "promptfooconfig.yaml").read_text()
    config = load_yaml(text)
    assert config["providers"] == ["openai:gpt-6-luna", "openai:gpt-6-sol"]
    assert len(config["tests"]) == 3


@pytest.mark.parametrize("case", json.loads((FIXTURES / "init_config_golden.json").read_text()))
def test_config_template_matches_nunjucks(case: dict[str, Any]):
    assert render_string(onboarding.templates()["CONFIG_TEMPLATE"], case["ctx"]) == case["out"]


# --- reportProviderAPIKeyWarnings -------


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_api_key_warning_for_openai(no_keys):
    assert onboarding.report_provider_api_key_warnings(["openai:gpt-4o"]) == [
        "Warning: OPENAI_API_KEY environment variable is not set.\n"
        "Please set this environment variable like: export OPENAI_API_KEY=<my-api-key>"
    ]


def test_api_key_warnings_multiple_and_objects(no_keys):
    warnings = onboarding.report_provider_api_key_warnings(
        ["anthropic:messages:claude", {"id": "openai:gpt-4o"}, {"config": {}}]
    )
    assert [w.split(" ")[1] for w in warnings] == ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]


def test_api_key_warnings_only_for_missing_keys(no_keys):
    no_keys.setenv("OPENAI_API_KEY", "sk")
    warnings = onboarding.report_provider_api_key_warnings(["openai:x", "vertex:y", "cohere:z"])
    assert [w.split(" ")[1] for w in warnings] == ["GOOGLE_API_KEY", "COHERE_API_KEY"]


# --- interactive wizard -------


def test_wizard_rag_python_writes_context_py(tmp_path: Path, no_keys):
    # action 3 = RAG, language 2 = Python, provider 6 = Local Python script
    result = run(["init"], input_="3\n2\n6\n")
    assert result.exit_code == 0, result.output
    config = load_yaml((tmp_path / "promptfooconfig.yaml").read_text())
    assert config["providers"] == ["file://provider.py"]
    assert config["tests"][0]["vars"]["context"] == "file://context.py"
    assert config["tests"][1]["assert"][1] == {"type": "python", "value": "1 / (len(output) + 1)"}
    templates = onboarding.templates()
    assert (tmp_path / "context.py").read_text() == templates["PYTHON_VAR"]
    assert (tmp_path / "provider.py").read_text() == templates["PYTHON_PROVIDER"]
    assert (tmp_path / "README.md").read_text().startswith("# Promptfoo RAG evaluation")


def test_wizard_agent_openai_uses_tool_config(tmp_path: Path, no_keys):
    result = run(["init"], input_="4\n1\n2\n")
    assert result.exit_code == 0, result.output
    config = load_yaml((tmp_path / "promptfooconfig.yaml").read_text())
    assert config["prompts"] == ["Fulfill this user helpdesk ticket: {{inquiry}}"]
    assert config["providers"][0]["id"] == "openai:chat:gpt-6-sol"
    assert config["providers"][0]["config"]["tools"][0]["function"]["name"] == (
        "get_current_weather"
    )
    assert (tmp_path / "context.py").exists()


def test_wizard_reports_vertex_prefixes(tmp_path: Path, no_keys):
    out = run(["init"], input_="2\n4\n")
    assert out.exit_code == 0, out.output
    config = load_yaml((tmp_path / "promptfooconfig.yaml").read_text())
    assert config["providers"][0] == {
        "id": "vertex:gemini-3.8-flash",
        "config": {"region": "global"},
    }
    assert len(config["prompts"]) == 1  # more than 3 providers: one prompt


def test_wizard_offers_current_models():
    names = [name for name, _ in onboarding.provider_choices("compare")]
    anthropic = dict(onboarding.provider_choices("compare"))[names[2]]
    assert "anthropic:messages:claude-opus-4-1" not in anthropic
    assert "anthropic:messages:claude-haiku-4-5" in anthropic
    cohere = dict(onboarding.provider_choices("compare"))["[Cohere] Command A+, Command A, ..."]
    assert cohere == ["cohere:command-a-plus-05-2026", "cohere:command-a-03-2025"]


def test_wizard_prompts_before_overwriting(tmp_path: Path, no_keys, caplog):
    caplog.set_level("INFO", logger="promptphoo")
    (tmp_path / "promptfooconfig.yaml").write_text("description: keep me")
    (tmp_path / "README.md").write_text("keep readme")
    result = run(["init"], input_="1\n1\nn\nn\n")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "promptfooconfig.yaml").read_text() == "description: keep me"
    messages = [r.getMessage() for r in caplog.records]
    assert "⏩ Skipping README.md" in messages
    assert any(m.startswith("⚠️ Skipping required file promptfooconfig.yaml") for m in messages)


def test_wizard_eof_pauses_with_exit_130(caplog):
    caplog.set_level("INFO", logger="promptphoo")
    result = run(["init"], input_="")
    assert result.exit_code == 130
    assert any(
        "Initialization paused. To continue setup later, use the command: promptphoo init"
        in r.getMessage()
        for r in caplog.records
    )


def test_wizard_redteam_is_not_supported(tmp_path: Path, caplog):
    result = run(["init"], input_="5\n")
    assert result.exit_code == 1
    assert "Red teaming is not supported by promptphoo" in caplog.text
    assert not (tmp_path / "promptfooconfig.yaml").exists()


def test_redteam_directory_confirmation(tmp_path: Path, caplog):
    result = run(["init", "redteam"], input_="y\n")
    assert result.exit_code == 0
    assert 'Use Node promptfoo: "npx promptfoo@latest redteam init"' in caplog.text
    assert not (tmp_path / "redteam").exists()


# --- examples -------


@pytest.mark.parametrize(
    "example",
    [
        variant
        for root in ("github-models", "provider-github-models")
        for variant in (
            root,
            f"{root}/nested",
            f"./{root}",
            f"config-js/../{root}",
            f".\\{root}",
            f"config-js\\..\\{root}",
            f"./{root}/nested",
        )
    ]
    + [
        "%70rovider-github-models",
        "%67ithub-models",
        "%70rovider-github-models%2FREADME.md",
        "%67ithub-models%2Fnested",
    ],
)
def test_unsupported_examples_are_rejected_before_fetching(example: str, tmp_path: Path):
    with respx.mock(assert_all_called=False) as mock:
        with pytest.raises(init_mod.ExampleError, match="GitHub Models has been retired"):
            asyncio.run(init_mod.download_directory(example, str(tmp_path), ["0.122.2"]))
        assert not mock.calls


@pytest.mark.parametrize(
    ("example", "effective"),
    [
        ("config-js", "config-js"),
        ("./config-js", "config-js"),
        ("provider-http/../config-js", "config-js"),
        ("provider-github-models/../config-js", "config-js"),
        ("provider-http/basic", "provider-http/basic"),
        ("provider-http\\basic", "provider-http/basic"),
    ],
)
def test_supported_effective_paths_download(example: str, effective: str, tmp_path: Path):
    with respx.mock() as mock:
        listing = mock.get(f"{GITHUB}/contents/examples/{effective}", params={"ref": "0.122.2"})
        listing.mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "type": "file",
                        "name": "promptfooconfig.yaml",
                        "download_url": "https://example.com/promptfooconfig.yaml",
                    }
                ],
            )
        )
        mock.get("https://example.com/promptfooconfig.yaml").mock(
            return_value=httpx.Response(200, text="description: supported example")
        )
        asyncio.run(init_mod.download_directory(example, str(tmp_path), ["0.122.2"]))
        assert len(mock.calls) == 2
    assert (tmp_path / "promptfooconfig.yaml").read_text() == "description: supported example"


def test_encoded_example_keeps_request_path(tmp_path: Path):
    with respx.mock() as mock:
        route = mock.get(url__regex=r"https://api\.github\.com/.*").mock(
            return_value=httpx.Response(
                200,
                json=[{"type": "file", "name": "a.yaml", "download_url": "https://x.test/a"}],
            )
        )
        mock.get("https://x.test/a").mock(return_value=httpx.Response(200, text="encoded"))
        asyncio.run(init_mod.download_directory("%63onfig-js", str(tmp_path), ["0.122.2"]))
    assert str(route.calls[0].request.url) == (
        f"{GITHUB}/contents/examples/%63onfig-js?ref=0.122.2"
    )


def test_directory_contents_fall_back_to_main_then_fail(tmp_path: Path):
    with respx.mock() as mock:
        route = mock.get(url__regex=r".*/contents/examples/example.*").mock(
            return_value=httpx.Response(404)
        )
        with pytest.raises(
            init_mod.ExampleError,
            match=r"Failed to fetch directory contents for refs: 0\.125\.1 \(404 Not Found\), "
            r"main \(404 Not Found\)",
        ):
            asyncio.run(init_mod.download_directory("example", str(tmp_path)))
        assert [c.request.url.params["ref"] for c in route.calls] == ["0.125.1", "main"]


def test_directory_contents_main_succeeds(tmp_path: Path):
    with respx.mock() as mock:
        mock.get(url__regex=r".*ref=0\.125\.1").mock(return_value=httpx.Response(404))
        mock.get(url__regex=r".*ref=main").mock(return_value=httpx.Response(200, json=[]))
        asyncio.run(init_mod.download_directory("example", str(tmp_path)))
        assert len(mock.calls) == 2


def test_nested_directories_download(tmp_path: Path):
    with respx.mock() as mock:
        mock.get(f"{GITHUB}/contents/examples/ex", params={"ref": "main"}).mock(
            return_value=httpx.Response(200, json=[{"type": "dir", "name": "sub"}])
        )
        mock.get(f"{GITHUB}/contents/examples/ex/sub", params={"ref": "main"}).mock(
            return_value=httpx.Response(
                200, json=[{"type": "file", "name": "f.txt", "download_url": "https://x.test/f"}]
            )
        )
        mock.get("https://x.test/f").mock(return_value=httpx.Response(200, text="nested"))
        asyncio.run(init_mod.download_directory("ex", str(tmp_path), ["main"]))
    assert (tmp_path / "sub" / "f.txt").read_text() == "nested"


def test_download_file_failure(tmp_path: Path):
    with respx.mock() as mock:
        mock.get("https://x.test/f").mock(return_value=httpx.Response(404))
        with pytest.raises(init_mod.ExampleError, match="Failed to download file: Not Found"):
            asyncio.run(init_mod.download_file("https://x.test/f", str(tmp_path / "f")))


def test_download_example_cleans_up_new_directory(tmp_path: Path):
    target = tmp_path / "new"
    with respx.mock() as mock:
        mock.get(url__regex=r".*").mock(return_value=httpx.Response(500))
        with pytest.raises(init_mod.ExampleError, match="^Failed to download example: "):
            asyncio.run(init_mod.download_example("x", str(target), ["main"]))
    assert not target.exists()


def test_download_example_keeps_existing_directory(tmp_path: Path):
    target = tmp_path / "existing"
    target.mkdir()
    with respx.mock() as mock:
        mock.get(url__regex=r".*").mock(return_value=httpx.Response(500))
        with pytest.raises(init_mod.ExampleError):
            asyncio.run(init_mod.download_example("x", str(target), ["main"]))
    assert target.exists()


TREE = {
    "tree": [
        {"path": "examples/provider-github-models/promptfooconfig.yaml", "type": "blob"},
        {"path": "examples/github-models/promptfooconfig.yaml", "type": "blob"},
        {"path": "examples/provider-github-models/nested/promptfooconfig.yaml", "type": "blob"},
        {"path": "examples/config-js/promptfooconfig.js", "type": "blob"},
        {"path": "examples/provider-http/basic/promptfooconfig.yaml", "type": "blob"},
        {"path": "examples/provider-http/README.md", "type": "blob"},
        {"path": "examples/provider-http/basic/server.js", "type": "blob"},
    ]
}


@pytest.mark.parametrize("version_unavailable", [False, True])
def test_examples_list(version_unavailable: bool):
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{GITHUB}/git/trees/0.125.1", params={"recursive": "1"}).mock(
            return_value=httpx.Response(404 if version_unavailable else 200, json=TREE)
        )
        mock.get(f"{GITHUB}/git/trees/main", params={"recursive": "1"}).mock(
            return_value=httpx.Response(200, json=TREE)
        )
        assert asyncio.run(init_mod.get_examples_list()) == ["config-js", "provider-http/basic"]
        assert len(mock.calls) == (2 if version_unavailable else 1)


def test_examples_list_failure_returns_empty(caplog):
    with respx.mock() as mock:
        mock.get(url__regex=r".*").mock(return_value=httpx.Response(500))
        assert asyncio.run(init_mod.get_examples_list()) == []
    assert "Failed to fetch examples list: GitHub tree request failed for ref 'main'" in (
        caplog.text
    )


def _serve_example(mock: respx.MockRouter, name: str, ref: str, files: list[str]) -> None:
    mock.get(f"{GITHUB}/contents/examples/{name}", params={"ref": ref}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {"type": "file", "name": f, "download_url": f"https://x.test/{f}"} for f in files
            ],
        )
    )
    mock.get(url__startswith="https://x.test/").mock(return_value=httpx.Response(200, text="x"))


def test_alias_resolution(tmp_path: Path, caplog):
    caplog.set_level("INFO", logger="promptphoo")
    with respx.mock() as mock:
        _serve_example(mock, "eval-g-eval", "0.125.1", ["promptfooconfig.yaml", "README.md"])
        result = run(["init", "--example", "g-eval"])
    assert result.exit_code == 0
    assert "Note: 'g-eval' has been renamed to 'eval-g-eval'. Using new name." in caplog.text
    assert (tmp_path / "eval-g-eval" / "promptfooconfig.yaml").exists()
    assert "✅ Example project 'eval-g-eval' written to: eval-g-eval" in caplog.text
    assert "View the README at eval-g-eval/README.md or run:" in caplog.text
    assert "`cd eval-g-eval && promptphoo eval`" in caplog.text


def test_replacement_messaging(tmp_path: Path, caplog):
    caplog.set_level("INFO", logger="promptphoo")
    target = init_mod.example_aliases()["integration-hermeneutic"]
    with respx.mock() as mock:
        _serve_example(mock, target, "0.125.1", ["promptfooconfig.yaml"])
        assert run(["init", "--example", "integration-hermeneutic"]).exit_code == 0
    assert f"Downloading '{target}' instead." in caplog.text
    assert "Example docs: https://github.com/promptfoo/promptfoo/tree/main/examples/" in (
        caplog.text
    )


def test_removed_example_uses_legacy_ref(tmp_path: Path, caplog):
    caplog.set_level("INFO", logger="promptphoo")
    with respx.mock() as mock:
        _serve_example(mock, "assistant-cli", "0.120.26", ["README.md"])
        assert run(["init", "--example", "assistant-cli"]).exit_code == 0
    assert "Note: assistant-cli was removed because it is redundant" in caplog.text
    assert "Downloaded legacy example 'assistant-cli' from ref '0.120.26'." in caplog.text
    assert "View the README file at assistant-cli/README.md to get started!" in caplog.text


def test_unsupported_example_exits_1(caplog):
    result = run(["init", "--example", "github-models", "--no-interactive"])
    assert result.exit_code == 1
    assert (
        "Failed to download example: Failed to download example: Example 'github-models' is "
        "unavailable. GitHub Models has been retired, so this example is no longer supported."
    ) in caplog.text


def test_declined_retry_shows_help(tmp_path: Path, caplog):
    caplog.set_level("INFO", logger="promptphoo")
    with respx.mock() as mock:
        mock.get(url__regex=r".*").mock(return_value=httpx.Response(404))
        result = run(["init", "--example", "config-%ZZ"], input_="n\n")
    assert result.exit_code == 0
    assert "Failed to fetch directory contents for refs:" in caplog.text
    assert "No example downloaded. To get started, try:\n\n  promptphoo init --example" in (
        caplog.text
    )
    assert "✅ Example project" not in caplog.text
    assert not os.path.exists(tmp_path / "promptfooconfig.yaml")


def test_example_flag_without_name_selects_from_list(tmp_path: Path):
    with respx.mock() as mock:
        mock.get(f"{GITHUB}/git/trees/0.125.1", params={"recursive": "1"}).mock(
            return_value=httpx.Response(200, json=TREE)
        )
        # "1" = None (initialize with dummy files) -> falls through to the wizard; EOF pauses it
        result = run(["init", "--example", "--no-interactive"], input_="1\n")
    assert result.exit_code == 0
    assert (tmp_path / "promptfooconfig.yaml").exists()
