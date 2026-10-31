"""`eval` flags. Ports the tag, var, suggestion and option cases of test/commands/eval.test.ts
and test/evaluator/generatedPrompts.test.ts / test/suggestions.test.ts."""

from __future__ import annotations

import json
import logging

import pytest
from click.testing import CliRunner

from promptphoo import db
from promptphoo.cli.eval import resolve_suggestion_options
from promptphoo.cli.main import build_cli
from promptphoo.suggestions import (
    PromptSuggestionsRejectedError,
    generate_prompts,
    maybe_add_generated_prompts,
)

CONFIG = """\
description: flags
prompts:
  - "Hi {{name}}"
providers:
  - echo
tests:
  - vars: {name: a}
  - vars: {name: b}
"""


def _eval(tmp_path, monkeypatch, *extra, config=CONFIG):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "promptfooconfig.yaml").write_text(config)
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-table", *extra]
    try:
        result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)
        out = tmp_path / "out.json"
        data = json.loads(out.read_text()) if out.exists() else None
    finally:
        db.close_db()
    return result, data


def test_runtime_tags_merge_over_config_and_command_line_tags(tmp_path, monkeypatch):
    config = CONFIG + "tags: {team: a, keep: x}\ncommandLineOptions:\n  tags: {team: b, clo: y}\n"
    result, data = _eval(
        tmp_path, monkeypatch, "--tag", "team=c", "--tag", "eq=a=b", "--tag", "empty=",
        config=config,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert data["config"]["tags"] == {"team": "c", "keep": "x", "clo": "y", "eq": "a=b",
                                      "empty": ""}  # fmt: skip


def test_var_description_repeat_and_prefix(tmp_path, monkeypatch):
    config = CONFIG.replace('"Hi {{name}}"', '"Hi {{name}}{{extra}}"')
    result, data = _eval(
        tmp_path, monkeypatch, "--var", "name=override", "--var", "extra=!",
        "--description", "from cli", "--repeat", "2", "--prompt-prefix", "[",
        "--prompt-suffix", "]", config=config,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    rows = data["results"]["results"]
    assert len(rows) == 4
    # --var lands in defaultTest.vars, so the tests' own vars still win.
    assert sorted(r["response"]["output"] for r in rows) == ["[Hi a!]", "[Hi a!]", "[Hi b!]",
                                                             "[Hi b!]"]  # fmt: skip
    assert data["config"]["description"] == "from cli"


def test_multiple_outputs_and_verbose(tmp_path, monkeypatch):
    result, data = _eval(tmp_path, monkeypatch, "-o", "out.yaml", "out.csv", "--verbose")
    assert result.exit_code == 0, result.output
    # Repeated variadic -o flags accumulate, as commander's do.
    assert data is not None
    assert (tmp_path / "out.yaml").exists() and (tmp_path / "out.csv").exists()


def test_extension_option_accepted(tmp_path, monkeypatch):
    (tmp_path / "hook.py").write_text(
        "def afterAll(context, extra):\n"
        "    open('after.txt', 'w').write(str(len(context['results'])))\n"
    )
    result, _ = _eval(tmp_path, monkeypatch, "-x", "file://hook.py:afterAll")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "after.txt").read_text() == "2"


@pytest.mark.parametrize("value", ["0", "51", "x"])
def test_suggest_prompts_validation(tmp_path, monkeypatch, value):
    result, _ = _eval(tmp_path, monkeypatch, "--suggest-prompts", value)
    assert result.exit_code == 2


def test_resolve_suggestion_options():
    out = resolve_suggestion_options({"suggest_prompts": 3}, {}, {"generateSuggestions": False})
    assert out == {"generateSuggestions": True, "suggestionsCount": 3}
    assert resolve_suggestion_options({}, {}, {}) == {}
    assert resolve_suggestion_options({}, {"generateSuggestions": False}, {}) == {
        "generateSuggestions": False
    }
    assert resolve_suggestion_options({}, {}, {"generateSuggestions": True}) == {
        "generateSuggestions": True
    }
    assert resolve_suggestion_options(
        {}, {"generateSuggestions": True, "suggestionsCount": 4}, {"suggestionsCount": 9}
    ) == {"generateSuggestions": True, "suggestionsCount": 4}


class SuggestionsProvider:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.payloads: list[str] = []

    def id(self):
        return "suggest"

    async def call_api(self, prompt, context=None, options=None):
        self.payloads.append(prompt)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def suggestions(monkeypatch):
    def install(outputs, answers=()):
        provider = SuggestionsProvider(outputs)

        async def defaults(env=None):
            return {"suggestionsProvider": provider}

        answer_list = list(answers)
        monkeypatch.setattr("promptphoo.providers.defaults.get_default_providers", defaults)
        monkeypatch.setattr("promptphoo.suggestions.prompt_user", lambda q: answer_list.pop(0))
        return provider

    return install


async def test_generate_prompts_payload_and_partial_failure(suggestions, caplog):
    from promptphoo.prompts.grading import SUGGEST_PROMPTS_SYSTEM_MESSAGE

    provider = suggestions(
        [{"output": "v1", "tokenUsage": {"total": 3}}, RuntimeError("boom"), {"output": ""}]
    )
    with caplog.at_level(logging.WARNING, logger="promptphoo"):
        out = await generate_prompts("Base prompt", 3)
    assert out["prompts"] == ["v1"]
    assert out["tokensUsed"]["total"] == 3
    assert json.loads(provider.payloads[0]) == [
        SUGGEST_PROMPTS_SYSTEM_MESSAGE,
        {"role": "user", "content": "Generate a variant for the following prompt:"},
        {"role": "user", "content": "Base prompt"},
    ]
    assert "Variant 2/3 failed: boom" in caplog.text
    assert "Variant 3/3 failed: Unknown error" in caplog.text
    assert "Generated 1/3 prompt variants; 2 failed." in caplog.text


async def test_generate_prompts_errors(suggestions):
    suggestions([{"error": "e1"}, {"error": "e2"}])
    out = await generate_prompts("p", 2)
    assert out["error"] == "Variant 1: e1; Variant 2: e2"
    assert "between 1 and 50 (got 0)" in (await generate_prompts("p", 0))["error"]


async def test_maybe_add_generated_prompts(suggestions):
    suite = {"prompts": [{"raw": "orig", "label": "orig"}]}
    await maybe_add_generated_prompts(suite, {})
    assert len(suite["prompts"]) == 1

    suggestions([{"output": "A"}, {"output": "B"}], answers=["y", "n"])
    await maybe_add_generated_prompts(suite, {"generateSuggestions": True, "suggestionsCount": 2})
    assert suite["prompts"][1:] == [{"raw": "A", "label": "A"}]

    suggestions([{"output": "C"}], answers=[""])
    with pytest.raises(PromptSuggestionsRejectedError):
        await maybe_add_generated_prompts(suite, {"generateSuggestions": True})

    suggestions([{"error": "down"}])
    with pytest.raises(RuntimeError, match="Failed to generate prompts: Variant 1: down"):
        await maybe_add_generated_prompts(suite, {"generateSuggestions": True})


def test_suggest_prompts_end_to_end(tmp_path, monkeypatch, suggestions):
    suggestions([{"output": "Yo {{name}}"}], answers=["yes"])
    result, data = _eval(tmp_path, monkeypatch, "--suggest-prompts", "1")
    assert result.exit_code == 0, result.output
    outputs = sorted(r["response"]["output"] for r in data["results"]["results"])
    assert outputs == ["Hi a", "Hi b", "Yo a", "Yo b"]


def test_suggest_prompts_rejected_exits_1(tmp_path, monkeypatch, suggestions, caplog):
    suggestions([{"output": "Yo {{name}}"}], answers=["n"])
    result, _ = _eval(tmp_path, monkeypatch, "--suggest-prompts", "1")
    assert result.exit_code == 1
    assert "No prompts selected. Aborting." in caplog.text
