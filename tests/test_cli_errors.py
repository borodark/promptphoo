"""User mistakes in a config print a message and exit 1, without a traceback (D67)."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from promptphoo import db
from promptphoo.cli.errors import ConfigLoadError
from promptphoo.cli.eval import _recoverable_errors
from promptphoo.cli.main import build_cli

GOOD_HEAD = "prompts:\n  - id: p1\n    raw: 'Hi {{x}}'\nproviders: [echo]\n"

CASES = {
    "assertion_without_type": (
        GOOD_HEAD + "tests:\n  - vars: {x: 1}\n    assert:\n      - type: python\n"
        "      - value: file://script.py\n",
        "Missing required 'type' property",
    ),
    "unknown_test_provider": (
        GOOD_HEAD + "tests:\n  - vars: {x: 1}\n    providers: [openai]\n",
        'references provider "openai" which does not exist',
    ),
    "unknown_test_prompt": (
        GOOD_HEAD + "tests:\n  - vars: {x: 1}\n    prompts: [p2]\n",
        'references prompt "p2" which does not exist',
    ),
    "unknown_provider_id": (
        "prompts: [hi]\nproviders: [nope:thing]\ntests:\n  - vars: {x: 1}\n",
        "Could not identify provider: nope:thing",
    ),
    "yaml_syntax_error": ("prompts: [hi\n", "expected ',' or ']'"),
    "missing_prompt_file": (
        "prompts: [file://missing.txt]\nproviders: [echo]\ntests:\n  - vars: {x: 1}\n",
        "missing.txt",
    ),
}


def _invoke(tmp_path, monkeypatch, args, config):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bad.yaml").write_text(config)
    try:
        return CliRunner().invoke(build_cli(args), args, catch_exceptions=False)
    finally:
        db.close_db()


@pytest.mark.parametrize("case", sorted(CASES))
def test_eval_reports_config_mistakes_without_traceback(tmp_path, monkeypatch, caplog, case):
    config, message = CASES[case]
    result = _invoke(tmp_path, monkeypatch, ["eval", "-c", "bad.yaml", "--no-write"], config)
    assert result.exit_code == 1
    text = result.output + caplog.text
    assert message in text
    assert "Traceback" not in text


@pytest.mark.parametrize("case", ["assertion_without_type", "unknown_test_prompt"])
def test_generate_reports_config_mistakes_without_traceback(tmp_path, monkeypatch, caplog, case):
    config, message = CASES[case]
    args = ["generate", "dataset", "-c", "bad.yaml"]
    result = _invoke(tmp_path, monkeypatch, args, config)
    assert result.exit_code == 1
    text = result.output + caplog.text
    assert message in text
    assert "Traceback" not in text


def test_watch_survives_a_bad_edit():
    # The watch loop's rerun catches the same tuple as the first run.
    assert ConfigLoadError in _recoverable_errors()
