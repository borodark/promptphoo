"""Prompt files that fall back to their raw text warn instead of doing so silently (D69)."""

from __future__ import annotations

import json
import logging

import pytest

from promptphoo.prompts.processors import (
    process_csv_prompts,
    process_json_file,
    process_yaml_file,
)


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_invalid_yaml_warns_and_keeps_text(tmp_path, caplog):
    path = tmp_path / "p.yaml"
    path.write_text("- role: user\n  content: [unclosed\n")
    caplog.set_level(logging.DEBUG)
    [prompt] = process_yaml_file(str(path), {})
    assert prompt["raw"] == path.read_text()
    [message] = _warnings(caplog)
    assert "is not valid YAML" in message and "used as the prompt as is" in message


@pytest.mark.parametrize(
    ("name", "text", "processor"),
    [
        (
            "p.yaml",
            "{% for m in msgs %}\n- role: user\n  content: [{{ m }}\n{% endfor %}\n",
            "yaml",
        ),
        (
            "p.json",
            '[{% for m in msgs %}{"role": "user", "content": "{{ m }}"}{% endfor %}]',
            "json",
        ),
    ],
)
def test_templated_prompt_file_falls_back_quietly(tmp_path, caplog, name, text, processor):
    path = tmp_path / name
    path.write_text(text)
    process = process_yaml_file if processor == "yaml" else process_json_file
    [prompt] = process(str(path), {})
    assert prompt["raw"] == text
    assert _warnings(caplog) == []


def test_invalid_json_warns(tmp_path, caplog):
    path = tmp_path / "p.json"
    path.write_text('[{"role": "user", "content": "hi"},]')
    [prompt] = process_json_file(str(path), {})
    assert prompt["raw"] == path.read_text()
    assert "is not valid JSON" in _warnings(caplog)[0]


@pytest.mark.parametrize("ext", ["json", "yaml"])
def test_missing_nested_reference_warns(tmp_path, monkeypatch, caplog, ext):
    monkeypatch.chdir(tmp_path)
    messages = [{"role": "user", "content": "file://missing.txt"}]
    text = (
        json.dumps(messages) if ext == "json" else "- role: user\n  content: file://missing.txt\n"
    )
    path = tmp_path / f"p.{ext}"
    path.write_text(text)
    process = process_json_file if ext == "json" else process_yaml_file
    [prompt] = process(str(path), {})
    assert prompt["raw"] == text
    [message] = _warnings(caplog)
    assert "could not load a file:// reference" in message


def test_valid_files_do_not_warn(tmp_path, caplog):
    (tmp_path / "p.json").write_text('[{"role": "user", "content": "hi {{q}}"}]')
    (tmp_path / "p.yaml").write_text("- role: user\n  content: hi {{q}}\n")
    process_json_file(str(tmp_path / "p.json"), {})
    process_yaml_file(str(tmp_path / "p.yaml"), {})
    assert _warnings(caplog) == []


def test_malformed_csv_warns_and_splits_lines(tmp_path, caplog):
    path = tmp_path / "p.csv"
    path.write_text('prompt,label\n"unterminated, quote\nsecond line\n')
    prompts = process_csv_prompts(str(path), {"raw": str(path), "label": None})
    assert [p["raw"] for p in prompts] == ["prompt,label", '"unterminated, quote', "second line"]
    assert "is not valid CSV" in _warnings(caplog)[0]
