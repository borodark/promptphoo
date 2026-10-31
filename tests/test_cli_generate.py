"""`generate dataset` / `generate assertions`. Ports test/commands/generate/dataset.test.ts and
test/commands/generate/assertions.test.ts, plus CLI runs against a fake Python provider."""

from __future__ import annotations

import json
import logging

import pytest
from click.testing import CliRunner

from promptphoo.cli.generate import (
    GenerateError,
    do_generate_assertions,
    do_generate_dataset,
    serialize_object_array_as_csv,
)
from promptphoo.cli.main import build_cli
from promptphoo.yamlload import load_yaml

CONFIG = """\
prompts:
  - "Answer {{question}} about {{topic}}"
tests:
  - vars: {topic: cats, question: why}
"""

PROVIDER = """\
import json


def call_api(prompt, options, context):
    if "user personas" in prompt:
        return {"output": json.dumps({"personas": ["Student"]})}
    if "Fully embody this persona" in prompt:
        return {"output": json.dumps({"vars": [{"topic": "dogs", "question": "how"},
                                               {"topic": "fish", "question": 'say "hi"'}]})}
    if "metric design" in prompt:
        return {"output": json.dumps({"questions": [{"label": "Short", "question": "Short?"}]})}
    return {"output": "return {'pass': len(output) < 10, 'score': 1.0}"}
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "promptfooconfig.yaml").write_text(CONFIG)
    (tmp_path / "fake.py").write_text(PROVIDER)
    return tmp_path


@pytest.fixture
def mock_synthesis(monkeypatch):
    results = {"value": [{"var1": "value1"}, {"var1": "value2"}]}

    async def fake(test_suite, options):
        if isinstance(results["value"], Exception):
            raise results["value"]
        return results["value"]

    monkeypatch.setattr("promptphoo.synthesis.dataset.synthesize_from_test_suite", fake)
    monkeypatch.setattr("promptphoo.synthesis.assertions.synthesize_from_test_suite", fake)
    return results


def _opts(**kw):
    return {"config": "promptfooconfig.yaml", "num_personas": "5",
            "num_test_cases_per_persona": "3", "write": False, "default_config": {},
            "default_config_path": "promptfooconfig.yaml", **kw}  # fmt: skip


async def test_dataset_yaml_and_csv_output(project, mock_synthesis):
    await do_generate_dataset(_opts(output="output.yaml"))
    assert load_yaml((project / "output.yaml").read_text()) == {
        "tests": [{"vars": {"var1": "value1"}}, {"vars": {"var1": "value2"}}]
    }
    await do_generate_dataset(_opts(output="output.csv"))
    assert (project / "output.csv").read_text() == 'var1\n"value1"\n"value2"\n'


@pytest.mark.parametrize("output", ["output.txt", "output"])
async def test_dataset_unsupported_output(project, mock_synthesis, output):
    with pytest.raises(GenerateError, match=f"Unsupported output file type: {output}"):
        await do_generate_dataset(_opts(output=output))


async def test_dataset_write_to_config(project, mock_synthesis):
    await do_generate_dataset(_opts(write=True))
    written = load_yaml((project / "promptfooconfig.yaml").read_text())
    assert written["tests"] == [
        {"vars": {"topic": "cats", "question": "why"}},
        {"vars": {"var1": "value1"}},
        {"vars": {"var1": "value2"}},
    ]


async def test_no_config_and_synthesis_errors(project, mock_synthesis):
    with pytest.raises(GenerateError, match="Could not find a config file"):
        await do_generate_dataset(_opts(config=None, default_config_path=None))
    with pytest.raises(GenerateError, match="Could not find a config file"):
        await do_generate_assertions(_opts(config=None, default_config_path=None))
    mock_synthesis["value"] = RuntimeError("Synthesis failed")
    with pytest.raises(RuntimeError, match="Synthesis failed"):
        await do_generate_dataset(_opts(output="output.yaml"))
    with pytest.raises(RuntimeError, match="Synthesis failed"):
        await do_generate_assertions(_opts(output="output.yaml"))


async def test_assertions_yaml_output_and_write(project, mock_synthesis):
    mock_synthesis["value"] = [{"type": "pi", "metric": "M", "value": "Q?"}]
    await do_generate_assertions(_opts(output="output.yaml"))
    assert load_yaml((project / "output.yaml").read_text()) == {
        "assert": [{"type": "pi", "metric": "M", "value": "Q?"}]
    }
    for output in ("output.csv", "output"):
        with pytest.raises(GenerateError, match=f"Unsupported output file type: {output}"):
            await do_generate_assertions(_opts(output=output))
    await do_generate_assertions(_opts(write=True))
    written = load_yaml((project / "promptfooconfig.yaml").read_text())
    assert written["defaultTest"] == {"assert": [{"type": "pi", "metric": "M", "value": "Q?"}]}


def test_serialize_object_array_as_csv():
    assert serialize_object_array_as_csv([{"a": 'x"y', "b": 1}, {"a": "z", "b": True}]) == (
        'a,b\n"x""y","1"\n"z","true"\n'
    )


def _cli(*args):
    return CliRunner().invoke(build_cli(list(args)), list(args), catch_exceptions=False)


def test_generate_dataset_cli(project, caplog):
    with caplog.at_level(logging.INFO, logger="promptphoo"):
        result = _cli("generate", "dataset", "--provider", "file://fake.py", "--numPersonas", "1",
                      "--numTestCasesPerPersona", "2", "-w")  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Cache is disabled." in caplog.text
    assert f"Wrote 2 new test cases to {project / 'promptfooconfig.yaml'}" in caplog.text
    tests = load_yaml((project / "promptfooconfig.yaml").read_text())["tests"]
    assert tests[1:] == [
        {"vars": {"topic": "dogs", "question": "how"}},
        {"vars": {"topic": "fish", "question": 'say "hi"'}},
    ]


def test_generate_assertions_cli(project, caplog):
    with caplog.at_level(logging.INFO, logger="promptphoo"):
        result = _cli("generate", "assertions", "--provider", "file://fake.py", "-t", "g-eval",
                      "--numAssertions", "1", "-o", "asserts.yaml")  # fmt: skip
    assert result.exit_code == 0, result.output
    assert load_yaml((project / "asserts.yaml").read_text()) == {
        "assert": [{"type": "python", "metric": "Short",
                    "value": "return {'pass': len(output) < 10, 'score': 1.0}"}]
    }  # fmt: skip
    assert "Wrote 1 new assertions to asserts.yaml" in caplog.text


def test_generate_assertions_rejects_unknown_type(project):
    result = _cli("generate", "assertions", "-t", "bogus")
    assert result.exit_code == 2
    assert "Option --type must be one of: pi, g-eval, llm-rubric." in result.output


def test_generate_without_config_exits_1(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    result = _cli("generate", "dataset")
    assert result.exit_code == 1
    assert "Could not find a config file" in caplog.text


def test_generate_uses_default_synthesize_provider(project, monkeypatch):
    calls: list[str] = []

    class Provider:
        config: dict = {}

        def id(self):
            return "default"

        async def call_api(self, prompt, context=None, options=None):
            calls.append(prompt)
            if "user personas" in prompt:
                return {"output": json.dumps(["P"])}
            return {"output": {"vars": [{"topic": "t", "question": "q"}]}}

    async def defaults(env=None):
        return {"synthesizeProvider": Provider()}

    monkeypatch.setattr("promptphoo.providers.defaults.get_default_providers", defaults)
    result = _cli("generate", "dataset", "--numPersonas", "1", "--numTestCasesPerPersona", "1",
                  "-o", "out.yaml")  # fmt: skip
    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert load_yaml((project / "out.yaml").read_text()) == {
        "tests": [{"vars": {"topic": "t", "question": "q"}}]
    }
