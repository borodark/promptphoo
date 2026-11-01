"""Regression tests for the security fixes, built from the payloads that proved each hole.

D73: a model output kept with storeOutputAs is inserted literally in the next prompt.
D74: in safe mode, test files are data: no code, no file access, no script providers.
D75: the state directory's database, cache, logs and blobs are owner-only.
D76: var values are rendered without the environment.
D77: CSV and JSONL datasets cannot carry code unless trusted.
"""

from __future__ import annotations

import json
import os
import stat
import textwrap

import pytest
from click.testing import CliRunner

from promptphoo import db
from promptphoo.cli.main import build_cli

# The model answers the first prompt with the payload, read from a file so that it never passes
# through the var pipeline itself, and echoes every other prompt.
PAYLOAD_PROVIDER = """\
from pathlib import Path


def call_api(prompt, options, context):
    if prompt == "first":
        return {"output": Path("payload.txt").read_text()}
    return {"output": prompt}
"""


def _eval(tmp_path, monkeypatch, config: str, *extra: str, files: dict[str, str] | None = None):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "promptfooconfig.yaml").write_text(textwrap.dedent(config))
    for name, text in (files or {}).items():
        (tmp_path / name).write_text(textwrap.dedent(text))
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-table", "--no-cache"]
    args += list(extra)
    try:
        result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)
    finally:
        db.close_db()
    out = tmp_path / "out.json"
    rows = json.loads(out.read_text())["results"]["results"] if out.exists() else []
    return result, rows


# --- D73: stored model output is not a template ---------------------------------------------

STORE_CONFIG = """\
prompts: ["{{ q }}"]
providers: [file://provider.py]
tests:
  - vars: {q: first}
    options: {storeOutputAs: prev}
  - vars: {q: "next: {{ prev }}"}
"""


@pytest.mark.parametrize(
    "payload",
    [
        "{{ env.PP_TEST_SECRET }}",  # reads a secret into the next prompt in Node
        "file:///etc/hostname",  # loaded as a file in Node
        "{{ q }}",  # substituted with another var in Node
        "{% for i in range(3) %}x{% endfor %}",  # runs template code in Node
    ],
)
def test_stored_output_is_inserted_literally(tmp_path, monkeypatch, payload):
    monkeypatch.setenv("PP_TEST_SECRET", "s3cr3t-value")
    files = {"provider.py": PAYLOAD_PROVIDER, "payload.txt": payload}
    result, rows = _eval(tmp_path, monkeypatch, STORE_CONFIG, files=files)
    assert result.exit_code == 0, result.output
    second = next(r for r in rows if r["testIdx"] == 1)
    assert second["prompt"]["raw"] == "next: " + payload
    assert "s3cr3t-value" not in json.dumps(rows)


def test_stored_output_still_loses_one_trailing_newline(tmp_path, monkeypatch):
    files = {"provider.py": PAYLOAD_PROVIDER, "payload.txt": "ok\n"}
    result, rows = _eval(tmp_path, monkeypatch, STORE_CONFIG, files=files)
    assert result.exit_code == 0, result.output
    assert next(r for r in rows if r["testIdx"] == 1)["prompt"]["raw"] == "next: ok"


# --- D74: safe mode -------------------------------------------------------------------------

BASE_CONFIG = 'prompts: ["Hi {{ name }}"]\nproviders: [echo]\n'
CSV_CONFIG = BASE_CONFIG + "tests: file://tests.csv\n"

MARKER = "__import__('pathlib').Path('PWNED').write_text('ran') or True"


def test_csv_python_assertion_refused_by_default(tmp_path, monkeypatch, caplog):
    # D77: a dataset cannot carry code even outside safe mode.
    csv = f'name,__expected\nAnn,"python: {MARKER}"\n'
    result, rows = _eval(tmp_path, monkeypatch, CSV_CONFIG, files={"tests.csv": csv})
    assert result.exit_code == 1
    text = result.output + caplog.text
    assert "has a python assertion" in text and "PROMPTPHOO_TRUST_TEST_DATA" in text
    assert not (tmp_path / "PWNED").exists() and rows == []


def test_trusted_csv_python_assertion_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTPHOO_TRUST_TEST_DATA", "true")
    csv = f'name,__expected\nAnn,"python: {MARKER}"\n'
    result, _ = _eval(tmp_path, monkeypatch, CSV_CONFIG, files={"tests.csv": csv})
    assert result.exit_code == 0, result.output
    assert (tmp_path / "PWNED").exists()


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("tests.csv", "name,__expected\nAnn,file://evil.py\n"),
        ("tests.csv", "name\nfile://evil.py:get_var\n"),
        ("tests.jsonl", '{"vars": {"name": "a"}, "options": {"provider": "exec:touch PWNED"}}\n'),
        (
            "tests.jsonl",
            '{"vars": {"name": "a"}, "assert": [{"type": "not-python", "value": "1"}]}\n',
        ),
    ],
)
def test_dataset_code_refused_by_default(tmp_path, monkeypatch, caplog, name, body):
    (tmp_path / "evil.py").write_text(
        "open('PWNED', 'w').write('ran')\n"
        "def get_assert(output, context):\n    return True\n"
        "def get_var(var_name, prompt, other_vars):\n    return {'output': 'x'}\n"
    )
    config = BASE_CONFIG + f"tests: file://{name}\n"
    result, rows = _eval(tmp_path, monkeypatch, config, files={name: body})
    assert result.exit_code == 1
    assert "cannot run code (D77)" in result.output + caplog.text
    assert not (tmp_path / "PWNED").exists() and rows == []


def test_dataset_plain_file_reference_still_loads(tmp_path, monkeypatch):
    (tmp_path / "doc.txt").write_text("from a file")
    csv = "name\nfile://doc.txt\n"
    result, rows = _eval(tmp_path, monkeypatch, CSV_CONFIG, files={"tests.csv": csv})
    assert result.exit_code == 0, result.output
    assert rows[0]["response"]["output"] == "Hi from a file"


def test_yaml_test_file_stays_trusted_outside_safe_mode(tmp_path, monkeypatch):
    body = "- vars: {name: a}\n  assert: [{type: python, value: 'True'}]\n"
    config = BASE_CONFIG + "tests: file://tests.yaml\n"
    result, rows = _eval(tmp_path, monkeypatch, config, files={"tests.yaml": body})
    assert result.exit_code == 0, result.output
    assert rows[0]["success"]


# --- D76: var values cannot read the environment ----------------------------------------------


def test_dataset_var_cannot_read_environment(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("PP_TEST_SECRET", "s3cr3t-value")
    csv = 'name\n"{{ env.PP_TEST_SECRET }}"\n'
    result, rows = _eval(tmp_path, monkeypatch, CSV_CONFIG, files={"tests.csv": csv})
    assert result.exit_code == 0, result.output
    assert rows[0]["response"]["output"] == "Hi "
    assert "s3cr3t-value" not in json.dumps(rows)
    assert "PROMPTPHOO_VARS_READ_ENV" in caplog.text


def test_vars_read_env_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("PP_TEST_SECRET", "s3cr3t-value")
    monkeypatch.setenv("PROMPTPHOO_VARS_READ_ENV", "true")
    csv = 'name\n"{{ env.PP_TEST_SECRET }}"\n'
    result, rows = _eval(tmp_path, monkeypatch, CSV_CONFIG, files={"tests.csv": csv})
    assert rows[0]["response"]["output"] == "Hi s3cr3t-value"


def test_config_vars_and_prompts_still_read_environment(tmp_path, monkeypatch):
    # Config-authored env templates render at load and in prompts, as in Node.
    monkeypatch.setenv("PP_TEST_VALUE", "from-env")
    config = (
        'prompts: ["{{ name }} - {{ env.PP_TEST_VALUE }}"]\nproviders: [echo]\n'
        'tests: [{vars: {name: "{{ env.PP_TEST_VALUE }}"}}]\n'
    )
    result, rows = _eval(tmp_path, monkeypatch, config)
    assert result.exit_code == 0, result.output
    assert rows[0]["response"]["output"] == "from-env - from-env"


@pytest.mark.parametrize(
    ("csv", "message"),
    [
        (f'name,__expected\nAnn,"python: {MARKER}"\n', "has a python assertion"),
        (f'name,__expected\nAnn,"not-python: {MARKER}"\n', "has a python assertion"),
        ("name,__expected\nAnn,file://evil.py\n", "has a python assertion"),
        ("name\nfile:///etc/hostname\n", 'refers to "file:///etc/hostname" at vars.name'),
    ],
)
def test_safe_mode_refuses_code_and_files_in_csv(tmp_path, monkeypatch, caplog, csv, message):
    (tmp_path / "evil.py").write_text(
        "def get_assert(output, context):\n    open('PWNED', 'w').write('ran')\n    return True\n"
    )
    result, rows = _eval(tmp_path, monkeypatch, CSV_CONFIG, "--safe-mode", files={"tests.csv": csv})
    assert result.exit_code == 1
    assert message in result.output + caplog.text and "Safe mode:" in result.output + caplog.text
    assert "Traceback" not in result.output + caplog.text
    assert not (tmp_path / "PWNED").exists() and rows == []


YAML_TEST_CASES = {
    "exec_grader": ("- vars: {name: a}\n  options: {provider: 'exec:touch PWNED'}\n", "exec:"),
    "exec_assert_provider": (
        "- vars: {name: a}\n  assert:\n    - type: llm-rubric\n      value: ok\n"
        "      provider: {text: {id: 'exec:touch PWNED'}}\n",
        "script provider",
    ),
    "python_provider": ("- vars: {name: a}\n  provider: file://evil.py\n", "evil.py"),
    "transform_file": ("- vars: {name: a}\n  options: {transform: file://evil.py}\n", "evil.py"),
    "vars_file": ("- vars: file://vars.yaml\n", "loads its vars"),
    "vars_list": ("- vars: [vars.yaml]\n", "loads its vars"),
    "ref": ("- vars: {name: {$ref: /etc/hostname}}\n", "$ref"),
    "nested_test_file": ("- other-tests.yaml\n", "refers to another test file"),
    "assert_set": (
        "- vars: {name: a}\n  assert:\n    - type: assert-set\n      assert:\n"
        "        - type: python\n          value: 'True'\n",
        "has a python assertion",
    ),
}


@pytest.mark.parametrize("case", sorted(YAML_TEST_CASES))
@pytest.mark.parametrize("ext", ["yaml", "json"])
def test_safe_mode_refuses_code_and_files_in_test_files(tmp_path, monkeypatch, caplog, case, ext):
    from promptphoo.yamlload import load_yaml

    text, message = YAML_TEST_CASES[case]
    body = text if ext == "yaml" else json.dumps(load_yaml(text))
    config = f'prompts: ["Hi {{{{ name }}}}"]\nproviders: [echo]\ntests: file://tests.{ext}\n'
    files = {f"tests.{ext}": body, "evil.py": "open('PWNED', 'w').write('ran')\n"}
    result, rows = _eval(tmp_path, monkeypatch, config, "--safe-mode", files=files)
    assert result.exit_code == 1, result.output
    assert message in result.output + caplog.text
    assert not (tmp_path / "PWNED").exists() and rows == []


def test_safe_mode_from_the_config_itself(tmp_path, monkeypatch, caplog):
    config = CSV_CONFIG + "commandLineOptions: {safeMode: true}\n"
    csv = f'name,__expected\nAnn,"python: {MARKER}"\n'
    result, _ = _eval(tmp_path, monkeypatch, config, files={"tests.csv": csv})
    assert result.exit_code == 1
    assert "has a python assertion" in result.output + caplog.text
    assert not (tmp_path / "PWNED").exists()


def test_safe_mode_checks_scenario_test_files(tmp_path, monkeypatch, caplog):
    config = """\
    prompts: ["Hi {{ name }}"]
    providers: [echo]
    scenarios:
      - config: [{vars: {name: a}}]
        tests: file://scenario-tests.yaml
    """
    files = {"scenario-tests.yaml": "- assert: [{type: python, value: 'True'}]\n"}
    result, _ = _eval(tmp_path, monkeypatch, config, "--safe-mode", files=files)
    assert result.exit_code == 1
    assert "has a python assertion" in result.output + caplog.text


def test_safe_mode_allows_plain_data(tmp_path, monkeypatch):
    csv = (
        "name,note,__expected\n"
        'Ann,"exec: is just text here",contains: Ann\n'
        'Bob,"python: also text",icontains: bob\n'
    )
    result, rows = _eval(tmp_path, monkeypatch, CSV_CONFIG, "--safe-mode", files={"tests.csv": csv})
    assert result.exit_code == 0, result.output
    assert [r["success"] for r in rows] == [True, True]


def test_safe_mode_refuses_inline_python_but_runs_file_assertions(tmp_path, monkeypatch):
    # Config-inline tests stay trusted, except inline code, as upstream refuses inline JS.
    config = f"""\
    prompts: ["Hi {{{{ name }}}}"]
    providers: [echo]
    tests:
      - vars: {{name: Ann}}
        assert:
          - type: python
            value: "{MARKER}"
          - type: python
            value: file://check.py
    """
    files = {"check.py": "def get_assert(output, context):\n    return 'Ann' in output\n"}
    result, rows = _eval(tmp_path, monkeypatch, config, "--safe-mode", files=files)
    assert not (tmp_path / "PWNED").exists()
    inline, from_file = rows[0]["gradingResult"]["componentResults"]
    assert not inline["pass"] and "disabled in safe mode" in inline["reason"]
    assert from_file["pass"]


# --- D75: permissions -----------------------------------------------------------------------


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_new_state_directory_and_files_are_owner_only(tmp_path, monkeypatch):
    home = tmp_path / "fresh" / "state"
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(home))
    monkeypatch.delenv("IS_TESTING", raising=False)
    monkeypatch.delenv("PROMPTFOO_DISABLE_DEBUG_LOG", raising=False)
    old_umask = os.umask(0o022)
    try:
        config = 'prompts: ["Hi {{ name }}"]\nproviders: [echo]\ntests: [{vars: {name: a}}]\n'
        monkeypatch.chdir(tmp_path)
        (tmp_path / "promptfooconfig.yaml").write_text(config)
        args = ["eval", "-c", "promptfooconfig.yaml", "--no-table"]
        try:
            result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)
        finally:
            db.close_db()
        assert result.exit_code == 0, result.output
        import asyncio

        from promptphoo import cache as cache_mod
        from promptphoo.blobs.filesystem_provider import FilesystemBlobStorageProvider

        FilesystemBlobStorageProvider().store(b"blob", "text/plain")
        monkeypatch.setattr(cache_mod, "_instance", None)
        monkeypatch.setattr(cache_mod, "_enabled", True)
        asyncio.run(cache_mod.get_cache().set("key", "value"))
        asyncio.run(cache_mod._get_instance().flush())
        from promptphoo import logfiles

        logfiles.initialize_run_logging()  # main() does this; CliRunner skips it
        logfiles.close_run_logging()
    finally:
        os.umask(old_umask)
    assert _mode(home) == 0o700
    assert _mode(home / "promptfoo.db") == 0o600
    assert _mode(home / "cache" / "cache-py.json") == 0o600
    assert _mode(home / "logs") == 0o700
    logs = list((home / "logs").iterdir())
    assert logs and all(_mode(p) == 0o600 for p in logs)
    blobs = [p for p in (home / "blobs").rglob("*") if p.is_file()]
    assert blobs and all(_mode(p) == 0o600 for p in blobs)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_existing_readable_database_is_narrowed(tmp_path, monkeypatch):
    home = tmp_path / "shared"
    home.mkdir(mode=0o775)
    os.chmod(home, 0o775)
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(home))
    monkeypatch.delenv("IS_TESTING", raising=False)
    db.close_db()
    db.run_db_migrations()
    db.close_db()
    os.chmod(home / "promptfoo.db", 0o644)
    db.get_db()
    db.close_db()
    assert _mode(home / "promptfoo.db") == 0o600
    assert _mode(home) == 0o775  # a directory shared on purpose stays shared
