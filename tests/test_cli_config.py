"""`promptphoo config` and the global config file: ports test/commands/config.test.ts and the
email/read/write cases of test/globalConfig/{globalConfig,accounts}.test.ts."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from promptphoo import globalconfig
from promptphoo.cli.main import build_cli
from promptphoo.yamlload import load_yaml


@pytest.fixture(autouse=True)
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.delenv("PROMPTFOO_API_KEY", raising=False)
    caplog.set_level("INFO", logger="promptphoo")
    return tmp_path / "home"


def run(args: list[str], input_: str | None = None):
    return CliRunner().invoke(build_cli(args), args, input=input_, catch_exceptions=False)


def stored(home: Path) -> dict:
    return load_yaml((home / "promptfoo.yaml").read_text())


def log(caplog) -> list[tuple[str, str]]:
    return [(r.levelname, r.getMessage()) for r in caplog.records]


def login(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "promptfoo.yaml").write_text("id: abc\ncloud:\n  apiKey: k\n")


# --- set email -------


def test_set_email_refused_when_logged_in(home: Path, caplog):
    login(home)
    assert run(["config", "set", "email", "a@b.co"]).exit_code == 1
    assert log(caplog) == [
        (
            "ERROR",
            "Cannot update email while logged in. Email is managed through Node promptfoo's "
            "'promptfoo auth login'. Please use 'promptfoo auth logout' there first if you want to "
            "use a different email.",
        )
    ]
    assert "account" not in stored(home)


def test_set_email_when_logged_out(home: Path, caplog):
    assert run(["config", "set", "email", "test@example.com"]).exit_code == 0
    assert log(caplog) == [("INFO", "Email set to test@example.com")]
    assert stored(home)["account"] == {"email": "test@example.com"}


def test_set_email_validates_format(home: Path, caplog):
    assert run(["config", "set", "email", "invalid-email"]).exit_code == 1
    assert log(caplog) == [("ERROR", "Invalid email address: invalid-email")]


@pytest.mark.parametrize("email", ["a.b+c@x.io", "o'neil@sub.example.org", "x_y@a-b.co"])
def test_valid_emails(email: str):
    assert run(["config", "set", "email", email]).exit_code == 0


@pytest.mark.parametrize("email", [".a@x.io", "a@x", "a@x.c", "a..b@x.io", "a@-x.io", "a@x.io\n"])
def test_invalid_emails(email: str):
    assert run(["config", "set", "email", email]).exit_code == 1


# --- unset email -------


def test_unset_refused_when_logged_in(home: Path, caplog):
    login(home)
    assert run(["config", "unset", "email", "--force"]).exit_code == 1


def test_unset_with_force(home: Path, caplog):
    run(["config", "set", "email", "a@b.co"])
    caplog.clear()
    assert run(["config", "unset", "email", "-f"]).exit_code == 0
    assert log(caplog) == [("INFO", "Email has been unset.")]
    assert stored(home)["account"] == {"email": ""}


def test_unset_with_confirmation(home: Path, caplog):
    run(["config", "set", "email", "a@b.co"])
    caplog.clear()
    result = run(["config", "unset", "email"], input_="y\n")
    assert result.exit_code == 0
    assert 'Are you sure you want to unset the email "a@b.co"?' in result.output
    assert log(caplog) == [("INFO", "Email has been unset.")]


def test_unset_cancelled(home: Path, caplog):
    run(["config", "set", "email", "a@b.co"])
    caplog.clear()
    assert run(["config", "unset", "email"], input_="n\n").exit_code == 0
    assert log(caplog) == [("INFO", "Operation cancelled.")]
    assert stored(home)["account"] == {"email": "a@b.co"}


def test_unset_without_email(caplog):
    assert run(["config", "unset", "email"]).exit_code == 0
    assert log(caplog) == [("INFO", "No email is currently set.")]


# --- get email -------


def test_get_email(caplog):
    run(["config", "set", "email", "a@b.co"])
    caplog.clear()
    assert run(["config", "get", "email"]).exit_code == 0
    assert log(caplog) == [("INFO", "a@b.co")]


def test_get_email_unset(caplog):
    run(["config", "get", "email"])
    assert log(caplog) == [
        ("INFO", 'No email set. Use "promptphoo config set email <email>" to set one.')
    ]


def test_get_email_auth_guidance(monkeypatch: pytest.MonkeyPatch, caplog):
    monkeypatch.setenv("PROMPTFOO_API_KEY", "key")
    run(["config", "get", "email"])
    assert log(caplog) == [
        (
            "INFO",
            "Email is managed through Node promptfoo's 'promptfoo auth login'. Run 'promptfoo auth "
            "whoami' there to view the current account.",
        )
    ]


# --- globalConfig -------


def test_read_creates_file_with_id(home: Path):
    config = globalconfig.read_global_config()
    assert set(config) == {"id"}
    assert (home / "promptfoo.yaml").read_text() == f"id: {config['id']}\n"
    assert globalconfig.read_global_config() == config


def test_read_adds_missing_id_and_keeps_keys(home: Path):
    home.mkdir()
    (home / "promptfoo.yaml").write_text("account:\n  email: a@b.co\n")
    config = globalconfig.read_global_config()
    assert config["account"] == {"email": "a@b.co"}
    assert stored(home)["id"] == config["id"]


def test_partial_write_merges_and_removes(home: Path):
    globalconfig.write_global_config({"id": "x", "a": 1, "b": {"c": [1, 2]}})
    globalconfig.write_global_config_partial({"a": None, "d": "e"})
    assert (home / "promptfoo.yaml").read_text() == "id: x\nb:\n  c:\n    - 1\n    - 2\nd: e\n"


def test_node_file_round_trips(home: Path):
    home.mkdir()
    text = (
        "id: 0c89b1df-cf47-4387-aa49-9592f6ae88cb\n"
        "account:\n  email: a@b.co\n"
        "cloud:\n  appUrl: https://www.promptfoo.app\n"
    )
    (home / "promptfoo.yaml").write_text(text)
    globalconfig.set_user_email("a@b.co")
    assert (home / "promptfoo.yaml").read_text() == text
