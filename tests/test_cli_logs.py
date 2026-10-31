"""Log files and `promptfoo logs`: ports test/commands/logs.test.ts, test/util/logs.test.ts and
the file-transport cases of test/logger.test.ts."""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from promptphoo import logfiles
from promptphoo.cli import logs as logs_cmd
from promptphoo.cli.main import build_cli


@pytest.fixture(autouse=True)
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.delenv("PROMPTFOO_LOG_DIR", raising=False)
    caplog.set_level("INFO", logger="promptphoo")
    yield tmp_path / "home"
    logfiles.close_run_logging()


def run(args: list[str]):
    return CliRunner().invoke(build_cli(args), args, catch_exceptions=False)


def make_log(home: Path, name: str, content: str = "", age: float = 0) -> Path:
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    path.write_text(content)
    if age:
        t = time.time() - age
        os.utime(path, (t, t))
    return path


def msgs(caplog, level: str | None = None) -> list[str]:
    return [r.getMessage() for r in caplog.records if level is None or r.levelname == level]


# --- logfiles -------


def test_run_log_file_names_match_node(home: Path):
    date = datetime(2026, 10, 31, 5, 24, 47, 109000, tzinfo=UTC)
    path = logfiles.create_run_log_file("debug", date)
    assert path == str(home / "logs" / "promptfoo-debug-2026-10-31_05-24-47-109Z.log")


def test_log_dir_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("PROMPTFOO_LOG_DIR", str(tmp_path / "custom"))
    assert logfiles.get_log_directory() == str(tmp_path / "custom")


def test_initialize_writes_debug_and_error_files(home: Path):
    logfiles.initialize_run_logging()
    assert logfiles.debug_log_file and logfiles.error_log_file
    log = logging.getLogger("promptphoo")
    log.info("hello info")
    log.error("bad thing\nsecond line")
    logfiles.close_run_logging()
    debug = Path(sorted((home / "logs").glob("promptfoo-debug-*"))[0]).read_text()
    error = Path(sorted((home / "logs").glob("promptfoo-error-*"))[0]).read_text()
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z \[INFO\]: hello info\n"
        r"\S+ \[ERROR\]: bad thing\nsecond line\n",
        debug,
    )
    assert "[INFO]" not in error and "[ERROR]: bad thing" in error


def test_formatter_levels_and_debug_location():
    fmt = logfiles.FileFormatter()
    rec = logging.LogRecord("x", logging.WARNING, "/a/b/mod.py", 7, "careful", None, None)
    assert fmt.format(rec).endswith(" [WARN]: careful")
    rec = logging.LogRecord("x", logging.DEBUG, "/a/b/mod.py", 7, "dbg", None, None)
    assert fmt.format(rec).endswith(" [DEBUG] [mod.py:7]: dbg")


def test_disable_env_vars(monkeypatch: pytest.MonkeyPatch, home: Path):
    monkeypatch.setenv("PROMPTFOO_DISABLE_DEBUG_LOG", "true")
    monkeypatch.setenv("PROMPTFOO_DISABLE_ERROR_LOG", "true")
    logfiles.initialize_run_logging()
    assert logfiles.debug_log_file is None and logfiles.error_log_file is None


def test_old_log_files_are_pruned(home: Path):
    for i in range(52):
        make_log(home, f"promptfoo-debug-old-{i:02d}.log", age=1000 + i)
    logfiles.create_run_log_file("debug")
    assert len(list((home / "logs").iterdir())) == 50
    assert not (home / "logs" / "promptfoo-debug-old-51.log").exists()


def test_get_log_files_sorted_and_filtered(home: Path):
    make_log(home, "promptfoo-debug-a.log", "x", age=100)
    make_log(home, "promptfoo-error-b.log", "yy", age=10)
    make_log(home, "other.log")
    names = [f.name for f in logfiles.get_log_files()]
    assert names == ["promptfoo-error-b.log", "promptfoo-debug-a.log"]
    assert [f.name for f in logfiles.get_log_files("debug")] == ["promptfoo-debug-a.log"]
    assert logfiles.get_log_files("error")[0].size == 2


def test_find_log_file(home: Path):
    path = make_log(home, "promptfoo-debug-2026.log")
    assert logfiles.find_log_file(str(path)) == str(path)
    assert logfiles.find_log_file("promptfoo-debug-2026.log") == str(path)
    assert logfiles.find_log_file("2026") == str(path)
    assert logfiles.find_log_file("nope") is None


@pytest.mark.parametrize(
    ("size", "text"),
    [(0, "0 B"), (512, "512 B"), (1024, "1.0 KB"), (1536, "1.5 KB"), (1048576, "1.0 MB")],
)
def test_format_file_size(size: int, text: str):
    assert logfiles.format_file_size(size) == text


def test_print_error_information(home: Path, caplog):
    error_log = make_log(home, "promptfoo-error-x.log", "")
    logfiles.print_error_information(str(error_log), "/d.log")
    assert msgs(caplog) == []
    error_log.write_text("2026 [ERROR]: x\n")
    logfiles.print_error_information(str(error_log), "/d.log")
    assert msgs(caplog) == [
        "\nThere were some errors during the operation. See logs for more details.\n"
        f"Error log: {error_log}\nDebug log: /d.log"
    ]


# --- promptfoo logs -------


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["--type", "x"], "Invalid log type: x. Must be one of: debug, error, all"),
        (["-n", "0"], "--lines must be a positive number"),
        (["-n", "abc"], "--lines must be a positive number"),
        (["--head", "-1"], "--head must be a positive number"),
        (["-g", "("], 'Invalid grep pattern: "(" is not a valid regular expression'),
    ],
)
def test_option_validation(args: list[str], error: str, caplog):
    assert run(["logs", *args]).exit_code == 1
    assert msgs(caplog, "ERROR") == [error]


def test_no_log_files(home: Path, caplog):
    assert run(["logs"]).exit_code == 1
    assert msgs(caplog, "ERROR") == [
        "No log files found.\n\nLog files are created when running commands like promptfoo "
        f"eval.\nLog directory: {home / 'logs'}"
    ]


def test_unknown_file(home: Path, caplog):
    make_log(home, "promptfoo-debug-a.log", "x")
    assert run(["logs", "zzz"]).exit_code == 1
    assert msgs(caplog, "ERROR") == [
        "Log file not found: zzz\n\nRun promptfoo logs --list to see available log files."
    ]


def test_show_most_recent_with_header(home: Path, caplog):
    make_log(home, "promptfoo-debug-old.log", "old", age=100)
    path = make_log(home, "promptfoo-debug-new.log", "a [INFO]: one\nb [ERROR]: two\n")
    assert run(["logs", "--no-color"]).exit_code == 0
    out = msgs(caplog)
    assert out[1:4] == ["promptfoo-debug-new.log", f"Path: {path}", "Size: 29 B"]
    assert out[4].startswith("Modified: ")
    assert out[-1] == "a [INFO]: one\nb [ERROR]: two"


def test_lines_head_and_grep(home: Path, caplog):
    make_log(home, "promptfoo-debug-a.log", "\n".join(f"line {i}" for i in range(10)) + "\n")
    run(["logs", "-n", "2", "--no-color"])
    assert msgs(caplog)[-1] == "line 8\nline 9"
    caplog.clear()
    run(["logs", "--head", "2", "--no-color"])
    assert msgs(caplog)[-1] == "line 0\nline 1"
    caplog.clear()
    run(["logs", "-g", "LINE [35]", "--no-color"])
    assert msgs(caplog)[-1] == "line 3\nline 5"
    caplog.clear()
    run(["logs", "-g", "nomatch"])
    assert msgs(caplog)[-1] == "No lines matching pattern found."


def test_empty_file(home: Path, caplog):
    make_log(home, "promptfoo-error-a.log", "")
    assert run(["logs", "--type", "error"]).exit_code == 0
    assert msgs(caplog)[-1] == "Log file is empty."


def test_highlighting_is_stateful():
    lines = ["pre", "x [ERROR]: a", "  stack", "x [INFO]: b"]
    assert logs_cmd.highlight_log_lines(lines, True) == "\n".join(lines)
    from promptphoo.output.ansi import chalk

    colored = logs_cmd.highlight_log_lines(lines, False).split("\n")
    assert colored[2] == chalk("  stack", "red")


@pytest.mark.parametrize("args", [["--list"], ["list"], ["list", "--type", "bogus"]])
def test_list(args: list[str], home: Path, caplog):
    make_log(home, "promptfoo-debug-a.log", "x" * 418)
    make_log(home, "promptfoo-error-a.log", "", age=60)
    assert run(["logs", *args]).exit_code == 0
    table = re.sub(r"\x1b\[\d+m", "", msgs(caplog)[0])
    assert "│ 1  │ promptfoo-debug-a.log" in table
    assert "│ debug  │ 418 B    │" in table
    assert msgs(caplog)[-2:] == [
        "Run promptfoo logs <filename> to view a specific log file.",
        f"Log directory: {home / 'logs'}",
    ]


def test_list_empty(home: Path, caplog):
    assert run(["logs", "list"]).exit_code == 0
    assert msgs(caplog)[0].startswith("No log files found.")


def test_locale_string_is_en_us():
    ts = datetime(2026, 10, 31, 13, 5, 9).timestamp()
    assert logs_cmd.locale_string(ts) == "10/31/2026, 1:05:09 PM"
    ts = datetime(2026, 1, 2, 0, 0, 0).timestamp()
    assert logs_cmd.locale_string(ts) == "1/2/2026, 12:00:00 AM"
