"""`promptfoo logs [file]` and `promptfoo logs list`. Ports src/commands/logs.ts.

Reads the per-run log files that both Node and promptphoo write to `<configDir>/logs` (or
PROMPTFOO_LOG_DIR); see `promptphoo.logfiles`.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import sys
import time
from datetime import datetime
from typing import Any

import click

from promptphoo import logfiles
from promptphoo.cli.common import common_options, exit_with
from promptphoo.output.ansi import chalk

logger = logging.getLogger("promptphoo")

VALID_TYPES = ("debug", "error", "all")
ONE_MB = 1024 * 1024


def js_parse_int(value: str) -> int | None:
    match = re.match(r"\s*([+-]?\d+)", value)
    return int(match.group(1)) if match else None


def locale_string(timestamp: float) -> str:
    """Date#toLocaleString() in the en-US locale, local time."""
    dt = datetime.fromtimestamp(timestamp)
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{dt.month}/{dt.day}/{dt.year}, {hour}:{dt.minute:02d}:{dt.second:02d} {ampm}"


def highlight_log_lines(lines: list[str], no_color: bool) -> str:
    if no_color:
        return "\n".join(lines)
    color = "gray"
    out = []
    for line in lines:
        if "[ERROR]" in line:
            color = "red"
        elif "[WARN]" in line:
            color = "yellow"
        elif "[DEBUG]" in line:
            color = "cyan"
        elif "[INFO]" in line:
            color = "white"
        out.append(chalk(line, color))
    return "\n".join(out)


def _print_border() -> None:
    from promptphoo.cli.debug import print_border

    print_border()


def print_log_header(log_path: str, is_current_session: bool) -> None:
    st = os.stat(log_path)
    _print_border()
    logger.info(os.path.basename(log_path))
    logger.info(f"Path: {log_path}")
    logger.info(f"Size: {logfiles.format_file_size(st.st_size)}")
    logger.info(f"Modified: {locale_string(st.st_mtime)}")
    if is_current_session:
        logger.info("Note: This is the log file for the current CLI session.")
    _print_border()


def _read_file_content(path: str) -> list[str]:
    size = os.stat(path).st_size
    if size > ONE_MB:
        logger.warning(
            f"Log file is large ({logfiles.format_file_size(size)}).\n"
            "Consider using -n <lines> to limit output."
        )
    return logfiles.read_last_lines(path, sys.maxsize)


def print_log_content(
    log_path: str,
    lines_count: int | None,
    head: int | None,
    grep: re.Pattern[str] | None,
    no_color: bool,
) -> None:
    if os.stat(log_path).st_size == 0:
        logger.info("Log file is empty.")
        return
    if lines_count:
        lines = logfiles.read_last_lines(log_path, lines_count)
    elif head:
        lines = logfiles.read_first_lines(log_path, head)
    else:
        lines = _read_file_content(log_path)
    if grep:
        lines = [line for line in lines if grep.search(line)]
        if not lines:
            logger.info("No lines matching pattern found.")
            return
    logger.info(highlight_log_lines(lines, no_color))


def follow_log_file(log_path: str, no_color: bool) -> None:
    """tail -f: print the last 20 lines, then poll for appended content until interrupted."""
    try:
        position = os.stat(log_path).st_size
    except OSError:
        position = 0
    logger.info(f"Following {os.path.basename(log_path)}... (Ctrl+C to stop)\n")
    if position > 0:
        last = logfiles.read_last_lines(log_path, 20)
        if last:
            logger.info(highlight_log_lines(last, no_color))
        position = os.stat(log_path).st_size

    stop = False

    def _stop(*_: Any) -> None:
        nonlocal stop
        stop = True

    previous = signal.signal(signal.SIGTERM, _stop)
    try:
        while not stop:
            time.sleep(0.1)
            try:
                size = os.stat(log_path).st_size
            except OSError as error:
                logger.debug(f"Error reading log file: {error}")
                continue
            if size > position:
                with open(log_path, "rb") as f:
                    f.seek(position)
                    chunk = f.read(size - position).decode("utf-8", errors="replace")
                new_lines = [line for line in chunk.split("\n") if line]
                if new_lines:
                    sys.stdout.write(highlight_log_lines(new_lines, no_color) + "\n")
                    sys.stdout.flush()
                position = size
            elif size < position:
                logger.info("Log file was rotated, resetting position...")
                position = size
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous)


def list_log_files(log_type: str) -> None:
    from promptphoo.output.table import wrap_table

    files = logfiles.get_log_files(log_type)  # type: ignore[arg-type]
    if not files:
        logger.info(
            "No log files found.\n\n"
            "Log files are created when running commands like promptphoo eval.\n"
            f"Log directory: {logfiles.get_log_directory()}"
        )
        return
    rows = [
        {
            "#": str(i + 1),
            "filename": f.name,
            "type": f.type,
            "size": logfiles.format_file_size(f.size),
            "modified": locale_string(f.mtime),
        }
        for i, f in enumerate(files)
    ]
    widths = {"#": 4, "filename": 50, "type": 8, "size": 10, "modified": 22}
    logger.info(wrap_table(rows, widths))
    _print_border()
    logger.info("Run promptphoo logs <filename> to view a specific log file.")
    logger.info(f"Log directory: {logfiles.get_log_directory()}")


def resolve_log_path(file: str | None, log_type: str) -> str | None:
    if file:
        return logfiles.find_log_file(file, log_type)  # type: ignore[arg-type]
    if log_type in ("all", "debug") and logfiles.debug_log_file:
        if os.path.exists(logfiles.debug_log_file):
            return logfiles.debug_log_file
    if log_type in ("all", "error") and logfiles.error_log_file:
        if os.path.exists(logfiles.error_log_file):
            return logfiles.error_log_file
    files = logfiles.get_log_files(log_type)  # type: ignore[arg-type]
    return files[0].path if files else None


def _invalid_type(log_type: str) -> int:
    logger.error(f"Invalid log type: {log_type}. Must be one of: {', '.join(VALID_TYPES)}")
    return 1


def do_logs(
    file: str | None,
    log_type: str,
    lines: str | None,
    head: str | None,
    follow: bool,
    list_: bool,
    grep: str | None,
    color: bool,
) -> int:
    from promptphoo.jsregex import compile_js_regex

    try:
        if log_type not in VALID_TYPES:
            return _invalid_type(log_type)
        if lines:
            count = js_parse_int(lines)
            if count is None or count <= 0:
                logger.error("--lines must be a positive number")
                return 1
        if head:
            count = js_parse_int(head)
            if count is None or count <= 0:
                logger.error("--head must be a positive number")
                return 1
        grep_pattern = None
        if grep:
            try:
                grep_pattern = compile_js_regex(grep, "i")
            except (re.error, ValueError):
                logger.error(f'Invalid grep pattern: "{grep}" is not a valid regular expression')
                return 1
        if list_:
            list_log_files(log_type)
            return 0
        log_path = resolve_log_path(file, log_type)
        if not log_path:
            if file:
                logger.error(
                    f"Log file not found: {file}\n\n"
                    "Run promptphoo logs --list to see available log files."
                )
            else:
                logger.error(
                    "No log files found.\n\n"
                    "Log files are created when running commands like promptphoo eval.\n"
                    f"Log directory: {logfiles.get_log_directory()}"
                )
            return 1
        if not os.access(log_path, os.R_OK):
            logger.error(f"Permission denied: Cannot read {log_path}")
            return 1
        is_current = log_path in (logfiles.debug_log_file, logfiles.error_log_file)
        if follow:
            follow_log_file(log_path, not color)
            return 0
        print_log_header(log_path, is_current)
        print_log_content(
            log_path,
            js_parse_int(lines) if lines else None,
            js_parse_int(head) if head else None,
            grep_pattern,
            not color,
        )
        return 0
    except Exception as error:  # noqa: BLE001 - upstream reports and sets exit code 1
        logger.error(f"Failed to read logs: {error}")
        return 1


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    # Commander matches the `list` subcommand before the optional [file] argument; a click group
    # would bind "list" to [file] instead, so `logs list [--type]` is dispatched here.
    @cli.command("logs", help="View promptfoo log files")
    @click.argument("file", required=False)
    @click.option("--type", "log_type", default="all", help="Log type: debug, error, or all")
    @click.option("-n", "--lines", help="Number of lines to display from end")
    @click.option("--head", help="Number of lines to display from start")
    @click.option(
        "-f", "--follow", is_flag=True, default=False, help="Follow log file in real-time"
    )
    @click.option(
        "-l", "--list", "list_", is_flag=True, default=False, help="List available log files"
    )
    @click.option("-g", "--grep", help="Filter lines matching pattern (case-insensitive regex)")
    @click.option(
        "--no-color", "no_color", is_flag=True, default=False, help="Disable syntax highlighting"
    )
    @common_options
    def logs_cmd(file: str | None, log_type: str, no_color: bool, **opts: Any) -> None:
        if file == "list":
            # Commander binds a trailing --type to the parent command, so upstream's
            # `logs list` always lists every type.
            list_log_files("all")
            return
        exit_with(
            do_logs(
                file,
                log_type,
                opts["lines"],
                opts["head"],
                opts["follow"],
                opts["list_"],
                opts["grep"],
                not no_color,
            )
        )

    return logs_cmd
