"""Per-run log files under `<configDir>/logs`, shared in layout and naming with Node promptfoo.

Ports the file-transport half of src/logger.ts (setupLogDirectory, createRunLogFile,
initializeRunLogging, fileFormatter), src/util/logFiles.ts, src/util/logs.ts and
printErrorInformation from src/util/errors/index.ts.

The debug file records what the root logger lets through: info and up, or everything with
--verbose. Node's debug file always holds debug records (see D45).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from promptphoo import envars
from promptphoo.paths import get_config_directory_path
from promptphoo.private_files import make_private_dirs, restrict

logger = logging.getLogger("promptphoo")

MAX_LOG_FILES = 50
LogType = Literal["debug", "error", "all"]

debug_log_file: str | None = None
error_log_file: str | None = None
_handlers: list[logging.Handler] = []

_LEVEL_NAMES = {"WARNING": "WARN", "CRITICAL": "ERROR"}


def _iso_now() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class FileFormatter(logging.Formatter):
    """`${timestamp} [LEVEL] [file:line]: message`; the location is only on debug lines."""

    def format(self, record: logging.LogRecord) -> str:
        level = _LEVEL_NAMES.get(record.levelname, record.levelname)
        location = (
            f" [{os.path.basename(record.pathname)}:{record.lineno}]"
            if record.levelno <= logging.DEBUG
            else ""
        )
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        return f"{_iso_now()} [{level}]{location}: {message}"


def get_log_directory() -> str:
    config_dir = get_config_directory_path(True)
    custom = envars.get_env_string("PROMPTFOO_LOG_DIR")
    return os.path.abspath(custom) if custom else os.path.join(config_dir, "logs")


@dataclass
class LogFileInfo:
    name: str
    path: str
    mtime: float
    type: Literal["debug", "error"]
    size: int


def get_log_files(log_type: LogType = "all") -> list[LogFileInfo]:
    """Log files newest first (src/util/logs.ts getLogFiles)."""
    log_dir = get_log_directory()
    if not os.path.isdir(log_dir):
        return []
    try:
        names = os.listdir(log_dir)
    except OSError as error:
        logger.error(f"Error reading log directory: {error}")
        return []
    files: list[LogFileInfo] = []
    for name in names:
        if not name.startswith("promptfoo-") or not name.endswith(".log"):
            continue
        if log_type != "all" and f"-{log_type}-" not in name:
            continue
        path = os.path.join(log_dir, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        kind: Literal["debug", "error"] = "error" if "-error-" in name else "debug"
        files.append(LogFileInfo(name, path, st.st_mtime, kind, st.st_size))
    return sorted(files, key=lambda f: f.mtime, reverse=True)


def find_log_file(identifier: str, log_type: LogType = "all") -> str | None:
    if os.path.isabs(identifier) and os.path.exists(identifier):
        return identifier
    full_path = os.path.join(get_log_directory(), identifier)
    if os.path.exists(full_path):
        return full_path
    for f in get_log_files(log_type):
        if identifier in f.name or f.name.startswith(identifier):
            return f.path
    return None


def format_file_size(size: int) -> str:
    if size == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = min(math.floor(math.log(size) / math.log(1024)), len(units) - 1)
    value = size / 1024**i
    return f"{value:.{1 if i > 0 else 0}f} {units[i]}"


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def read_last_lines(path: str, count: int) -> list[str]:
    lines = [line.removesuffix("\r") for line in _read_lines(path)]
    return lines[-count:]


def read_first_lines(path: str, count: int) -> list[str]:
    return [line.removesuffix("\r") for line in _read_lines(path)][:count]


def _setup_log_directory() -> str:
    log_dir = get_log_directory()
    make_private_dirs(log_dir)
    try:
        files = get_log_files("all")
        if len(files) >= MAX_LOG_FILES:
            for f in files[MAX_LOG_FILES:]:
                try:
                    os.unlink(f.path)
                except OSError as error:
                    logger.warning(f"Error removing old log file: {f.name} {error}")
    except Exception as error:  # noqa: BLE001 - upstream logs and carries on
        logger.warning(f"Error cleaning up old log files: {error}")
    return log_dir


def create_run_log_file(level: Literal["debug", "error"], date: datetime | None = None) -> str:
    log_dir = _setup_log_directory()
    date = date or datetime.now(UTC)
    iso = date.strftime("%Y-%m-%dT%H:%M:%S.") + f"{date.microsecond // 1000:03d}Z"
    timestamp = iso.replace(":", "-").replace(".", "-").replace("T", "_")
    return os.path.join(log_dir, f"promptfoo-{level}-{timestamp}.log")


def initialize_run_logging() -> None:
    """Attach the debug and error file handlers to the root logger, as Node does at startup."""
    global debug_log_file, error_log_file
    root = logging.getLogger()
    try:
        date = datetime.now(UTC)
        if not envars.get_env_bool("PROMPTFOO_DISABLE_DEBUG_LOG", False):
            debug_log_file = create_run_log_file("debug", date)
            _add_handler(root, debug_log_file, logging.DEBUG)
        if not envars.get_env_bool("PROMPTFOO_DISABLE_ERROR_LOG", False):
            error_log_file = create_run_log_file("error", date)
            _add_handler(root, error_log_file, logging.ERROR)
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Error creating run log file: {error}")


def _add_handler(root: logging.Logger, path: str, level: int) -> None:
    handler = logging.FileHandler(path, encoding="utf-8")
    restrict(path)  # a debug log records prompts and responses (D75)
    handler.setLevel(level)
    handler.setFormatter(FileFormatter())
    root.addHandler(handler)
    _handlers.append(handler)


def close_run_logging() -> None:
    global debug_log_file, error_log_file
    root = logging.getLogger()
    for handler in _handlers:
        root.removeHandler(handler)
        handler.close()
    _handlers.clear()
    debug_log_file = error_log_file = None


def print_error_information(error_log: str | None = None, debug_log: str | None = None) -> None:
    """After a command: point at the log files when the error log has contents."""
    error_log = error_log if error_log is not None else error_log_file
    debug_log = debug_log if debug_log is not None else debug_log_file
    for handler in _handlers:
        handler.flush()
    try:
        has_contents = bool(error_log) and os.path.isfile(error_log) and os.path.getsize(error_log)
    except OSError:
        has_contents = False
    if not has_contents:
        return
    lines = [
        "There were some errors during the operation. See logs for more details.",
        f"Error log: {error_log}",
    ]
    if debug_log:
        lines.append(f"Debug log: {debug_log}")
    logger.info("\n" + "\n".join(lines))
