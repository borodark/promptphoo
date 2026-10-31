"""`promptfoo export eval <id|latest>` and `promptfoo export logs`. Ports src/commands/export.ts
(plus `getLogDirectory`/`getLogFiles` from src/util/logs.ts).

`export eval` writes through `output.writer` (any `--output` format `eval -o` supports) or prints
`createOutputData` as indented JSON. `--include-media` is accepted, but promptphoo has no blob
store until Phase 6, so no `blobAssets` are embedded (see D43). `export logs` builds the same
minimal tar inside a level-9 gzip stream that upstream hand-rolls.
"""

from __future__ import annotations

import logging
import os
import re
import zlib
from datetime import UTC, datetime
from typing import Any

import click

from promptphoo import db, envars
from promptphoo.jsvalues import json_stringify
from promptphoo.paths import get_config_directory_path

logger = logging.getLogger("promptphoo")


def _js_error(error: BaseException) -> str:
    """`${error}` for an Error: `Error: <message>`."""
    return f"Error: {error}"


# --- export eval ------------------------------------------------------------------------------


def export_eval(eval_id: str, output: str | None = None, include_media: bool = False) -> int:
    """The `export eval` action. Returns the process exit code."""
    from promptphoo.models.eval import Eval
    from promptphoo.output.writer import create_output_data, write_output

    try:
        result = Eval.latest() if eval_id == "latest" else Eval.find_by_id(eval_id)
        if not result:
            logger.error("No eval found with ID %s", eval_id)
            return 1
        if include_media:
            logger.debug("--include-media: no blob store in promptphoo yet; nothing to embed")
        if output:
            write_output(output, result, None)
            logger.info("Eval with ID %s has been successfully exported to %s.", eval_id, output)
        else:
            logger.info("%s", json_stringify(create_output_data(result, None, include_media), 2))
        return 0
    except Exception as error:
        logger.error("Failed to export eval: %s", _js_error(error))
        return 1


# --- export logs ------------------------------------------------------------------------------


def get_log_directory() -> str:
    """getLogDirectory (src/util/logs.ts)."""
    config_dir = get_config_directory_path(True)
    custom = envars.get_env_string("PROMPTFOO_LOG_DIR")
    return os.path.abspath(custom) if custom else os.path.join(config_dir, "logs")


def get_log_files(kind: str = "all") -> list[dict[str, Any]]:
    """getLogFiles (src/util/logs.ts): promptfoo-*.log files, newest first."""
    log_dir = get_log_directory()
    if not os.path.isdir(log_dir):
        return []
    try:
        names = os.listdir(log_dir)
    except OSError as error:
        logger.error("Error reading log directory: %s", _js_error(error))
        return []
    files = []
    for name in names:
        if not name.startswith("promptfoo-") or not name.endswith(".log"):
            continue
        if kind != "all" and f"-{kind}-" not in name:
            continue
        path = os.path.join(log_dir, name)
        try:
            stats = os.stat(path)
        except OSError:
            continue
        files.append(
            {
                "name": name,
                "path": path,
                "mtime": stats.st_mtime,
                "type": "error" if "-error-" in name else "debug",
                "size": stats.st_size,
            }
        )
    # Array#sort is stable; mtime compared at millisecond precision like Date#getTime.
    return sorted(files, key=lambda f: -int(f["mtime"] * 1000))


def _octal(value: int, width: int) -> bytes:
    return format(value, "o").rjust(width, "0").encode("ascii")


def tar_header(file_name: str, size: int, mtime_s: int) -> bytes:
    """The 512-byte ustar-less header upstream writes for each log file."""
    header = bytearray(512)
    name = file_name.encode("utf-8")[: min(len(file_name), 100)]
    header[0 : len(name)] = name
    header[100:108] = b"0000644 "
    header[108:116] = b"0000000 "
    header[116:124] = b"0000000 "
    header[124:136] = _octal(size, 11) + b" "
    header[136:148] = _octal(mtime_s, 11) + b" "
    header[148:156] = b"        "
    header[156] = 0x30
    checksum = sum(header)
    header[148:156] = _octal(checksum, 6) + b"\0 "
    return bytes(header)


def create_log_archive(log_files: list[str], output_path: str) -> None:
    """createLogArchive: tar entries (header, content, zero padding) + two zero blocks, gzip -9."""
    gzip = zlib.compressobj(9, zlib.DEFLATED, 31)
    with open(output_path, "wb") as out:
        for log_file in log_files:
            try:
                with open(log_file, "rb") as f:
                    content = f.read()
                stats = os.stat(log_file)
            except FileNotFoundError:
                continue
            out.write(gzip.compress(tar_header(os.path.basename(log_file), stats.st_size,
                                               int(stats.st_mtime))))  # fmt: skip
            out.write(gzip.compress(content))
            padding = 512 - (len(content) % 512)
            if padding < 512:
                out.write(gzip.compress(bytes(padding)))
        out.write(gzip.compress(bytes(1024)))
        out.write(gzip.flush())
    logger.info("Created log archive: %s (%d bytes)", output_path, os.path.getsize(output_path))


def _js_parse_int(value: str) -> int | None:
    """parseInt(value, 10); None for NaN."""
    m = re.match(r"^[\s\ufeff\xa0]*([+-]?\d+)", value)
    return int(m.group(1)) if m else None


def _iso_now() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def export_logs(count: str | None = None, output: str | None = None) -> int:
    """The `export logs` action. Returns the process exit code."""
    try:
        log_dir = get_log_directory()
        if not os.access(log_dir, os.F_OK):
            logger.error(
                'No log directory found. Logs are created when running commands like "promptfoo '
                'eval".\nLog directory: %s',
                log_dir,
            )
            return 1
        all_log_files = get_log_files()
        if not all_log_files:
            logger.error(
                "No log files found in the logs directory. Logs are created when running commands "
                'like "promptfoo eval".\nLog directory: %s',
                log_dir,
            )
            return 1
        log_files = all_log_files
        if count:
            n = _js_parse_int(count)
            if n is None or n <= 0:
                logger.error("Count must be a positive number")
                return 1
            log_files = all_log_files[:n]
        output_path = output
        if not output_path:
            timestamp = re.sub(r"[:.]", "-", _iso_now()).replace("T", "_")
            output_path = f"promptfoo-logs-{timestamp}.gz"
        if not output_path.endswith(".gz"):
            output_path += ".gz"
        logger.info("Collecting %d log file(s)...", len(log_files))
        create_log_archive([f["path"] for f in log_files], output_path)
        logger.info("Log files have been collected in: %s", output_path)
        logger.info("You can now share this file for debugging purposes.")
        return 0
    except Exception as error:
        logger.error("Failed to collect logs: %s", _js_error(error))
        return 1


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    del default_config, default_config_path

    @cli.group("export", help="Export eval records or logs")
    def export_group() -> None:
        pass

    @export_group.command("eval", help="Export an eval record to a JSON file")
    @click.argument("eval_id", metavar="<evalId>")
    @click.option("-o", "--output", "output", help="Output path for the exported file")
    @click.option(
        "--include-media",
        is_flag=True,
        help="Embed referenced blob media bytes for portable imports",
    )
    def export_eval_cmd(eval_id: str, output: str | None, include_media: bool) -> None:
        db.run_db_migrations()
        code = export_eval(eval_id, output, include_media)
        if code:
            raise SystemExit(code)

    @export_group.command("logs", help="Collect and zip log files for debugging")
    @click.option("-n", "--count", help="Number of recent log files to include (default: all)")
    @click.option("-o", "--output", "output", help="Output path for the compressed log file")
    def export_logs_cmd(count: str | None, output: str | None) -> None:
        code = export_logs(count, output)
        if code:
            raise SystemExit(code)
