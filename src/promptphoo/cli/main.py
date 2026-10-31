"""CLI entry point. Ports src/main.ts; commands are registered per module as they land."""

from __future__ import annotations

import logging
import sys

import click

from promptphoo import UPSTREAM_VERSION, __version__
from promptphoo.envfile import env_paths_from_argv, setup_env


class _StreamHandler(logging.Handler):
    """Info to stdout, warnings and errors to stderr, plain text like Node's console logger."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = sys.stderr if record.levelno >= logging.WARNING else sys.stdout
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)


def setup_logging(verbose: bool = False) -> None:
    root = logging.getLogger()
    if not any(isinstance(h, _StreamHandler) for h in root.handlers):
        handler = _StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(
    __version__, "-V", "--version", message=f"%(version)s (promptfoo {UPSTREAM_VERSION})"
)
def cli() -> None:
    """Python port of promptfoo: LLM eval & testing toolkit."""


# Command modules under promptphoo.cli, registered in upstream's main.ts order. Each exposes
# `register(cli, default_config, default_config_path)`; a module not yet ported is skipped.
_COMMAND_MODULES = (
    "eval", "init", "share", "cache", "config", "debug", "delete", "export", "generate",
    "feedback", "import_", "list_", "logs", "retry", "validate", "show",
)  # fmt: skip


def build_cli(argv: list[str]) -> click.Group:
    import importlib

    from promptphoo.cli import eval as eval_cmd

    skip_default = any(a in ("-h", "--help", "-V", "--version") for a in argv[:1])
    default_config, default_config_path = (
        ({}, None) if skip_default else (eval_cmd.load_default_config())
    )
    for name in _COMMAND_MODULES:
        try:
            module = importlib.import_module(f"promptphoo.cli.{name}")
        except ModuleNotFoundError as err:
            if err.name != f"promptphoo.cli.{name}":
                raise
            continue
        module.register(cli, default_config, default_config_path)
    return cli


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    setup_logging(verbose="--verbose" in args)
    env_paths = env_paths_from_argv(args)
    if env_paths:
        setup_env(env_paths)
    from promptphoo import logfiles

    logfiles.initialize_run_logging()
    try:
        build_cli(args).main(args=args, prog_name="promptfoo")
    finally:
        logfiles.print_error_information()
        logfiles.close_run_logging()
