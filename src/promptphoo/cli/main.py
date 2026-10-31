"""CLI entry point. Ports src/main.ts; commands are registered per module as they land."""

import click

from promptphoo import UPSTREAM_VERSION, __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(
    __version__, "-V", "--version", message=f"%(version)s (promptfoo {UPSTREAM_VERSION})"
)
def cli() -> None:
    """Python port of promptfoo: LLM eval & testing toolkit."""


def main() -> None:
    cli()
