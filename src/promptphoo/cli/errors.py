"""Errors caused by what the user gave the CLI, reported as one message, not a traceback (D67).

Node lets these reach its top level, which prints the stack. promptphoo logs the message and exits
1, as Node's exit code is. Anything raised outside config loading that is not listed here is a bug
and keeps its traceback.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from typing import Any, TypeVar

logger = logging.getLogger("promptphoo")

T = TypeVar("T")


class ConfigLoadError(Exception):
    """A ValueError, OSError or YAML error raised while loading the config: a missing or
    unreadable file, a syntax error, a bad `$ref`, an unknown provider id."""


def user_errors() -> tuple[type[BaseException], ...]:
    from promptphoo.assertions.validate import AssertValidationError
    from promptphoo.config.references import (
        PromptReferenceValidationError,
        ProviderReferenceValidationError,
    )
    from promptphoo.db import DatabaseVersionError
    from promptphoo.output.writer import OutputWriteError

    return (
        AssertValidationError,
        ProviderReferenceValidationError,
        PromptReferenceValidationError,
        ConfigLoadError,
        DatabaseVersionError,
        OutputWriteError,
    )


async def load_configs(coro: Awaitable[T]) -> T:
    """Await a `resolve_configs` call, turning the errors a bad config causes into
    ConfigLoadError. The traceback still goes to the debug log."""
    import yaml

    try:
        return await coro
    except user_errors():
        raise
    except (ValueError, OSError, yaml.YAMLError) as e:
        logger.debug("Loading the config failed", exc_info=True)
        raise ConfigLoadError(_message(e)) from e


def _message(error: Any) -> str:
    text = str(error)
    if isinstance(error, OSError) and error.filename and error.strerror:
        return f"{error.strerror}: {error.filename}"
    return text or type(error).__name__
