"""`promptfoo cache clear`. Ports src/commands/cache.ts.

Clears promptphoo's own cache (`<configDir>/cache/cache-py.json` and the claims directory);
Node's `cache.json` is left alone, as the two caches are kept separate.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import click

from promptphoo.cli.common import common_options
from promptphoo.envfile import setup_env

logger = logging.getLogger("promptphoo")

CUTE_MESSAGES = (
    "Scrubbing bits...",
    "Sweeping stale data...",
    "Defragmenting memory...",
    "Flushing temporary files...",
    "Tuning hyperparameters...",
    "Purging expired entries...",
    "Resetting cache counters...",
    "Pruning the neural net...",
    "Removing overfitting...",
    "Invalidating cached queries...",
    "Aligning embeddings...",
    "Refreshing data structures...",
)
MESSAGE_INTERVAL_S = 8.0


async def clear_with_messages() -> None:
    from promptphoo.cache import clear_cache

    async def ticker() -> None:
        index = 0
        while True:
            await asyncio.sleep(MESSAGE_INTERVAL_S)
            logger.info(CUTE_MESSAGES[index % len(CUTE_MESSAGES)])
            index += 1

    task = asyncio.create_task(ticker())
    try:
        await clear_cache()
    finally:
        task.cancel()


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.group("cache", help="Manage cache")
    def cache_group() -> None:
        pass

    @cache_group.command("clear", help="Clear cache")
    @common_options
    def clear_cmd(env_file: tuple[str, ...], verbose: bool) -> None:
        setup_env(list(env_file) or None)
        logger.info("Clearing cache...")
        asyncio.run(clear_with_messages())

    return cache_group
