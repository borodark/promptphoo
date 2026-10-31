"""Response cache. Ports the public surface of src/cache.ts.

Node's KeyvFile cache (<configDir>/cache/cache.json) is rewritten whole without locking, so
sharing it between runtimes would lose writes. promptphoo keeps the same envelope format
({"value", "expires"} per key, absolute ms expiry) in its own file, cache-py.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from promptphoo import envars
from promptphoo.paths import get_config_directory_path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL_SECONDS = 60 * 60 * 24 * 14
CACHE_FILE_NAME = "cache-py.json"

_enabled = envars.get_env_bool("PROMPTFOO_CACHE_ENABLED", True)
_enabled_override: ContextVar[bool | None] = ContextVar("cache_enabled", default=None)
_namespace: ContextVar[str | None] = ContextVar("cache_namespace", default=None)
_clear_generation = 0


def get_cache_ttl_ms() -> int:
    return envars.get_env_int("PROMPTFOO_CACHE_TTL", DEFAULT_CACHE_TTL_SECONDS) * 1000


class Cache:
    """In-memory map with optional JSON-file persistence."""

    def __init__(self, path: str | None, ttl_ms: int) -> None:
        self.path = path
        self.ttl_ms = ttl_ms
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._lock = asyncio.Lock()
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("[Cache] Ignoring unreadable cache file %s: %s", path, e)

    async def get(self, key: str) -> Any:
        entry = self._data.get("keyv:" + key)
        if entry is None:
            return None
        expires = entry.get("expires")
        if expires is not None and expires < time.time() * 1000:
            self._data.pop("keyv:" + key, None)
            return None
        return entry.get("value")

    async def set(self, key: str, value: Any, ttl_ms: int | None = None) -> None:
        ttl = self.ttl_ms if ttl_ms is None else ttl_ms
        self._data["keyv:" + key] = {
            "value": value,
            "expires": int(time.time() * 1000 + ttl) if ttl else None,
        }
        self._dirty = True
        await self.flush()

    async def delete(self, key: str) -> None:
        if self._data.pop("keyv:" + key, None) is not None:
            self._dirty = True
            await self.flush()

    async def clear(self) -> None:
        global _clear_generation
        self._data.clear()
        self._dirty = True
        await self.flush()
        _clear_generation += 1

    async def flush(self) -> None:
        if not self.path or not self._dirty:
            return
        async with self._lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            self._dirty = False


class _NamespacedCache:
    def __init__(self, cache: Cache, namespace: str) -> None:
        self._cache = cache
        self._ns = namespace

    async def get(self, key: str) -> Any:
        return await self._cache.get(f"{self._ns}:{key}")

    async def set(self, key: str, value: Any, ttl_ms: int | None = None) -> None:
        await self._cache.set(f"{self._ns}:{key}", value, ttl_ms)

    async def delete(self, key: str) -> None:
        await self._cache.delete(f"{self._ns}:{key}")

    async def clear(self) -> None:
        """Clear only this namespace's keys (clearNamespacedCache in src/cache.ts)."""
        global _clear_generation
        prefix = f"keyv:{self._ns}:"
        data = self._cache._data
        for key in [k for k in data if k.startswith(prefix)]:
            del data[key]
        self._cache._dirty = True
        await self._cache.flush()
        _clear_generation += 1


_instance: Cache | None = None


def _get_instance() -> Cache:
    global _instance
    if _instance is None:
        cache_type = envars.get_env_string("PROMPTFOO_CACHE_TYPE") or "disk"
        path = None
        if cache_type == "disk" and _enabled:
            directory = envars.get_env_string("PROMPTFOO_CACHE_PATH") or os.path.join(
                get_config_directory_path(), "cache"
            )
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, CACHE_FILE_NAME)
        _instance = Cache(path, get_cache_ttl_ms())
    return _instance


def get_cache() -> Cache | _NamespacedCache:
    ns = _namespace.get()
    return _NamespacedCache(_get_instance(), ns) if ns else _get_instance()


def get_scoped_cache_key(key: str, namespace: str | None = None) -> str:
    namespace = _namespace.get() if namespace is None else namespace
    return f"{namespace}:{key}" if namespace else key


def get_cache_clear_generation() -> int:
    """Bumped whenever a cache is cleared, so in-process provider caches can invalidate."""
    return _clear_generation


def is_cache_enabled() -> bool:
    override = _enabled_override.get()
    return _enabled if override is None else override


def enable_cache() -> None:
    global _enabled
    _enabled = True


def disable_cache() -> None:
    global _enabled
    _enabled = False


async def clear_cache() -> None:
    await _get_instance().clear()


@contextmanager
def with_cache_enabled(enabled: bool | None) -> Iterator[None]:
    if enabled is None:
        yield
        return
    token = _enabled_override.set(enabled)
    try:
        yield
    finally:
        _enabled_override.reset(token)


@contextmanager
def with_cache_namespace(namespace: str | None) -> Iterator[None]:
    parent = _namespace.get()
    if not namespace or parent == namespace:
        yield
        return
    token = _namespace.set(f"{parent}:{namespace}" if parent else namespace)
    try:
        yield
    finally:
        _namespace.reset(token)
