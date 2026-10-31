"""A minimal Node `EventEmitter` (on / off / emit / removeAllListeners)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Listener = Callable[[Any], Any]


class EventEmitter:
    def __init__(self) -> None:
        self._listeners: dict[str, list[Listener]] = {}

    def on(self, event: str, listener: Listener) -> EventEmitter:
        self._listeners.setdefault(event, []).append(listener)
        return self

    def off(self, event: str, listener: Listener) -> EventEmitter:
        listeners = self._listeners.get(event)
        if listeners and listener in listeners:
            listeners.remove(listener)
        return self

    def emit(self, event: str, data: Any = None) -> bool:
        listeners = list(self._listeners.get(event) or [])
        for listener in listeners:
            listener(data)
        return bool(listeners)

    def listener_count(self, event: str) -> int:
        return len(self._listeners.get(event) or [])

    def remove_all_listeners(self) -> None:
        self._listeners.clear()
