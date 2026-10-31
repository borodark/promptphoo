"""Provider interface. Ports ApiProvider from src/types/providers.ts.

Responses are plain dicts with the ProviderResponse fields (output, error, raw, cached, cost,
tokenUsage, latencyMs, metadata, ...), matching what Node serializes.
"""

from __future__ import annotations

from typing import Any

ProviderResponse = dict[str, Any]


class ApiProvider:
    """Base class; subclasses implement `call_api` and usually `default_id`."""

    handles_own_delay = False
    handles_own_retries = False

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        self.options = options
        self._id_override: str | None = options.get("id")
        self.label: str | None = options.get("label")
        self.config: dict[str, Any] = options.get("config") or {}
        self.env: dict[str, Any] | None = options.get("env")
        self.transform: Any = None
        self.delay: Any = options.get("delay")
        self.inputs: Any = None
        self.prompts: Any = None

    def default_id(self) -> str:
        raise NotImplementedError

    def id(self) -> str:
        return self._id_override or self.default_id()

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        raise NotImplementedError

    async def cleanup(self) -> None:
        return None

    def to_json(self) -> dict[str, Any]:
        """Serialized form used in stored results (provider id, label, config)."""
        out: dict[str, Any] = {"id": self.id()}
        if self.label:
            out["label"] = self.label
        return out

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id()}>"


def is_api_provider(value: Any) -> bool:
    return isinstance(value, ApiProvider)
