"""Ports src/tracing/providers/index.ts: external trace backends (Tempo, Langfuse, Braintrust)."""

from __future__ import annotations

from typing import Any

from promptphoo.tracing.providers.types import TraceProvider, TraceProviderError

__all__ = [
    "TraceProvider",
    "TraceProviderError",
    "create_trace_provider",
    "is_external_trace_provider",
]


def create_trace_provider(config: dict[str, Any]) -> TraceProvider:
    """Create a trace provider instance for `config["id"]`. Raises for unsupported or
    misconfigured providers."""
    provider_id = config.get("id")
    if provider_id == "braintrust":
        from promptphoo.tracing.providers.braintrust import BraintrustProvider

        return BraintrustProvider(config)
    if provider_id == "langfuse":
        from promptphoo.tracing.providers.langfuse import LangfuseProvider

        return LangfuseProvider(config)
    if provider_id == "tempo":
        from promptphoo.tracing.providers.tempo import TempoProvider

        return TempoProvider(config)
    # Future providers: jaeger.
    if provider_id == "local":
        raise ValueError(
            "Local provider id should use TraceStore directly, not createTraceProvider()"
        )
    from promptphoo.jsvalues import js_string

    raise ValueError(f"Unknown trace provider id: {js_string(provider_id)}")


def is_external_trace_provider(config: dict[str, Any] | None = None) -> bool:
    """True when the config names a backend other than 'local' and has an endpoint."""
    return isinstance(config, dict) and config.get("id") != "local" and bool(config.get("endpoint"))
