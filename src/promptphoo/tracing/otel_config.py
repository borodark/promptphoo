"""OpenTelemetry tracing configuration from the environment. Ports src/tracing/otelConfig.ts."""

from __future__ import annotations

from typing import TypedDict

from promptphoo.envars import get_env_bool, get_env_string


class _OtelConfigRequired(TypedDict):
    enabled: bool
    serviceName: str
    localExport: bool
    debug: bool


class OtelConfig(_OtelConfigRequired, total=False):
    # External OTLP endpoint URL for trace export; without it spans are only stored locally.
    endpoint: str


def get_otel_config_from_env() -> OtelConfig:
    endpoint = get_env_string("PROMPTFOO_OTEL_ENDPOINT") or get_env_string(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    config: OtelConfig = {
        "enabled": get_env_bool("PROMPTFOO_OTEL_ENABLED", False),
        "serviceName": get_env_string("PROMPTFOO_OTEL_SERVICE_NAME", "promptfoo"),
        "localExport": get_env_bool("PROMPTFOO_OTEL_LOCAL_EXPORT", True),
        "debug": get_env_bool("PROMPTFOO_OTEL_DEBUG", False),
    }
    if endpoint:
        config["endpoint"] = endpoint
    return config


def get_default_otel_config() -> OtelConfig:
    """The env config with tracing switched on (tracing was already found to be enabled)."""
    return {**get_otel_config_from_env(), "enabled": True}
