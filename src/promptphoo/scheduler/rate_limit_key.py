"""Rate-limit pool keys. Ports src/scheduler/rateLimitKey.ts."""

from __future__ import annotations

import hashlib
from typing import Any

from promptphoo.jsvalues import is_truthy, js_string


def get_rate_limit_key(provider: Any) -> str:
    """A key identifying a unique rate-limit pool.

    The same provider with different API keys or regions gets different keys.
    """
    provider_id = provider.id()

    # Extract config that affects rate limiting
    config = getattr(provider, "config", None) or {}
    if not isinstance(config, dict):
        config = {}
    relevant_config: dict[str, str] = {}

    # Use last 4 chars of API key for differentiation (safe partial identifier)
    api_key = config.get("apiKey")
    if isinstance(api_key, str) and len(api_key) > 4:
        relevant_config["apiKeyTail"] = api_key[-4:]
    if is_truthy(config.get("apiBaseUrl")):
        relevant_config["apiBaseUrl"] = js_string(config["apiBaseUrl"])
    if is_truthy(config.get("region")):
        relevant_config["region"] = js_string(config["region"])
    if is_truthy(config.get("organization")):
        relevant_config["organization"] = js_string(config["organization"])

    # Create a stable hash
    config_parts = "|".join(f"{k}:{v}" for k, v in sorted(relevant_config.items()))

    if config_parts:
        # Use 12 hex chars (48 bits) for low collision probability
        digest = hashlib.sha256(config_parts.encode("utf-8")).hexdigest()[:12]
        return f"{provider_id}[{digest}]"

    return provider_id
