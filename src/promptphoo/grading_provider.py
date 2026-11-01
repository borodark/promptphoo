"""Grading provider references to configured providers. Ports src/util/gradingProvider.ts.

A grader written as the id or label of a provider under `providers:` reuses that loaded
instance instead of constructing a second one.
"""

from __future__ import annotations

import logging
from typing import Any

from promptphoo.providers.base import ApiProvider, is_api_provider

logger = logging.getLogger(__name__)

GRADING_PROVIDER_TYPE_KEYS = ("text", "embedding", "classification", "moderation")


def _is_typed_provider_value(value: Any) -> bool:
    if isinstance(value, str):
        return len(value) > 0
    if is_api_provider(value):
        return True
    if not isinstance(value, dict):
        return False
    provider_id = value.get("id")
    return isinstance(provider_id, str) and len(provider_id) > 0


def is_provider_type_map(provider: Any) -> bool:
    return (
        isinstance(provider, dict)
        and "id" not in provider
        and any(
            key in provider and _is_typed_provider_value(provider[key])
            for key in GRADING_PROVIDER_TYPE_KEYS
        )
    )


def build_configured_provider_map(providers: list[Any]) -> dict[str, ApiProvider]:
    """Look up configured providers by id() and label. Ids are added first; labels only fill
    keys no id has claimed, so a label can never shadow another provider's id."""
    provider_map: dict[str, ApiProvider] = {}
    for provider in providers:
        provider_map[provider.id()] = provider
    for provider in providers:
        label = getattr(provider, "label", None)
        if label and label not in provider_map:
            provider_map[label] = provider
    return provider_map


class ReuseWarningProviderMap(dict):
    """A configured-provider map that warns, once per reference, when a grader named by a
    provider's id (not its label) reuses that provider's non-empty config. Upstream reuses it
    silently, so a target's `max_tokens` or `temperature` quietly applies to grading (D72)."""

    def __init__(self, provider_map: dict[str, ApiProvider]):
        super().__init__(provider_map)
        self._warned: set[str] = set()

    def get(self, key: Any, default: Any = None) -> Any:
        provider = super().get(key, default)
        if (
            provider is not default
            and isinstance(key, str)
            and key not in self._warned
            and key == provider.id()
            and key != getattr(provider, "label", None)
            and getattr(provider, "config", None)
        ):
            self._warned.add(key)
            keys = ", ".join(sorted(provider.config))
            logger.warning(
                'Grader "%s" is the id of a configured provider, so grading reuses that provider '
                "and its config (%s). To grade with other settings, give the grader its own "
                "entry: provider: {id: %s, config: {...}}.",
                key,
                keys,
                key,
            )
        return provider


def _resolve_typed_provider_value(provider: Any, provider_map: dict[str, ApiProvider]) -> Any:
    if isinstance(provider, str):
        return provider_map.get(provider, provider)
    if is_api_provider(provider):
        return provider
    # An id-only typed entry is a reference; entries with options stay inline.
    if isinstance(provider, dict) and len(provider) == 1 and isinstance(provider.get("id"), str):
        configured = provider_map.get(provider["id"])
        if configured is not None:
            return configured
    return provider


def resolve_configured_provider_reference(
    provider: Any, provider_map: dict[str, ApiProvider]
) -> Any:
    """Unconfigured typed entries stay lazy and inherit the evaluation scope when loaded."""
    if isinstance(provider, str):
        return provider_map.get(provider, provider)
    if not is_provider_type_map(provider):
        return provider
    resolved: dict[str, Any] | None = None
    for provider_type in GRADING_PROVIDER_TYPE_KEYS:
        nested = provider.get(provider_type)
        if not nested:
            continue
        resolved_provider = _resolve_typed_provider_value(nested, provider_map)
        if resolved_provider is not nested:
            if resolved is None:
                resolved = dict(provider)
            resolved[provider_type] = resolved_provider
    return resolved if resolved is not None else provider


def _resolve_assertion_provider_references(
    assertion: dict[str, Any], provider_map: dict[str, ApiProvider]
) -> dict[str, Any]:
    if assertion.get("type") == "assert-set":
        children = assertion.get("assert") or []
        resolved_children = [
            _resolve_assertion_provider_references(child, provider_map) for child in children
        ]
        if all(a is b for a, b in zip(resolved_children, children, strict=True)):
            return assertion
        return {**assertion, "assert": resolved_children}
    if "provider" not in assertion:
        return assertion
    provider = resolve_configured_provider_reference(assertion.get("provider"), provider_map)
    return (
        assertion if provider is assertion.get("provider") else {**assertion, "provider": provider}
    )


def resolve_runtime_grading_provider_references(
    test_case: dict[str, Any], provider_map: dict[str, ApiProvider]
) -> None:
    """Point `options.provider` and assertion `provider`s that name a configured provider at
    that loaded instance (resolveRuntimeGradingProviderReferences in src/evaluator.ts)."""
    options = test_case.get("options")
    if isinstance(options, dict) and options.get("provider"):
        provider = resolve_configured_provider_reference(options["provider"], provider_map)
        if provider is not options["provider"]:
            test_case["options"] = {**options, "provider": provider}

    assertions = test_case.get("assert")
    if assertions:
        resolved = [_resolve_assertion_provider_references(a, provider_map) for a in assertions]
        if any(a is not b for a, b in zip(resolved, assertions, strict=True)):
            test_case["assert"] = resolved
