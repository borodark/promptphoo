"""Moderation matcher. Ports src/matchers/moderation.ts (matchesModeration)."""

from __future__ import annotations

from typing import Any

from promptphoo import envars
from promptphoo.matchers.providers import call_grading_provider, get_and_check_provider
from promptphoo.matchers.shared import normalize_matcher_token_usage

# src/redteam/constants/plugins.ts. No Replicate provider is ported, so loading it fails loudly.
LLAMA_GUARD_REPLICATE_PROVIDER = "replicate:moderation:meta/llama-guard-4-12b"


async def matches_moderation(
    user_prompt: str,
    assistant_response: str,
    categories: list[str] | None = None,
    grading: dict[str, Any] | None = None,
) -> dict[str, Any]:
    categories = categories or []
    if not assistant_response:
        return {"pass": True, "score": 1, "reason": "No output to moderate"}

    from promptphoo.providers.defaults import get_default_providers

    default_providers = await get_default_providers()

    # Only try to use Replicate if OpenAI is not available.
    has_openai_key = envars.get_env_string("OPENAI_API_KEY")
    has_replicate_key = not has_openai_key and (
        envars.get_env_string("REPLICATE_API_KEY") or envars.get_env_string("REPLICATE_API_TOKEN")
    )
    if has_replicate_key:
        from promptphoo.providers import load_api_provider

        default_moderation_provider = await load_api_provider(LLAMA_GUARD_REPLICATE_PROVIDER)
    else:
        default_moderation_provider = default_providers.get("moderationProvider")

    provider = await get_and_check_provider(
        "moderation",
        (grading or {}).get("provider"),
        default_moderation_provider,
        "moderation check",
    )
    if not provider:
        raise ValueError("Invariant failed: Moderation provider must be defined")

    resp = await call_grading_provider(
        provider,
        "moderation",
        lambda _context: provider.call_moderation_api(user_prompt, assistant_response),
    )
    token_usage_result = (
        {"tokensUsed": normalize_matcher_token_usage(resp["tokenUsage"])}
        if resp.get("tokenUsage")
        else {}
    )
    if resp.get("error"):
        # A provider/transport error is not evidence about the content; tag it as a grader
        # failure so `not-moderation` propagates it instead of flipping it into a pass.
        return {
            "pass": False,
            "score": 0,
            "reason": f"Moderation API error: {resp['error']}",
            **token_usage_result,
            "metadata": {"graderError": True},
        }

    flags = resp.get("flags")
    if not flags:
        return {
            "pass": True,
            "score": 1,
            "reason": "No moderation flags detected",
            **token_usage_result,
        }
    filtered = flags if not categories else [f for f in flags if f.get("code") in categories]
    if filtered:
        return {
            "pass": False,
            "score": 0,
            "reason": "Moderation flags detected: "
            + ", ".join(str(f.get("description")) for f in filtered),
            **token_usage_result,
        }
    return {
        "pass": True,
        "score": 1,
        "reason": "No relevant moderation flags detected",
        **token_usage_result,
    }
