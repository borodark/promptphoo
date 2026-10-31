"""Legacy Anthropic Text Completions provider. Ports src/providers/anthropic/completion.ts.

Sends POST {base}/v1/complete directly instead of going through the SDK.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from promptphoo import cache as cache_mod
from promptphoo import envars
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.anthropic.generic import (
    AnthropicAPIError,
    AnthropicGenericProvider,
    hash_anthropic_cache_value,
    post_anthropic,
)
from promptphoo.tokens import create_empty_token_usage

logger = logging.getLogger(__name__)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"


def _sdk_error_string(err: BaseException) -> str:
    """String(err) for an SDK error: `Error: <message>`."""
    if isinstance(err, AnthropicAPIError | ConnectionError):
        return f"Error: {err}"
    return f"{type(err).__name__}: {err}" if str(err) else type(err).__name__


class AnthropicCompletionProvider(AnthropicGenericProvider):
    # All legacy completion models are retired; kept for reference.
    ANTHROPIC_COMPLETION_MODELS = ["claude-2.0", "claude-2.1"]

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is not set. Set the ANTHROPIC_API_KEY environment variable or "
                "add `apiKey` to the provider config."
            )
        env_stop = envars.get_env_string("ANTHROPIC_STOP")
        try:
            stop = json.loads(env_stop) if env_stop else ["<|im_end|>", "<|endoftext|>"]
        except ValueError as err:
            raise ValueError(
                f"ANTHROPIC_STOP is not a valid JSON string: SyntaxError: {err}"
            ) from err

        scoped_raw = (self.env or {}).get("ANTHROPIC_TEMPERATURE")
        scoped_temperature = envars.parse_env_float("" if scoped_raw is None else str(scoped_raw))
        temperature = self.config.get("temperature")
        if temperature is None:
            temperature = (
                envars.get_env_float("ANTHROPIC_TEMPERATURE", 0)
                if scoped_temperature is None
                else scoped_temperature
            )
        max_tokens = self.config.get("max_tokens_to_sample")
        if max_tokens is None:
            max_tokens = envars.get_env_int("ANTHROPIC_MAX_TOKENS", 1024)
        params = {
            "model": self.model_name,
            "prompt": f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "stop_sequences": stop,
        }

        cache = cache_mod.get_cache()
        cache_key = (
            f"anthropic:completion:{self.model_name}:{self.get_cache_identity_hash()}:"
            f"{self.get_cache_namespace()}:{hash_anthropic_cache_value(params)}"
        )
        ephemeral_key = cache_mod.get_scoped_cache_key(cache_key)
        clear_generation = cache_mod.get_cache_clear_generation()
        should_use_cache = cache_mod.is_cache_enabled() and not self.has_custom_headers()

        if should_use_cache:
            cached = await self.get_cached_response(
                cache, cache_key, ephemeral_key, clear_generation
            )
            if cached:
                return {
                    "output": json.loads(cached),
                    "tokenUsage": create_empty_token_usage(),
                    "cached": True,
                }

        try:
            response = await post_anthropic(self, "/v1/complete", params, {})
        except Exception as err:
            return {"error": f"API call error: {_sdk_error_string(err)}"}
        completion = response["completion"]
        if should_use_cache:
            try:
                await self.set_cached_response(
                    cache,
                    cache_key,
                    ephemeral_key,
                    clear_generation,
                    cache_mod.get_cache_ttl_ms(),
                    json_stringify(completion),
                )
            except Exception as err:
                logger.error("Failed to cache response: %s", err)
        return {
            "output": completion,
            "tokenUsage": {**create_empty_token_usage(), "numRequests": 1},
        }
