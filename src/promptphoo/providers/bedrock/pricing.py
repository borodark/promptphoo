"""Bedrock per-token pricing. Ports src/providers/bedrock/pricing.ts.

Rates are USD per 1M tokens (us-east-1 on-demand unless a regional table applies). Lookup is
first-match `in` (JS `includes()`) in insertion order, so more specific ids come first.
"""

from __future__ import annotations

import re
from typing import Any

from promptphoo.providers.anthropic.util import (
    CLAUDE_REGIONAL_ENDPOINT_PREMIUM,
    calculate_cache_input_cost,
    is_claude_fable_or_mythos5_model,
    is_claude_opus5_model,
    is_claude_opus55_model,
    is_claude_regional_premium_model,
    is_claude_sonnet5_model,
    is_claude_sonnet55_model,
)

# Nova bills cache reads at 25% of the input rate and cache writes at $0.
NOVA_CACHE_READ_RATIO = 0.25

BEDROCK_PRICING: dict[str, Any] = {
  # Claude 5
  "anthropic.claude-fable-5-1": { "input": 10, "output": 50 },
  "anthropic.claude-mythos-5-1": { "input": 10, "output": 50 },
  "anthropic.claude-fable-5": { "input": 10, "output": 50 },
  # Claude Opus 5.5. Must precede Opus 5: lookup is first-match `includes()`, and
  # `anthropic.claude-opus-5-5` contains `anthropic.claude-opus-5`. Cache reads bill at
  # 0.05x input ($0.20) per the AWS price list (see calculateCacheInputCost).
  "anthropic.claude-opus-5-5": { "input": 4, "output": 20 },
  # Claude Opus 5 (same list rates as Opus 4.8; full 1M context bills at the standard rate)
  "anthropic.claude-opus-5": { "input": 5, "output": 25 },
  # Claude Opus 4.8
  "anthropic.claude-opus-4-8": { "input": 5, "output": 25 },
  # Claude Opus 4.7
  "anthropic.claude-opus-4-7": { "input": 5, "output": 25 },
  # Claude Opus 4.6
  "anthropic.claude-opus-4-6": { "input": 5, "output": 25 },
  # Claude Opus 4.5
  "anthropic.claude-opus-4-5": { "input": 5, "output": 25 },
  # Claude Opus 4/4.1
  "anthropic.claude-opus-4": { "input": 15, "output": 75 },
  # Claude Sonnet 5.5 ($2/$10 global, $2.20/$11 regional per the AWS price list on 2026-09-29).
  # Must precede Sonnet 5: lookup is first-match `includes()`, and `anthropic.claude-sonnet-5-5`
  # contains `anthropic.claude-sonnet-5`.
  "anthropic.claude-sonnet-5-5": { "input": 2, "output": 10 },
  # Claude Sonnet 5 (standard list pricing; full 1M context bills at the standard rate)
  "anthropic.claude-sonnet-5": { "input": 3, "output": 15 },
  # Claude Sonnet 4.x — the point releases must precede the bare `-4` prefix: lookup is first-match
  # on `includes()`, and AWS publishes long-context meters only for Sonnet 4 itself. 4.5 and
  # 4.6 bill their full context flat, so they must not inherit Sonnet 4"s surcharge — add an
  # explicit entry here for any future 4.x release for the same reason.
  "anthropic.claude-sonnet-4-6": { "input": 3, "output": 15 },
  "anthropic.claude-sonnet-4-5": { "input": 3, "output": 15 },
  # Claude Sonnet 4 is the only Bedrock Claude model with published long-context meters
  # (verified 2026-09-01 in us-east-1, us-east-2, us-west-2, eu-west-1 and ap-northeast-1):
  # above 200K input tokens input doubles to $6 and output is 1.5x at $22.50. The matching
  # `-cache-read/-cache-write-...-long-context-...` meters are 10% and 1.25x of the tier"s
  # input rate, which falls out of deriving the cache rates from it.
  "anthropic.claude-sonnet-4": {
    "input": 3,
    "output": 15,
    "longContext": { "threshold": 200_000, "input": 6, "output": 22.5 },
  },
  # Claude Haiku 4.5
  "anthropic.claude-haiku-4": { "input": 1, "output": 5 },
  # Claude 3.x
  "anthropic.claude-3-opus": { "input": 15, "output": 75 },
  "anthropic.claude-3-5-sonnet": { "input": 3, "output": 15 },
  "anthropic.claude-3-7-sonnet": { "input": 3, "output": 15 },
  "anthropic.claude-3-5-haiku": { "input": 0.8, "output": 4 },
  "anthropic.claude-3-haiku": { "input": 0.25, "output": 1.25 },
  # Amazon Nova
  "amazon.nova-micro": { "input": 0.035, "output": 0.14 },
  "amazon.nova-lite": { "input": 0.06, "output": 0.24 },
  "amazon.nova-pro": { "input": 0.8, "output": 3.2 },
  "amazon.nova-premier": { "input": 2.5, "output": 12.5 },
  # Amazon Nova 2 (reasoning models). The cross-region global profile is cheaper
  # ($0.30/$2.50); this is the plain us-east-1 on-demand rate.
  "amazon.nova-2-lite": { "input": 0.33, "output": 2.75 },
  # Amazon Titan Text
  "amazon.titan-text-lite": { "input": 0.15, "output": 0.2 },
  "amazon.titan-text-express": { "input": 0.2, "output": 0.6 },
  "amazon.titan-text-premier": { "input": 0.5, "output": 1.5 },
  # Meta Llama
  "meta.llama3-1-8b": { "input": 0.22, "output": 0.22 },
  "meta.llama3-1-70b": { "input": 0.72, "output": 0.72 },
  "meta.llama3-1-405b": { "input": 5.32, "output": 16 },
  "meta.llama3-2-1b": { "input": 0.1, "output": 0.1 },
  "meta.llama3-2-3b": { "input": 0.15, "output": 0.15 },
  "meta.llama3-2-11b": { "input": 0.16, "output": 0.16 },
  "meta.llama3-2-90b": { "input": 0.72, "output": 0.72 },
  "meta.llama3-3-70b": { "input": 0.72, "output": 0.72 },
  "meta.llama4-scout": { "input": 0.17, "output": 0.66 },
  "meta.llama4-maverick": { "input": 0.24, "output": 0.97 },
  "meta.llama4": { "input": 1.0, "output": 3.0 },
  # Mistral
  "mistral.mistral-7b": { "input": 0.15, "output": 0.2 },
  "mistral.mixtral-8x7b": { "input": 0.45, "output": 0.7 },
  "mistral.mistral-large": { "input": 4, "output": 12 },
  "mistral.mistral-small": { "input": 1, "output": 3 },
  "mistral.pixtral-large": { "input": 2, "output": 6 },
  # AI21 Jamba
  "ai21.jamba-1-5-mini": { "input": 0.2, "output": 0.4 },
  "ai21.jamba-1-5-large": { "input": 2, "output": 8 },
  # Cohere
  "cohere.command-r-plus": { "input": 3, "output": 15 },
  "cohere.command-r": { "input": 0.5, "output": 1.5 },
  # DeepSeek
  "deepseek.deepseek-r1": { "input": 1.35, "output": 5.4 },
  "deepseek.r1": { "input": 1.35, "output": 5.4 },
  # Qwen
  "qwen.qwen3-32b": { "input": 0.15, "output": 0.6 },
  "qwen.qwen3-235b": { "input": 0.18, "output": 0.54 },
  "qwen.qwen3-coder-30b": { "input": 0.15, "output": 0.6 },
  "qwen.qwen3-coder-480b": { "input": 1.5, "output": 7.5 },
  "qwen.qwen3": { "input": 0.5, "output": 1.5 },
  # Writer Palmyra
  "writer.palmyra-vision": { "input": 0.15, "output": 0.6 },
  "writer.palmyra-x5": { "input": 0.6, "output": 6 },
  "writer.palmyra-x4": { "input": 2.5, "output": 10 },
  # Newer OpenAI Chat-compatible families (base rates from https:#aws.amazon.com/bedrock/pricing/;
  # verify there as rates change). Keys are variant-specific because rates differ within a
  # family, and `includes()` matches the first key in insertion order — so list more specific
  # ids first (e.g. `glm-4.7-flash` before `glm-4.7`). `includes()` also tolerates geo prefixes
  # (e.g. `us.zai.glm-5`). Region-specific overrides are applied below where AWS publishes them.
  # Z.AI GLM
  "zai.glm-5": { "input": 1.0, "output": 3.2 },
  "zai.glm-4.7-flash": { "input": 0.07, "output": 0.4 },
  "zai.glm-4.7": { "input": 0.6, "output": 2.2 },
  # MiniMax (M2 / M2.1 / M2.5 share a rate)
  "minimax.minimax-m2": { "input": 0.3, "output": 1.2 },
  # Moonshot Kimi
  "kimi-k2.5": { "input": 0.6, "output": 3.0 },
  "kimi-k2-thinking": { "input": 0.6, "output": 2.5 },
  # NVIDIA Nemotron
  "nemotron-nano-12b-v2": { "input": 0.2, "output": 0.6 },
  "nemotron-nano-3-30b": { "input": 0.06, "output": 0.24 },
  "nemotron-nano": { "input": 0.06, "output": 0.23 },
  "nemotron-super": { "input": 0.15, "output": 0.65 },
  # Google Gemma 3
  "gemma-3-4b": { "input": 0.04, "output": 0.08 },
  "gemma-3-12b": { "input": 0.09, "output": 0.29 },
  "gemma-3-27b": { "input": 0.23, "output": 0.38 },
  # Open-weight GPT-OSS Runtime entries. Frontier Mantle selectors use separate billing
  # through the OpenAI-compatible adapters (see src/providers/bedrock/openaiResponses.ts).
  "openai.gpt-oss-120b": { "input": 0.15, "output": 0.6 },
  "openai.gpt-oss-20b": { "input": 0.07, "output": 0.3 },
}  # fmt: skip

GPT_OSS_REGION_PRICING: dict[str, Any] = {
  "ap-northeast-1": {
    "openai.gpt-oss-120b": { "input": 0.18, "output": 0.73 },
    "openai.gpt-oss-20b": { "input": 0.08, "output": 0.36 },
  },
  "ap-south-1": {
    "openai.gpt-oss-120b": { "input": 0.18, "output": 0.71 },
    "openai.gpt-oss-20b": { "input": 0.08, "output": 0.35 },
  },
  "ap-southeast-2": {
    "openai.gpt-oss-120b": { "input": 0.1545, "output": 0.618 },
    "openai.gpt-oss-20b": { "input": 0.0721, "output": 0.309 },
  },
  "ap-southeast-3": {
    "openai.gpt-oss-120b": { "input": 0.16, "output": 0.62 },
    "openai.gpt-oss-20b": { "input": 0.07, "output": 0.31 },
  },
  "eu-central-1": {
    "openai.gpt-oss-120b": { "input": 0.2, "output": 0.79 },
    "openai.gpt-oss-20b": { "input": 0.09, "output": 0.4 },
  },
  "eu-north-1": {
    "openai.gpt-oss-120b": { "input": 0.15, "output": 0.6 },
    "openai.gpt-oss-20b": { "input": 0.07, "output": 0.3 },
  },
  "eu-south-1": {
    "openai.gpt-oss-120b": { "input": 0.2, "output": 0.79 },
    "openai.gpt-oss-20b": { "input": 0.09, "output": 0.4 },
  },
  "eu-west-1": {
    "openai.gpt-oss-120b": { "input": 0.18, "output": 0.7 },
    "openai.gpt-oss-20b": { "input": 0.08, "output": 0.35 },
  },
  "eu-west-2": {
    "openai.gpt-oss-120b": { "input": 0.23, "output": 0.93 },
    "openai.gpt-oss-20b": { "input": 0.11, "output": 0.47 },
  },
  "sa-east-1": {
    "openai.gpt-oss-120b": { "input": 0.18, "output": 0.73 },
    "openai.gpt-oss-20b": { "input": 0.08, "output": 0.36 },
  },
  "us-east-1": {
    "openai.gpt-oss-120b": { "input": 0.15, "output": 0.6 },
    "openai.gpt-oss-20b": { "input": 0.07, "output": 0.3 },
  },
  "us-east-2": {
    "openai.gpt-oss-120b": { "input": 0.15, "output": 0.6 },
    "openai.gpt-oss-20b": { "input": 0.07, "output": 0.3 },
  },
  "us-gov-east-1": {
    "openai.gpt-oss-120b": { "input": 0.18, "output": 0.72 },
    "openai.gpt-oss-20b": { "input": 0.084, "output": 0.36 },
  },
  "us-gov-west-1": {
    "openai.gpt-oss-120b": { "input": 0.18, "output": 0.72 },
    "openai.gpt-oss-20b": { "input": 0.084, "output": 0.36 },
  },
  "us-west-2": {
    "openai.gpt-oss-120b": { "input": 0.15, "output": 0.6 },
    "openai.gpt-oss-20b": { "input": 0.07, "output": 0.3 },
  },
}  # fmt: skip

AP_NORTHEAST_1_AND_SA_EAST_1_PRICING: dict[str, Any] = {
  "zai.glm-5": { "input": 1.2, "output": 3.84 },
  "zai.glm-4.7-flash": { "input": 0.08, "output": 0.48 },
  "zai.glm-4.7": { "input": 0.72, "output": 2.64 },
  "minimax.minimax-m2.5": { "input": 0.36, "output": 1.44 },
  "minimax.minimax-m2.1": { "input": 0.36, "output": 1.44 },
  "minimax.minimax-m2": { "input": 0.36, "output": 1.45 },
  "kimi-k2.5": { "input": 0.72, "output": 3.6 },
  "kimi-k2-thinking": { "input": 0.73, "output": 3.03 },
  "nemotron-nano-12b-v2": { "input": 0.24, "output": 0.73 },
  "nemotron-nano-3-30b": { "input": 0.07, "output": 0.29 },
  "nemotron-nano": { "input": 0.07, "output": 0.28 },
  "nemotron-super": { "input": 0.18, "output": 0.78 },
  "gemma-3-4b": { "input": 0.05, "output": 0.1 },
  "gemma-3-12b": { "input": 0.11, "output": 0.35 },
  "gemma-3-27b": { "input": 0.28, "output": 0.46 },
}  # fmt: skip

AP_SOUTHEAST_3_EU_CENTRAL_1_AND_EU_NORTH_1_PRICING: dict[str, Any] = {
  "zai.glm-5": { "input": 1.2, "output": 3.84 },
  "zai.glm-4.7-flash": { "input": 0.08, "output": 0.48 },
  "zai.glm-4.7": { "input": 0.72, "output": 2.64 },
  "minimax.minimax-m2.5": { "input": 0.36, "output": 1.44 },
  "minimax.minimax-m2.1": { "input": 0.36, "output": 1.44 },
  "minimax.minimax-m2": { "input": 0.36, "output": 1.44 },
  "kimi-k2.5": { "input": 0.72, "output": 3.6 },
  "kimi-k2-thinking": { "input": 0.72, "output": 3.0 },
  "nemotron-nano-12b-v2": { "input": 0.24, "output": 0.72 },
  "nemotron-nano-3-30b": { "input": 0.072, "output": 0.288 },
  "nemotron-nano": { "input": 0.072, "output": 0.276 },
  "nemotron-super": { "input": 0.18, "output": 0.78 },
  "gemma-3-4b": { "input": 0.048, "output": 0.096 },
  "gemma-3-12b": { "input": 0.108, "output": 0.348 },
  "gemma-3-27b": { "input": 0.276, "output": 0.456 },
}  # fmt: skip

EU_SOUTH_1_AND_EU_WEST_1_PRICING: dict[str, Any] = {
  "zai.glm-5": { "input": 1.2, "output": 3.84 },
  "zai.glm-4.7-flash": { "input": 0.08, "output": 0.48 },
  "zai.glm-4.7": { "input": 0.72, "output": 2.64 },
  "minimax.minimax-m2.5": { "input": 0.36, "output": 1.44 },
  "minimax.minimax-m2.1": { "input": 0.36, "output": 1.44 },
  "minimax.minimax-m2": { "input": 0.35, "output": 1.41 },
  "kimi-k2.5": { "input": 0.72, "output": 3.6 },
  "kimi-k2-thinking": { "input": 0.72, "output": 3.0 },
  "nemotron-nano-12b-v2": { "input": 0.23, "output": 0.7 },
  "nemotron-nano-3-30b": { "input": 0.07, "output": 0.28 },
  "nemotron-nano": { "input": 0.07, "output": 0.27 },
  "nemotron-super": { "input": 0.18, "output": 0.78 },
  "gemma-3-4b": { "input": 0.05, "output": 0.09 },
  "gemma-3-12b": { "input": 0.11, "output": 0.34 },
  "gemma-3-27b": { "input": 0.27, "output": 0.45 },
}  # fmt: skip

US_GOV_PRICING: dict[str, Any] = {
  # AWS publishes GovCloud rates directly; do not apply the commercial regional premium again.
  "anthropic.claude-opus-4-8": { "input": 6, "output": 30 },
  "nemotron-nano-12b-v2": { "input": 0.24, "output": 0.72 },
  "nemotron-nano-3-30b": { "input": 0.072, "output": 0.288 },
  "nemotron-nano": { "input": 0.072, "output": 0.276 },
  "nemotron-super": { "input": 0.18, "output": 0.78 },
}  # fmt: skip

BEDROCK_REGION_PRICING: dict[str, Any] = {
  "ap-northeast-1": AP_NORTHEAST_1_AND_SA_EAST_1_PRICING,
  "ap-south-1": {
    "zai.glm-5": { "input": 1.2, "output": 3.84 },
    "zai.glm-4.7-flash": { "input": 0.08, "output": 0.48 },
    "zai.glm-4.7": { "input": 0.72, "output": 2.64 },
    "minimax.minimax-m2.5": { "input": 0.36, "output": 1.44 },
    "minimax.minimax-m2.1": { "input": 0.36, "output": 1.44 },
    "minimax.minimax-m2": { "input": 0.35, "output": 1.41 },
    "kimi-k2.5": { "input": 0.72, "output": 3.6 },
    "kimi-k2-thinking": { "input": 0.71, "output": 2.94 },
    "nemotron-nano-12b-v2": { "input": 0.24, "output": 0.71 },
    "nemotron-nano-3-30b": { "input": 0.07, "output": 0.28 },
    "nemotron-nano": { "input": 0.07, "output": 0.27 },
    "nemotron-super": { "input": 0.18, "output": 0.78 },
    "gemma-3-4b": { "input": 0.05, "output": 0.09 },
    "gemma-3-12b": { "input": 0.11, "output": 0.34 },
    "gemma-3-27b": { "input": 0.27, "output": 0.45 },
  },
  # These are AWS"s published ap-southeast-2 meters, not the US rate scaled by a regional
  # uplift. Four entries had been derived that way and drifted from the real values.
  "ap-southeast-2": {
    "zai.glm-5": { "input": 1.03, "output": 3.3 },
    "zai.glm-4.7-flash": { "input": 0.07, "output": 0.41 },
    "zai.glm-4.7": { "input": 0.62, "output": 2.27 },
    "minimax.minimax-m2.5": { "input": 0.31, "output": 1.24 },
    "minimax.minimax-m2.1": { "input": 0.31, "output": 1.24 },
    "minimax.minimax-m2": { "input": 0.309, "output": 1.236 },
    "kimi-k2.5": { "input": 0.62, "output": 3.09 },
    "kimi-k2-thinking": { "input": 0.618, "output": 2.575 },
    "nemotron-nano-12b-v2": { "input": 0.206, "output": 0.618 },
    "nemotron-nano-3-30b": { "input": 0.0618, "output": 0.2472 },
    "nemotron-nano": { "input": 0.0618, "output": 0.2369 },
    "nemotron-super": { "input": 0.15, "output": 0.67 },
    "gemma-3-4b": { "input": 0.0412, "output": 0.0824 },
    "gemma-3-12b": { "input": 0.0927, "output": 0.2987 },
    "gemma-3-27b": { "input": 0.2369, "output": 0.3914 },
  },
  "ap-southeast-3": AP_SOUTHEAST_3_EU_CENTRAL_1_AND_EU_NORTH_1_PRICING,
  "eu-central-1": AP_SOUTHEAST_3_EU_CENTRAL_1_AND_EU_NORTH_1_PRICING,
  "eu-north-1": AP_SOUTHEAST_3_EU_CENTRAL_1_AND_EU_NORTH_1_PRICING,
  "eu-south-1": EU_SOUTH_1_AND_EU_WEST_1_PRICING,
  "eu-west-1": EU_SOUTH_1_AND_EU_WEST_1_PRICING,
  "eu-west-2": {
    "zai.glm-5": { "input": 1.55, "output": 4.96 },
    "zai.glm-4.7-flash": { "input": 0.11, "output": 0.62 },
    "zai.glm-4.7": { "input": 0.93, "output": 3.41 },
    "minimax.minimax-m2.5": { "input": 0.47, "output": 1.86 },
    "minimax.minimax-m2.1": { "input": 0.47, "output": 1.86 },
    "minimax.minimax-m2": { "input": 0.47, "output": 1.86 },
    "kimi-k2.5": { "input": 0.93, "output": 4.65 },
    "kimi-k2-thinking": { "input": 0.93, "output": 3.875 },
    "nemotron-nano-12b-v2": { "input": 0.31, "output": 0.93 },
    "nemotron-nano-3-30b": { "input": 0.09, "output": 0.37 },
    "nemotron-nano": { "input": 0.09, "output": 0.36 },
    "nemotron-super": { "input": 0.23, "output": 1.01 },
    "gemma-3-4b": { "input": 0.06, "output": 0.12 },
    "gemma-3-12b": { "input": 0.14, "output": 0.45 },
    "gemma-3-27b": { "input": 0.36, "output": 0.59 },
  },
  "sa-east-1": AP_NORTHEAST_1_AND_SA_EAST_1_PRICING,
  "us-gov-east-1": US_GOV_PRICING,
  "us-gov-west-1": US_GOV_PRICING,
}  # fmt: skip


BEDROCK_REGION_PRICING_MODEL_PREFIXES = (
    "zai.glm-",
    "minimax.minimax-",
    "kimi-k2",
    "nemotron-",
    "gemma-3-",
    "openai.gpt-oss-",
)

BEDROCK_INVOKE_PRICING_MODEL_PREFIXES = (
    "zai.glm-",
    "minimax.minimax-",
    "kimi-k2",
    "nemotron-",
    "gemma-3-",
    "writer.palmyra-",
    "openai.gpt-oss-",
)


def _is_nova_prompt_caching_model(normalized_model_id: str) -> bool:
    return "amazon.nova-" in normalized_model_id


def _get_bedrock_region_pricing(
    normalized_model_id: str, region: str | None = None
) -> dict[str, Any] | None:
    if region:
        return BEDROCK_REGION_PRICING.get(region.lower())
    if normalized_model_id.startswith("us-gov.") or "/us-gov." in normalized_model_id:
        return US_GOV_PRICING
    return None


def get_bedrock_pricing(normalized_model_id: str, region: str | None = None) -> dict | None:
    if "openai.gpt-oss-" in normalized_model_id and region:
        pricing = GPT_OSS_REGION_PRICING.get(region.lower())
        if not pricing:
            return None
        for prefix, model_pricing in pricing.items():
            if prefix in normalized_model_id:
                return model_pricing
        return None

    region_pricing = _get_bedrock_region_pricing(normalized_model_id, region)
    if region_pricing:
        for prefix, pricing in region_pricing.items():
            if prefix in normalized_model_id:
                return pricing
        if any(p in normalized_model_id for p in BEDROCK_REGION_PRICING_MODEL_PREFIXES):
            return None

    for prefix, pricing in BEDROCK_PRICING.items():
        if prefix in normalized_model_id:
            return pricing
    return None


def _has_published_regional_pricing(normalized_model_id: str, region: str | None) -> bool:
    region_pricing = _get_bedrock_region_pricing(normalized_model_id, region)
    return bool(region_pricing) and any(p in normalized_model_id for p in region_pricing or {})


def is_bedrock_grok46_profile(model_id: str) -> bool:
    """Only the documented US and global Grok 4.6 Runtime inference profiles."""
    return (
        re.search(r"(?:^|inference-profile/)(?:us|global)\.xai\.grok-4\.6$", model_id.lower())
        is not None
    )


def calculate_bedrock_cost(
    model_id: str,
    prompt_tokens: float | None = None,
    completion_tokens: float | None = None,
    cache_read_tokens: float | None = 0,
    cache_write_tokens: float | None = 0,
    region: str | None = None,
    service_tier: dict[str, Any] | None = None,
    cache_write_1h_tokens: float | None = 0,
) -> float | None:
    """Cost of one Bedrock call from token usage."""
    cache_read_tokens = cache_read_tokens or 0
    cache_write_tokens = cache_write_tokens or 0
    cache_write_1h_tokens = cache_write_1h_tokens or 0
    if prompt_tokens is None or completion_tokens is None:
        return None
    tier_type = (service_tier or {}).get("type")
    # Reserved throughput is billed as fixed monthly capacity, not per token.
    if tier_type == "reserved":
        return None

    normalized = model_id.lower()
    if is_bedrock_grok46_profile(normalized):
        if (tier_type and tier_type != "default") or cache_write_tokens > 0:
            return None
        global_profile = normalized.startswith("global.") or "/global." in normalized
        input_rate, output_rate, cached_rate = (2, 6, 0.5) if global_profile else (2.2, 6.6, 0.55)
        return (
            prompt_tokens * input_rate
            + cache_read_tokens * cached_rate
            + completion_tokens * output_rate
        ) / 1_000_000
    pricing = get_bedrock_pricing(normalized, region)
    if not pricing:
        return None
    is_global = normalized.startswith("global.") or "/global." in normalized
    endpoint_multiplier = (
        CLAUDE_REGIONAL_ENDPOINT_PREMIUM
        if is_claude_regional_premium_model(normalized)
        and not is_global
        and not _has_published_regional_pricing(normalized, region)
        else 1
    )
    tier_multiplier = 1.75 if tier_type == "priority" else 0.5 if tier_type == "flex" else 1
    multiplier = endpoint_multiplier * tier_multiplier
    # Long-context tiers are billed on total input size, including cache reads and writes.
    total_input = prompt_tokens + cache_read_tokens + cache_write_tokens
    long_context = pricing.get("longContext")
    tier = long_context if long_context and total_input >= long_context["threshold"] else pricing

    input_rate = (tier["input"] / 1_000_000) * multiplier
    if "anthropic.claude" in normalized:
        input_cost = calculate_cache_input_cost(
            input_rate,
            prompt_tokens,
            cache_read_tokens,
            cache_write_tokens,
            normalized,
            cache_write_1h_tokens,
        )
    elif _is_nova_prompt_caching_model(normalized):
        input_cost = (
            prompt_tokens * input_rate + cache_read_tokens * input_rate * NOVA_CACHE_READ_RATIO
        )
    else:
        input_cost = prompt_tokens * input_rate
    output_cost = (completion_tokens / 1_000_000) * tier["output"] * multiplier
    return input_cost + output_cost


def calculate_bedrock_invoke_model_cost(
    model_id: str,
    prompt_tokens: float | None = None,
    completion_tokens: float | None = None,
    cache_read_tokens: float | None = 0,
    cache_write_tokens: float | None = 0,
    region: str | None = None,
    cache_write_1h_tokens: float | None = 0,
) -> float | None:
    """InvokeModel cost, only for models whose Runtime rates are verified (fail closed)."""
    normalized = model_id.lower()
    if (
        not is_bedrock_grok46_profile(normalized)
        and not is_claude_fable_or_mythos5_model(normalized)
        and not is_claude_opus5_model(normalized)
        and not is_claude_opus55_model(normalized)
        and not is_claude_sonnet55_model(normalized)
        and not is_claude_sonnet5_model(normalized)
        and not any(p in normalized for p in BEDROCK_INVOKE_PRICING_MODEL_PREFIXES)
    ):
        return None
    return calculate_bedrock_cost(
        model_id,
        prompt_tokens,
        completion_tokens,
        cache_read_tokens,
        cache_write_tokens,
        region,
        None,
        cache_write_1h_tokens,
    )
