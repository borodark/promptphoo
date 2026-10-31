"""Adaptive rate limit scheduler. Ports src/scheduler/index.ts.

Learns rate limits from response headers and adapts concurrency dynamically. Zero
configuration required.
"""

from promptphoo.scheduler.adaptive_concurrency import (
    WARNING_THRESHOLD,
    AdaptiveConcurrency,
    ConcurrencyChangeResult,
)
from promptphoo.scheduler.cancellation import (
    AbortController,
    AbortError,
    AbortSignal,
    sleep_with_abort,
)
from promptphoo.scheduler.header_parser import (
    ParsedRateLimitHeaders,
    parse_rate_limit_headers,
    parse_retry_after,
)
from promptphoo.scheduler.provider_call_execution_context import (
    get_provider_call_execution_context,
    get_provider_call_tracing_context,
)
from promptphoo.scheduler.provider_rate_limit_state import (
    ProviderMetrics,
    ProviderRateLimitState,
)
from promptphoo.scheduler.provider_wrapper import (
    create_provider_rate_limit_options,
    is_rate_limit_wrapped,
    wrap_provider_with_rate_limiting,
    wrap_providers_with_rate_limiting,
)
from promptphoo.scheduler.rate_limit_key import get_rate_limit_key
from promptphoo.scheduler.rate_limit_registry import RateLimitRegistry, RateLimitRegistryOptions
from promptphoo.scheduler.retry_policy import (
    DEFAULT_RETRY_POLICY,
    RetryPolicy,
    get_retry_delay,
    should_retry,
)
from promptphoo.scheduler.slot_queue import SlotQueue, SlotQueueOptions
from promptphoo.scheduler.types import RateLimitExecuteOptions

__all__ = [
    "DEFAULT_RETRY_POLICY",
    "WARNING_THRESHOLD",
    "AbortController",
    "AbortError",
    "AbortSignal",
    "AdaptiveConcurrency",
    "ConcurrencyChangeResult",
    "ParsedRateLimitHeaders",
    "ProviderMetrics",
    "ProviderRateLimitState",
    "RateLimitExecuteOptions",
    "RateLimitRegistry",
    "RateLimitRegistryOptions",
    "RetryPolicy",
    "SlotQueue",
    "SlotQueueOptions",
    "create_provider_rate_limit_options",
    "get_provider_call_execution_context",
    "get_provider_call_tracing_context",
    "get_rate_limit_key",
    "get_retry_delay",
    "is_rate_limit_wrapped",
    "parse_rate_limit_headers",
    "parse_retry_after",
    "should_retry",
    "sleep_with_abort",
    "wrap_provider_with_rate_limiting",
    "wrap_providers_with_rate_limiting",
]
