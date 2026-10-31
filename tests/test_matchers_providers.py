"""Ports test/matchers/getGradingProvider.test.ts and test/matchers/callProviderWithContext.test.ts,
plus getAndCheckProvider and callEmbeddingProvider from src/matchers/providers.ts."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from promptphoo import cli_state
from promptphoo import providers as providers_mod
from promptphoo.matchers import providers as mp
from promptphoo.providers.base import ApiProvider
from promptphoo.scheduler.cancellation import AbortController, AbortError
from promptphoo.scheduler.provider_call_execution_context import (
    with_provider_call_execution_context,
    with_provider_call_tracing_context,
)
from promptphoo.scheduler.provider_call_queue import ProviderGroupedCallQueue
from promptphoo.scheduler.provider_wrapper import wrap_provider_with_rate_limiting


class MockProvider(ApiProvider):
    def __init__(self, pid: str = "test-grader", response: Any = None) -> None:
        super().__init__()
        self._pid = pid
        self.response = {"output": "ok"} if response is None else response
        self.calls: list[tuple] = []

    def default_id(self) -> str:
        return self._pid

    async def call_api(self, prompt, *args):
        self.calls.append((prompt, *args))
        return self.response


class EmbeddingProvider(MockProvider):
    supports_embedding_cancellation = False

    async def call_embedding_api(self, text, *args):
        self.calls.append((text, *args))
        return self.response


class ClassificationProvider(MockProvider):
    async def call_classification_api(self, text):
        return {"classification": {}}


class ModerationProvider(MockProvider):
    async def call_moderation_api(self, prompt, output):
        return {"flags": []}


class SimilarityProvider(MockProvider):
    async def call_similarity_api(self, a, b):
        return {"similarity": 1}


async def _settle() -> None:
    """Let spawned tasks run until they block (enqueueing a job takes a few loop turns)."""
    for _ in range(10):
        await asyncio.sleep(0)


class FakeRegistry:
    def __init__(self) -> None:
        self.executed: list[tuple] = []

    async def execute(self, provider, call_fn, options=None):
        self.executed.append((provider, call_fn, options))
        return await call_fn(None)


@pytest.fixture
def load_calls(monkeypatch):
    """Replace load_api_provider; records (path, kwargs) and returns `load_calls.result`."""

    class Recorder(list):
        result: Any = None

    calls = Recorder()
    calls.result = MockProvider("loaded")

    async def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return calls.result

    monkeypatch.setattr(providers_mod, "load_api_provider", fake_load)
    return calls


@pytest.fixture
def config():
    def set_config(value):
        return cli_state.with_config(value)

    return set_config


# --- getGradingProvider -------------------------------------------------------------------


class TestExplicitProvider:
    async def test_string(self, load_calls):
        result = await mp.get_grading_provider("text", "openai:gpt-4", None)
        assert load_calls == [("openai:gpt-4", {"base_path": None})]
        assert result is load_calls.result

    async def test_api_provider_instance(self, load_calls):
        provider = MockProvider()
        assert await mp.get_grading_provider("text", provider, None) is provider
        assert load_calls == []

    async def test_none_is_unspecified(self, load_calls):
        default = MockProvider()
        assert await mp.get_grading_provider("text", None, default) is default
        assert load_calls == []

    async def test_provider_options(self, load_calls):
        options = {"id": "openai:gpt-4", "config": {"temperature": 0.5}}
        result = await mp.get_grading_provider("text", options, None)
        assert load_calls == [("openai:gpt-4", {"options": options, "base_path": None})]
        assert result is load_calls.result

    async def test_base_path_comes_from_cli_state(self, load_calls):
        with cli_state.with_base_path("/suite"):
            await mp.get_grading_provider("text", "openai:gpt-4", None)
        assert load_calls == [("openai:gpt-4", {"base_path": "/suite"})]


class TestProviderTypeMap:
    @pytest.mark.parametrize(
        ("kind", "pid"),
        [
            ("embedding", "openai:embedding"),
            ("classification", "openai:gpt-4"),
            ("text", "anthropic:claude-3-sonnet"),
        ],
    )
    async def test_typed_string(self, load_calls, kind, pid):
        result = await mp.get_grading_provider(kind, {kind: pid}, None)
        assert load_calls == [(pid, {"base_path": None, "env": None})]
        assert result is load_calls.result

    async def test_typed_entries_load_with_the_suite_env(self, load_calls):
        typed = {"id": "openai:gpt-4", "config": {"temperature": 0}}
        with cli_state.with_env({"OPENAI_API_KEY": "suite"}):
            await mp.get_grading_provider("text", {"text": typed}, None)
        assert load_calls == [
            ("openai:gpt-4",
             {"options": typed, "base_path": None, "env": {"OPENAI_API_KEY": "suite"}})
        ]  # fmt: skip

    async def test_typed_api_provider_instance(self, load_calls):
        provider = MockProvider()
        assert await mp.get_grading_provider("text", {"text": provider}, None) is provider
        assert load_calls == []


class TestDefaultTestFallback:
    async def test_default_test_options_provider(self, load_calls, config):
        with config({"defaultTest": {"options": {"provider": "azureopenai:chat:gpt-4"}}}):
            result = await mp.get_grading_provider("text", None, None)
        assert load_calls == [("azureopenai:chat:gpt-4", {"base_path": None})]
        assert result is load_calls.result

    async def test_default_test_provider(self, load_calls, config):
        with config({"defaultTest": {"provider": "azureopenai:chat:gpt-4"}}):
            await mp.get_grading_provider("text", None, None)
        assert load_calls == [("azureopenai:chat:gpt-4", {"base_path": None})]

    async def test_skips_simulated_user(self, load_calls, config):
        default = MockProvider("default-provider")
        simulated = {"id": "promptfoo:simulated-user", "config": {"maxTurns": 3}}
        with config({"defaultTest": {"provider": simulated}}):
            assert await mp.get_grading_provider("text", None, default) is default
        assert load_calls == []

    async def test_simulated_user_falls_through_to_options_provider(self, load_calls, config):
        simulated = {"id": "promptfoo:simulated-user", "config": {"maxTurns": 3}}
        cfg = {"defaultTest": {"provider": simulated,
                               "options": {"provider": "azureopenai:chat:gpt-4"}}}  # fmt: skip
        with config(cfg):
            await mp.get_grading_provider("text", None, None)
        assert load_calls == [("azureopenai:chat:gpt-4", {"base_path": None})]

    async def test_options_provider_text(self, load_calls, config):
        cfg = {"defaultTest": {"options": {"provider": {"text": "azureopenai:chat:gpt-4"}}}}
        with config(cfg):
            await mp.get_grading_provider("text", None, None)
        assert load_calls == [("azureopenai:chat:gpt-4", {"base_path": None})]

    async def test_default_test_provider_wins_over_options_provider(self, load_calls, config):
        cfg = {"defaultTest": {"provider": "azureopenai:chat:gpt-4",
                               "options": {"provider": "openai:gpt-4"}}}  # fmt: skip
        with config(cfg):
            await mp.get_grading_provider("text", None, None)
        assert load_calls == [("azureopenai:chat:gpt-4", {"base_path": None})]

    @pytest.mark.parametrize("cfg", [{"defaultTest": {}}, None, {}])
    async def test_falls_back_to_default_provider(self, load_calls, config, cfg):
        default = MockProvider("default-provider")
        with config(cfg):
            assert await mp.get_grading_provider("text", None, default) is default
        assert load_calls == []

    async def test_none_without_default(self, config):
        with config({}):
            assert await mp.get_grading_provider("text", None, None) is None

    async def test_full_azure_provider_options(self, load_calls, config):
        provider = {
            "id": "azureopenai:chat:gpt-4o",
            "config": {"apiHost": "https://my-resource.openai.azure.com", "apiKey": "my-api-key",
                       "deploymentName": "gpt-4o"},
        }  # fmt: skip
        with config({"defaultTest": {"options": {"provider": provider}}}):
            await mp.get_grading_provider("text", None, None)
        assert load_calls == [("azureopenai:chat:gpt-4o", {"options": provider, "base_path": None})]

    async def test_explicit_provider_wins(self, load_calls, config):
        explicit = MockProvider("explicit-provider")
        with config({"defaultTest": {"options": {"provider": "azureopenai:chat:gpt-4"}}}):
            assert await mp.get_grading_provider("text", explicit, None) is explicit
            await mp.get_grading_provider("text", "openai:gpt-4o", None)
        assert load_calls == [("openai:gpt-4o", {"base_path": None})]


class TestErrors:
    async def test_array(self):
        with pytest.raises(
            ValueError, match="Provider must be an object or string, but received an array"
        ):
            await mp.get_grading_provider("text", ["openai:gpt-4"], None)

    async def test_invalid_definition(self):
        with pytest.raises(ValueError, match="Invalid provider definition for output type 'text'"):
            await mp.get_grading_provider("text", {"foo": "bar"}, None)


# --- getAndCheckProvider ------------------------------------------------------------------


class TestGetAndCheckProvider:
    async def test_no_provider_and_no_default(self, config):
        with config({}), pytest.raises(ValueError, match="No provider of type text found for 'x'"):
            await mp.get_and_check_provider("text", None, None, "x")

    @pytest.mark.parametrize(
        ("kind", "cls"),
        [
            ("embedding", EmbeddingProvider),
            ("embedding", SimilarityProvider),
            ("classification", ClassificationProvider),
            ("moderation", ModerationProvider),
            ("text", MockProvider),
        ],
    )
    async def test_valid_types(self, kind, cls):
        provider = cls()
        assert await mp.get_and_check_provider(kind, provider, None, "check") is provider

    async def test_explicit_wrong_type_raises(self):
        with pytest.raises(
            ValueError, match="Provider test-grader is not a valid embedding provider for 'sim'"
        ):
            await mp.get_and_check_provider("embedding", MockProvider(), EmbeddingProvider(), "sim")

    async def test_wrong_typed_default_test_falls_back_to_default(self, config):
        default = EmbeddingProvider("default-embedding")
        with config({"defaultTest": {"provider": MockProvider()}}):
            assert await mp.get_and_check_provider("embedding", None, default, "sim") is default

    async def test_wrong_type_without_default_raises(self, config):
        with (
            config({"defaultTest": {"provider": MockProvider()}}),
            pytest.raises(ValueError, match="is not a valid moderation provider"),
        ):
            await mp.get_and_check_provider("moderation", None, None, "mod")

    async def test_rate_limit_wrapped_provider_keeps_its_api_methods(self):
        wrapped = wrap_provider_with_rate_limiting(EmbeddingProvider(), FakeRegistry())  # type: ignore[arg-type]
        assert await mp.get_and_check_provider("embedding", wrapped, None, "sim") is wrapped


# --- callProviderWithContext ----------------------------------------------------------------

VARS = {"question": "What is two plus two?"}
CTX = {"isGrading": True, "prompt": {"raw": "grade this", "label": "rubric"}, "vars": VARS}


class TestCallProviderWithContext:
    async def test_direct_call_without_scheduler(self):
        response = {"output": "direct response"}
        provider = MockProvider(response=response)
        assert await mp.call_provider_with_context(provider, "grade this", "rubric", VARS) is (
            response
        )
        assert provider.calls == [("grade this", CTX)]

    async def test_prompt_config_is_attached(self):
        provider = MockProvider()
        await mp.call_provider_with_context(provider, "p", "l", {}, None, {"temperature": 0})
        assert provider.calls[0][1]["prompt"] == {"raw": "p", "label": "l",
                                                  "config": {"temperature": 0}}  # fmt: skip

    async def test_uses_scheduler_registry(self):
        provider = MockProvider()
        registry = FakeRegistry()
        await with_provider_call_execution_context(
            {"rateLimitRegistry": registry},
            lambda: mp.call_provider_with_context(provider, "grade this", "rubric", VARS),
        )
        assert len(registry.executed) == 1
        executed_provider, _, options = registry.executed[0]
        assert executed_provider is provider
        for key in ("getHeaders", "isRateLimited", "getRetryAfter"):
            assert callable(options[key])
        assert provider.calls == [("grade this", CTX)]

    async def test_propagates_abort_signal(self):
        provider = MockProvider()
        registry = FakeRegistry()
        controller = AbortController()
        await with_provider_call_execution_context(
            {"abortSignal": controller.signal, "rateLimitRegistry": registry},
            lambda: mp.call_provider_with_context(provider, "grade this", "rubric", VARS),
        )
        assert registry.executed[0][2]["abortSignal"] is controller.signal
        assert provider.calls == [("grade this", CTX, {"abortSignal": controller.signal})]

    async def test_tracing_preserves_context(self):
        provider = MockProvider()
        registry = FakeRegistry()
        controller = AbortController()
        traceparent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
        original = {
            "isGrading": False,
            "prompt": {"raw": "original input", "label": "target"},
            "vars": {"original": "value"},
            "evaluationId": "evaluation-123",
            "testIdx": 2,
            "bustCache": True,
            "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-aaaaaaaaaaaaaaaa-01",
            "tracestate": "vendor=state",
        }
        spans: list[dict] = []

        async def with_provider_span(options, invoke):
            spans.append(options)
            return await invoke({**options["callContext"], "traceparent": traceparent})

        async def with_grader_span(_options, invoke):
            return await invoke()

        await with_provider_call_execution_context(
            {"abortSignal": controller.signal, "rateLimitRegistry": registry},
            lambda: with_provider_call_tracing_context(
                {"getActiveTraceparent": lambda: traceparent, "withGraderSpan": with_grader_span,
                 "withProviderSpan": with_provider_span},
                lambda: mp.call_provider_with_context(
                    provider, "grade this", "rubric", VARS, original
                ),
            ),
        )  # fmt: skip
        assert original["isGrading"] is False
        assert len(registry.executed) == 1
        assert spans[0]["provider"] is provider
        assert spans[0]["role"] == "grader" and spans[0]["promptLabel"] == "rubric"
        assert provider.calls == [(
            "grade this",
            {**CTX, "evaluationId": "evaluation-123", "testIdx": 2, "bustCache": True,
             "traceparent": traceparent, "tracestate": "vendor=state"},
            {"abortSignal": controller.signal},
        )]  # fmt: skip

    async def test_execution_context_is_scoped(self):
        provider = MockProvider()
        registry = FakeRegistry()
        await with_provider_call_execution_context(
            {"rateLimitRegistry": registry},
            lambda: mp.call_provider_with_context(provider, "scheduled", "rubric", VARS),
        )
        await mp.call_provider_with_context(provider, "direct", "rubric", VARS)
        assert len(registry.executed) == 1
        assert len(provider.calls) == 2
        assert provider.calls[-1] == (
            "direct",
            {"isGrading": True, "prompt": {"raw": "direct", "label": "rubric"}, "vars": VARS},
        )

    async def test_no_double_scheduling_of_wrapped_providers(self):
        provider = MockProvider()
        wrapper_registry = FakeRegistry()
        context_registry = FakeRegistry()
        wrapped = wrap_provider_with_rate_limiting(provider, wrapper_registry)  # type: ignore[arg-type]
        await with_provider_call_execution_context(
            {"rateLimitRegistry": context_registry},
            lambda: mp.call_provider_with_context(wrapped, "grade this", "rubric", VARS),
        )
        assert context_registry.executed == []
        assert len(wrapper_registry.executed) == 1
        assert provider.calls == [("grade this", CTX, None)]

    @pytest.mark.parametrize("with_signal", [False, True])
    async def test_queues_calls(self, with_signal):
        signal = AbortController().signal if with_signal else None
        response = {"output": "queued response"}
        provider = MockProvider(response=response)
        queue = ProviderGroupedCallQueue()
        task = asyncio.ensure_future(
            with_provider_call_execution_context(
                {"providerCallQueue": queue, "abortSignal": signal},
                lambda: mp.call_provider_with_context(provider, "grade this", "rubric", VARS),
            )
        )
        await _settle()
        assert provider.calls == []
        group = queue.take_next_group()
        assert len(group) == 1
        assert group[0].providerId == "test-grader"
        await queue.run(group[0])
        assert await task is response
        assert provider.calls[0] == (
            ("grade this", CTX, {"abortSignal": signal}) if signal else ("grade this", CTX)
        )

    async def test_queued_call_does_not_start_after_cancellation(self):
        provider = MockProvider()
        controller = AbortController()
        queue = ProviderGroupedCallQueue()
        task = asyncio.ensure_future(
            with_provider_call_execution_context(
                {"abortSignal": controller.signal, "providerCallQueue": queue},
                lambda: mp.call_provider_with_context(provider, "grade this", "rubric", VARS),
            )
        )
        await _settle()
        reason = RuntimeError("eval paused")
        controller.abort(reason)
        for job in queue.take_next_group():
            await queue.run(job)
        with pytest.raises(RuntimeError) as excinfo:
            await task
        assert excinfo.value is reason
        assert provider.calls == []


# --- callGradingProvider ------------------------------------------------------------------


class TestCallGradingProvider:
    async def test_traces_non_text_calls_without_changing_the_response(self):
        provider = MockProvider()
        response = {"classification": {"safe": 0.9}}
        spans: list[dict] = []

        async def with_provider_span(options, invoke):
            spans.append(options)
            return await invoke(options["callContext"])

        async def invoke(_context):
            return response

        result = await with_provider_call_tracing_context(
            {"withProviderSpan": with_provider_span},
            lambda: mp.call_grading_provider(provider, "classification", invoke),
        )
        assert result is response
        assert spans[0]["provider"] is provider
        assert spans[0]["role"] == "grader" and spans[0]["promptLabel"] == "classification"

    async def test_operation_name_reaches_the_span(self):
        spans: list[dict] = []

        async def with_provider_span(options, invoke):
            spans.append(options)
            return await invoke(None)

        async def invoke(_context):
            return {"embedding": [1]}

        await with_provider_call_tracing_context(
            {"withProviderSpan": with_provider_span},
            lambda: mp.call_grading_provider(
                MockProvider(), "sim", invoke, operation_name="embeddings"
            ),
        )
        assert spans[0]["operationName"] == "embeddings"

    async def test_embedding_calls_reuse_rate_limiting_and_grouping(self):
        provider = MockProvider()
        registry = FakeRegistry()
        queue = ProviderGroupedCallQueue()
        invocations: list[Any] = []

        async def invoke(context):
            invocations.append(context)
            return {"embedding": [1, 0, 0]}

        task = asyncio.ensure_future(
            with_provider_call_execution_context(
                {"providerCallQueue": queue, "rateLimitRegistry": registry},
                lambda: mp.call_grading_provider(provider, "similarity.embedding", invoke),
            )
        )
        await _settle()
        assert invocations == []
        [job] = queue.take_next_group()
        await queue.run(job)
        assert await task == {"embedding": [1, 0, 0]}
        assert len(registry.executed) == 1
        assert len(invocations) == 1

    @pytest.mark.parametrize("selected", [False, True])
    async def test_cancels_grouped_non_text_grading_before_invocation(self, selected):
        controller = AbortController()
        survivor_controller = AbortController()
        provider = MockProvider()
        queue = ProviderGroupedCallQueue()
        cancelled_calls: list[Any] = []
        survivor_calls: list[Any] = []

        async def cancelled_invoke(context):
            cancelled_calls.append(context)
            return {"embedding": [0, 1]}

        async def survivor_invoke(context):
            survivor_calls.append(context)
            return {"embedding": [1, 0]}

        pending = asyncio.ensure_future(
            with_provider_call_execution_context(
                {"providerCallQueue": queue, "abortSignal": controller.signal},
                lambda: mp.call_grading_provider(provider, "cancelled.embedding", cancelled_invoke),
            )
        )
        survivor = asyncio.ensure_future(
            with_provider_call_execution_context(
                {"providerCallQueue": queue, "abortSignal": survivor_controller.signal},
                lambda: mp.call_grading_provider(provider, "survivor.embedding", survivor_invoke),
            )
        )
        await _settle()
        selected_jobs = queue.take_next_group() if selected else None
        controller.abort()
        with pytest.raises(AbortError):
            await pending
        assert controller.signal._listeners == []
        group = selected_jobs if selected_jobs is not None else queue.take_next_group()
        assert len(group) == (2 if selected else 1)
        for job in group:
            await queue.run(job)
        assert await survivor == {"embedding": [1, 0]}
        assert cancelled_calls == []
        assert len(survivor_calls) == 1
        assert survivor_controller.signal._listeners == []
        assert not queue.has_jobs()


# --- callEmbeddingProvider ----------------------------------------------------------------


class CancellableEmbeddingProvider(EmbeddingProvider):
    supports_embedding_cancellation = True


class TestCallEmbeddingProvider:
    async def test_plain_call_without_context(self):
        provider = CancellableEmbeddingProvider(response={"embedding": [1]})
        assert await mp.call_embedding_provider(provider, "hello") == {"embedding": [1]}
        assert provider.calls == [("hello",)]

    async def test_cancellable_provider_gets_context_and_signal(self):
        provider = CancellableEmbeddingProvider(response={"embedding": [1]})
        controller = AbortController()
        await with_provider_call_execution_context(
            {"abortSignal": controller.signal},
            lambda: mp.call_embedding_provider(provider, "hello", {"vars": {}}),
        )
        assert provider.calls == [("hello", {"vars": {}}, {"abortSignal": controller.signal})]

    async def test_non_cancellable_provider_gets_input_only(self):
        provider = EmbeddingProvider(response={"embedding": [1]})
        await mp.call_embedding_provider(provider, "hello", {"vars": {}})
        assert provider.calls == [("hello",)]

    async def test_error_after_abort_raises_the_abort_reason(self):
        provider = EmbeddingProvider(response={"error": "aborted"})
        controller = AbortController()
        controller.abort()

        async def run():
            return await mp.call_embedding_provider(provider, "hello")

        with pytest.raises(AbortError):
            await with_provider_call_execution_context({"abortSignal": controller.signal}, run)


def test_remote_grading_is_disabled():
    assert mp.should_use_remote_grading() is False
    assert mp.should_use_remote_grading({"anything": True}) is False
    assert mp.get_remote_grading_context() == {}


def test_grading_call_options():
    assert mp.get_grading_provider_call_options() is None


# --- cancellation through the real scheduler ----------------------------------------------


def test_text_grading_settles_when_aborted_during_registry_backoff(monkeypatch):
    from scheduler_fakes import FakeTime, advance, flush

    from promptphoo.scheduler.rate_limit_registry import RateLimitRegistry

    monkeypatch.setenv("PROMPTFOO_DISABLE_ADAPTIVE_SCHEDULER", "false")
    fake = FakeTime(monkeypatch)

    async def scenario() -> None:
        registry = RateLimitRegistry({"maxConcurrency": 1})
        controller = AbortController()
        rate_limited = {
            "error": "429 rate limit",
            "metadata": {"http": {"status": 429, "statusText": "Too Many Requests",
                                  "headers": {"retry-after-ms": "60000"}}},
        }  # fmt: skip
        provider = MockProvider(response=rate_limited)
        try:
            pending = asyncio.ensure_future(
                with_provider_call_execution_context(
                    {"rateLimitRegistry": registry, "abortSignal": controller.signal},
                    lambda: mp.call_provider_with_context(provider, "grade this", "rubric", {}),
                )
            )
            await flush()
            assert len(provider.calls) == 1
            controller.abort()
            await flush()
            with pytest.raises(AbortError):
                await pending
            await advance(120_000)
            assert len(provider.calls) == 1
            assert next(iter(registry.get_metrics().values()))["activeRequests"] == 0
        finally:
            registry.dispose()

    fake.run(scenario)
