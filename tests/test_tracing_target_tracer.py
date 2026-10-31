"""Ports test/tracing/targetTracer.test.ts and test/tracing/executionHierarchy.test.ts.

Upstream mocks the tracer for the targetTracer cases; here every case runs against a real
in-house BasicTracerProvider with a SimpleSpanProcessor over an InMemorySpanExporter.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from promptphoo.scheduler.provider_call_execution_context import (
    with_provider_call_tracing_context,
)
from promptphoo.tracing import store as store_module
from promptphoo.tracing.evaluator_tracing import generate_trace_context_if_needed
from promptphoo.tracing.genai_tracer import (
    GenAIAttributes,
    PromptfooAttributes,
    get_genai_tracer,
    with_genai_span,
)
from promptphoo.tracing.otel_api import SpanKind, SpanStatusCode, propagation, trace
from promptphoo.tracing.otel_trace_sdk import (
    BasicTracerProvider,
    InMemorySpanExporter,
    SimpleSpanProcessor,
)
from promptphoo.tracing.span_filter import is_relevant_span
from promptphoo.tracing.span_roles import SPAN_ROLE_ATTRIBUTE, get_active_traceparent
from promptphoo.tracing.target_tracer import (
    MAX_GRADING_EXPLANATION_LENGTH,
    GraderAttributes,
    TargetAttributes,
    get_target_type,
    with_grader_span,
    with_target_span,
    with_test_case_span,
    with_traced_provider_call,
)

TRACEPARENT = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"


class FakeStore:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create_trace(self, trace_record: dict[str, Any]) -> None:
        self.created.append(trace_record)


class Provider:
    def __init__(self, provider_id: str, label: str | None = None, **attrs: Any) -> None:
        self._id = provider_id
        self.label = label
        for key, value in attrs.items():
            setattr(self, key, value)

    def id(self) -> str:
        return self._id

    async def call_api(self, prompt: str, context: Any = None, options: Any = None) -> Any:
        return {"output": "done"}


@pytest.fixture
def exporter():
    exporter = InMemorySpanExporter()
    provider = BasicTracerProvider(span_processors=[SimpleSpanProcessor(exporter)])
    provider.register()
    yield exporter
    trace.disable()
    propagation.disable()


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
    monkeypatch.setattr(store_module, "get_trace_store", lambda: fake)
    return fake


def by_name(exporter: InMemorySpanExporter) -> dict[str, Any]:
    return {span.name: span for span in exporter.get_finished_spans()}


def parent_id(span: Any) -> str | None:
    return span.parent_span_context.span_id if span.parent_span_context else None


CALL_CONTEXT = {"prompt": {"raw": "test prompt", "label": "target"}, "vars": {}}


# --- targetTracer.test.ts -----------------------------------------------------------------


async def test_records_target_context_and_cached_responses(exporter) -> None:
    result = await with_target_span(
        {
            "targetType": "provider",
            "providerId": "python:customer_provider.py",
            "label": "Customer provider",
            "traceparent": TRACEPARENT,
            "promptLabel": "test prompt",
            "evalId": "eval-1",
            "testIndex": 3,
        },
        lambda _span: _value({"cached": True, "output": "ok"}),
    )
    assert result == {"cached": True, "output": "ok"}
    [span] = exporter.get_finished_spans()
    assert span.name == "Customer provider"
    assert span.kind == SpanKind.CLIENT
    assert span.span_context().trace_id == "0123456789abcdef0123456789abcdef"
    assert parent_id(span) == "0123456789abcdef"
    assert span.attributes == {
        PromptfooAttributes.PROVIDER_ID: "python:customer_provider.py",
        SPAN_ROLE_ATTRIBUTE: "target",
        TargetAttributes.TARGET_TYPE: "provider",
        TargetAttributes.TARGET_LABEL: "Customer provider",
        PromptfooAttributes.PROMPT_LABEL: "test prompt",
        PromptfooAttributes.EVAL_ID: "eval-1",
        PromptfooAttributes.TEST_INDEX: 3,
        PromptfooAttributes.CACHE_HIT: True,
    }
    assert span.status.code == SpanStatusCode.OK
    assert span.ended


async def test_records_provider_error_responses_without_swallowing(exporter) -> None:
    result = await with_target_span(
        {"targetType": "provider", "providerId": "a2a:customer-agent", "traceparent": TRACEPARENT},
        lambda _span: _value({"error": "agent unavailable"}),
    )
    assert result == {"error": "agent unavailable"}
    [span] = exporter.get_finished_spans()
    assert span.name == "a2a:customer-agent"
    assert (span.status.code, span.status.message) == (SpanStatusCode.ERROR, "agent unavailable")
    [event] = span.events
    assert event.name == "exception"
    assert event.attributes == {
        "exception.type": "Error",
        "exception.message": "agent unavailable",
        "exception.stacktrace": "Error: agent unavailable",
    }


async def test_grader_provider_spans_have_no_target_only_metadata(exporter) -> None:
    await with_target_span(
        {
            "targetType": "provider",
            "providerId": "openai:judge",
            "label": "Judge provider",
            "role": "grader",
            "traceparent": TRACEPARENT,
        },
        lambda _span: _value({"output": "pass"}),
    )
    [span] = exporter.get_finished_spans()
    assert span.name == "grader provider Judge provider"
    assert span.attributes == {
        PromptfooAttributes.PROVIDER_ID: "openai:judge",
        SPAN_ROLE_ATTRIBUTE: "grader",
    }


async def test_records_and_rethrows_provider_exceptions(exporter) -> None:
    async def fail(_span: Any) -> Any:
        raise RuntimeError("target unavailable")

    with pytest.raises(RuntimeError, match="target unavailable"):
        await with_target_span(
            {"targetType": "http", "providerId": "http:customer-api", "traceparent": TRACEPARENT},
            fail,
        )
    [span] = exporter.get_finished_spans()
    assert span.ended
    assert span.status.code == SpanStatusCode.ERROR
    assert span.events[0].attributes["exception.message"] == "target unavailable"


@pytest.mark.parametrize(
    ("provider_id", "expected"),
    [
        ("mcp:server", "mcp"),
        ("ws://host", "websocket"),
        ("https://customer.example/chat", "http"),
        ("http:customer", "http"),
        ("openai:gpt-4.1", "provider"),
    ],
)
def test_target_type(provider_id: str, expected: str) -> None:
    assert get_target_type(provider_id) == expected


async def test_traced_provider_call_without_traceparent_is_a_passthrough(exporter) -> None:
    seen: list[Any] = []

    async def invoke(context: Any) -> Any:
        seen.append(context)
        return {"output": "x"}

    assert await with_traced_provider_call(
        {"provider": Provider("echo"), "callContext": CALL_CONTEXT}, invoke
    ) == {"output": "x"}
    assert seen == [CALL_CONTEXT]
    assert exporter.get_finished_spans() == []


async def test_traced_provider_call_injects_the_target_traceparent(exporter) -> None:
    seen: list[Any] = []

    async def invoke(context: Any) -> Any:
        seen.append(context)
        return {"output": "x"}

    await with_traced_provider_call(
        {
            "provider": Provider("echo", label="echo-label"),
            "callContext": {**CALL_CONTEXT, "traceparent": TRACEPARENT, "evaluationId": "ev"},
            "testIndex": 2,
        },
        invoke,
    )
    [span] = exporter.get_finished_spans()
    assert span.name == "echo-label"
    ctx = span.span_context()
    assert seen[0]["traceparent"] == f"00-{ctx.trace_id}-{ctx.span_id}-01"
    assert span.attributes[PromptfooAttributes.PROMPT_LABEL] == "target"
    assert span.attributes[PromptfooAttributes.EVAL_ID] == "ev"
    assert span.attributes[PromptfooAttributes.TEST_INDEX] == 2
    assert CALL_CONTEXT.get("traceparent") is None


# --- executionHierarchy.test.ts -----------------------------------------------------------


async def test_root_span_is_the_traceparent_for_one_provider_execution(
    exporter, fake_store: FakeStore
) -> None:
    trace_context = await generate_trace_context_if_needed(
        {"metadata": {"tracingEnabled": True, "evaluationId": "eval-1", "testCaseId": "test-1"}},
        {},
        2,
        3,
        None,
        {"providerId": "python:customer.py", "promptLabel": "customer prompt", "repeatIndex": 4},
    )
    assert trace_context is not None
    root_context = trace_context["rootSpan"].span_context()
    assert trace_context["traceparent"] == (f"00-{root_context.trace_id}-{root_context.span_id}-01")
    [created] = fake_store.created
    assert created["traceId"] == root_context.trace_id
    assert created["metadata"]["providerId"] == "python:customer.py"
    assert created["metadata"]["repeatIndex"] == 4

    await with_test_case_span(
        trace_context["rootSpan"], lambda: _value([{"score": 1, "success": True}])
    )

    [root] = exporter.get_finished_spans()
    assert root.name == "promptfoo.test_case"
    assert root.parent_span_context is None
    assert root.status.code == SpanStatusCode.OK
    expected = {
        SPAN_ROLE_ATTRIBUTE: "test_case",
        "promptfoo.provider.id": "python:customer.py",
        "promptfoo.prompt.index": 3,
        "promptfoo.repeat.index": 4,
        "promptfoo.test.success": True,
        "promptfoo.test.score": 1,
    }
    assert {k: root.attributes.get(k) for k in expected} == expected


async def test_root_status_is_error_with_the_row_error(exporter) -> None:
    root = get_genai_tracer().start_span("test case row error")
    await with_test_case_span(
        root, lambda: _value([{"score": 0, "success": False, "error": "boom"}])
    )
    [span] = exporter.get_finished_spans()
    assert (span.status.code, span.status.message) == (SpanStatusCode.ERROR, "boom")


async def test_target_and_model_call_nest_under_the_root(exporter) -> None:
    root = get_genai_tracer().start_span("test case model")
    provider = Provider("openai:gpt-4.1")

    async def run() -> Any:
        await with_traced_provider_call(
            {"provider": provider, "callContext": CALL_CONTEXT},
            lambda target_context: with_genai_span(
                {
                    "system": "openai",
                    "operationName": "chat",
                    "model": "gpt-4.1",
                    "providerId": provider.id(),
                    "traceparent": (target_context or {}).get("traceparent"),
                },
                lambda _span: _value({"output": "done"}),
            ),
        )
        return [{"score": 1, "success": True}]

    await with_test_case_span(root, run)
    spans = by_name(exporter)
    root_span, target, model = (
        spans["test case model"],
        spans["openai:gpt-4.1"],
        spans["chat gpt-4.1"],
    )
    assert parent_id(target) == root_span.span_context().span_id
    assert parent_id(model) == target.span_context().span_id
    assert model.attributes[SPAN_ROLE_ATTRIBUTE] == "target"


async def test_honors_an_explicit_parent_within_the_active_trace(exporter) -> None:
    root = get_genai_tracer().start_span("test case explicit parent")
    tracer = get_genai_tracer()

    async def run() -> Any:
        async def in_requested(requested: Any) -> None:
            ctx = requested.span_context()
            explicit = f"00-{ctx.trace_id}-{ctx.span_id}-01"

            async def in_active(active: Any) -> None:
                try:
                    await with_genai_span(
                        {"system": "openai", "operationName": "chat", "model": "gpt-4.1",
                         "providerId": "openai:gpt-4.1", "traceparent": explicit},
                        lambda _span: _value({"output": "done"}),
                    )  # fmt: skip
                finally:
                    active.end()

            try:
                await tracer.start_active_span("different active parent", in_active)
            finally:
                requested.end()

        await tracer.start_active_span("requested parent", in_requested)
        return [{"score": 1, "success": True}]

    await with_test_case_span(root, run)
    spans = by_name(exporter)
    model = spans["chat gpt-4.1"]
    assert parent_id(model) == spans["requested parent"].span_context().span_id
    assert parent_id(model) != spans["different active parent"].span_context().span_id


async def test_target_and_grading_branches_share_the_root(exporter) -> None:
    root = get_genai_tracer().start_span("test case shared")
    target = Provider("http:customer-agent")
    judge = Provider("openai:judge")

    async def grade() -> Any:
        await with_traced_provider_call(
            {"provider": judge, "callContext": CALL_CONTEXT, "role": "grader"},
            lambda grading_context: with_genai_span(
                {"system": "openai", "operationName": "chat", "model": "judge-model",
                 "providerId": judge.id(),
                 "traceparent": (grading_context or {}).get("traceparent")},
                lambda _span: _value({"output": "judge response"}),
            ),
        )  # fmt: skip
        return {"pass": False, "score": 0, "reason": "The response failed the rubric."}

    async def run() -> Any:
        await with_traced_provider_call(
            {"provider": target, "callContext": CALL_CONTEXT},
            lambda target_context: target.call_api("test prompt", target_context),
        )
        await with_grader_span({"graderId": "llm-rubric", "evalId": "e", "testIndex": 1}, grade)
        return [{"score": 0, "success": False}]

    await with_test_case_span(root, run)
    spans = by_name(exporter)
    root_id = spans["test case shared"].span_context().span_id
    grader = spans["grader llm-rubric"]
    grading_provider = spans["grader provider openai:judge"]
    grading_model = spans["chat judge-model"]
    assert parent_id(spans["http:customer-agent"]) == root_id
    assert parent_id(grader) == root_id
    assert parent_id(grading_provider) == grader.span_context().span_id
    assert parent_id(grading_model) == grading_provider.span_context().span_id
    assert grader.kind == SpanKind.INTERNAL
    assert grader.attributes == {
        GraderAttributes.GRADER_ID: "llm-rubric",
        GenAIAttributes.EVALUATION_NAME: "llm-rubric",
        SPAN_ROLE_ATTRIBUTE: "grader",
        PromptfooAttributes.EVAL_ID: "e",
        PromptfooAttributes.TEST_INDEX: 1,
        GenAIAttributes.EVALUATION_SCORE_LABEL: "fail",
        GenAIAttributes.EVALUATION_SCORE_VALUE: 0,
        GenAIAttributes.EVALUATION_EXPLANATION: "The response failed the rubric.",
    }
    assert grading_model.attributes[SPAN_ROLE_ATTRIBUTE] == "grader"
    assert is_relevant_span({"attributes": grading_model.attributes}) is False


async def test_tracing_context_hooks_parent_grader_provider_spans(exporter) -> None:
    root = get_genai_tracer().start_span("test case direct judge")
    judge = Provider("openai:judge")

    async def run() -> Any:
        await with_grader_span(
            {"graderId": "judge", "testIndex": 6},
            lambda: with_traced_provider_call(
                {"provider": judge, "callContext": CALL_CONTEXT, "role": "grader"},
                lambda ctx: judge.call_api("evaluate", ctx),
            ),
        )
        return [{"score": 1, "success": True}]

    await with_provider_call_tracing_context(
        {
            "getActiveTraceparent": get_active_traceparent,
            "testIndex": 6,
            "withGraderSpan": with_grader_span,
            "withProviderSpan": with_traced_provider_call,
        },
        lambda: with_test_case_span(root, run),
    )
    spans = by_name(exporter)
    grader, provider_span = spans["grader judge"], spans["grader provider openai:judge"]
    assert parent_id(provider_span) == grader.span_context().span_id
    assert grader.attributes[PromptfooAttributes.TEST_INDEX] == 6
    assert provider_span.attributes[SPAN_ROLE_ATTRIBUTE] == "grader"


async def test_http_target_records_no_model_inference_span(exporter) -> None:
    root = get_genai_tracer().start_span("test case http target")
    provider = Provider("https://customer.example/chat")

    async def run() -> Any:
        await with_traced_provider_call(
            {"provider": provider, "callContext": CALL_CONTEXT},
            lambda ctx: provider.call_api("test prompt", ctx),
        )
        return [{"score": 1, "success": True}]

    await with_test_case_span(root, run)
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["https://customer.example/chat", "test case http target"]
    assert spans[0].attributes[SPAN_ROLE_ATTRIBUTE] == "target"
    assert spans[0].attributes[TargetAttributes.TARGET_TYPE] == "http"
    assert GenAIAttributes.OPERATION_NAME not in spans[0].attributes
    assert GenAIAttributes.REQUEST_MODEL not in spans[0].attributes


async def test_embedding_inference_nests_under_the_grading_provider(exporter) -> None:
    root = get_genai_tracer().start_span("test case embedding grader")
    provider = Provider("azure:text-embedding-3-small", modelName="text-embedding-3-small")

    async def grade() -> Any:
        await with_traced_provider_call(
            {"provider": provider, "role": "grader", "operationName": "embeddings"},
            lambda _ctx: _value({"embedding": [1, 0], "tokenUsage": {"prompt": 7, "total": 7}}),
        )
        return {"pass": True, "score": 1}

    async def run() -> Any:
        await with_grader_span({"graderId": "similarity"}, grade)
        return [{"score": 1, "success": True}]

    await with_test_case_span(root, run)
    spans = by_name(exporter)
    provider_span = spans["grader provider azure:text-embedding-3-small"]
    embedding = spans["embeddings text-embedding-3-small"]
    assert parent_id(embedding) == provider_span.span_context().span_id
    expected = {
        GenAIAttributes.OPERATION_NAME: "embeddings",
        GenAIAttributes.PROVIDER_NAME: "azure.ai.openai",
        GenAIAttributes.USAGE_INPUT_TOKENS: 7,
        SPAN_ROLE_ATTRIBUTE: "grader",
    }
    assert {k: embedding.attributes.get(k) for k in expected} == expected


async def test_grader_explanations_are_sanitized_and_bounded(exporter) -> None:
    root = get_genai_tracer().start_span("test case grader explanation")
    reason = f"api_key=abcdefghijklmnop {'long explanation ' * 100}"

    async def run() -> Any:
        await with_grader_span(
            {"graderId": "contains"}, lambda: _value({"pass": True, "score": 1, "reason": reason})
        )
        return [{"score": 1, "success": True}]

    await with_test_case_span(root, run)
    explanation = by_name(exporter)["grader contains"].attributes[
        GenAIAttributes.EVALUATION_EXPLANATION
    ]
    assert "api_key=<REDACTED>" in explanation
    assert "abcdefghijklmnop" not in explanation
    assert len(explanation) == MAX_GRADING_EXPLANATION_LENGTH
    assert explanation.endswith("…")


async def test_root_stays_open_until_deferred_grading_finishes(exporter) -> None:
    root = get_genai_tracer().start_span("test case deferred")
    deferred: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    result = [{"score": 0, "success": False}]

    assert await with_test_case_span(root, lambda: _value(result), lambda _r: deferred) is result
    assert exporter.get_finished_spans() == []

    result[0] = {"score": 1, "success": True}
    deferred.set_result(None)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    [finished] = exporter.get_finished_spans()
    assert finished.attributes["promptfoo.test.success"] is True
    assert finished.attributes["promptfoo.test.score"] == 1


async def test_separate_targets_get_independent_root_traces(exporter, fake_store) -> None:
    suite = {"tracing": {"enabled": True}}
    test = {"metadata": {"evaluationId": "eval-1", "testCaseId": "shared-test"}}
    first = await generate_trace_context_if_needed(
        test, {}, 0, 0, suite, {"providerId": "http:target-a"}
    )
    second = await generate_trace_context_if_needed(
        test, {}, 0, 0, suite, {"providerId": "http:target-b"}
    )
    assert first and second
    assert first["rootSpan"].span_context().trace_id != second["rootSpan"].span_context().trace_id
    ok = [{"score": 1, "success": True}]
    await asyncio.gather(
        with_test_case_span(first["rootSpan"], lambda: _value(ok)),
        with_test_case_span(second["rootSpan"], lambda: _value(ok)),
    )
    ids = {s.attributes.get("promptfoo.provider.id") for s in exporter.get_finished_spans()}
    assert ids == {"http:target-a", "http:target-b"}


async def test_target_exceptions_mark_target_and_root_spans(exporter) -> None:
    root = get_genai_tracer().start_span("test case failure")
    provider = Provider("http:broken-target")

    async def broken(_ctx: Any) -> Any:
        raise RuntimeError("target unavailable")

    with pytest.raises(RuntimeError, match="target unavailable"):
        await with_test_case_span(
            root,
            lambda: with_traced_provider_call(
                {"provider": provider, "callContext": CALL_CONTEXT}, broken
            ),
        )
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["http:broken-target", "test case failure"]
    for span in spans:
        assert span.status.code == SpanStatusCode.ERROR
    assert get_active_traceparent() is None


async def _value(value: Any) -> Any:
    return value
