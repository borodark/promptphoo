"""Ports test/tracing/evaluatorTracing.test.ts: trace ids, the per-test trace context and the
shared OTLP receiver's lifecycle (startOTLPReceiver / stopOTLPReceiver are faked)."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest

from promptphoo import cli_state
from promptphoo.tracing import evaluator_tracing as et
from promptphoo.tracing import otlp_receiver as receiver_module
from promptphoo.tracing import store as store_module

TRACEPARENT = re.compile(r"^00-[a-f0-9]{32}-[a-f0-9]{16}-01$")


class FakeStore:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.deleted: list[float] = []

    async def create_trace(self, trace):
        self.created.append(trace)

    async def delete_old_traces(self, days):
        self.deleted.append(days)


class FakeReceiver:
    def __init__(self) -> None:
        self.starts: list[tuple] = []
        self.stops = 0
        self.policies: list[dict] = []
        self.deregistered: list[str] = []
        self.start_gate: asyncio.Future | None = None
        self.stop_gate: asyncio.Future | None = None
        self.start_error: Exception | None = None

    async def start(self, port, host, accept_formats, extra=None):
        self.starts.append((port, host, accept_formats, extra or {}))
        if self.start_error is not None:
            error, self.start_error = self.start_error, None
            raise error
        if self.start_gate is not None:
            gate, self.start_gate = self.start_gate, None
            await gate

    async def stop(self):
        self.stops += 1
        if self.stop_gate is not None:
            gate, self.stop_gate = self.stop_gate, None
            await gate


@pytest.fixture
def store(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(store_module, "get_trace_store", lambda: fake)
    return fake


@pytest.fixture
def receiver(monkeypatch, store):
    fake = FakeReceiver()
    monkeypatch.setattr(receiver_module, "start_otlp_receiver", fake.start)
    monkeypatch.setattr(receiver_module, "stop_otlp_receiver", fake.stop)
    monkeypatch.setattr(receiver_module, "register_otlp_receiver_trace_policy",
                        fake.policies.append)  # fmt: skip
    monkeypatch.setattr(receiver_module, "deregister_otlp_receiver_trace_policy",
                        fake.deregistered.append)  # fmt: skip
    monkeypatch.delenv("PROMPTFOO_TRACING_ENABLED", raising=False)
    et.reset_tracing_state()
    yield fake
    et.reset_tracing_state()


def _suite(http: dict | None = None, **tracing: Any) -> dict[str, Any]:
    tracing = {"enabled": True, **tracing}
    if http is not None:
        tracing["otlp"] = {"http": {"enabled": True, "port": 4318, "host": "127.0.0.1", **http}}
    return {"providers": [], "prompts": [], "tracing": tracing}


# --- ids ------------------------------------------------------------------------------------------


def test_ids_and_traceparent():
    trace_ids = {et.generate_trace_id() for _ in range(50)}
    span_ids = {et.generate_span_id() for _ in range(50)}
    assert len(trace_ids) == 50 and all(re.fullmatch(r"[0-9a-f]{32}", t) for t in trace_ids)
    assert len(span_ids) == 50 and all(re.fullmatch(r"[0-9a-f]{16}", s) for s in span_ids)
    assert et.generate_traceparent("a" * 32, "b" * 16, True) == f"00-{'a' * 32}-{'b' * 16}-01"
    assert et.generate_traceparent("a" * 32, "b" * 16, False) == f"00-{'a' * 32}-{'b' * 16}-00"
    assert et.generate_traceparent("a" * 32, "b" * 16).endswith("-01")


# --- generateTraceContextIfNeeded ---------------------------------------------------------


async def test_no_context_without_tracing(receiver, store):
    assert await et.generate_trace_context_if_needed({"vars": {"foo": "bar"}}, {}, 0, 0) is None
    assert store.created == []


async def test_context_from_metadata(receiver, store):
    test = {
        "vars": {"foo": "bar"},
        "metadata": {"tracingEnabled": True, "evaluationId": "eval-123", "testCaseId": "test-456"},
    }
    context = await et.generate_trace_context_if_needed(test, {}, 0, 0)
    assert context is not None
    assert TRACEPARENT.match(context["traceparent"])
    assert context["evaluationId"] == "eval-123" and context["testCaseId"] == "test-456"
    [trace] = store.created
    assert trace["traceId"] == context["traceparent"].split("-")[1]
    # Absent options are JS undefined, which JSON.stringify drops when the row is written.
    assert {k: v for k, v in trace["metadata"].items() if v is not et.UNDEFINED} == {
        "testIdx": 0, "promptIdx": 0, "vars": {"foo": "bar"},
    }  # fmt: skip


async def test_context_from_env(receiver, store, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_TRACING_ENABLED", "true")
    context = await et.generate_trace_context_if_needed({"vars": {"foo": "bar"}}, {}, 5, 10)
    assert context is not None and TRACEPARENT.match(context["traceparent"])
    assert context["evaluationId"].startswith("eval-")
    assert context["testCaseId"] == "5-10"


async def test_context_from_yaml_config(receiver, store):
    context = await et.generate_trace_context_if_needed({"vars": {}}, {}, 3, 7, _suite())
    assert context is not None and context["testCaseId"] == "3-7"
    assert context["evaluationId"].startswith("eval-")


async def test_event_source_names_the_evaluation(receiver, store):
    test = {"metadata": {"tracingEnabled": True}}
    context = await et.generate_trace_context_if_needed(test, {"eventSource": "eval-src"}, 1, 2)
    assert context is not None and context["evaluationId"] == "eval-src"


async def test_trace_metadata_carries_command_tool_names_and_redaction(receiver, store):
    suite = _suite({"redactAttributes": ["authorization"]}, commandToolNames=["bash"])
    await et.generate_trace_context_if_needed(
        {"vars": {"foo": "bar"}}, {}, 2, 4, suite, {"providerId": "echo", "repeatIndex": 1}
    )
    [trace] = store.created
    assert trace["testCaseId"] == "2-4"
    assert trace["metadata"] == {
        "testIdx": 2, "promptIdx": 4, "providerId": "echo", "repeatIndex": 1,
        "vars": {"foo": "bar"}, "commandToolNames": ["bash"],
        "otlpHttpRedactAttributes": ["authorization"],
    }  # fmt: skip


async def test_create_trace_failure_still_returns_a_context(receiver, store, caplog):
    async def boom(trace):
        raise RuntimeError("db down")

    store.create_trace = boom  # type: ignore[method-assign]
    context = await et.generate_trace_context_if_needed({"metadata": {"tracingEnabled": True}},
                                                        {}, 0, 0)  # fmt: skip
    assert context is not None
    assert "[EvaluatorTracing] Failed to create trace: db down" in caplog.text


# --- isTracingEnabled ----------------------------------------------------------------------------


def test_is_tracing_enabled(monkeypatch):
    monkeypatch.delenv("PROMPTFOO_TRACING_ENABLED", raising=False)
    assert et.is_tracing_enabled({}) is False
    assert et.is_tracing_enabled({"metadata": {"tracingEnabled": True}}) is True
    assert et.is_tracing_enabled({}, {"tracing": {"enabled": True}}) is True
    assert et.is_tracing_enabled({}, {"tracing": {"enabled": False}}) is False
    assert et.is_tracing_enabled({"metadata": {"tracingEnabled": True}},
                                 {"tracing": {"enabled": False}}) is True  # fmt: skip
    monkeypatch.setenv("PROMPTFOO_TRACING_ENABLED", "true")
    assert et.is_tracing_enabled({}) is True


# --- startOtlpReceiverIfNeeded / stopOtlpReceiverIfNeeded ---------------------------------------


def test_receiver_not_started_initially(receiver):
    assert et.is_otlp_receiver_started() is False


async def test_passes_accept_formats(receiver):
    suite = _suite({"host": "0.0.0.0", "acceptFormats": ["protobuf"]})
    assert await et.start_otlp_receiver_if_needed(suite) is True
    assert receiver.starts == [(4318, "0.0.0.0", ["protobuf"], {})]
    assert cli_state.active_otlp_receiver() == {
        "host": "0.0.0.0", "port": 4318, "acceptFormats": ["protobuf"],
    }  # fmt: skip


async def test_accept_formats_default_to_both(receiver):
    await et.start_otlp_receiver_if_needed(_suite({"acceptFormats": ["xml"]}))
    assert receiver.starts[0][2] == ["json", "protobuf"]


async def test_passes_redaction_command_tools_and_the_evaluation_policy(receiver):
    suite = _suite({"redactAttributes": ["authorization"]}, commandToolNames=["bash"])
    await et.start_otlp_receiver_if_needed(suite, "eval-1")
    assert receiver.starts == [(4318, "127.0.0.1", ["json", "protobuf"], {
        "commandToolNames": ["bash"], "redactAttributes": ["authorization"],
        "tracePolicy": {"evaluationId": "eval-1", "commandToolNames": ["bash"],
                        "redactAttributes": ["authorization"]},
    })]  # fmt: skip


async def test_start_errors_are_swallowed_by_default(receiver, caplog):
    receiver.start_error = OSError("EADDRINUSE: port 4318")
    assert await et.start_otlp_receiver_if_needed(_suite({})) is False
    assert et.is_otlp_receiver_started() is False
    assert "Failed to start OTLP receiver: EADDRINUSE: port 4318" in caplog.text


async def test_start_errors_rethrown_with_fail_on_receiver_start_failure(receiver):
    receiver.start_error = OSError("EADDRINUSE: port 4318")
    with pytest.raises(RuntimeError, match="Failed to start OTLP tracing receiver: EADDRINUSE"):
        await et.start_otlp_receiver_if_needed(_suite({}, failOnReceiverStartFailure=True))


async def test_prunes_with_retention_days(receiver, store, monkeypatch):
    await et.start_otlp_receiver_if_needed(_suite({}, storage={"type": "sqlite",
                                                               "retentionDays": 7}))  # fmt: skip
    assert store.deleted == [7]
    # Without an HTTP receiver, and with tracing enabled by the environment.
    await et.start_otlp_receiver_if_needed(_suite(storage={"retentionDays": 7}))
    monkeypatch.setenv("PROMPTFOO_TRACING_ENABLED", "true")
    await et.start_otlp_receiver_if_needed({"tracing": {"storage": {"retentionDays": 3}}})
    assert store.deleted == [7, 7, 3]
    assert len(receiver.starts) == 1


async def test_no_prune_without_retention_days(receiver, store):
    await et.start_otlp_receiver_if_needed(_suite({}, storage={"type": "sqlite"}))
    assert store.deleted == []


async def test_prunes_when_already_running(receiver, store):
    await et.start_otlp_receiver_if_needed(_suite({}))
    await et.start_otlp_receiver_if_needed(_suite({}, storage={"retentionDays": 7}))
    assert len(receiver.starts) == 1 and store.deleted == [7]


async def test_reuse_registers_the_evaluation_policy_without_new_defaults(receiver):
    await et.start_otlp_receiver_if_needed(_suite({}), "eval-a")
    await et.start_otlp_receiver_if_needed(
        _suite({"redactAttributes": ["authorization"], "acceptFormats": ["json"]},
               commandToolNames=["bash"]),
        "eval-b",
    )  # fmt: skip
    assert len(receiver.starts) == 1
    assert receiver.policies == [{"evaluationId": "eval-b", "commandToolNames": ["bash"],
                                  "redactAttributes": ["authorization"]}]  # fmt: skip


async def test_overlapping_startup_keeps_the_first_defaults(receiver):
    gate = asyncio.get_running_loop().create_future()
    receiver.start_gate = gate
    first = asyncio.ensure_future(et.start_otlp_receiver_if_needed(_suite(
        {"acceptFormats": ["json"], "redactAttributes": ["authorization"]},
        commandToolNames=["bash"])))  # fmt: skip
    second = asyncio.ensure_future(et.start_otlp_receiver_if_needed(_suite(
        {"acceptFormats": ["protobuf"], "redactAttributes": ["cookie"]},
        commandToolNames=["shell"])))  # fmt: skip
    while not receiver.starts:
        await asyncio.sleep(0)
    gate.set_result(None)
    assert await asyncio.gather(first, second) == [True, True]
    assert receiver.starts == [(4318, "127.0.0.1", ["json"], {
        "commandToolNames": ["bash"], "redactAttributes": ["authorization"]})]  # fmt: skip
    assert cli_state.active_otlp_receiver() == {
        "host": "127.0.0.1", "port": 4318, "acceptFormats": ["json"],
    }  # fmt: skip


async def test_first_endpoint_kept_until_the_last_lease_ends(receiver):
    first = await et.start_otlp_receiver_if_needed(
        _suite({"host": "127.0.0.2", "port": 14318, "acceptFormats": ["protobuf"]})
    )
    second = await et.start_otlp_receiver_if_needed(
        _suite({"host": "127.0.0.3", "port": 24318, "acceptFormats": ["json"]})
    )
    assert cli_state.active_otlp_receiver() == {
        "host": "127.0.0.2", "port": 14318, "acceptFormats": ["protobuf"],
    }  # fmt: skip
    await et.stop_otlp_receiver_if_needed(first)
    active = cli_state.active_otlp_receiver()
    assert active is not None and active["port"] == 14318
    await et.stop_otlp_receiver_if_needed(second)
    assert cli_state.active_otlp_receiver() is None


async def test_receiver_lives_until_every_lease_is_released(receiver):
    first = await et.start_otlp_receiver_if_needed(_suite({}), "eval-1")
    second = await et.start_otlp_receiver_if_needed(_suite({}), "eval-2")
    await et.stop_otlp_receiver_if_needed(first, "eval-1")
    assert receiver.stops == 0 and et.is_otlp_receiver_started() is True
    await et.stop_otlp_receiver_if_needed(second, "eval-2")
    assert receiver.stops == 1 and et.is_otlp_receiver_started() is False
    assert receiver.deregistered == ["eval-1", "eval-2"]


async def test_unacquired_release_is_a_no_op(receiver):
    await et.start_otlp_receiver_if_needed(_suite({}))
    await et.stop_otlp_receiver_if_needed(False, "eval-1")
    assert receiver.stops == 0 and receiver.deregistered == []


async def test_restart_waits_for_an_in_flight_shutdown(receiver):
    suite = _suite({})
    lease = await et.start_otlp_receiver_if_needed(suite)
    gate = asyncio.get_running_loop().create_future()
    receiver.stop_gate = gate
    stopping = asyncio.ensure_future(et.stop_otlp_receiver_if_needed(lease))
    restarting = asyncio.ensure_future(et.start_otlp_receiver_if_needed(suite))
    while receiver.stops == 0:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(receiver.starts) == 1
    gate.set_result(None)
    assert await asyncio.gather(stopping, restarting) == [None, True]
    assert len(receiver.starts) == 2


async def test_failed_overlapping_shutdown_preserves_the_lease(receiver):
    suite = _suite({})
    lease = await et.start_otlp_receiver_if_needed(suite)
    gate = asyncio.get_running_loop().create_future()
    receiver.stop_gate = gate
    stopping = asyncio.ensure_future(et.stop_otlp_receiver_if_needed(lease))
    restarting = asyncio.ensure_future(et.start_otlp_receiver_if_needed(suite))
    while receiver.stops == 0:
        await asyncio.sleep(0)
    gate.set_exception(RuntimeError("shutdown failed"))
    assert await asyncio.gather(stopping, restarting) == [None, True]
    assert len(receiver.starts) == 1 and et.is_otlp_receiver_started() is True
    active = cli_state.active_otlp_receiver()
    assert active is not None and active["port"] == 4318
    await et.stop_otlp_receiver_if_needed(True)
    assert receiver.stops == 2 and et.is_otlp_receiver_started() is False
    assert cli_state.active_otlp_receiver() is None
