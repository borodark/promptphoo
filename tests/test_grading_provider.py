"""src/util/gradingProvider.ts and the grader plumbing of getFinalTest (src/assertions/utils.ts)."""

from __future__ import annotations

import threading

from promptphoo.assertions.index import get_final_test
from promptphoo.grading_provider import (
    build_configured_provider_map,
    is_provider_type_map,
    resolve_configured_provider_reference,
    resolve_runtime_grading_provider_references,
)
from promptphoo.providers.base import ApiProvider


class P(ApiProvider):
    def __init__(self, pid: str, label: str | None = None) -> None:
        super().__init__({"label": label} if label else None)
        self._pid = pid
        self.lock = threading.Lock()  # not deep-copyable, like real provider clients

    def default_id(self) -> str:
        return self._pid


def test_provider_map_ids_win_over_labels():
    a = P("openai:gpt-4", label="grader")
    b = P("grader", label="other")
    provider_map = build_configured_provider_map([a, b])
    assert provider_map == {"openai:gpt-4": a, "grader": b, "other": b}


def test_is_provider_type_map():
    assert is_provider_type_map({"text": "openai:gpt-4"})
    assert is_provider_type_map({"embedding": {"id": "x"}})
    assert not is_provider_type_map({"text": ""})
    assert not is_provider_type_map({"id": "x", "text": "y"})
    assert not is_provider_type_map({"foo": "bar"})
    assert not is_provider_type_map("openai:gpt-4")


def test_resolve_configured_provider_reference():
    grader = P("openai:gpt-4", label="grader")
    provider_map = build_configured_provider_map([grader])
    assert resolve_configured_provider_reference("grader", provider_map) is grader
    assert resolve_configured_provider_reference("unknown", provider_map) == "unknown"
    options = {"id": "openai:gpt-4", "config": {}}
    assert resolve_configured_provider_reference(options, provider_map) is options
    typed = {"text": "openai:gpt-4", "embedding": {"id": "openai:gpt-4"}, "moderation": "x"}
    resolved = resolve_configured_provider_reference(typed, provider_map)
    assert resolved == {"text": grader, "embedding": grader, "moderation": "x"}
    inline = {"text": {"id": "openai:gpt-4", "config": {"temperature": 0}}}
    assert resolve_configured_provider_reference(inline, provider_map) is inline


def test_resolve_runtime_grading_provider_references():
    grader = P("openai:gpt-4")
    provider_map = build_configured_provider_map([grader])
    child = {"type": "llm-rubric", "provider": "openai:gpt-4"}
    untouched = {"type": "contains", "value": "x"}
    test = {
        "options": {"provider": "openai:gpt-4", "rubricPrompt": "r"},
        "assert": [{"type": "assert-set", "assert": [child]}, untouched],
    }
    resolve_runtime_grading_provider_references(test, provider_map)
    assert test["options"] == {"provider": grader, "rubricPrompt": "r"}
    assert test["assert"][0]["assert"][0]["provider"] is grader
    assert test["assert"][1] is untouched
    assert child["provider"] == "openai:gpt-4"  # the original is not mutated


def test_get_final_test_forwards_grading_options_and_shares_providers():
    grader = P("grader")
    target = P("target")
    test = {
        "vars": {"a": 1},
        "provider": target,
        "options": {"provider": grader, "rubricPrompt": "test rubric", "transform": "output"},
        "assert": [{"type": "llm-rubric", "provider": grader}],
    }
    final = get_final_test(test, {"type": "llm-rubric"})
    assert final["provider"] is target
    assert final["options"]["provider"] is grader
    assert final["options"]["rubricPrompt"] == "test rubric"
    assert final["options"]["transform"] == "output"
    assert final["assert"][0]["provider"] is grader
    assert final["vars"] == {"a": 1} and final["vars"] is not test["vars"]

    override = get_final_test(
        test, {"type": "llm-rubric", "provider": "openai:gpt-4", "rubricPrompt": ["msg"]}
    )
    assert override["options"]["provider"] == "openai:gpt-4"
    assert override["options"]["rubricPrompt"] == ["msg"]
    assert test["options"]["provider"] is grader
