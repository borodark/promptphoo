"""Load-time provider and prompt reference checks. Cases from
test/util/validateTestProviderReferences.test.ts and the validateTestPromptReferences block of
test/test-prompts-filter.test.ts."""

from __future__ import annotations

import asyncio

import pytest

from promptphoo.config.load import resolve_configs
from promptphoo.config.references import (
    PromptReferenceValidationError,
    ProviderReferenceValidationError,
    validate_test_prompt_references,
    validate_test_provider_references,
)


class _Provider:
    def __init__(self, pid: str, label: str | None = None):
        self._id, self.label = pid, label

    def id(self) -> str:
        return self._id


PROVIDERS = [
    _Provider("openai:gpt-4", "smart-model"),
    _Provider("openai:gpt-3.5-turbo", "fast-model"),
]


# --- providers --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tests",
    [
        [{"vars": {"foo": "bar"}}],
        [{"providers": ["smart-model"]}],
        [{"providers": ["openai:gpt-4"]}],
        [{"providers": ["openai:*"]}],
        [{"providers": ["openai"]}],
        [{"providers": ["smart-model"]}, {"providers": ["fast-model"]}],
    ],
)
def test_valid_provider_refs(tests):
    validate_test_provider_references(tests, PROVIDERS)


def test_invalid_provider_ref_message():
    with pytest.raises(ProviderReferenceValidationError) as err:
        validate_test_provider_references([{"providers": ["nonexistent"]}], PROVIDERS)
    assert str(err.value) == (
        'Test #1 references provider "nonexistent" which does not exist. Available providers: '
        "smart-model (openai:gpt-4), fast-model (openai:gpt-3.5-turbo)"
    )


@pytest.mark.parametrize(
    ("tests", "default_test", "scenarios", "message"),
    [
        ([{"description": "My test case", "providers": ["x"]}], None, None, "My test case"),
        ([], {"providers": ["x"]}, None, "defaultTest references provider"),
        ([{"providers": "not-an-array"}], None, None, "must be an array"),
        ([{"providers": ["smart-model"]}, {"providers": ["x"]}], None, None, "Test #2"),
        (
            [],
            None,
            [{"config": [{}], "tests": [{"providers": ["nonexistent"]}]}],
            'Scenario #1 test #1 references provider "nonexistent"',
        ),
        (
            [],
            None,
            [{"config": [{"providers": ["nonexistent"]}], "tests": []}],
            r'Scenario #1 config\[0\] references provider "nonexistent"',
        ),
        (
            [],
            None,
            [{"description": "My scenario", "config": [{}], "tests": [{"providers": ["x"]}]}],
            r'Scenario #1 \("My scenario"\)',
        ),
        (
            [],
            None,
            [
                {"config": [{}], "tests": [{"providers": ["smart-model"]}]},
                {"config": [{}], "tests": [{"providers": ["nonexistent"]}]},
            ],
            "Scenario #2",
        ),
    ],
)
def test_invalid_provider_refs(tests, default_test, scenarios, message):
    with pytest.raises(ProviderReferenceValidationError, match=message):
        validate_test_provider_references(tests, PROVIDERS, default_test, scenarios)


def test_valid_default_and_scenario_provider_refs():
    validate_test_provider_references([], PROVIDERS, {"providers": ["smart-model"]})
    scenarios = [{"config": [{}], "tests": [{"providers": ["smart-model"]}]}]
    validate_test_provider_references([], PROVIDERS, None, scenarios)


# --- prompts ----------------------------------------------------------------------------------

PROMPTS = [
    {"id": "prompt-a", "label": "Factual Assistant", "raw": "content a"},
    {"id": "prompt-b", "label": "Creative Writer", "raw": "content b"},
    {"label": "Math:Basic", "raw": "content c"},
    {"label": "Math:Advanced", "raw": "content d"},
]


@pytest.mark.parametrize(
    "tests",
    [
        [{"prompts": ["Factual Assistant"]}],
        [{"prompts": ["prompt-a"]}],
        [{"prompts": ["Factual Assistant", "prompt-b"]}],
        [{"vars": {"topic": "test"}}],
        [{"prompts": []}],
        [{"prompts": ["Math:*"]}],
        [{"prompts": ["Math"]}],
        [{"prompts": ["Factual Assistant"]}, {"prompts": ["prompt-b"]}, {"prompts": ["Math:*"]}],
    ],
)
def test_valid_prompt_refs(tests):
    validate_test_prompt_references(tests, PROMPTS)


def test_invalid_prompt_ref_message():
    with pytest.raises(PromptReferenceValidationError) as err:
        validate_test_prompt_references([{"prompts": ["Nonexistent Prompt"]}], PROMPTS)
    assert str(err.value) == (
        'Test #1 references prompt "Nonexistent Prompt" which does not exist.\n'
        "Available prompts: [Creative Writer, Factual Assistant, Math:Advanced, Math:Basic, "
        "prompt-a, prompt-b]"
    )


@pytest.mark.parametrize(
    ("tests", "default_test", "message"),
    [
        ([{"prompts": ["nonexistent-id"]}], None, "does not exist"),
        (
            [{"description": "My important test", "prompts": ["Nonexistent Prompt"]}],
            None,
            r'Test #1 \("My important test"\) references prompt',
        ),
        ([{"prompts": ["Science:*"]}], None, "does not exist"),
        ([], {"prompts": ["Nonexistent"]}, 'defaultTest references prompt "Nonexistent"'),
        (
            [{"prompts": ["Factual Assistant"]}, {"prompts": ["Invalid1"]}, {"prompts": ["x"]}],
            None,
            'Test #2 references prompt "Invalid1"',
        ),
    ],
)
def test_invalid_prompt_refs(tests, default_test, message):
    with pytest.raises(PromptReferenceValidationError, match=message):
        validate_test_prompt_references(tests, PROMPTS, default_test)


def test_default_test_without_prompts():
    validate_test_prompt_references([], PROMPTS, {"prompts": ["Factual Assistant"]})
    validate_test_prompt_references([], PROMPTS, {"threshold": 0.5})


@pytest.mark.parametrize(
    ("tests", "default_test", "prompts", "lines", "prefix"),
    [
        (
            [{"description": "Missing prompt", "prompts": ["Missing Prompt"]}],
            None,
            PROMPTS,
            [
                "providers:",
                "  - openai:gpt-4.1-mini",
                "prompts:",
                "  - id: prompt-a",
                "    label: Factual Assistant",
                "tests:",
                "  - description: Missing prompt",
                "    prompts:",
                "      - Missing Prompt",
            ],
            'promptfooconfig.yaml:9:9: Test #1 ("Missing prompt")',
        ),
        (
            [
                {
                    "description": "Missing prompt",
                    "vars": {"question": "Missing Prompt"},
                    "prompts": ["Missing Prompt"],
                }
            ],
            None,
            PROMPTS,
            [
                "prompts:",
                "  - id: prompt-a",
                "    label: Factual Assistant",
                "tests:",
                "  - description: Missing prompt",
                "    vars:",
                "      question: Missing Prompt",
                "    assert:",
                "      - type: contains",
                "        value: Missing Prompt",
                "    prompts:",
                "      - Missing Prompt",
            ],
            'promptfooconfig.yaml:12:9: Test #1 ("Missing prompt")',
        ),
        (
            [],
            {"prompts": ["Missing Default"]},
            PROMPTS,
            ["defaultTest:", "  prompts:", "    - Missing Default"],
            "promptfooconfig.yaml:3:7: defaultTest",
        ),
        (
            [{"description": "Target duplicate", "prompts": ["Missing Prompt"]}],
            None,
            PROMPTS,
            [
                "prompts:",
                "  - id: prompt-a",
                "    label: Factual Assistant",
                "tests:",
                "  - prompts:",
                "      - Missing Prompt",
                "    description: First duplicate",
                "  - prompts:",
                "      - Missing Prompt",
                "    description: Target duplicate",
            ],
            'promptfooconfig.yaml:9:9: Test #1 ("Target duplicate")',
        ),
        (
            [{"description": "substr", "prompts": ["Ghost Extended", "Ghost"]}],
            None,
            [{"raw": "x", "label": "Ghost Extended"}],
            [
                "prompts:",
                "  - raw: x",
                "    label: Ghost Extended",
                "tests:",
                "  - description: substr",
                "    prompts:",
                "      - Ghost Extended",
                "      - Ghost",
            ],
            'promptfooconfig.yaml:8:9: Test #1 ("substr") references prompt "Ghost"',
        ),
        (
            [{"description": "Uses missing prompt", "prompts": ["Missing Prompt"]}],
            {"prompts": ["Missing Prompt"]},
            PROMPTS,
            [
                "prompts:",
                "  - id: prompt-a",
                "    label: Factual Assistant",
                "tests:",
                "  - description: Uses missing prompt",
                "    prompts:",
                "      - Missing Prompt",
                "defaultTest:",
                "  prompts:",
                "    - Missing Prompt",
            ],
            "promptfooconfig.yaml:10:7: defaultTest",
        ),
    ],
)
def test_prompt_ref_yaml_location(tests, default_test, prompts, lines, prefix):
    sources = [{"path": "promptfooconfig.yaml", "content": "\n".join(lines)}]
    with pytest.raises(PromptReferenceValidationError) as err:
        validate_test_prompt_references(tests, prompts, default_test, sources)
    assert str(err.value).startswith(prefix)


# --- through resolve_configs ------------------------------------------------------------------


def test_resolve_configs_checks_references(tmp_path):
    config = tmp_path / "promptfooconfig.yaml"
    config.write_text(
        "prompts:\n  - id: p1\n    raw: hi\nproviders: [echo]\n"
        "tests:\n  - description: bad\n    vars: {x: 1}\n    prompts:\n      - p2\n"
    )
    with pytest.raises(PromptReferenceValidationError) as err:
        asyncio.run(resolve_configs({"config": [str(config)]}, {}))
    assert str(err.value).startswith(f'{config}:9:9: Test #1 ("bad") references prompt "p2"')

    config.write_text(
        "prompts: [hi]\nproviders: [echo]\ntests:\n  - vars: {x: 1}\n    providers: [openai]\n"
    )
    with pytest.raises(ProviderReferenceValidationError, match="Available providers: echo"):
        asyncio.run(resolve_configs({"config": [str(config)]}, {}))
