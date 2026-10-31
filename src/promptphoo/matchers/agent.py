"""agent-rubric matcher. Ports src/matchers/agent.ts and isAgentic(Grading)Provider from
src/providers/agentic-utils.ts.

promptphoo ports no coding-agent provider, so there is no Codex default grader: agent-rubric
needs an explicitly configured provider whose id is agentic (see D26). The grader is never
pointed at a target workspace, because promptphoo creates no copy_working_dir workspaces.
"""

from __future__ import annotations

from typing import Any

from promptphoo.matchers.providers import get_grading_provider
from promptphoo.matchers.rubric import run_json_grading_prompt
from promptphoo.matchers.shared import try_parse
from promptphoo.prompts.grading import DEFAULT_AGENT_GRADING_PROMPT

AGENTIC_PROVIDER_IDS = (
    "anthropic:claude-agent-sdk",
    "anthropic:claude-code",
    "openai:codex",
    "openai:codex-app-server",
    "openai:codex-desktop",
    "openai:codex-security",
    "openai:codex-sdk",
    "openinterpreter",
    "opencode",
    "opencode:sdk",
)


def is_agentic_provider(provider: Any) -> bool:
    if not provider or not callable(getattr(provider, "id", None)):
        return False
    try:
        provider_id = provider.id()
    except Exception:
        return False
    return any(provider_id == a or provider_id.startswith(f"{a}:") for a in AGENTIC_PROVIDER_IDS)


def is_agentic_grading_provider(provider: Any) -> bool:
    """Security scanners run agentic workflows but cannot return rubric grading verdicts."""
    if not provider or not is_agentic_provider(provider):
        return False
    provider_id = provider.id()
    return provider_id != "openai:codex-security" and not provider_id.startswith(
        "openai:codex-security:"
    )


def is_agent_workspace(directory: Any) -> bool:
    """Upstream only trusts live workspaces this process created; promptphoo creates none."""
    return False


async def matches_agent_rubric(
    rubric: Any,
    llm_output: str,
    grading: dict[str, Any] | None = None,
    vars: dict[str, Any] | None = None,  # noqa: A002 - Node's parameter name
    assertion: dict[str, Any] | None = None,
    provider_call_context: dict[str, Any] | None = None,
    target_working_dir: Any = None,
) -> dict[str, Any]:
    if grading is None:
        raise ValueError(
            "Cannot grade output without grading config. Specify --grader option or grading config."
        )
    configured = (
        await get_grading_provider("text", grading["provider"], None)
        if grading.get("provider")
        else None
    )
    # Upstream falls back to getCodexDefaultProviders().llmRubricProvider (the Codex SDK).
    agent_provider = configured
    if not agent_provider or not is_agentic_grading_provider(agent_provider):
        raise ValueError(
            "agent-rubric assertion requires an agentic grading provider. "
            "Use openai:codex-sdk, openai:codex-app-server, anthropic:claude-agent-sdk, "
            "openinterpreter, or opencode:sdk."
        )
    extra: dict[str, Any] = {}
    if isinstance(target_working_dir, str) and is_agent_workspace(target_working_dir):
        extra["provider_prompt_config"] = {"working_dir": target_working_dir}
    result = await run_json_grading_prompt(
        assertion=assertion,
        check_name="agent-rubric check",
        default_prompt=DEFAULT_AGENT_GRADING_PROMPT,
        grading={**grading, "provider": agent_provider},
        label="agent-rubric",
        provider_call_context=provider_call_context,
        vars={**(vars or {}), "output": try_parse(llm_output), "rubric": rubric},
        **extra,
    )
    return {
        **result,
        "metadata": {**(result.get("metadata") or {}), "agentProvider": agent_provider.id()},
    }
