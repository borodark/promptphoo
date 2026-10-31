"""`promptfoo init` project scaffolding. Ports src/onboarding.ts.

The file templates are vendored verbatim in `_data/init_templates.json`
(regenerate with tools/gen_init_templates.mjs), and the config template is rendered with the
Nunjucks-compatible engine, so `init --no-interactive` writes byte-identical files.

The interactive wizard asks the same questions with numbered click prompts instead of
@inquirer's arrow-key menus. Choosing "Run a red team evaluation" stops with a not-supported
message, because redteam is out of scope.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import sys
from functools import cache
from importlib import resources
from typing import Any

import click

from promptphoo import envars
from promptphoo.promptfoo_command import promptfoo_command
from promptphoo.templating import render_string

logger = logging.getLogger("promptphoo")


@cache
def templates() -> dict[str, Any]:
    text = (resources.files("promptphoo") / "_data" / "init_templates.json").read_text("utf-8")
    return json.loads(text)


def get_default_readme(action: str | None = None) -> str:
    use_case = (
        "RAG evaluation"
        if action == "rag"
        else "agent evaluation"
        if action == "agent"
        else "prompt evaluation"
    )
    text: str = templates()["README"]
    for sub in ("eval", "view"):
        text = text.replace(f"@@PROMPTFOO_COMMAND:{sub}@@", promptfoo_command(sub))
    return text.replace("@@USE_CASE@@", use_case)


def report_provider_api_key_warnings(provider_choices: list[Any]) -> list[str]:
    ids = [(c.get("id") or "") if isinstance(c, dict) else c for c in provider_choices]
    mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "vertex": "GOOGLE_API_KEY",
        "google": "GOOGLE_API_KEY",
        "cohere": "COHERE_API_KEY",
    }
    return [
        f"Warning: {key} environment variable is not set.\n"
        f"Please set this environment variable like: export {key}=<my-api-key>"
        for prefix, key in mapping.items()
        if any(i.startswith(prefix) for i in ids) and not envars.get_env_string(key)
    ]


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
            },
            "required": ["location"],
        },
    },
}


def provider_choices(action: str) -> list[tuple[str, list[Any]]]:
    """The wizard's model-provider menu, in upstream order."""
    agent = action == "agent"
    return [
        ("I'll choose later", ["openai:gpt-6-luna", "openai:gpt-6-sol"]),
        (
            "[OpenAI] GPT-6 Sol" if agent else "[OpenAI] GPT-6 Luna and Sol",
            [
                {
                    "id": "openai:chat:gpt-6-sol",
                    "config": {"reasoning_effort": "none", "tools": [_WEATHER_TOOL]},
                }
            ]
            if agent
            else ["openai:gpt-6-luna", "openai:gpt-6-sol"],
        ),
        (
            "[Anthropic] Claude Fable, Opus, Sonnet, Haiku, ...",
            [
                "anthropic:messages:claude-fable-5",
                "anthropic:messages:claude-opus-5-5",
                "anthropic:messages:claude-opus-4-8",
                "anthropic:messages:claude-sonnet-5-5",
                "anthropic:messages:claude-sonnet-4-6",
                "anthropic:messages:claude-opus-4-6",
                "anthropic:messages:claude-haiku-4-5",
            ],
        ),
        (
            "[Google] Gemini 3.8 Flash, 3.7 Flash, 3.6 Flash, 3.5 Flash-Lite, ...",
            [
                *(
                    {"id": f"vertex:{m}", "config": {"region": "global"}}
                    for m in (
                        "gemini-3.8-flash",
                        "gemini-3.7-flash",
                        "gemini-3.6-flash",
                        "gemini-3.5-flash-lite",
                    )
                ),
                "vertex:gemini-3.1-pro-preview",
                "vertex:gemini-2.5-pro",
            ],
        ),
        (
            "[HuggingFace] Llama, Phi, Gemma, ...",
            [
                "huggingface:text-generation:meta-llama/Meta-Llama-3.1-8B-Instruct",
                "huggingface:text-generation:microsoft/Phi-4-mini-instruct",
                "huggingface:text-generation:google/gemma-3-4b-it",
            ],
        ),
        ("Local Python script", ["file://provider.py"]),
        ("Local Javascript script", ["file://provider.js"]),
        ("Local executable", ["exec:provider.bat" if _is_windows() else "exec:provider.sh"]),
        ("HTTP endpoint", ["https://example.com/api/generate"]),
        (
            "[Azure] OpenAI, DeepSeek, Llama, ...",
            [
                {
                    "id": "azure:chat:deploymentNameHere",
                    "config": {"apiHost": "xxxxxxxx.openai.azure.com"},
                }
            ],
        ),
        ("[AWS Bedrock] Claude, Llama, Titan, ...", ["bedrock:us.anthropic.claude-sonnet-5"]),
        (
            "[Cohere] Command A+, Command A, ...",
            ["cohere:command-a-plus-05-2026", "cohere:command-a-03-2025"],
        ),
        ("[Ollama] Llama, Qwen, Phi, ...", ["ollama:chat:llama3.2:3b", "ollama:chat:phi4"]),
        (
            "[WatsonX] Llama, IBM Granite, ...",
            [
                "watsonx:meta-llama/llama-3-2-11b-vision-instruct",
                "watsonx:ibm/granite-3-3-8b-instruct",
            ],
        ),
    ]


ACTION_CHOICES = [
    ("Not sure yet", "compare", "Get started with a basic prompt comparison"),
    (
        "Compare prompts and models",
        "compare",
        "Test different prompts, models, or parameters side by side",
    ),
    ("Improve RAG performance", "rag", "Evaluate retrieval-augmented generation pipelines"),
    (
        "Improve agent/chain of thought performance",
        "agent",
        "Test agent workflows and tool-calling behavior",
    ),
    (
        "Run a red team evaluation",
        "redteam",
        "Scan for security vulnerabilities and compliance risks",
    ),
]

LANGUAGE_CHOICES = [
    ("Not sure yet", "not_sure"),
    ("Python", "python"),
    ("Javascript", "javascript"),
]


def path_join(*parts: str) -> str:
    """Node's path.join, which (unlike os.path.join) keeps earlier parts before an absolute one."""
    joined = "/".join(p for p in parts if p)
    return posixpath.normpath(joined) if joined else "."


def _is_windows() -> bool:
    return sys.platform == "win32"


def select(message: str, choices: list[tuple[str, Any]]) -> Any:
    """A numbered stand-in for @inquirer/prompts `select`; the first choice is the default."""
    click.echo(f"? {message}")
    for i, (name, *_rest) in enumerate(choices, 1):
        click.echo(f"  {i}. {name}")
    picked = click.prompt(
        "Select",
        type=click.IntRange(1, len(choices)),
        default=1,
        show_default=True,
    )
    return choices[picked - 1][1]


def confirm(message: str, default: bool = False) -> bool:
    return click.confirm(f"? {message}", default=default)


class InitAborted(Exception):
    """Raised for redteam, which promptphoo does not scaffold."""


def create_dummy_files(directory: str | None, interactive: bool = True) -> dict[str, Any]:
    out_directory = directory or "."
    out_dir_absolute = path_join(os.getcwd(), out_directory)

    def write_file(file: str, contents: str, required: bool) -> None:
        relative_path = path_join(out_directory, file)
        absolute_path = path_join(out_dir_absolute, file)
        if interactive and os.path.exists(absolute_path):
            required_text = "(required)" if required else "(optional)"
            if not confirm(
                f"{relative_path} {required_text} already exists. Do you want to overwrite it?",
                default=False,
            ):
                if required:
                    logger.warning(
                        f"⚠️ Skipping required file {relative_path} - configuration may be "
                        "incomplete"
                    )
                else:
                    logger.info(f"⏩ Skipping {relative_path}")
                return
        with open(absolute_path, "w", encoding="utf-8", newline="") as f:
            f.write(contents)
        logger.info(f"📝 Wrote {relative_path}")

    t = templates()
    prompts: list[str] = []
    providers: list[Any] = []
    os.makedirs(out_dir_absolute, exist_ok=True)

    if interactive:
        logger.info(
            "\nWelcome to promptphoo!\nWe'll set up a configuration file to get you started.\n"
        )
        action = select("What would you like to do?", [(n, v) for n, v, _d in ACTION_CHOICES])
        if action == "redteam":
            raise InitAborted(
                "Red teaming is not supported by promptphoo. Use Node promptfoo's "
                f'"{promptfoo_command("redteam init")}" to set up a red teaming project.'
            )
        language = "not_sure"
        if action in ("rag", "agent"):
            language = select(
                "What programming language are you developing the app in?", LANGUAGE_CHOICES
            )
        choice = select("Which model provider would you like to use?", provider_choices(action))
        chosen: list[Any] = choice if isinstance(choice, list) else [choice]
        for warning in report_provider_api_key_warnings(chosen):
            logger.warning(warning)
        if chosen:
            if len(chosen) > 3:
                providers.extend(c[0] if isinstance(c, list) else c for c in chosen)
            else:
                providers.extend(chosen)
            ids = [c for c in chosen if isinstance(c, str)]
            if any(i.startswith("file://") and i.endswith(".js") for i in ids):
                write_file("provider.js", t["JAVASCRIPT_PROVIDER"], True)
            if any(i.startswith("exec:") for i in ids):
                windows = _is_windows()
                write_file(
                    "provider.bat" if windows else "provider.sh",
                    t["WINDOWS_PROVIDER"] if windows else t["BASH_PROVIDER"],
                    True,
                )
            if any(
                i.startswith("python:") or (i.startswith("file://") and i.endswith(".py"))
                for i in ids
            ):
                write_file("provider.py", t["PYTHON_PROVIDER"], True)
        else:
            providers.extend(["openai:gpt-6-luna", "openai:gpt-6-sol"])

        if action == "compare":
            prompts.append("Write a tweet about {{topic}}")
            if len(providers) < 3:
                prompts.append("Write a concise, funny tweet about {{topic}}")
        elif action == "rag":
            prompts.append(
                "Write a customer service response to:\n\n{{inquiry}}\n\nUse these documents:"
                "\n\n{{context}}"
            )
        elif action == "agent":
            prompts.append("Fulfill this user helpdesk ticket: {{inquiry}}")

        if action in ("rag", "agent"):
            if language == "javascript":
                write_file("context.js", t["JAVASCRIPT_VAR"], True)
            else:
                write_file("context.py", t["PYTHON_VAR"], True)
    else:
        action = "compare"
        language = "not_sure"
        prompts.extend(
            ["Write a tweet about {{topic}}", "Write a concise, funny tweet about {{topic}}"]
        )
        providers.extend(["openai:gpt-6-luna", "openai:gpt-6-sol"])

    config = render_string(
        t["CONFIG_TEMPLATE"],
        {"prompts": prompts, "providers": providers, "type": action, "language": language},
    )
    write_file("README.md", get_default_readme(action), False)
    write_file("promptfooconfig.yaml", config, True)

    def prefix(provider: Any) -> str:
        provider_id = provider if isinstance(provider, str) else provider.get("id")
        return provider_id.split(":")[0] if provider_id is not None else "unknown"

    return {
        "numPrompts": len(prompts),
        "providerPrefixes": [prefix(p) for p in providers],
        "action": action,
        "language": language,
        "outDirectory": out_directory,
    }


def initialize_project(directory: str | None, interactive: bool = True) -> int:
    """Scaffold the project and print next steps. Returns the exit code."""
    try:
        result = create_dummy_files(directory, interactive)
    except InitAborted as err:
        logger.warning(str(err))
        return 1
    except (click.Abort, EOFError, KeyboardInterrupt):
        logger.info(
            "\nInitialization paused. To continue setup later, use the command: "
            + promptfoo_command("init")
        )
        logger.info("For help or feedback, run: promptphoo feedback")
        return 130
    out_directory = result["outDirectory"]
    run_command = promptfoo_command("eval")
    view_command = promptfoo_command("view")
    logger.info("")
    if out_directory == ".":
        logger.info("✅ Setup complete! Next steps:\n")
        logger.info(f"  1. Run {run_command} to evaluate your prompts")
        logger.info(f"  2. Run {view_command} (Node promptfoo) to view results in your browser")
    else:
        logger.info(f"✅ Wrote promptfooconfig.yaml to ./{out_directory}\n")
        logger.info(f"  1. Run cd {out_directory}")
        logger.info(f"  2. Run {run_command} to evaluate your prompts")
        logger.info(f"  3. Run {view_command} (Node promptfoo) to view results in your browser")
    logger.info("")
    logger.info(
        "  Config docs (promptfoo, same format): https://promptfoo.dev/docs/configuration/guide"
    )
    return 0
