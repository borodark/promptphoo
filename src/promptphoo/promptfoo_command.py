"""How to spell the next command for the user. Replaces src/util/promptfooCommand.ts.

Upstream spells it `promptfoo <sub>` or `npx promptfoo@latest <sub>` depending on how promptfoo
was installed. promptphoo always names itself, so a user never mistakes one tool for the other;
commands that only Node promptfoo has (`view`, `redteam`, `auth`) are spelled as Node commands,
which work on the same ~/.promptfoo state (D62).
"""

from __future__ import annotations

CLI_NAME = "promptphoo"
NODE_ONLY_COMMANDS = ("view", "redteam", "auth")


def promptfoo_command(subcommand: str) -> str:
    if subcommand.split(" ", 1)[0] in NODE_ONLY_COMMANDS:
        return f"npx promptfoo@latest {subcommand}"
    return f"{CLI_NAME} {subcommand}" if subcommand else CLI_NAME
