"""`promptfoo init [directory]`. Ports src/commands/init.ts.

`--example <name>` downloads the example from the promptfoo GitHub repository through the
shared fetch layer, as upstream does (tests mock it with respx). The interactive prompts use
the numbered click menus from `promptphoo.onboarding`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import posixpath
import shutil
from typing import Any
from urllib.parse import unquote

import click

from promptphoo import UPSTREAM_VERSION, onboarding
from promptphoo.cli.common import common_options, exit_with
from promptphoo.promptfoo_command import promptfoo_command
from promptphoo.whatwg_url import URL

logger = logging.getLogger("promptphoo")

GITHUB_API_BASE = "https://api.github.com"
GITHUB_EXAMPLES_PATH = "/repos/promptfoo/promptfoo/contents/examples/"
DEFAULT_EXAMPLE_REFS = [UPSTREAM_VERSION, "main"]
EXAMPLE_CONFIG_FILENAMES = {
    "promptfooconfig.yaml",
    "promptfooconfig.yml",
    "promptfooconfig.js",
    "promptfooconfig.cjs",
    "promptfooconfig.mjs",
    "promptfooconfig.ts",
}


class ExampleError(Exception):
    pass


def example_aliases() -> dict[str, str]:
    return onboarding.templates()["EXAMPLE_ALIASES"]


def example_replacements() -> dict[str, str]:
    return onboarding.templates()["EXAMPLE_REPLACEMENTS"]


def removed_examples() -> dict[str, dict[str, str]]:
    return onboarding.templates()["REMOVED_EXAMPLES"]


def get_example_directory_url(dir_path: str, ref: str) -> URL:
    return URL(f"{GITHUB_API_BASE}{GITHUB_EXAMPLES_PATH}{dir_path}?ref={ref}")


def get_unsupported_example_reason(example_name: str) -> str | None:
    pathname = get_example_directory_url(example_name, UPSTREAM_VERSION).pathname
    try:
        pathname = unquote(pathname, errors="strict")
    except UnicodeDecodeError:
        pass
    root = (
        pathname[len(GITHUB_EXAMPLES_PATH) :].split("/")[0]
        if pathname.startswith(GITHUB_EXAMPLES_PATH)
        else None
    )
    if root in ("github-models", "provider-github-models"):
        return "GitHub Models has been retired, so this example is no longer supported."
    return None


def _github_headers() -> dict[str, str]:
    return {"Accept": "application/vnd.github.v3+json", "User-Agent": "promptfoo-cli"}


async def _fetch(url: str, options: dict[str, Any] | None = None) -> Any:
    from promptphoo.fetch import fetch_with_proxy

    return await fetch_with_proxy(url, options or {})


async def fetch_examples_tree(ref: str) -> list[dict[str, Any]]:
    response = await _fetch(
        f"{GITHUB_API_BASE}/repos/promptfoo/promptfoo/git/trees/{ref}?recursive=1",
        {"headers": _github_headers()},
    )
    if not response.ok:
        raise ExampleError(
            f"GitHub tree request failed for ref '{ref}': {response.status} {response.status_text}"
        )
    return response.json().get("tree") or []


def extract_runnable_examples(tree: list[dict[str, Any]]) -> list[str]:
    examples: set[str] = set()
    for item in tree:
        path = item.get("path") or ""
        if item.get("type") != "blob" or not path.startswith("examples/"):
            continue
        if posixpath.basename(path) not in EXAMPLE_CONFIG_FILENAMES:
            continue
        example_dir = posixpath.dirname(path).removeprefix("examples/")
        if example_dir and example_dir != "." and not get_unsupported_example_reason(example_dir):
            examples.add(example_dir)
    return sorted(examples, key=str.casefold)


def has_root_promptfoo_config(example_dir: str) -> bool:
    try:
        return any(e in EXAMPLE_CONFIG_FILENAMES for e in os.listdir(example_dir))
    except OSError:
        return False


def is_legacy_refs(refs: list[str]) -> bool:
    return len(refs) == 1 and refs[0] not in DEFAULT_EXAMPLE_REFS


def get_example_docs_url(example_name: str, refs: list[str]) -> str:
    docs_ref = refs[0] if is_legacy_refs(refs) else "main"
    return f"https://github.com/promptfoo/promptfoo/tree/{docs_ref}/examples/{example_name}"


async def fetch_example_directory_contents(dir_path: str, refs: list[str]) -> list[dict]:
    failed: list[str] = []
    for ref in refs:
        url = str(get_example_directory_url(dir_path, ref))
        response = await _fetch(url, {"headers": _github_headers()})
        if response.ok:
            return response.json()
        failed.append(f"{ref} ({response.status} {response.status_text})")
    raise ExampleError(
        f"Failed to fetch directory contents for refs: {', '.join(failed) or ', '.join(refs)}"
    )


async def download_file(url: str, file_path: str) -> None:
    response = await _fetch(url)
    if not response.ok:
        raise ExampleError(f"Failed to download file: {response.status_text}")
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        f.write(response.text())


async def download_directory(dir_path: str, target_dir: str, refs: list[str] | None = None) -> None:
    refs = refs or DEFAULT_EXAMPLE_REFS
    reason = get_unsupported_example_reason(dir_path)
    if reason:
        raise ExampleError(f"Example '{dir_path}' is unavailable. {reason}")
    for item in await fetch_example_directory_contents(dir_path, refs):
        item_path = onboarding.path_join(target_dir, item["name"])
        if item.get("type") == "file" and item.get("download_url"):
            await download_file(item["download_url"], item_path)
        elif item.get("type") == "dir":
            os.makedirs(item_path, exist_ok=True)
            await download_directory(f"{dir_path}/{item['name']}", item_path, refs)


async def download_example(
    example_name: str, target_dir: str, refs: list[str] | None = None
) -> None:
    dir_already_exists = os.path.exists(target_dir)
    try:
        os.makedirs(target_dir, exist_ok=True)
        await download_directory(example_name, target_dir, refs or DEFAULT_EXAMPLE_REFS)
    except Exception as error:
        if not dir_already_exists:
            try:
                shutil.rmtree(target_dir, ignore_errors=False)
            except FileNotFoundError:
                pass
            except OSError as rm_error:
                logger.error(f"Failed to remove directory: {rm_error}")
        raise ExampleError(f"Failed to download example: {error}") from error


async def get_examples_list() -> list[str]:
    try:
        try:
            return extract_runnable_examples(await fetch_examples_tree(UPSTREAM_VERSION))
        except Exception:  # noqa: BLE001 - fall back to main when VERSION isn't available
            return extract_runnable_examples(await fetch_examples_tree("main"))
    except Exception as error:  # noqa: BLE001
        logger.error(f"Failed to fetch examples list: {error}")
        return []


def resolve_example_selection(example: str) -> tuple[str, list[str]]:
    removed = removed_examples().get(example)
    if removed:
        logger.warning(f"Note: {removed['reason']}")
        logger.info(
            f"Downloading the legacy '{example}' example from promptfoo@{removed['legacyRef']}."
        )
        return example, [removed["legacyRef"]]
    resolved = example_aliases().get(example)
    if resolved:
        replacement = example_replacements().get(example)
        if replacement:
            logger.info(f"Note: {replacement} Downloading '{resolved}' instead.")
        else:
            logger.info(f"Note: '{example}' has been renamed to '{resolved}'. Using new name.")
        return resolved, DEFAULT_EXAMPLE_REFS
    return example, DEFAULT_EXAMPLE_REFS


def log_example_instructions(example_name: str, directory: str | None, refs: list[str]) -> None:
    example_path = onboarding.path_join(directory or ".", example_name)
    readme_path = onboarding.path_join(example_path, "README.md")
    readme_exists = os.path.exists(readme_path)
    docs_url = get_example_docs_url(example_name, refs)
    cd_command = f"cd {example_path}"
    if "redteam" in example_name or not has_root_promptfoo_config(example_path):
        if readme_exists:
            logger.info(f"View the README file at {readme_path} to get started!")
        else:
            logger.info(f"View the example at {docs_url} to get started!")
        return
    run_command = promptfoo_command("eval")
    if readme_exists:
        logger.info(
            f"View the README at {readme_path} or run:\n\n"
            f"`{cd_command} && {run_command}`\n\nto get started!"
        )
    else:
        logger.info(
            f"Run:\n\n`{cd_command} && {run_command}`\n\nto get started.\nExample docs: {docs_url}"
        )


async def select_example() -> str:
    examples = await get_examples_list()
    choices = [("None (initialize with dummy files)", ""), *((e, e) for e in examples)]
    return onboarding.select("Choose an example to download:", choices)


async def handle_example_download(
    directory: str | None, example: str | bool | None
) -> tuple[str | None, int]:
    """Returns (example name or None, exit code)."""
    example_name: str | None = None
    refs = DEFAULT_EXAMPLE_REFS
    if example is True:
        example_name = await select_example()
    elif isinstance(example, str):
        example_name, refs = resolve_example_selection(example)

    attempt = True
    while attempt and example_name:
        target_dir = onboarding.path_join(directory or ".", example_name)
        try:
            await download_example(example_name, target_dir, refs)
            logger.info(f"✅ Example project '{example_name}' written to: {target_dir}")
            if is_legacy_refs(refs):
                logger.info(f"Downloaded legacy example '{example_name}' from ref '{refs[0]}'.")
            attempt = False
        except Exception as error:  # noqa: BLE001
            logger.error(f"Failed to download example: {error}")
            if get_unsupported_example_reason(example_name):
                return example_name, 1
            attempt = onboarding.confirm(
                "Would you like to try downloading a different example?", default=True
            )
            if attempt:
                example_name = await select_example()
                refs = DEFAULT_EXAMPLE_REFS
            else:
                logger.info(
                    "No example downloaded. To get started, try:\n\n"
                    "  promptphoo init --example    (browse and select an example)\n"
                    "  promptphoo init              (create a basic project)"
                )
                return example_name, 0
    if not example_name:
        return None, 0
    log_example_instructions(example_name, directory, refs)
    return example_name, 0


_SELECT_EXAMPLE = "\x00select"


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.command(
        "init", help="Set up a new promptphoo project with prompts, providers, and test cases"
    )
    @click.argument("directory", required=False)
    @click.option(
        "--interactive/--no-interactive",
        default=True,
        help="Do not run in interactive mode",
    )
    @click.option(
        "--example",
        is_flag=False,
        flag_value=_SELECT_EXAMPLE,
        default=None,
        help="Download an example from the promptfoo repo",
    )
    @common_options
    def init_cmd(
        directory: str | None, interactive: bool, example: str | bool | None, **_: Any
    ) -> None:
        if example == _SELECT_EXAMPLE:
            example = True
        if directory == "redteam" and interactive:
            try:
                use_redteam = onboarding.confirm(
                    'You specified "redteam" as the directory. Did you mean to write '
                    '"npx promptfoo@latest redteam init" (Node promptfoo) instead?',
                    default=False,
                )
            except click.Abort:
                use_redteam = False
            if use_redteam:
                logger.warning(
                    "Red teaming is not part of promptphoo. Use Node promptfoo: "
                    '"npx promptfoo@latest redteam init".'
                )
                return
        try:
            example_name, code = asyncio.run(handle_example_download(directory, example))
        except click.Abort:
            example_name, code = None, 130
        if code or example_name:
            exit_with(code)
            return
        exit_with(onboarding.initialize_project(directory, interactive))

    return init_cmd
