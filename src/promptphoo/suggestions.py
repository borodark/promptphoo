"""Prompt suggestions (`eval --suggest-prompts N`). Ports src/suggestions.ts, and
`maybeAddGeneratedPrompts` / `PromptSuggestionsRejectedError` from src/evaluator.ts, and
`promptYesNo` from src/util/readline.ts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from promptphoo.jsvalues import js_string, json_stringify
from promptphoo.tokens import accumulate_token_usage, create_empty_token_usage

logger = logging.getLogger("promptphoo")

MAX_SUGGESTIONS_COUNT = 50
SUGGESTIONS_CONCURRENCY = 4


class PromptSuggestionsRejectedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("No prompts selected. Aborting.")


async def generate_prompts(prompt: str, num: int) -> dict[str, Any]:
    """Ask the default suggestions provider for `num` variants of `prompt`.

    Returns `{prompts, tokensUsed}` or `{error, tokensUsed}`; one failed variant does not
    discard the others."""
    from promptphoo.prompts.grading import SUGGEST_PROMPTS_SYSTEM_MESSAGE
    from promptphoo.providers.defaults import get_default_providers

    if not isinstance(num, int) or isinstance(num, bool) or num < 1 or num > MAX_SUGGESTIONS_COUNT:
        return {
            "error": f"generatePrompts: num must be an integer between 1 and "
            f"{MAX_SUGGESTIONS_COUNT} (got {js_string(num)})",
            "tokensUsed": create_empty_token_usage(),
        }
    provider = (await get_default_providers())["suggestionsProvider"]
    payload = json_stringify(
        [
            SUGGEST_PROMPTS_SYSTEM_MESSAGE,
            {"role": "user", "content": "Generate a variant for the following prompt:"},
            {"role": "user", "content": prompt},
        ]
    )
    semaphore = asyncio.Semaphore(SUGGESTIONS_CONCURRENCY)

    async def one(i: int) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            try:
                return i, await provider.call_api(payload)
            except Exception as err:
                return i, {"error": str(err)}

    responses = await asyncio.gather(*(one(i) for i in range(num)))
    tokens_used = create_empty_token_usage()
    prompts: list[str] = []
    errors: list[str] = []
    for i, resp in responses:
        if isinstance(resp.get("tokenUsage"), dict):
            accumulate_token_usage(tokens_used, resp["tokenUsage"])
        if resp.get("error") or not resp.get("output"):
            message = resp.get("error") or "Unknown error"
            errors.append(f"Variant {i + 1}: {message}")
            logger.warning("[suggestions] Variant %d/%d failed: %s", i + 1, num, message)
            continue
        output = resp["output"]
        prompts.append(output if isinstance(output, str) else js_string(output))

    if not prompts:
        return {
            "error": "; ".join(errors) if errors else "No prompts generated",
            "tokensUsed": tokens_used,
        }
    if errors:
        logger.warning(
            "[suggestions] Generated %d/%d prompt variants; %d failed.",
            len(prompts), num, len(errors),
        )  # fmt: skip
    return {"prompts": prompts, "tokensUsed": tokens_used}


def prompt_user(question: str) -> str:
    return input(question)


async def prompt_yes_no(question: str, default_yes: bool = False) -> bool:
    suffix = "(Y/n): " if default_yes else "(y/N): "
    answer = await asyncio.to_thread(prompt_user, f"{question} {suffix}")
    if default_yes:
        return not answer.strip().lower().startswith("n")
    return answer.strip().lower().startswith("y")


async def maybe_add_generated_prompts(test_suite: dict[str, Any], options: dict[str, Any]) -> None:
    """With `generateSuggestions`, generate variants of the first prompt and append the ones
    the user accepts. Raises PromptSuggestionsRejectedError when none is accepted."""
    if not options.get("generateSuggestions"):
        return
    raw_count = options.get("suggestionsCount")
    raw_count = 1 if raw_count is None else raw_count
    requested = raw_count if isinstance(raw_count, int) and raw_count >= 1 else 1
    if requested > MAX_SUGGESTIONS_COUNT:
        logger.warning(
            "suggestionsCount=%s exceeds max of %d; clamping to %d.",
            js_string(raw_count), MAX_SUGGESTIONS_COUNT, MAX_SUGGESTIONS_COUNT,
        )  # fmt: skip
        requested = MAX_SUGGESTIONS_COUNT
    logger.info("Generating prompt variations...")
    generated = await generate_prompts(test_suite["prompts"][0]["raw"], requested)
    new_prompts = generated.get("prompts")
    if generated.get("error") or not new_prompts:
        raise RuntimeError(f"Failed to generate prompts: {generated.get('error')}")
    if len(new_prompts) < requested:
        logger.warning(
            "Only %d of %d requested prompt variants were generated. See warnings above for "
            "details.",
            len(new_prompts), requested,
        )  # fmt: skip

    logger.info("Generated prompts:")
    num_added = 0
    for prompt in new_prompts:
        logger.info("--------------------------------------------------------")
        logger.info("%s", prompt)
        logger.info("--------------------------------------------------------")
        if await prompt_yes_no("Do you want to test this prompt?", False):
            test_suite["prompts"].append({"raw": prompt, "label": prompt})
            num_added += 1
        else:
            logger.info("Skipping this prompt.")
    if num_added > 0:
        return
    logger.info("No prompts selected. Aborting.")
    raise PromptSuggestionsRejectedError()
