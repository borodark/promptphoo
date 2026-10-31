"""Extension hooks (beforeAll, beforeEach, afterEach, afterAll).

Ports `runExtensionHook`, `getExtensionHookName` and `restorePromptFunctions` from
src/evaluatorHelpers.ts, and `runBeforeAllExtensions` / `synchronizeLegacyTransportHeaders`
from src/evaluator.ts. Hooks are called through `transform.transform`, so a Python file runs
in-process with JSON-round-tripped arguments (as Node's runPython does) and a JavaScript file
fails with a clear error.
"""

from __future__ import annotations

import logging
from typing import Any

from promptphoo.files import is_javascript_file
from promptphoo.numeric import filter_finite_scores

logger = logging.getLogger("promptphoo")

EXTENSION_HOOK_NAMES = frozenset({"beforeAll", "beforeEach", "afterEach", "afterAll"})

# The beforeAll suite keys a hook may replace; everything else is kept from the input suite.
_BEFORE_ALL_MUTABLE_KEYS = (
    "prompts", "providerPromptMap", "tests", "scenarios", "defaultTest", "nunjucksFilters",
    "derivedMetrics", "redteam",
)  # fmt: skip


class ExtensionHookError(RuntimeError):
    pass


def get_extension_hook_name(extension: str) -> str | None:
    """The function name of `file://path:fn`, or None. A drive-letter colon is not a name."""
    if not extension.startswith("file://"):
        return None
    last_colon = extension.rfind(":")
    if last_colon > 8:
        return extension[last_colon + 1 :] or None
    return None


def _extension_file(extension: str) -> str | None:
    if not extension.startswith("file://"):
        return None
    target = extension[len("file://") :]
    idx = target.rfind(":")
    return target[:idx] if idx > 1 else target


def check_extensions_supported(extensions: Any) -> None:
    """Reject JavaScript extension files before the eval starts (promptphoo runs Python only)."""
    if not isinstance(extensions, list):
        return
    for extension in extensions:
        path = _extension_file(extension) if isinstance(extension, str) else None
        if path and is_javascript_file(path):
            raise ExtensionHookError(
                f"JavaScript extension hooks are not supported by promptphoo: {extension}. "
                "Use a Python file instead (e.g. file://hooks.py:extension_hook)."
            )


def restore_prompt_functions(returned_prompts: Any, original_prompts: Any) -> Any:
    """Re-attach the `function` of `file://...:fn` prompts lost in the hook's JSON round-trip.

    A returned prompt gets its original's function back when it has the same label and an
    unchanged `raw`. Labels shared by several function-carrying originals are left alone.
    """
    if not isinstance(returned_prompts, list) or not isinstance(original_prompts, list):
        return returned_prompts

    originals_by_label: dict[str, dict[str, Any] | None] = {}
    for original in original_prompts:
        if (
            not isinstance(original, dict)
            or not callable(original.get("function"))
            or original.get("label") is None
        ):
            continue
        label = original["label"]
        # None marks an ambiguous (duplicated) label that must not be restored.
        originals_by_label[label] = None if label in originals_by_label else original
    if not originals_by_label:
        return returned_prompts

    def restore(returned: Any) -> Any:
        if (
            not isinstance(returned, dict)
            or callable(returned.get("function"))
            or returned.get("label") is None
        ):
            return returned
        original = originals_by_label.get(returned["label"])
        if original is not None and original.get("raw") == returned.get("raw"):
            return {**returned, "function": original["function"]}
        return returned

    restored = [restore(p) for p in returned_prompts]

    labels_with_function: set[str] = set()
    returned_labels: set[str] = set()
    for prompt in restored:
        if isinstance(prompt, dict) and prompt.get("label") is not None:
            returned_labels.add(prompt["label"])
            if callable(prompt.get("function")):
                labels_with_function.add(prompt["label"])
    for label, original in originals_by_label.items():
        if label in labels_with_function:
            continue
        if original is None:
            if label in returned_labels:
                logger.warning(
                    'beforeAll extension hook: multiple prompts share the label "%s", so their '
                    "prompt functions cannot be safely restored after the hook's serialized "
                    "round-trip. These prompts will render their raw source as text. Give each "
                    "prompt a unique label to fix this.",
                    label,
                )
        elif label in returned_labels:
            logger.debug(
                'beforeAll extension hook rewrote prompt "%s"; its original prompt function was '
                "not re-attached.",
                label,
            )
        elif any(
            isinstance(p, dict)
            and not callable(p.get("function"))
            and p.get("raw") == original.get("raw")
            for p in restored
        ):
            logger.warning(
                'beforeAll extension hook: prompt "%s" carried a prompt function, but the hook '
                "returned its content under a different label. The function cannot be restored, "
                "so the prompt will render its raw source as text. Keep prompt labels unchanged "
                "to preserve prompt functions.",
                label,
            )
        else:
            logger.debug(
                'beforeAll extension hook removed prompt "%s"; its prompt function no longer '
                "applies.",
                label,
            )
    return restored


def _get(value: Any, key: str) -> Any:
    """`value.key` in JS: reading a property of a non-object throws a TypeError."""
    if isinstance(value, dict):
        return value.get(key)
    if value is None:
        raise TypeError(f"Cannot read properties of {value!r} (reading '{key}')")
    return None


async def run_extension_hook(
    extensions: Any, hook_name: str, context: dict[str, Any]
) -> dict[str, Any]:
    """Run each extension for `hook_name` and return the context with its mutable fields
    updated from the hooks' return values (beforeAll: the suite's prompts, tests, ...;
    beforeEach: the test; afterEach: the result's namedScores, metadata and response.metadata;
    afterAll: return values are ignored)."""
    from promptphoo.transform import transform

    if not extensions or not isinstance(extensions, list):
        return context

    logger.debug("Running %s hook with %d extension(s)", hook_name, len(extensions))
    updated: dict[str, Any] = {**context}

    for extension in extensions:
        if not isinstance(extension, str):
            raise ExtensionHookError("extension must be a string")
        extension_hook_name = get_extension_hook_name(extension)
        if (
            extension_hook_name
            and extension_hook_name in EXTENSION_HOOK_NAMES
            and extension_hook_name != hook_name
        ):
            logger.debug(
                "Skipping extension %s for hook %s (extension targets %s only)",
                extension, hook_name, extension_hook_name,
            )  # fmt: skip
            continue

        # A known hook name uses the new convention fn(context, {hookName}); a custom
        # function name (or none) the legacy fn(hookName, context).
        new_convention = bool(extension_hook_name and extension_hook_name in EXTENSION_HOOK_NAMES)
        logger.debug(
            "Running extension %s for hook %s (%s convention)",
            extension, hook_name, "new" if new_convention else "legacy",
        )  # fmt: skip
        try:
            check_extensions_supported([extension])
            if new_convention:
                ret = await transform(extension, updated, {"hookName": hook_name}, False)
            else:
                ret = await transform(extension, hook_name, updated, False)
        except Exception as error:
            raise ExtensionHookError(
                f'Extension hook "{hook_name}" failed for {extension}: {error}'
            ) from error

        if not _is_truthy(ret):
            continue
        if hook_name == "beforeAll":
            current = updated["suite"]
            returned = _get(ret, "suite")
            suite = {**current}
            for key in _BEFORE_ALL_MUTABLE_KEYS:
                value = _get(returned, key)
                if key == "prompts":
                    value = restore_prompt_functions(value, current.get("prompts"))
                if isinstance(returned, dict) and key in returned:
                    suite[key] = value
                else:
                    suite.pop(key, None)  # JS: the key becomes undefined
            updated = {"suite": suite}
        elif hook_name == "beforeEach":
            updated = {"test": _get(ret, "test")}
        elif hook_name == "afterEach":
            returned_result = _get(ret, "result")
            if _is_truthy(returned_result):
                current = updated["result"]
                returned_response = _get(returned_result, "response")
                returned_response_metadata = (
                    _get(returned_response, "metadata") if returned_response is not None else None
                )
                if _is_truthy(current.get("response")) and _is_truthy(returned_response_metadata):
                    merged_response = {
                        **current["response"],
                        "metadata": {
                            **(current["response"].get("metadata") or {}),
                            **(
                                returned_response_metadata
                                if isinstance(returned_response_metadata, dict)
                                else {}
                            ),
                        },
                    }
                else:
                    merged_response = current.get("response")
                valid_scores = filter_finite_scores(_get(returned_result, "namedScores") or {})
                result = {
                    **current,
                    "namedScores": {**(current.get("namedScores") or {}), **valid_scores},
                    "metadata": {
                        **(current.get("metadata") or {}),
                        **(_get(returned_result, "metadata") or {}),
                    },
                }
                if "response" in current or merged_response is not None:
                    result["response"] = merged_response
                updated = {"test": updated["test"], "result": result}
        # afterAll runs after results are persisted and is for side effects only.
    return updated


def _is_truthy(value: Any) -> bool:
    from promptphoo.jsvalues import is_truthy

    return is_truthy(value)


async def run_before_all_extensions(test_suite: dict[str, Any]) -> dict[str, Any]:
    """Ports runBeforeAllExtensions: seed defaultTest and providerPromptMap, run beforeAll,
    and keep only the provider prompt selectors the hook actually changed."""
    if not test_suite.get("extensions"):
        return test_suite
    if not test_suite.get("defaultTest"):
        test_suite["defaultTest"] = {}
    default_test = test_suite["defaultTest"]
    if isinstance(default_test, dict) and not default_test.get("assert"):
        default_test["assert"] = []

    seeded: list[tuple[str, list[Any]]] | None = None
    if not test_suite.get("providerPromptMap"):
        prompt_map: dict[str, list[Any]] = {}
        for provider in test_suite["providers"]:
            prompts = provider.prompts
            selectors = list(
                prompts if prompts is not None else [p.get("label") for p in test_suite["prompts"]]
            )
            prompt_map[provider.id()] = selectors
            if provider.label:
                prompt_map[provider.label] = selectors
        seeded = [(key, list(selectors)) for key, selectors in prompt_map.items()]
        test_suite = {**test_suite, "providerPromptMap": prompt_map}

    out = await run_extension_hook(test_suite["extensions"], "beforeAll", {"suite": test_suite})
    suite = out["suite"]
    if seeded is not None:
        # Untouched seeds must not merge duplicate providers or exclude newly added prompts,
        # so only entries the hook changed override the providers' own prompt filters. The
        # hook's copy never shares identity with the seed (JSON round-trip), so an entry
        # equal to its seed counts as unchanged.
        returned_map = suite.get("providerPromptMap")
        overrides: dict[str, Any] = dict(returned_map) if isinstance(returned_map, dict) else {}
        for key, selectors in seeded:
            if key not in overrides:
                overrides[key] = ["*"]
            elif overrides[key] == selectors:
                del overrides[key]
        suite["providerPromptMap"] = overrides
    return suite


def synchronize_legacy_transport_headers(
    original_metadata: dict[str, Any] | None,
    original_response_metadata: dict[str, Any] | None,
    hook_metadata: dict[str, Any] | None,
    hook_response_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Drop a stale top-level `metadata.headers` copied from transport headers that an
    afterEach hook replaced in `response.metadata`."""
    original_headers = (original_metadata or {}).get("headers")
    if (
        original_headers is None
        or original_headers != (original_response_metadata or {}).get("headers")
        or (hook_metadata or {}).get("headers") != original_headers
    ):
        return hook_metadata
    without_headers = {k: v for k, v in (hook_metadata or {}).items() if k != "headers"}
    if (hook_response_metadata or {}).get("headers") is None:
        return without_headers
    return {**without_headers, "headers": hook_response_metadata["headers"]}  # type: ignore[index]
