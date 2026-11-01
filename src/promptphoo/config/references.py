"""Load-time checks that tests name providers and prompts that exist. Ports
src/util/validateTestProviderReferences.ts and src/util/validateTestPromptReferences.ts.

A bad prompt reference is reported with a best-effort `path:line:column` into the YAML config,
found by scanning the text without a YAML parser, exactly as upstream does.
"""

from __future__ import annotations

import re
from typing import Any

from promptphoo.jsvalues import is_truthy


class ProviderReferenceValidationError(Exception):
    pass


class PromptReferenceValidationError(Exception):
    pass


# --- providers --------------------------------------------------------------------------------


def _validate_provider_ref(ref: str, providers: list[Any], context: str) -> None:
    from promptphoo.evaluator.run import does_provider_ref_match, get_provider_description

    if not any(does_provider_ref_match(ref, p) for p in providers):
        available = ", ".join(get_provider_description(p) for p in providers)
        raise ProviderReferenceValidationError(
            f'{context} references provider "{ref}" which does not exist. '
            f"Available providers: {available}"
        )


def _validate_test_providers(test: dict[str, Any], providers: list[Any], context: str) -> None:
    refs = test.get("providers")
    if not is_truthy(refs):
        return
    if not isinstance(refs, list):
        raise ProviderReferenceValidationError(f"{context}.providers must be an array")
    desc = f' ("{test["description"]}")' if is_truthy(test.get("description")) else ""
    for ref in refs:
        _validate_provider_ref(ref, providers, f"{context}{desc}")


def validate_test_provider_references(
    tests: list[dict[str, Any]],
    providers: list[Any],
    default_test: dict[str, Any] | None = None,
    scenarios: list[dict[str, Any]] | None = None,
) -> None:
    if default_test:
        _validate_test_providers(default_test, providers, "defaultTest")
    for i, test in enumerate(tests):
        _validate_test_providers(test, providers, f"Test #{i + 1}")
    for i, scenario in enumerate(scenarios or []):
        desc = scenario.get("description")
        scenario_desc = f' ("{desc}")' if is_truthy(desc) else ""
        for j, item in enumerate(scenario.get("config") or []):
            _validate_test_providers(
                item, providers, f"Scenario #{i + 1}{scenario_desc} config[{j}]"
            )
        for j, test in enumerate(scenario.get("tests") or []):
            _validate_test_providers(
                test, providers, f"Scenario #{i + 1}{scenario_desc} test #{j + 1}"
            )


# --- prompts ----------------------------------------------------------------------------------


def _is_yaml_key(trimmed_line: str, key: str) -> bool:
    return trimmed_line == f"{key}:" or trimmed_line.startswith(f"{key}: ")


def _unquote(value: str) -> str:
    if value and value[0] in "\"'" and len(value) >= 2 and value.endswith(value[0]):
        return value[1:-1]
    return value


def _ref_location(path: str, line: str, line_number: int, ref: str) -> str | None:
    column = line.find(ref)
    if column == -1:
        return None
    # String.indexOf counts UTF-16 code units.
    column = len(line[:column].encode("utf-16-le")) // 2
    return f"{path}:{line_number}:{column + 1}"


def _collect_section_items(
    source: dict[str, str], ref: str, section: str
) -> list[dict[str, str | None]]:
    """Split a source's `tests:` or `defaultTest:` section into list items, recording each
    item's description and the first line inside its `prompts:` block that contains `ref`.
    An exact list-item match is kept apart from a substring match, so that `- GhostBuster`
    cannot take the location of a missing `Ghost`."""
    lines = re.split(r"\r?\n", source["content"])
    in_section = False
    item_indent: int | None = None
    prompts_indent: int | None = None
    items: list[dict[str, str | None]] = [{}]

    for i, line in enumerate(lines):
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        is_list_item = trimmed == "-" or trimmed.startswith("- ")

        if not in_section:
            if indent == 0 and _is_yaml_key(trimmed, section):
                in_section = True
                items[0]["location"] = _ref_location(source["path"], line, i + 1, ref)
            continue
        if indent == 0 and not is_list_item:
            break

        if is_list_item:
            if item_indent is None:
                item_indent = indent
            if indent <= item_indent:
                items.append({})

        item = items[-1]
        stripped = re.sub(r"^-\s*", "", trimmed)
        if _is_yaml_key(stripped, "description"):
            if item.get("description") is None:
                item["description"] = _unquote(stripped[len("description:") :].strip())
            continue
        # Only search inside the item's `prompts:` block, so the same string in vars or
        # assertions does not take the location.
        if _is_yaml_key(stripped, "prompts"):
            prompts_indent = indent + (len(trimmed) - len(stripped))
        elif prompts_indent is None or indent <= prompts_indent:
            prompts_indent = None
            continue
        location = _ref_location(source["path"], line, i + 1, ref)
        if location:
            key = "location" if is_list_item and _unquote(stripped) == ref else "loose"
            if item.get(key) is None:
                item[key] = location

    return [
        {"description": it.get("description"), "location": it.get("location") or it.get("loose")}
        for it in items
    ]


def _find_prompt_reference_location(
    ref: str, sources: list[dict[str, str]] | None, section: str, description: Any = None
) -> str | None:
    """Prefer the item whose description matches the failing test's, else the first
    occurrence in the right section."""
    target = description if section == "tests" else None
    fallback = None
    for source in sources or []:
        with_ref = [it for it in _collect_section_items(source, ref, section) if it["location"]]
        if target is None:
            match = with_ref[0] if with_ref else None
        else:
            match = next((it for it in with_ref if it["description"] == target), None)
        if match:
            return match["location"]
        if fallback is None and with_ref:
            fallback = with_ref[0]["location"]
    return fallback


def validate_test_prompt_references(
    tests: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    default_test: dict[str, Any] | None = None,
    prompt_reference_sources: list[dict[str, str]] | None = None,
) -> None:
    from promptphoo.evaluator.run import does_prompt_ref_match

    def check(refs: Any, label: str, section: str, description: Any = None) -> None:
        if not isinstance(refs, list):
            return
        for ref in refs:
            if any(does_prompt_ref_match(ref, p) for p in prompts):
                continue
            names = {v for p in prompts for v in (p.get("label"), p.get("id")) if v}
            message = (
                f'{label} references prompt "{ref}" which does not exist.\n'
                f"Available prompts: [{', '.join(sorted(names))}]"
            )
            location = _find_prompt_reference_location(
                ref, prompt_reference_sources, section, description
            )
            raise PromptReferenceValidationError(f"{location}: {message}" if location else message)

    check((default_test or {}).get("prompts"), "defaultTest", "defaultTest")
    for i, test in enumerate(tests):
        desc = test.get("description")
        test_desc = f' ("{desc}")' if is_truthy(desc) else ""
        check(test.get("prompts"), f"Test #{i + 1}{test_desc}", "tests", desc)
