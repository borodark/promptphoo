"""CSV test-case helpers. Ports src/csv.ts and src/validation/parseCommaSeparatedValues.ts."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from promptphoo.assertions.types import BASE_ASSERTION_TYPES
from promptphoo.files import is_javascript_file

logger = logging.getLogger(__name__)

DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD = 0.8

# Longer names first: `similar` is a prefix of `similar:cosine` and the value also starts at `:`.
_TYPES_ALT = "|".join(
    re.escape(t) for t in sorted(BASE_ASSERTION_TYPES, key=len, reverse=True)
)  # stable sort keeps declaration order among equal lengths, as Array.prototype.sort does
_ASSERTION_RE = re.compile(rf"^(not-)?({_TYPES_ALT})(?:\((\d+(?:\.\d+)?)\))?(?::([\s\S]*))?$")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

_THRESHOLD_TYPES = {
    "answer-relevance", "classifier", "context-faithfulness", "context-recall",
    "context-relevance", "cost", "latency", "levenshtein", "perplexity-score", "perplexity",
    "rouge-l", "rouge-n", "rouge-s", "similar", "starts-with",
}  # fmt: skip
_LIST_TYPES = {"contains-all", "contains-any", "icontains-all", "icontains-any"}


def parse_finite_number(value: str | None) -> float | int | None:
    trimmed = (value or "").strip()
    if not _NUMBER_RE.match(trimmed):
        return None
    parsed = float(trimmed)
    if not math.isfinite(parsed):
        return None
    return int(parsed) if parsed.is_integer() and "e" not in trimmed.lower() else parsed


def parse_comma_separated_values(value: str) -> list[str]:
    results: list[str] = []
    i, n = 0, len(value)
    while i < n:
        while i < n:
            while i < n and value[i].isspace():
                i += 1
            if i >= n or value[i] != ",":
                break
            i += 1
        if i >= n:
            break
        quoted = value[i] == '"'
        field = ""
        if quoted:
            i += 1
            terminated = False
            while i < n:
                if value[i] == "\\" and i + 1 < n and value[i + 1] in ('"', "\\"):
                    field += value[i + 1]
                    i += 2
                elif value[i] == '"' and i + 1 < n and value[i + 1] == '"':
                    field += '"'
                    i += 2
                elif value[i] == '"':
                    i += 1
                    terminated = True
                    break
                else:
                    field += value[i]
                    i += 1
            if not terminated:
                raise ValueError(
                    "Invariant failed: Unterminated quoted field in contains assertion value"
                )
        else:
            comma = value.find(",", i)
            end = n if comma == -1 else comma
            field = value[i:end].strip()
            i = end
        results.append(field)
        while i < n and value[i].isspace():
            i += 1
        if quoted and i < n and value[i] != ",":
            raise ValueError(
                "Invariant failed: Expected comma after quoted field in contains assertion value"
            )
    return results


def assertion_from_string(expected: str) -> dict[str, Any]:
    for prefix in ("javascript:", "fn:", "eval:"):
        if expected.startswith(prefix):
            return {"type": "javascript", "value": expected[len(prefix) :].strip()}
    if expected.startswith("file://") and is_javascript_file(expected[len("file://") :]):
        return {"type": "javascript", "value": expected.strip()}
    if expected.startswith("grade:"):
        return {"type": "llm-rubric", "value": expected[6:]}
    if expected.startswith("llm-rubric:"):
        return {"type": "llm-rubric", "value": expected[11:]}
    if expected.startswith("python:") or (
        expected.startswith("file://") and (expected.endswith(".py") or ".py:" in expected)
    ):
        cut = len("python:") if expected.startswith("python:") else len("file://")
        return {"type": "python", "value": expected[cut:].strip()}

    m = _ASSERTION_RE.match(expected)
    if m:
        not_prefix, base_type, threshold_str, value = m.groups()
        full_type = f"not-{base_type}" if not_prefix else base_type
        threshold = parse_finite_number(threshold_str)
        if base_type in _LIST_TYPES:
            return {
                "type": full_type,
                **({"value": parse_comma_separated_values(value) if value else value}
                   if value is not None else {}),
            }  # fmt: skip
        if base_type in ("contains-json", "is-json"):
            return {"type": full_type, **({"value": value} if value is not None else {})}
        trimmed = {"value": value.strip()} if value is not None else {}
        if base_type in _THRESHOLD_TYPES:
            default = DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD if base_type == "similar" else 0.75
            return {
                "type": full_type,
                **trimmed,
                "threshold": default if threshold is None else threshold,
            }
        return {
            "type": full_type,
            **trimmed,
            **({} if threshold is None else {"threshold": threshold}),
        }
    return {"type": "equals", "value": expected}


_SPECIAL_KEYS = tuple(
    f"_{k}"
    for k in (
        "expected", "prefix", "suffix", "description", "providerOutput", "metric", "threshold",
        "metadata", "config",
    )
)  # fmt: skip
_warned: set[str] = set()


def test_case_from_csv_row(row: dict[str, str]) -> dict[str, Any]:
    vars_: dict[str, str] = {}
    asserts: list[dict[str, Any]] = []
    options: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    assertion_configs: dict[int, dict[str, Any]] = {}
    provider_output: str | None = None
    description: str | None = None
    metric: str | None = None
    threshold: float | int | None = None

    for raw_key, value in row.items():
        key = raw_key.strip()
        if not key.startswith("__") and key.startswith(_SPECIAL_KEYS) and key not in _warned:
            _warned.add(key)
            logger.warning(
                'You used a single underscore for the key "%s". Did you mean to use "%s" instead?',
                key,
                key.replace("_", "__", 1),
            )
        if key.startswith("__expected"):
            if value.strip():
                asserts.append(assertion_from_string(value.strip()))
        elif key == "__prefix":
            options["prefix"] = value
        elif key == "__suffix":
            options["suffix"] = value
        elif key == "__description":
            description = value
        elif key == "__providerOutput":
            provider_output = value
        elif key == "__metric":
            metric = value
        elif key == "__threshold":
            threshold = parse_finite_number(value)
        elif key.startswith("__metadata:"):
            meta_key = key[len("__metadata:") :]
            if meta_key.endswith("[]"):
                if value.strip():
                    parts = re.split(r"(?<!\\),", value)
                    metadata[meta_key[:-2]] = [p.strip().replace("\\,", ",") for p in parts]
            elif value.strip():
                metadata[meta_key] = value
        elif key == "__metadata":
            if key not in _warned:
                _warned.add(key)
                logger.warning(
                    'The "__metadata" column requires a key, e.g. "__metadata:category". '
                    "This column will be ignored."
                )
        elif key.startswith("__config:"):
            parts = key[len("__config:") :].split(":")
            if len(parts) != 2:
                logger.warning(
                    'Invalid __config column format: "%s". Expected format: '
                    "__config:__expected:threshold or __config:__expected<N>:threshold",
                    key,
                )
                continue
            expected_key, config_key = parts
            target: int | None = None
            if expected_key == "__expected":
                target = 0
            else:
                im = re.match(r"^__expected(\d+)$", expected_key)
                if im and int(im.group(1)) >= 1:
                    target = int(im.group(1)) - 1
            if target is None:
                raise ValueError(f'Invalid expected key "{expected_key}" in __config column')
            if config_key != "threshold":
                raise ValueError(f'Invalid config key "{config_key}" in __config column')
            parsed = parse_finite_number(value)
            if parsed is None:
                raise ValueError(f"Invalid numeric value for {config_key}")
            assertion_configs.setdefault(target, {})[config_key] = parsed
        else:
            vars_[key] = value

    for i, assertion in enumerate(asserts):
        if metric is not None:
            assertion["metric"] = metric
        for config_key, config_value in assertion_configs.get(i, {}).items():
            assertion[config_key] = config_value
            metadata[config_key] = config_value

    return {
        "vars": vars_,
        "assert": asserts,
        "options": options,
        **({"description": description} if description else {}),
        **({"providerOutput": provider_output} if provider_output else {}),
        **({} if threshold is None else {"threshold": threshold}),
        **({"metadata": metadata} if metadata else {}),
    }
