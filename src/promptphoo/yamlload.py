"""YAML loading with js-yaml semantics. Ports src/util/yamlLoad.ts.

promptfoo parses YAML with js-yaml's CORE schema (YAML 1.2) plus merge keys, binary, timestamp,
omap, pairs and set. PyYAML defaults to YAML 1.1, where `yes`/`no`/`on`/`off` are booleans and
`012` is octal; configs would silently change meaning, so the implicit resolvers are rebuilt
here to match the 1.2 core schema.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

import yaml


class _CoreLoader(yaml.SafeLoader):
    pass


_CoreLoader.yaml_implicit_resolvers = {}


def _add(tag: str, pattern: str, first: str | None) -> None:
    _CoreLoader.add_implicit_resolver(
        tag, re.compile(pattern, re.X), list(first) if first else None
    )


_CoreLoader.add_implicit_resolver(
    "tag:yaml.org,2002:null", re.compile(r"^(?:~|null|Null|NULL|)$"), ["~", "n", "N", ""]
)
_add("tag:yaml.org,2002:bool", r"^(?:true|True|TRUE|false|False|FALSE)$", "tTfF")
_add(
    "tag:yaml.org,2002:int",
    r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$",
    "-+0123456789",
)
_add(
    "tag:yaml.org,2002:float",
    r"""^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?
         |[-+]?\.(?:inf|Inf|INF)
         |\.(?:nan|NaN|NAN))$""",
    "-+0123456789.",
)
_add("tag:yaml.org,2002:merge", r"^(?:<<)$", "<")
_add(
    "tag:yaml.org,2002:timestamp",
    r"""^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]
         |[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?
          (?:[Tt]|[ \t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\.[0-9]*)?
          (?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$""",
    "0123456789",
)


def _construct_int(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> int:
    value = str(loader.construct_scalar(node))
    sign = -1 if value.startswith("-") else 1
    digits = value.lstrip("+-")
    if digits.startswith("0o"):
        return sign * int(digits[2:], 8)
    if digits.startswith("0x"):
        return sign * int(digits[2:], 16)
    return sign * int(digits, 10)


def _construct_float(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> float | int:
    value = str(loader.construct_scalar(node)).lower()
    if value.endswith(".inf"):
        return float("-inf") if value.startswith("-") else float("inf")
    if value == ".nan":
        return float("nan")
    return float(value)


def _construct_timestamp(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> str:
    """js-yaml yields a Date; downstream it only ever surfaces as JSON (ISO, UTC, millis)."""
    value = yaml.SafeLoader.construct_yaml_timestamp(loader, node)
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.UTC).replace(tzinfo=None)
    else:
        value = dt.datetime(value.year, value.month, value.day)
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"


def _construct_set(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[Any, None]:
    """js-yaml v4 legacy: a set is an object whose keys map to null."""
    return dict.fromkeys(loader.construct_mapping(node))


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False):
    """Reject duplicate explicit keys, as js-yaml does; keys from `<<` merges may be overridden."""
    seen: set[Any] = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=True)
        if isinstance(key, list | dict):
            continue
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicated mapping key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_CoreLoader.add_constructor("tag:yaml.org,2002:int", _construct_int)
_CoreLoader.add_constructor("tag:yaml.org,2002:float", _construct_float)
_CoreLoader.add_constructor("tag:yaml.org,2002:timestamp", _construct_timestamp)
_CoreLoader.add_constructor("tag:yaml.org,2002:set", _construct_set)


def _construct_yaml_map(loader: yaml.SafeLoader, node: yaml.MappingNode):
    data: dict[Any, Any] = {}
    yield data
    data.update(_construct_mapping(loader, node))


_CoreLoader.add_constructor("tag:yaml.org,2002:map", _construct_yaml_map)


def load_yaml(content: str) -> Any:
    """Load one YAML document. Empty input returns None (JS undefined); >1 document raises."""
    docs = list(yaml.load_all(content, Loader=_CoreLoader))  # noqa: S506 - SafeLoader subclass
    if not docs:
        return None
    if len(docs) > 1:
        raise yaml.YAMLError("expected a single document in the stream, but found more")
    return _stringify_keys(docs[0])


def _stringify_keys(value: Any) -> Any:
    """JS object keys are strings: `1: a` loads as {"1": "a"}, `true: x` as {"true": "x"}."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str):
                key = k
            else:
                from promptphoo.jsvalues import js_string

                key = js_string(k)
            out[key] = _stringify_keys(v)
        return out
    if isinstance(value, list):
        return [_stringify_keys(v) for v in value]
    return value


def dump_yaml(value: Any) -> str:
    """Dump like js-yaml's default dump closely enough for output files."""
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True, default_flow_style=False)
