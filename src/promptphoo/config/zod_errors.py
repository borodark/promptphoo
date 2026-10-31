"""Render JSON-schema validation errors the way `z.prettifyError` renders zod issues.

Upstream validates configs with zod (UnifiedConfigSchema) and prints `z.prettifyError(...)`;
promptphoo validates with the vendored `_data/config-schema.json`, which upstream generates
from those zod schemas. This maps each jsonschema error to the zod issue the same input would
raise (messages from zod's English locale, paths from `toDotPath`) and sorts them by path
length, as prettifyError does. It is an approximation where the two validators diverge, e.g.
issue order inside one object and `.refine` checks that the JSON schema cannot express.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema.exceptions import ValidationError

from promptphoo.jsvalues import number_to_string

# Validators whose zod counterparts are non-aborting checks (`continue: true`).
_CHECKS = {
    "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "pattern", "format", "multipleOf",
}  # fmt: skip

_FORMATS = {
    "email": "email address",
    "uri": "URL",
    "url": "URL",
    "uuid": "UUID",
    "date-time": "ISO datetime",
    "date": "ISO date",
    "time": "ISO time",
    "duration": "ISO duration",
    "ipv4": "IPv4 address",
    "ipv6": "IPv6 address",
    "regex": "input",
}


@dataclass
class Issue:
    path: list[str | int]
    message: str
    aborts: bool


def parsed_type(value: Any) -> str:
    """zod's util.parsedType."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "nan" if value != value else "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _type_name(type_: str, value: Any = None) -> str:
    if type_ == "number" and isinstance(value, float) and value in (float("inf"), float("-inf")):
        return "Infinity" if value > 0 else "-Infinity"
    return "NaN" if type_ == "nan" else type_


def _stringify_primitive(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return number_to_string(value)
    return str(value)


def _join_values(values: list[Any], sep: str) -> str:
    return sep.join(_stringify_primitive(v) for v in values)


def to_dot_path(path: list[str | int]) -> str:
    segs: list[str] = []
    for seg in path:
        if isinstance(seg, int):
            segs.append(f"[{seg}]")
        elif re.search(r"[^\w$]", seg, re.ASCII):
            segs.append(f"[{json.dumps(seg, ensure_ascii=False)}]")
        else:
            if segs:
                segs.append(".")
            segs.append(seg)
    return "".join(segs)


def _invalid_type(expected: Any, value: Any, missing: bool = False) -> str:
    received = "undefined" if missing else _type_name(parsed_type(value), value)
    if isinstance(expected, list):
        return "Invalid input"
    if expected == "integer":
        is_number = isinstance(value, int | float) and not isinstance(value, bool)
        expected = "int" if is_number and not missing else "number"
    return f"Invalid input: expected {_type_name(expected)}, received {received}"


def _values_message(values: list[Any]) -> str:
    if len(values) == 1:
        return f"Invalid input: expected {_stringify_primitive(values[0])}"
    return f"Invalid option: expected one of {_join_values(values, '|')}"


def _missing_message(schema: Any) -> str:
    """The zod issue for `undefined` where `schema` is required."""
    if not isinstance(schema, dict):
        return "Invalid input"
    if "const" in schema:
        return _values_message([schema["const"]])
    if "enum" in schema:
        return _values_message(list(schema["enum"]))
    if isinstance(schema.get("type"), str):
        return _invalid_type(schema["type"], None, missing=True)
    return "Invalid input"


def _resolve(root: Any, schema: Any) -> Any:
    """Follow local `#/...` refs against the root schema."""
    for _ in range(32):
        ref = schema.get("$ref") if isinstance(schema, dict) else None
        if not isinstance(ref, str) or not ref.startswith("#"):
            return schema
        target: Any = root
        for part in ref[1:].split("/")[1:]:
            target = target.get(part.replace("~1", "/").replace("~0", "~"), {})
        schema = target
    return schema


def _size(error: ValidationError, small: bool, inclusive: bool) -> str:
    name = str(error.validator)
    origin = "array" if "Items" in name else "string" if "Length" in name else "number"
    bound = number_to_string(error.validator_value)  # type: ignore[arg-type]
    adj = (">=" if inclusive else ">") if small else ("<=" if inclusive else "<")
    word = "Too small" if small else "Too big"
    if origin == "array":
        return f"{word}: expected array to have {adj}{bound} items"
    if origin == "string":
        return f"{word}: expected string to have {adj}{bound} characters"
    return f"{word}: expected number to be {adj}{bound}"


def issues_from_error(error: ValidationError, root: Any = None) -> list[Issue]:
    path = list(error.absolute_path)
    validator = str(error.validator)
    value = error.validator_value
    instance = error.instance
    if validator == "type":
        return [Issue(path, _invalid_type(value, instance), True)]
    if validator == "enum":
        return [Issue(path, _values_message(list(value)), True)]  # type: ignore[arg-type]
    if validator == "const":
        return [Issue(path, _values_message([value]), True)]
    if validator == "required":
        properties = error.schema.get("properties", {}) if isinstance(error.schema, dict) else {}
        out = []
        for key in value:  # type: ignore[union-attr]
            if isinstance(instance, dict) and key not in instance:
                schema = _resolve(root, properties.get(key))
                out.append(Issue([*path, key], _missing_message(schema), True))
        return out
    if validator == "additionalProperties" and value is False:
        schema = error.schema if isinstance(error.schema, dict) else {}
        known = set(schema.get("properties", {}))
        patterns = list(schema.get("patternProperties", {}))
        extras = [
            k
            for k in (instance if isinstance(instance, dict) else {})
            if k not in known and not any(re.search(p, k) for p in patterns)
        ]
        plural = "s" if len(extras) > 1 else ""
        return [Issue(path, f"Unrecognized key{plural}: {_join_values(extras, ', ')}", True)]
    if validator in ("anyOf", "oneOf"):
        branches: dict[int, list[ValidationError]] = {}
        for sub in error.context or []:
            branches.setdefault(int(sub.relative_schema_path[0]), []).append(sub)
        nonaborted = [
            subs for subs in branches.values() if all(str(s.validator) in _CHECKS for s in subs)
        ]
        if len(nonaborted) == 1 and len(branches) == len(error.validator_value):  # type: ignore[arg-type]
            return [i for sub in nonaborted[0] for i in issues_from_error(sub, root)]
        return [Issue(path, "Invalid input", True)]
    if validator == "minItems" or validator == "minLength" or validator == "minimum":
        return [Issue(path, _size(error, True, True), False)]
    if validator == "exclusiveMinimum":
        return [Issue(path, _size(error, True, False), False)]
    if validator in ("maxItems", "maxLength", "maximum"):
        return [Issue(path, _size(error, False, True), False)]
    if validator == "exclusiveMaximum":
        return [Issue(path, _size(error, False, False), False)]
    if validator == "pattern":
        return [Issue(path, f"Invalid string: must match pattern /{value}/", False)]
    if validator == "format":
        return [Issue(path, f"Invalid {_FORMATS.get(str(value), str(value))}", False)]
    if validator == "multipleOf":
        return [
            Issue(path, f"Invalid number: must be a multiple of {number_to_string(value)}", False)  # type: ignore[arg-type]
        ]
    return [Issue(path, "Invalid input", True)]


def collect_issues(errors: list[ValidationError], root: Any = None) -> list[Issue]:
    issues: list[Issue] = []
    seen_required: set[tuple[int, str]] = set()
    for error in errors:
        if str(error.validator) == "required":
            # jsonschema reports one error per missing key; render each key once.
            fresh = []
            for issue in issues_from_error(error, root):
                key = (id(error.instance), str(issue.path[-1]))
                if key not in seen_required:
                    seen_required.add(key)
                    fresh.append(issue)
            issues.extend(fresh)
            continue
        issues.extend(issues_from_error(error, root))
    return issues


def prettify_issues(issues: list[Issue]) -> str:
    lines: list[str] = []
    for issue in sorted(issues, key=lambda i: len(i.path)):
        lines.append(f"✖ {issue.message}")
        if issue.path:
            lines.append(f"  → at {to_dot_path(issue.path)}")
    return "\n".join(lines)
