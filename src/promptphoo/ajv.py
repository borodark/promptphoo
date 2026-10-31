"""JSON-schema validation reported the way Ajv does (getAjv() in src/util/json.ts).

Ajv (draft-07, allErrors: false) stops at the first failing keyword and its errorsText() reads
"data/path message[, ...]". jsonschema computes every error, so we pick the one Ajv would hit
first using Ajv's keyword evaluation order, and render Ajv's messages.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

import jsonschema
from jsonschema import Draft7Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from promptphoo.jsonutil import js_deep_equal
from promptphoo.jsvalues import json_stringify, number_to_string

# Ajv v8 draft-07 rule order: type, then untyped keywords, then the per-type groups.
_ORDER = [
    "$ref", "type",
    "const", "enum", "not", "anyOf", "oneOf", "allOf", "if", "then", "else",
    "maximum", "minimum", "exclusiveMaximum", "exclusiveMinimum", "multipleOf",
    "maxLength", "minLength", "pattern", "format",
    "maxItems", "minItems", "uniqueItems", "additionalItems", "items", "contains",
    "maxProperties", "minProperties", "required", "dependencies", "propertyNames",
    "additionalProperties", "properties", "patternProperties",
]  # fmt: skip
_RANK = {k: i for i, k in enumerate(_ORDER)}


class _Validator:
    def __init__(self, schema: Any) -> None:
        self.schema = schema
        self._validator = Draft7Validator(schema, format_checker=FormatChecker())
        self.errors: list[dict[str, Any]] | None = None

    def __call__(self, data: Any) -> bool:
        errors = list(self._validator.iter_errors(data))
        if not errors:
            self.errors = None
            return True
        first = min(errors, key=_sort_key)
        self.errors = _ajv_errors(first)
        return False


def _sort_key(error: ValidationError) -> tuple:
    key: list[Any] = []
    for part in error.relative_schema_path:
        if isinstance(part, int):
            key.append((0, part))
        elif part in _RANK:
            key.append((1, _RANK[part]))
        else:
            key.append((2, 0))  # property names keep jsonschema's (schema) order
    return tuple(key)


def _instance_path(error: ValidationError) -> str:
    return "".join(f"/{str(p).replace('~', '~0').replace('/', '~1')}" for p in error.absolute_path)


def _message(error: ValidationError) -> str:
    v, value = cast(str, error.validator), cast(Any, error.validator_value)
    if v == "type":
        return f"must be {','.join(value) if isinstance(value, list) else value}"
    if v == "required":
        missing = error.message.split("'")[1] if "'" in error.message else ""
        return f"must have required property '{missing}'"
    if v == "additionalProperties":
        return "must NOT have additional properties"
    if v == "enum":
        return "must be equal to one of the allowed values"
    if v == "const":
        return "must be equal to constant"
    ops = {"maximum": "<=", "minimum": ">=", "exclusiveMaximum": "<", "exclusiveMinimum": ">"}
    if v in ops:
        return f"must be {ops[v]} {number_to_string(value)}"
    if v in ("maxLength", "minLength", "maxItems", "minItems", "maxProperties", "minProperties"):
        more = "more" if v.startswith("max") else "fewer"
        unit = {"Length": "characters", "Items": "items", "Properties": "properties"}[v[3:]]
        return f"must NOT have {more} than {value} {unit}"
    if v == "pattern":
        return f'must match pattern "{value}"'
    if v == "format":
        return f'must match format "{value}"'
    if v == "multipleOf":
        return f"must be multiple of {number_to_string(value)}"
    if v == "uniqueItems":
        i, j = _duplicate_indices(error)
        return f"must NOT have duplicate items (items ## {j} and {i} are identical)"
    if v == "anyOf":
        return "must match a schema in anyOf"
    if v == "oneOf":
        return "must match exactly one schema in oneOf"
    if v == "not":
        return "must NOT be valid"
    if v == "contains":
        return "must contain at least 1 valid item(s)"
    if v == "dependencies":
        quoted = error.message.split("'")  # "'bar' is a dependency of 'foo'"
        if len(quoted) >= 4:
            return f"must have property {quoted[1]} when property {quoted[3]} is present"
    if v is False or error.validator == "false":
        return "boolean schema is false"
    return error.message


_SCALAR_TYPES = {"string", "number", "integer", "boolean", "null"}


def _duplicate_indices(error: ValidationError) -> tuple[int, int]:
    """Ajv's uniqueItems scan: a value map for scalar `items.type`, else a pairwise scan."""
    data = error.instance if isinstance(error.instance, list) else []
    items = error.schema.get("items") if isinstance(error.schema, dict) else None
    item_type = items.get("type") if isinstance(items, dict) else None
    types = set(item_type) if isinstance(item_type, list) else {item_type}
    if item_type is not None and types <= _SCALAR_TYPES:
        seen: dict[str, int] = {}
        for i in range(len(data) - 1, -1, -1):
            key = json_stringify(data[i])
            if key in seen:
                return i, seen[key]
            seen[key] = i
    else:
        for i in range(len(data) - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                if js_deep_equal(data[i], data[j]):
                    return i, j
    return 0, 0


def _schema_path(error: ValidationError) -> str:
    return "#/" + "/".join(
        str(p).replace("~", "~0").replace("/", "~1") for p in error.absolute_schema_path
    )


def _params(error: ValidationError) -> dict[str, Any]:
    v, value = cast(str, error.validator), cast(Any, error.validator_value)
    if v == "type":
        return {"type": value}
    if v == "required":
        return {"missingProperty": error.message.split("'")[1] if "'" in error.message else ""}
    if v == "additionalProperties":
        extra = error.message.split("'")
        return {"additionalProperty": extra[1] if len(extra) > 1 else ""}
    if v == "enum":
        return {"allowedValues": value}
    if v == "const":
        return {"allowedValue": value}
    ops = {"maximum": "<=", "minimum": ">=", "exclusiveMaximum": "<", "exclusiveMinimum": ">"}
    if v in ops:
        return {"comparison": ops[v], "limit": value}
    if v in ("maxLength", "minLength", "maxItems", "minItems", "maxProperties", "minProperties"):
        return {"limit": value}
    if v == "pattern":
        return {"pattern": value}
    if v == "format":
        return {"format": value}
    if v == "multipleOf":
        return {"multipleOf": value}
    if v == "uniqueItems":
        i, j = _duplicate_indices(error)
        return {"i": i, "j": j}
    if v == "oneOf":
        return {"passingSchemas": None}
    if v == "contains":
        return {"minContains": 1}
    if v == "dependencies":
        quoted = error.message.split("'")
        if len(quoted) >= 4:
            deps = (value or {}).get(quoted[3]) or []
            return {"property": quoted[3], "missingProperty": quoted[1],
                    "depsCount": len(deps), "deps": ", ".join(deps)}  # fmt: skip
    return {}


def _ajv_errors(error: ValidationError) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if error.validator in ("anyOf", "oneOf") and error.context:
        # Ajv keeps the first failure of each branch before the anyOf/oneOf error itself.
        by_branch: dict[Any, ValidationError] = {}
        for sub in error.context:
            branch = sub.schema_path[0] if sub.schema_path else 0
            if branch not in by_branch or _sort_key(sub) < _sort_key(by_branch[branch]):
                by_branch[branch] = sub
        for branch in sorted(by_branch):
            out.extend(_ajv_errors(by_branch[branch]))
    out.append(
        {"instancePath": _instance_path(error), "schemaPath": _schema_path(error),
         "keyword": error.validator, "params": _params(error), "message": _message(error)}
    )  # fmt: skip
    return out


@lru_cache(maxsize=256)
def _compile_cached(schema_json: str) -> _Validator:
    import json

    return _compile(json.loads(schema_json))


def _compile(schema: Any) -> _Validator:
    try:
        Draft7Validator.check_schema(schema)
    except jsonschema.SchemaError as e:
        raise ValueError(f"schema is invalid: {e.message}") from e
    return _Validator(schema)


def compile_schema(schema: Any) -> _Validator:
    """ajv.compile(schema): returns a callable validator with `.errors` after a failed call."""
    try:
        return _compile_cached(json_stringify(schema))
    except TypeError:
        return _compile(schema)


def errors_text(errors: list[dict[str, Any]] | None) -> str:
    if not errors:
        return "No errors"
    return ", ".join(f"data{e['instancePath']} {e['message']}" for e in errors)
