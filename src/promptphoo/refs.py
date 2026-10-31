"""JSON `$ref` dereferencing, standing in for @apidevtools/json-schema-ref-parser.

Supports internal pointers (`#/definitions/x`) and external files (`other.yaml#/x`), resolved
relative to the referencing document (the working directory for in-memory input, as RefParser
does). The result shares no structure with the input; circular references raise.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from promptphoo.yamlload import load_yaml


class RefError(ValueError):
    pass


def _pointer_get(doc: Any, pointer: str) -> Any:
    if pointer in ("", "/"):
        return doc
    if not pointer.startswith("/"):
        raise RefError(f"Invalid JSON pointer: #{pointer}")
    node = doc
    for raw in pointer[1:].split("/"):
        token = unquote(raw).replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            try:
                node = node[int(token)]
            except (ValueError, IndexError):
                raise RefError(f"Error resolving $ref pointer: #{pointer}") from None
        elif isinstance(node, dict) and token in node:
            node = node[token]
        else:
            raise RefError(f'Error resolving $ref pointer "#{pointer}". Token "{token}" not found')
    return node


def _load_file(path: str, cache: dict[str, Any]) -> Any:
    if path not in cache:
        text = Path(path).read_text(encoding="utf-8")
        cache[path] = json.loads(text) if path.endswith(".json") else load_yaml(text)
    return cache[path]


def dereference(value: Any, base_dir: str | None = None) -> Any:
    base = os.path.abspath(base_dir or os.getcwd())
    root_key = os.path.join(base, "<root>")
    files: dict[str, Any] = {root_key: value}

    def resolve(node: Any, doc_key: str, stack: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [resolve(item, doc_key, stack) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str):
            target_file, _, pointer = ref.partition("#")
            if target_file:
                doc_key = os.path.normpath(os.path.join(os.path.dirname(doc_key), target_file))
                doc = _load_file(doc_key, files)
            else:
                doc = files[doc_key]
            ident = f"{doc_key}#{pointer}"
            if ident in stack:
                raise RefError(f"Circular $ref pointer found: {ref}")
            target = resolve(_pointer_get(doc, pointer), doc_key, (*stack, ident))
            extra = {k: v for k, v in node.items() if k != "$ref"}
            if extra and isinstance(target, dict):
                return {**target, **{k: resolve(v, doc_key, stack) for k, v in extra.items()}}
            return copy.deepcopy(target)
        return {k: resolve(v, doc_key, stack) for k, v in node.items()}

    return resolve(value, root_key, ())
