"""Detect and parse an `import` file. Ports src/importers/parse.ts and src/importers/types.ts.

A file is a promptfoo eval export (one JSON document) or an OpenAI Evals output-items export
(one JSON row per line, or a single JSON row/array of rows).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from promptphoo.importers.openai_evals import convert_openai_evals_jsonl, is_openai_evals_jsonl_row

logger = logging.getLogger("promptphoo")

IMPORT_SOURCE_OPENAI_EVALS = "openai-evals-jsonl"
IMPORT_SOURCE_PROMPTFOO = "promptfoo-json"


class _NonJsonConstant(Exception):
    pass


def _reject_constant(name: str) -> Any:
    raise _NonJsonConstant(name)


def _v8_message(text: str, pos: int) -> str:
    """V8's JSON.parse SyntaxError wording (Node 20+), for the token at `pos`."""
    if pos >= len(text):
        return "Unexpected end of JSON input"
    snippet = text if len(text) <= 32 else f"...{text[max(pos - 10, 0) : pos + 10]}..."
    return f"Unexpected token '{text[pos]}', \"{snippet}\" is not valid JSON"


def json_parse(text: str) -> Any:
    """JSON.parse: json.loads with NaN/Infinity rejected and V8-style error messages."""
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except _NonJsonConstant as error:
        raise ValueError(_v8_message(text, text.find(str(error)))) from None
    except json.JSONDecodeError as error:
        raise ValueError(_v8_message(text, error.pos)) from None


# String.prototype.trim: WhiteSpace and LineTerminator code points, which include the BOM.
_JS_WHITESPACE = "\t\n\v\f\r " + "".join(
    map(chr, (0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF))
)


def js_trim(value: str) -> str:
    return value.strip(_JS_WHITESPACE)


def _try_convert_openai_evals(parsed: Any) -> dict[str, Any] | None:
    rows = parsed if isinstance(parsed, list) else [parsed]
    if rows and all(is_openai_evals_jsonl_row(row) for row in rows):
        return {"source": IMPORT_SOURCE_OPENAI_EVALS, "evalData": convert_openai_evals_jsonl(rows)}
    return None


def _parse_jsonl_lines(file_content: str) -> list[Any] | None:
    lines = [js_trim(line) for line in re.split(r"\r?\n", file_content)]
    lines = [line for line in lines if line]
    if not lines:
        return None
    try:
        return [json_parse(line) for line in lines]
    except ValueError as error:
        logger.debug("Failed to parse as OpenAI Evals JSONL: %s", error)
        return None


def parse_import_file(file_content: str) -> dict[str, Any]:
    """parseImportFile -> {"source": ..., "evalData": ...}."""
    try:
        parsed = json_parse(file_content)
    except ValueError:
        rows = _parse_jsonl_lines(file_content)
        if rows is not None:
            result = _try_convert_openai_evals(rows)
            if result is not None:
                return result
            invalid = next(i for i, row in enumerate(rows) if not is_openai_evals_jsonl_row(row))
            raise ValueError(
                f"File parsed as JSONL but line {invalid + 1} is not a valid OpenAI Evals row. "
                "Expected a promptfoo eval JSON export or an OpenAI Evals JSONL export."
            ) from None
        raise

    result = _try_convert_openai_evals(parsed)
    if result is not None:
        return result
    return {"source": IMPORT_SOURCE_PROMPTFOO, "evalData": parsed}
