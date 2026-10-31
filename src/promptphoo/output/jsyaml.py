"""`yaml.dump` from js-yaml 5.5.0 with its default options.

promptfoo writes `.yaml`/`.yml`/`.txt` outputs with `yaml.dump(outputData)`. js-yaml 5 builds an
AST from the value with its DUMP_SCHEMA (YAML 1.1 plus Core int/float forms, so strings that any
of them would resolve are quoted) and then presents it: two-space indent, block collections,
single quotes preferred, literal blocks for multiline strings, folded blocks for lines longer
than 80 columns, `&ref_N`/`*ref_N` anchors for repeated objects.

Not reproduced: lengths are measured in code points rather than UTF-16 units (only differs for
astral characters in very long lines), and non-JSON JS values (Date, Map, Set, binary) have no
Python counterpart here.
"""

from __future__ import annotations

import math
import re
from typing import Any

from promptphoo.jsvalues import UNDEFINED, js_string, number_to_string

PLAIN, SINGLE_QUOTED, DOUBLE_QUOTED, LITERAL_BLOCK, FOLDED_BLOCK = 1, 2, 3, 4, 5

STR_TAG = "tag:yaml.org,2002:str"
NULL_TAG = "tag:yaml.org,2002:null"
BOOL_TAG = "tag:yaml.org,2002:bool"
INT_TAG = "tag:yaml.org,2002:int"
FLOAT_TAG = "tag:yaml.org,2002:float"
TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp"
MERGE_TAG = "tag:yaml.org,2002:merge"

INDENT = 2
LINE_WIDTH = 80


class YAMLException(Exception):
    pass


# --- implicit scalar resolution (DUMP_SCHEMA) ----------------------------------------------

_NULL_VALUES = {"", "~", "null", "Null", "NULL"}
_BOOL_VALUES = {
    "true", "True", "TRUE", "y", "Y", "yes", "Yes", "YES", "on", "On", "ON",
    "false", "False", "FALSE", "n", "N", "no", "No", "NO", "off", "Off", "OFF",
}  # fmt: skip
_INT11 = re.compile(
    r"^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?0x[0-9a-fA-F_]+|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+"
    r"|[-+]?(?:0|[1-9][0-9_]*))$"
)
_INT_CORE = re.compile(r"^(?:0o[0-7]+|0x[0-9a-fA-F]+|[-+]?[0-9]+)$")
_FLOAT11 = re.compile(
    r"^(?:[-+]?(?:(?:[0-9][0-9_]*)?\.[0-9_]*)(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+"
    r"\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
)
_FLOAT_CORE = re.compile(
    r"^(?:[-+]?[0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?|[-+]?\.[0-9]+(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
)
_FLOAT_SPECIAL = re.compile(r"^(?:[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$")
_DATE = re.compile(r"^([0-9]{4})-([0-9]{2})-([0-9]{2})$")
_TIMESTAMP = re.compile(
    r"^([0-9]{4})-([0-9][0-9]?)-([0-9][0-9]?)(?:[Tt]|[ \t]+)([0-9][0-9]?):([0-9]{2}):([0-9]{2})"
    r"(?:\.([0-9]*))?(?:[ \t]*(Z|([-+])([0-9][0-9]?)(?::([0-9]{2}))?))?$"
)


def _js_parse_float(text: str) -> float:
    m = re.match(r"^[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)", text)
    if not m:
        return math.nan
    try:
        return float(m.group(0))
    except ValueError:
        return math.nan


def _finite_parse(text: str, base: int) -> bool:
    digits = {2: "01", 8: "01234567", 16: "0123456789abcdefABCDEF", 10: "0123456789"}[base]
    prefix = ""
    for ch in text:
        if ch not in digits:
            break
        prefix += ch
    return prefix != ""


def _int11(source: str) -> bool:
    if not _INT11.match(source):
        return False
    value = source.replace("_", "")
    if value[0] in "+-":
        value = value[1:]
    if value.startswith("0b"):
        return _finite_parse(value[2:], 2)
    if value.startswith("0x"):
        return _finite_parse(value[2:], 16)
    if ":" in value:
        return True
    if value != "0" and value[0] == "0":
        return _finite_parse(value, 8)
    return _finite_parse(value, 10)


def _int_core(source: str) -> bool:
    if not _INT_CORE.match(source):
        return False
    value = source[1:] if source[0] in "+-" else source
    if value.startswith("0o"):
        return _finite_parse(value[2:], 8)
    if value.startswith("0x"):
        return _finite_parse(value[2:], 16)
    return math.isfinite(float(value))


def _float11(source: str) -> bool:
    if not _FLOAT11.match(source):
        return False
    value = source.lower().replace("_", "")
    if value[0] in "+-":
        value = value[1:]
    if value in (".inf", ".nan"):
        return True
    if ":" in value:
        try:
            result = 0.0
            for part in value.split(":"):
                result = result * 60 + float(part)
        except ValueError:
            return False
    else:
        result = _js_parse_float(value)
    return math.isfinite(result) or bool(_FLOAT_SPECIAL.match(source))


def _float_core(source: str) -> bool:
    if not _FLOAT_CORE.match(source):
        return False
    value = source.lower()
    if value[0] in "+-":
        value = value[1:]
    if value in (".inf", ".nan"):
        return True
    return math.isfinite(_js_parse_float(value)) or bool(_FLOAT_SPECIAL.match(source))


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    return 30 if month in (4, 6, 9, 11) else 31


def _timestamp(source: str) -> bool:
    m = _DATE.match(source) or _TIMESTAMP.match(source)
    if not m:
        return False
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not 1 <= month <= 12 or not 1 <= day <= _days_in_month(year, month):
        return False
    if m.lastindex and m.lastindex >= 4 and m.group(4):
        if int(m.group(4)) > 23 or int(m.group(5)) > 59 or int(m.group(6)) > 59:
            return False
        if m.group(9):
            if int(m.group(10)) > 23 or int(m.group(11) or 0) > 59:
                return False
    return True


def resolve_implicit_tag(source: str) -> str:
    """Schema.resolveImplicitScalarTag for DUMP_SCHEMA, returning the tag name."""
    first = source[:1]
    if first in ("", "~", "n", "N") and source in _NULL_VALUES:
        return NULL_TAG
    if first in "yYnNtTfFoO" and first and source in _BOOL_VALUES:
        return BOOL_TAG
    if first and first in "-+0123456789" and (_int11(source) or _int_core(source)):
        return INT_TAG
    if first and first in "-+.0123456789" and (_float11(source) or _float_core(source)):
        return FLOAT_TAG
    if first and first in "0123456789" and _timestamp(source):
        return TIMESTAMP_TAG
    if source == "<<":
        return MERGE_TAG
    return STR_TAG


# --- AST building --------------------------------------------------------------------------


class _Node:
    __slots__ = ("kind", "tag", "value", "items", "anchor", "style")

    def __init__(self, kind: str, tag: str, value: Any = None) -> None:
        self.kind = kind
        self.tag = tag
        self.value = value
        self.items: list[Any] = []
        self.anchor: str | None = None
        self.style = PLAIN


_INVALID = object()


def _represent_float(value: float) -> str:
    if math.isnan(value):
        return ".nan"
    if value == math.inf:
        return ".inf"
    if value == -math.inf:
        return "-.inf"
    if value == 0 and math.copysign(1, value) < 0:
        return "-0.0"
    result = number_to_string(value)
    return result.replace("e", ".e", 1) if re.match(r"^[-+]?[0-9]+e", result) else result


_ARRAY_INDEX = re.compile(r"(?:0|[1-9][0-9]*)")


def v8_items(obj: dict[Any, Any]) -> list[tuple[str, Any]]:
    """Object.keys order: array-index keys ascending, then the rest in insertion order."""
    items = [(k if isinstance(k, str) else js_string(k), v) for k, v in obj.items()]
    index = [(k, v) for k, v in items if _ARRAY_INDEX.fullmatch(k) and int(k) < 2**32 - 1]
    if not index:
        return items
    index.sort(key=lambda kv: int(kv[0]))
    seen = {k for k, _ in index}
    return index + [(k, v) for k, v in items if k not in seen]


class _Builder:
    def __init__(self) -> None:
        self.refs: dict[int, _Node] = {}
        self.keep: list[Any] = []
        self.counter = 0

    def build(self, obj: Any) -> Any:
        if isinstance(obj, dict | list | tuple):
            existing = self.refs.get(id(obj))
            if existing is not None:
                if existing.anchor is None:
                    existing.anchor = f"ref_{self.counter}"
                    self.counter += 1
                alias = _Node("alias", "")
                alias.anchor = existing.anchor
                return alias
        if obj is None:
            return _Node("scalar", NULL_TAG, "null")
        if obj is UNDEFINED:
            return _INVALID
        if isinstance(obj, bool):
            return _Node("scalar", BOOL_TAG, "true" if obj else "false")
        if isinstance(obj, int):
            return _Node("scalar", INT_TAG, str(obj))
        if isinstance(obj, float):
            if math.isfinite(obj) and obj.is_integer() and "e" not in number_to_string(obj):
                if not (obj == 0 and math.copysign(1, obj) < 0):
                    return _Node("scalar", INT_TAG, number_to_string(obj))
            return _Node("scalar", FLOAT_TAG, _represent_float(obj))
        if isinstance(obj, str):
            return _Node("scalar", STR_TAG, obj)
        if isinstance(obj, list | tuple):
            node = _Node("sequence", "tag:yaml.org,2002:seq")
            self.refs[id(obj)] = node
            self.keep.append(obj)
            for item in obj:
                built = self.build(item)
                if built is _INVALID and item is UNDEFINED:
                    built = self.build(None)
                if built is _INVALID:
                    continue
                node.items.append(built)
            return node
        if isinstance(obj, dict):
            node = _Node("mapping", "tag:yaml.org,2002:map")
            self.refs[id(obj)] = node
            self.keep.append(obj)
            for key, value in v8_items(obj):
                built_key = self.build(key)
                built_value = self.build(value)
                if built_key is _INVALID or built_value is _INVALID:
                    continue
                node.items.append((built_key, built_value))
            return node
        if callable(obj):
            raise YAMLException("unacceptable kind of an object to dump [object Function]")
        if hasattr(obj, "to_json"):
            return self.build(obj.to_json())
        raise YAMLException(f"unacceptable kind of an object to dump {type(obj).__name__}")


# --- scalar styling ------------------------------------------------------------------------

_C_PRINTABLE = "\t\n\r\x20-\x7e\x85\xa0-퟿-�\U00010000-\U0010ffff"
_NB_CHAR = "(?:(?![\n\r﻿])[" + _C_PRINTABLE + "])"
_NS_CHAR = "(?:(?![ \t])" + _NB_CHAR + ")"
_NB_JSON = "[\t\x20-퟿-￿\U00010000-\U0010ffff]"
_C_INDICATOR = "[-?:,\\[\\]{}#&*!|>'\"%@`]"
_C_FLOW_INDICATOR = "[,\\[\\]{}]"
_SAFE_OUT = _NS_CHAR
_SAFE_IN = f"(?:(?!{_C_FLOW_INDICATOR}){_NS_CHAR})"
_FIRST_OUT = f"(?:(?:(?!{_C_INDICATOR}){_NS_CHAR})|[?:-](?={_SAFE_OUT}))"
_FIRST_IN = f"(?:(?:(?!{_C_INDICATOR}){_NS_CHAR})|[?:-](?={_SAFE_IN}))"
_CHAR_OUT = f"(?:(?:(?![:#]){_SAFE_OUT})|:(?={_SAFE_OUT}))#*"
_CHAR_IN = f"(?:(?:(?![:#]){_SAFE_IN})|:(?={_SAFE_IN}))#*"
_IN_LINE_OUT = f"(?:[ \t]*{_CHAR_OUT})*"
_IN_LINE_IN = f"(?:[ \t]*{_CHAR_IN})*"
_ONE_LINE_OUT = f"{_FIRST_OUT}#*{_IN_LINE_OUT}"
_ONE_LINE_IN = f"{_FIRST_IN}#*{_IN_LINE_IN}"
_NEXT_LINE_OUT = f"\n+{_CHAR_OUT}{_IN_LINE_OUT}"
_NEXT_LINE_IN = f"\n+{_CHAR_IN}{_IN_LINE_IN}"
NS_PLAIN_FLOW_OUT = re.compile(f"(?:{_ONE_LINE_OUT}(?:{_NEXT_LINE_OUT})*)")
NS_PLAIN_FLOW_IN = re.compile(f"(?:{_ONE_LINE_IN}(?:{_NEXT_LINE_IN})*)")
NS_PLAIN_BLOCK_KEY = re.compile(f"(?:{_ONE_LINE_OUT})")
NS_PLAIN_FLOW_KEY = re.compile(f"(?:{_ONE_LINE_IN})")
NB_SINGLE_ONE_LINE = re.compile(f"(?:{_NB_JSON})*")
NB_SINGLE_MULTI_LINE = re.compile(f"(?:{_NB_JSON}|\n)*")
BLOCK_SCALAR_CONTENT = re.compile(f"(?:{_NB_CHAR}|\n)*")
C_FORBIDDEN_FIRST_LINE = re.compile(r"^(?:---|\.\.\.)(?=$|[ \t\n\r])")
C_FORBIDDEN_CONTENT = re.compile(r"^(?:---|\.\.\.)(?=$|[ \t\n\r])", re.M)
_INVISIBLES = re.compile("[\t\x7f-\xa0  ﻿￾￿]")
_JS_WS = "\t\n\x0b\x0c\r \xa0  -     　﻿"
_WHITESPACE_ONLY = re.compile(f"[{_JS_WS}]+")


def _js_dollar(pattern: re.Pattern[str], text: str) -> bool:
    """`^(?:...)$` with JS semantics: `$` matches only at the very end (no trailing \\n)."""
    return pattern.fullmatch(text) is not None


class _Layout:
    def __init__(self, node: _Node, level: int, is_key: bool, flow_only: bool) -> None:
        self.node = node
        self.is_key = is_key
        self.flow_only = flow_only
        self.shift_of_parent = -1 if level == 0 else INDENT * (level - 1)
        self.shift_of_content = INDENT * max(1, level)
        self.shift_of_first_line = 0 if level == 0 else INDENT * level
        self.allowed = 0
        self.style = node.style


def _bit(style: int) -> int:
    return 1 << style


def _can_use_plain(layout: _Layout) -> bool:
    value = layout.node.value
    if value != "":
        if layout.is_key:
            pattern = NS_PLAIN_FLOW_KEY if layout.flow_only else NS_PLAIN_BLOCK_KEY
        else:
            pattern = NS_PLAIN_FLOW_IN if layout.flow_only else NS_PLAIN_FLOW_OUT
        if not _js_dollar(pattern, value):
            return False
        if layout.shift_of_first_line == 0 and C_FORBIDDEN_FIRST_LINE.search(value):
            return False
        if layout.shift_of_content == 0:
            idx = value.find("\n")
            if idx != -1 and C_FORBIDDEN_CONTENT.search(value[idx + 1 :]):
                return False
    resolved = resolve_implicit_tag(value)
    if resolved != layout.node.tag:
        return False
    return not (value == "=" and resolved == STR_TAG)


def _can_use_single_quoted(layout: _Layout) -> bool:
    value = layout.node.value
    if not _js_dollar(NB_SINGLE_ONE_LINE if layout.is_key else NB_SINGLE_MULTI_LINE, value):
        return False
    if re.search(r"[ \t]\n|\n[ \t]", value):
        return False
    if not layout.is_key and layout.shift_of_content == 0:
        idx = value.find("\n")
        if idx != -1 and C_FORBIDDEN_CONTENT.search(value[idx + 1 :]):
            return False
    return True


def _can_use_block(layout: _Layout) -> bool:
    value = layout.node.value
    if layout.flow_only or not _js_dollar(BLOCK_SCALAR_CONTENT, value):
        return False
    content_indent = layout.shift_of_content - layout.shift_of_parent
    if content_indent < 1:
        return False
    if content_indent > 9 and re.match(r"^\n* ", value):
        return False
    return not (layout.shift_of_content == 0 and C_FORBIDDEN_CONTENT.search(value))


def _detect_allowed(layout: _Layout) -> None:
    mask = _bit(DOUBLE_QUOTED)
    if _can_use_plain(layout):
        mask |= _bit(PLAIN)
    if _can_use_single_quoted(layout):
        mask |= _bit(SINGLE_QUOTED)
    if _can_use_block(layout):
        mask |= _bit(LITERAL_BLOCK) | _bit(FOLDED_BLOCK)
    layout.allowed = mask


def _preferred_quoted(layout: _Layout) -> int:
    return SINGLE_QUOTED if layout.allowed & _bit(SINGLE_QUOTED) else DOUBLE_QUOTED


def _apply_rules(layout: _Layout) -> None:
    value = layout.node.value
    # doubleQuoteForInvisibles
    if layout.style == PLAIN and _INVISIBLES.search(value):
        layout.style = DOUBLE_QUOTED
    # doubleQuoteWhitespaceOnly
    if layout.style == PLAIN and _WHITESPACE_ONLY.fullmatch(value):
        layout.style = DOUBLE_QUOTED
    # tryLongOrMultilineAsBlock
    if layout.style == PLAIN and not layout.is_key:
        multiline = "\n" in value
        if not layout.allowed & _bit(LITERAL_BLOCK):
            if multiline:
                layout.style = DOUBLE_QUOTED
        else:
            available = max(min(LINE_WIDTH, 40), LINE_WIDTH - layout.shift_of_content)
            should_fold = any(
                len(line) > available and line[:1] != " " and re.search(r" [^ \t]", line)
                for line in value.split("\n")
            )
            if should_fold:
                layout.style = FOLDED_BLOCK
            elif multiline:
                layout.style = LITERAL_BLOCK
    # quoteInvalidPlain
    if layout.style == PLAIN and not layout.allowed & _bit(PLAIN):
        layout.style = _preferred_quoted(layout)
    # fallbackToDoubleQuoted
    if not layout.allowed & _bit(layout.style):
        layout.style = DOUBLE_QUOTED


def _encode_flow_breaks(value: str, shift: int) -> str:
    idx = value.find("\n")
    if idx == -1:
        return value
    pad = " " * shift
    result = value[:idx]
    for m in re.finditer(r"(\n+)([^\n]*)", value[idx:]):
        result += "\n" * (len(m.group(1)) + 1) + pad + m.group(2)
    return result


def _indent_string(value: str, spaces: int) -> str:
    indent = " " * spaces
    out = re.findall(r"[^\n]*\n|[^\n]+\Z", value)
    return "".join((indent + line) if line and line != "\n" else line for line in out)


def _block_header(value: str, shift_parent: int, shift_content: int) -> str:
    indicator = str(shift_content - shift_parent) if re.match(r"^\n* ", value) else ""
    clip = value[-1:] == "\n"
    if clip and (value[-2:-1] == "\n" or value == "\n"):
        chomp = "+"
    elif clip:
        chomp = ""
    else:
        chomp = "-"
    return f"{indicator}{chomp}\n"


def _drop_ending_newline(value: str) -> str:
    return value[:-1] if value[-1:] == "\n" else value


def _more_indented(ch: str) -> bool:
    return ch in (" ", "\t")


def _fold_line(line: str, width: float) -> str:
    if line == "" or _more_indented(line[0]):
        return line
    start = 0
    curr = 0
    result = ""
    for m in re.finditer(r" [^ \t]", line):
        nxt = m.start()
        if nxt - start > width:
            end = curr if curr > start else nxt
            result += "\n" + line[start:end]
            start = end + 1
        curr = nxt
    result += "\n"
    if len(line) - start > width and curr > start:
        result += line[start:curr] + "\n" + line[curr + 1 :]
    else:
        result += line[start:]
    return result[1:]


def _fold_block_scalar(value: str, width: float) -> str:
    next_lf = value.find("\n")
    if next_lf == -1:
        next_lf = len(value)
    result = _fold_line(value[:next_lf], width)
    prev_more = value[:1] == "\n" or (value[:1] != "" and _more_indented(value[0]))
    for m in re.finditer(r"(\n+)([^\n]*)", value[next_lf:]):
        prefix, line = m.group(1), m.group(2)
        more = line != "" and _more_indented(line[0])
        sep = "\n" if not prev_more and not more and line != "" else ""
        result += prefix + sep + _fold_line(line, width)
        prev_more = more
    return result


_ESCAPES = {
    "\0": "\\0", "\x07": "\\a", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\x0b": "\\v",
    "\x0c": "\\f", "\r": "\\r", "\x1b": "\\e", '"': '\\"', "\\": "\\\\", "\x85": "\\N",
    "\xa0": "\\_", " ": "\\L", " ": "\\P",
}  # fmt: skip
_TO_ESCAPE = re.compile('["\\\\\x00-\x1f\x7f-\xa0  \ud800-\udfff﻿￾￿]')


def _escape_char(m: re.Match[str]) -> str:
    ch = m.group(0)
    if ch in _ESCAPES:
        return _ESCAPES[ch]
    code = ord(ch)
    hexs = format(code, "X")
    if code <= 255:
        return "\\x" + hexs.rjust(2, "0")
    return "\\u" + hexs.rjust(4, "0")


def _render_scalar(layout: _Layout) -> str:
    value = layout.node.value
    if layout.style == PLAIN:
        return _encode_flow_breaks(value, layout.shift_of_content)
    if layout.style == SINGLE_QUOTED:
        return "'" + _encode_flow_breaks(value, layout.shift_of_content).replace("'", "''") + "'"
    if layout.style == LITERAL_BLOCK:
        return (
            "|"
            + _block_header(value, layout.shift_of_parent, layout.shift_of_content)
            + _drop_ending_newline(_indent_string(value, layout.shift_of_content))
        )
    if layout.style == FOLDED_BLOCK:
        available = max(min(LINE_WIDTH, 40), LINE_WIDTH - layout.shift_of_content)
        folded = _fold_block_scalar(value, available)
        return (
            ">"
            + _block_header(value, layout.shift_of_parent, layout.shift_of_content)
            + _drop_ending_newline(_indent_string(folded, layout.shift_of_content))
        )
    return '"' + _TO_ESCAPE.sub(_escape_char, value) + '"'


# --- presenter -----------------------------------------------------------------------------


def _next_line(level: int) -> str:
    return "\n" + " " * (INDENT * level)


class _Presenter:
    def __init__(self) -> None:
        self.open_ended = False

    def write_flow_sequence(self, level: int, node: _Node) -> str:
        parts = [self.write_node(level, item, node, block=False)[0] for item in node.items]
        return "[" + ", ".join(parts) + "]"

    def write_block_sequence(self, level: int, node: _Node, compact: bool) -> str:
        result = ""
        for item in node.items:
            text = self.write_node(level + 1, item, node, block=True, compact=True)[0]
            if not compact or result != "":
                result += _next_line(level)
            result += "-" if text == "" or text[0] == "\n" else "- "
            result += text
        return result

    def write_flow_mapping(self, level: int, node: _Node) -> str:
        result = ""
        for key, value in node.items:
            pair = ", " if result != "" else ""
            key_text, _ = self.write_node(level, key, node, block=False, is_key=True)
            value_text = self.write_node(level, value, node, block=False)[0]
            sep = "" if value_text == "" else " "
            colon_sep = " " if key.kind == "alias" else ""
            pair += f"{key_text}{colon_sep}:{sep}{value_text}"
            result += pair
        return "{" + result + "}"

    def write_block_mapping(self, level: int, node: _Node, compact: bool) -> str:
        result = ""
        for key, value in node.items:
            pair = ""
            if not compact or result != "":
                pair += _next_line(level)
            key_is_block = (key.kind in ("mapping", "sequence") and key.items) or (
                key.kind == "scalar" and key.style in (LITERAL_BLOCK, FOLDED_BLOCK)
            )
            if key_is_block:
                key_text = self.write_node(level + 1, key, node, block=True, compact=True)[0]
            else:
                key_text = self.write_node(
                    level + 1, key, node, block=True, compact=True, is_key=True
                )[0]
            key_has_break = key.kind == "scalar" and "\n" in key.value
            key_too_long = len(key_text) > 1024
            explicit = bool(key_is_block or key_has_break or key_too_long)
            if explicit:
                pair += "?" if key_text and key_text[0] == "\n" else "? "
            pair += key_text
            if explicit:
                pair += _next_line(level)
            value_text = self.write_node(level + 1, value, node, block=True, compact=explicit)[0]
            colon_sep = " " if not explicit and key.kind == "alias" else ""
            if value_text == "" or value_text[0] == "\n":
                pair += f"{colon_sep}:"
            else:
                pair += f"{colon_sep}: "
            pair += value_text
            result += pair
        return result

    def write_node(
        self,
        level: int,
        node: _Node,
        parent: _Node | None,
        *,
        block: bool = False,
        compact: bool = False,
        is_key: bool = False,
    ) -> tuple[str, bool]:
        if node.kind == "alias":
            self.open_ended = False
            return f"*{node.anchor}", False
        has_anchor = node.anchor is not None
        if has_anchor:
            compact = False
        use_block = block and node.kind in ("mapping", "sequence") and bool(node.items)
        should_print_tag = False
        if node.kind == "mapping":
            body = (
                self.write_block_mapping(level, node, compact)
                if use_block
                else self.write_flow_mapping(level, node)
            )
        elif node.kind == "sequence":
            body = (
                self.write_block_sequence(level, node, compact)
                if use_block
                else self.write_flow_sequence(level, node)
            )
        else:
            layout = _Layout(node, level, is_key, not block)
            _detect_allowed(layout)
            _apply_rules(layout)
            node.style = layout.style
            body = _render_scalar(layout)
            self.open_ended = layout.style in (LITERAL_BLOCK, FOLDED_BLOCK) and (
                node.value == "\n" or node.value.endswith("\n\n")
            )
            should_print_tag = (
                body == ""
                and layout.flow_only
                and parent is not None
                and parent.kind == "sequence"
                and not has_anchor
            ) or (layout.style != PLAIN and node.tag != STR_TAG)
        if node.kind in ("mapping", "sequence") and not use_block:
            self.open_ended = False
        no_body = body == ""
        text = body
        if should_print_tag or has_anchor:
            props = []
            if has_anchor:
                props.append(f"&{node.anchor}")
            if should_print_tag:
                props.append(_short_tag(node.tag))
            sep = "" if body == "" or body[0] == "\n" else " "
            text = " ".join(props) + sep + body
        return text, no_body


def _short_tag(tag: str) -> str:
    prefix = "tag:yaml.org,2002:"
    return "!!" + tag[len(prefix) :] if tag.startswith(prefix) else f"!<{tag}>"


def dump(value: Any) -> str:
    """js-yaml `dump(value)` with default options."""
    builder = _Builder()
    root = builder.build(value)
    if root is _INVALID:
        return ""
    presenter = _Presenter()
    result = presenter.write_node(0, root, None, block=True, compact=True)[0] + "\n"
    if presenter.open_ended:
        result += "...\n"
    return result
