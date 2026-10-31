"""XML and HTML assertions. Ports src/assertions/xml.ts and src/assertions/html.ts.

Handlers: is-xml, contains-xml (both `handleIsXml` upstream, dispatching on baseType),
is-html, contains-html.

Node delegates to three npm parsers; this module replaces them as follows.

* saxes (strict XML 1.0 + namespaces well-formedness, used by is-xml) is replaced by the
  stdlib expat parser in namespace mode. The DOCTYPE policy checks (internal subsets
  rejected, malformed DOCTYPEs, external-subset entity passthrough) are ported verbatim and
  run on the DOCTYPE text extracted the way saxes extracts it. Divergences:
    - Error text after "XML parsing failed: " is expat's ("junk after document element:
      line 1, column 4"), not saxes' ("1:5: text data outside of root node."). The pass/fail
      verdict matches for every case in upstream's spec.
    - expat only knows the XML 1.0 4th-edition name tables, saxes uses the 5th edition.
      Before expat sees the input, every non-ASCII 5th-edition name-start character is
      mapped one-to-one onto a CJK ideograph, and every name-only character (U+00B7,
      U+0300-036F, U+203F-2040) is mapped onto ".". Names like `<𐀀>` are then accepted as
      saxes accepts them. Two distinct name-only characters in otherwise identical names
      can collide, but only in namespace-prefix comparisons.
    - Anything that expat and saxes treat differently beyond that (for example XML 1.1
      declarations) follows expat.
* fast-xml-parser (lenient data extraction, used for requiredElements and for every
  contains-xml candidate) is ported in-module (`_fxp_parse`). It follows the v5.7 option set
  used upstream: attributes under "@_", text under "#text", repeated siblings as arrays,
  `__proto__`/`constructor`/`prototype` rejected, the 100-level nesting cap, and the same
  error messages. Values are left as strings and never number-parsed, because only key
  presence and the object-versus-primitive shape matter for path checks.
* parse5 (HTML5 tree construction, used by is-html) is replaced by a compact HTML5
  tokenizer and tree builder (`_parse_html`). It covers what validateHtml looks at: implied
  html/head/body (with no source location), head hoisting, raw-text elements, void and
  foreign (svg/math) elements, table foster parenting and the after-body modes. The
  adoption agency algorithm and the full insertion-mode table are simplified, so the shape
  of unusual misnested markup can differ from parse5's. Top-level body text and the set
  of element names come out the same for ordinary input.
"""

from __future__ import annotations

import html as _html
import pyexpat
import re
from collections.abc import Callable
from typing import Any

from promptphoo.assertions.base import (
    AssertionParams,
    AssertionValueError,
    GradingResult,
    Handler,
)
from promptphoo.jsvalues import js_string

# JS String.prototype.trim / \s whitespace set.
_JS_WS = "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"  # noqa: E501
_JS_S = r"[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]"  # JS \s


def _js_trim(s: str) -> str:
    return s.strip(_JS_WS)


def _js_trim_start(s: str) -> str:
    return s.lstrip(_JS_WS)


def _js_len(s: str) -> int:
    """String.prototype.length (UTF-16 code units)."""
    return len(s) + sum(1 for c in s if ord(c) > 0xFFFF) if not s.isascii() else len(s)


# ---------------------------------------------------------------------------------------------
# XML: name helpers (xml.ts)
# ---------------------------------------------------------------------------------------------


def _is_xml_name_start_code_point(cp: int | None) -> bool:
    return cp is not None and (
        cp == 0x3A
        or cp == 0x5F
        or 0x41 <= cp <= 0x5A
        or 0x61 <= cp <= 0x7A
        or 0xC0 <= cp <= 0xD6
        or 0xD8 <= cp <= 0xF6
        or 0xF8 <= cp <= 0x2FF
        or 0x370 <= cp <= 0x37D
        or 0x37F <= cp <= 0x1FFF
        or 0x200C <= cp <= 0x200D
        or 0x2070 <= cp <= 0x218F
        or 0x2C00 <= cp <= 0x2FEF
        or 0x3001 <= cp <= 0xD7FF
        or 0xF900 <= cp <= 0xFDCF
        or 0xFDF0 <= cp <= 0xFFFD
        or 0x10000 <= cp <= 0xEFFFF
    )


def _is_xml_name_only_code_point(cp: int | None) -> bool:
    return cp is not None and (
        cp == 0x2D
        or cp == 0x2E
        or 0x30 <= cp <= 0x39
        or cp == 0xB7
        or 0x300 <= cp <= 0x36F
        or 0x203F <= cp <= 0x2040
    )


def _is_xml_name_code_point(cp: int | None) -> bool:
    return _is_xml_name_start_code_point(cp) or _is_xml_name_only_code_point(cp)


def _cp(s: str, i: int) -> int | None:
    return ord(s[i]) if 0 <= i < len(s) else None


def _read_xml_name(s: str, start: int) -> tuple[str, int] | None:
    if not _is_xml_name_start_code_point(_cp(s, start)):
        return None
    end = start + 1
    while _is_xml_name_code_point(_cp(s, end)):
        end += 1
    return s[start:end], end


def _has_doctype_internal_subset(doctype: str) -> bool:
    quote: str | None = None
    for char in doctype:
        if quote:
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char == "[":
            return True
    return False


def _skip_xml_whitespace(s: str, start: int) -> int:
    i = start
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    return i


def _read_quoted_literal(s: str, start: int) -> tuple[str, int] | None:
    quote = s[start] if start < len(s) else ""
    if quote not in ('"', "'"):
        return None
    closing = s.find(quote, start + 1)
    if closing == -1:
        return None
    return s[start + 1 : closing], closing + 1


_PUBLIC_ID_RE = re.compile(r"[\x20\r\na-zA-Z0-9'()+,./:=?;!*#@$_%-]*")


def _is_valid_public_id(value: str) -> bool:
    return _PUBLIC_ID_RE.fullmatch(value) is not None


def _is_valid_system_id(value: str) -> bool:
    return "#" not in value


def _has_valid_system_literal(s: str, start: int) -> bool:
    literal_start = _skip_xml_whitespace(s, start)
    if literal_start == start:
        return False
    literal = _read_quoted_literal(s, literal_start)
    return bool(
        literal
        and _is_valid_system_id(literal[0])
        and _skip_xml_whitespace(s, literal[1]) == len(s)
    )


def _is_well_formed_doctype(doctype: str) -> bool:
    index = _skip_xml_whitespace(doctype, 0)
    if index == 0:
        return False
    name = _read_xml_name(doctype, index)
    if not name:
        return False
    index = name[1]
    external_id_start = _skip_xml_whitespace(doctype, index)
    if external_id_start == len(doctype):
        return True
    if external_id_start == index:
        return False
    index = external_id_start
    if doctype.startswith("SYSTEM", index):
        return _has_valid_system_literal(doctype, index + len("SYSTEM"))
    if doctype.startswith("PUBLIC", index):
        index += len("PUBLIC")
        public_start = _skip_xml_whitespace(doctype, index)
        if public_start == index:
            return False
        public_literal = _read_quoted_literal(doctype, public_start)
        if not public_literal or not _is_valid_public_id(public_literal[0]):
            return False
        return _has_valid_system_literal(doctype, public_literal[1])
    return False


# ---------------------------------------------------------------------------------------------
# XML: contains-xml candidate extraction (xml.ts)
# ---------------------------------------------------------------------------------------------


def _find_tag_end(s: str, start: int) -> int:
    quote: str | None = None
    for i in range(start + 1, len(s)):
        char = s[i]
        if quote:
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char == "<":
            return -1
        elif char == ">":
            return i
    return -1


def _find_delimited_declaration_end(s: str, start: int, delimiter: str) -> int:
    idx = s.find(delimiter, start)
    return idx + len(delimiter) - 1 if idx != -1 else len(s) - 1


def _find_declaration_end(s: str, start: int) -> int:
    if s.startswith("<!--", start):
        return _find_delimited_declaration_end(s, start + 4, "-->")
    if s.startswith("<![CDATA[", start):
        return _find_delimited_declaration_end(s, start + 9, "]]>")
    quote: str | None = None
    depth = 0
    for i in range(start + 2, len(s)):
        char = s[i]
        if quote:
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]" and depth > 0:
            depth -= 1
        elif char == ">" and depth == 0:
            return i
    return len(s) - 1


_JS_S_RE = re.compile(_JS_S)


def _is_self_closing_tag(s: str, start: int, end: int) -> bool:
    last = end - 1
    while last > start and _JS_S_RE.match(s[last]):
        last -= 1
    return s[last] == "/"


# tag tuples: ("opening", start, end, name, self_closing) | ("closing", start, end, name)
# | ("declaration", start, end)
def _read_xml_tag(s: str, start: int) -> tuple[Any, ...] | None:
    if start + 1 >= len(s):
        return None
    nxt = s[start + 1]
    if nxt == "!":
        return ("declaration", start, _find_declaration_end(s, start))
    if nxt == "?":
        return ("declaration", start, _find_delimited_declaration_end(s, start + 2, "?>"))
    if nxt == "/":
        name = _read_xml_name(s, start + 2)
        if not name:
            return None
        end = _find_tag_end(s, start)
        return None if end == -1 else ("closing", start, end, name[0])
    name = _read_xml_name(s, start + 1)
    if not name:
        return None
    end = _find_tag_end(s, start)
    if end == -1:
        return None
    return ("opening", start, end, name[0], _is_self_closing_tag(s, start, end))


# candidates: (start, end, parent_start | None)
def _order_xml_candidates(
    candidates: list[tuple[int, int, int | None]],
) -> list[tuple[int, int, int | None]]:
    ordered: list[tuple[int, int, int | None]] = []
    seen: set[tuple[int, int]] = set()
    by_parent: dict[int | None, list[tuple[int, int, int | None]]] = {}
    candidate_starts = {c[0] for c in candidates}

    def add(c: tuple[int, int, int | None]) -> None:
        key = (c[0], c[1])
        if key not in seen:
            seen.add(key)
            ordered.append(c)

    for c in candidates:
        by_parent.setdefault(c[2], []).append(c)

    for parent_start, siblings in by_parent.items():
        if len(siblings) <= 1 or (parent_start is not None and parent_start in candidate_starts):
            continue
        siblings.sort(key=lambda c: (c[0], c[1]))
        add((siblings[0][0], siblings[-1][1], parent_start))

    maximal = sorted(
        (c for c in candidates if c[2] is None or c[2] not in candidate_starts),
        key=lambda c: (c[0], -c[1]),
    )
    for c in maximal:
        add(c)
    return ordered


def _extract_xml_candidates(s: str) -> list[tuple[int, int, int | None]]:
    stack: list[tuple[str, int]] = []
    candidates: list[tuple[int, int, int | None]] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] != "<":
            nxt = s.find("<", i)
            if nxt == -1:
                break
            i = nxt
        tag = _read_xml_tag(s, i)
        if not tag:
            i += 1
            continue
        kind = tag[0]
        if kind == "opening":
            if tag[4]:
                candidates.append((tag[1], tag[2] + 1, stack[-1][1] if stack else None))
            else:
                stack.append((tag[3], tag[1]))
        elif kind == "closing":
            open_tag = stack[-1] if stack else None
            if open_tag is not None and open_tag[0] == tag[3]:
                stack.pop()
                candidates.append((open_tag[1], tag[2] + 1, stack[-1][1] if stack else None))
            else:
                stack.clear()
        i = tag[2] + 1
    return _order_xml_candidates(candidates)


# ---------------------------------------------------------------------------------------------
# XML: fast-xml-parser port (XMLParser with the options xml.ts passes)
# ---------------------------------------------------------------------------------------------

_DANGEROUS_PROPERTY_NAMES = {
    "hasOwnProperty",
    "toString",
    "valueOf",
    "__defineGetter__",
    "__defineSetter__",
    "__lookupGetter__",
    "__lookupSetter__",
}
_CRITICAL_PROPERTIES = {"__proto__", "constructor", "prototype"}
_FXP_MAX_NESTED_TAGS = 100
_FXP_ATTRS_RE = re.compile(r"""([^\s=]+)\s*(=\s*(['"])([\s\S]*?)\3)?""")
_TEXT = "#text"


class _FxpNode:
    __slots__ = ("tagname", "child", "attrs")

    def __init__(self, tagname: str) -> None:
        self.tagname = tagname
        # child entries: ("#text", str) or (tagname, list_of_children, attrs | None)
        self.child: list[tuple[Any, ...]] = []
        self.attrs: dict[str, Any] | None = None

    def add_text(self, val: str) -> None:
        self.child.append((_TEXT, val))

    def add_child(self, node: _FxpNode) -> None:
        if node.tagname == "__proto__":
            node.tagname = "#__proto__"
        self.child.append((node.tagname, node.child, node.attrs or None))


def _fxp_sanitize_name(name: str) -> str:
    if name in _CRITICAL_PROPERTIES:
        raise ValueError(
            f'[SECURITY] Invalid name: "{name}" is a reserved JavaScript keyword that could '
            "cause prototype pollution"
        )
    # DANGEROUS_PROPERTY_NAMES go through onDangerousProperty, which xml.ts sets to identity.
    return name


def _fxp_build_attributes_map(attr_str: str) -> dict[str, Any] | None:
    attrs: dict[str, Any] = {}
    for m in _FXP_ATTRS_RE.finditer(attr_str):
        attr_name = m.group(1)
        if not attr_name:
            continue
        a_name = _fxp_sanitize_name("@_" + attr_name)
        if m.group(4) is not None:
            attrs[a_name] = _js_trim(m.group(4))
        else:  # allowBooleanAttributes
            attrs[a_name] = True
    return attrs or None


def _fxp_tag_exp_with_closing_index(s: str, i: int, closing: str) -> tuple[str, int] | None:
    boundary = ""
    c0 = closing[0]
    c1 = closing[1] if len(closing) > 1 else ""
    for index in range(i, len(s)):
        ch = s[index]
        if boundary:
            if ch == boundary:
                boundary = ""
        elif ch in "\"'":
            boundary = ch
        elif ch == c0:
            if not c1 or (index + 1 < len(s) and s[index + 1] == c1):
                return s[i:index].replace("\t", " "), index
    return None


_JS_S_SEARCH = re.compile(_JS_S)


def _fxp_read_tag_exp(s: str, i: int, closing: str = ">") -> dict[str, Any] | None:
    res = _fxp_tag_exp_with_closing_index(s, i + 1, closing)
    if res is None:
        return None
    tag_exp, close_index = res
    tag_name = tag_exp
    m = _JS_S_SEARCH.search(tag_exp)
    if m:
        tag_name = tag_exp[: m.start()]
        tag_exp = _js_trim_start(tag_exp[m.start() + 1 :])
    return {"tagName": tag_name, "tagExp": tag_exp, "closeIndex": close_index, "attrExp": True}


def _fxp_find_closing_index(s: str, needle: str, i: int, err: str) -> int:
    idx = s.find(needle, i)
    if idx == -1:
        raise ValueError(err)
    return idx + len(needle) - 1


def _fxp_read_doctype(s: str, i: int) -> int:
    """DocTypeReader.readDocType, reduced to finding the end (entities don't affect paths)."""
    if s[i + 3 : i + 9] != "OCTYPE":
        raise ValueError("Invalid Tag instead of DOCTYPE")
    i += 9
    count = 1
    has_body = comment = False
    while i < len(s):
        ch = s[i]
        if ch == "<" and not comment:
            if has_body and (
                s.startswith("!ENTITY", i + 1)
                or s.startswith("!ELEMENT", i + 1)
                or s.startswith("!ATTLIST", i + 1)
                or s.startswith("!NOTATION", i + 1)
            ):
                pass
            elif s.startswith("!--", i + 1):
                comment = True
            else:
                raise ValueError("Invalid DOCTYPE")
            count += 1
        elif ch == ">":
            if comment:
                if s[i - 1] == "-" and s[i - 2] == "-":
                    comment = False
                    count -= 1
            else:
                count -= 1
            if count == 0:
                break
        elif ch == "[":
            has_body = True
        i += 1
    if count != 0:
        raise ValueError("Unclosed DOCTYPE")
    return i


def _fxp_save_text(text: str, node: _FxpNode | None) -> str:
    if text and node is not None:
        val = _js_trim(text)
        if val:
            node.add_text(val)
    return ""


def _fxp_parse(xml: str) -> dict[str, Any]:
    s = re.sub(r"\r\n?", "\n", xml)
    root = _FxpNode("!xml")
    current: _FxpNode | None = root
    stack: list[_FxpNode | None] = []
    text = ""
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch != "<":
            nxt = s.find("<", i)
            if nxt == -1:
                text += s[i:]
                break
            text += s[i:nxt]
            i = nxt
            continue
        c1 = s[i + 1] if i + 1 < n else ""
        if c1 == "/":
            close = _fxp_find_closing_index(s, ">", i, "Closing Tag is not closed.")
            _fxp_sanitize_name(_js_trim(s[i + 2 : close]))
            if current is not None:
                text = _fxp_save_text(text, current)
            current = stack.pop() if stack else None
            text = ""
            i = close
        elif c1 == "?":
            tag = _fxp_read_tag_exp(s, i, "?>")
            if not tag:
                raise ValueError("Pi Tag is not closed.")
            text = _fxp_save_text(text, current)
            atts = _fxp_build_attributes_map(tag["tagExp"])
            node = _FxpNode(tag["tagName"])
            node.add_text("")
            if tag["tagName"] != tag["tagExp"]:
                node.attrs = atts
            if current is None:
                raise TypeError("Cannot read properties of undefined (reading 'addChild')")
            current.add_child(node)
            i = tag["closeIndex"] + 1
        elif c1 == "!" and s.startswith("--", i + 2):
            i = _fxp_find_closing_index(s, "-->", i + 4, "Comment is not closed.")
        elif c1 == "!" and s.startswith("D", i + 2):
            i = _fxp_read_doctype(s, i)
        elif c1 == "!" and s.startswith("[", i + 2):
            close = _fxp_find_closing_index(s, "]]>", i, "CDATA is not closed.") - 2
            cdata = s[i + 9 : close]
            text = _fxp_save_text(text, current)
            if current is None:
                raise TypeError("Cannot read properties of undefined (reading 'add')")
            current.add_text(cdata)
            i = close + 2
        else:
            tag = _fxp_read_tag_exp(s, i)
            if not tag:
                context = s[max(0, i - 50) : min(n, i + 50)]
                raise ValueError(
                    f'readTagExp returned undefined at position {i}. Context: "{context}"'
                )
            tag_name: str = _fxp_sanitize_name(tag["tagName"])
            tag_exp: str = tag["tagExp"]
            attr_exp_present: bool = tag["attrExp"]
            if tag_name == _TEXT:  # strictReservedNames
                raise ValueError(f"Invalid tag name: {tag_name}")
            if current is not None and text and current.tagname != "!xml":
                text = _fxp_save_text(text, current)
            self_closing = False
            if tag_exp and tag_exp.endswith("/"):
                self_closing = True
                if tag_name.endswith("/"):
                    tag_name = tag_name[:-1]
                    tag_exp = tag_name
                else:
                    tag_exp = tag_exp[:-1]
                attr_exp_present = tag_name != tag_exp
            attrs = (
                _fxp_build_attributes_map(tag_exp)
                if tag_name != tag_exp and attr_exp_present
                else None
            )
            if current is None:
                raise TypeError("Cannot read properties of undefined (reading 'addChild')")
            if self_closing:
                tag_name = _fxp_sanitize_name(tag_name)
                node = _FxpNode(tag_name)
                node.attrs = attrs
                current.add_child(node)
            else:
                node = _FxpNode(tag_name)
                if len(stack) > _FXP_MAX_NESTED_TAGS:
                    raise ValueError("Maximum nested tags exceeded")
                stack.append(current)
                node.attrs = attrs
                current.add_child(node)
                current = node
            text = ""
            i = tag["closeIndex"]
        i += 1
    return _fxp_compress(root.child)


def _fxp_compress(children: list[tuple[Any, ...]]) -> dict[str, Any]:
    text: str | None = None
    obj: dict[str, Any] = {}
    for entry in children:
        prop = entry[0]
        if prop == _TEXT:
            text = entry[1] if text is None else text + entry[1]
            continue
        val: Any = _fxp_compress(entry[1])
        attrs = entry[2]
        if attrs:
            val.update(attrs)
        elif len(val) == 1 and _TEXT in val:
            val = val[_TEXT]
        elif len(val) == 0:
            val = ""
        if prop in obj:
            if not isinstance(obj[prop], list):
                obj[prop] = [obj[prop]]
            obj[prop].append(val)
        else:
            obj[prop] = val
    if text:
        obj[_TEXT] = text
    return obj


# ---------------------------------------------------------------------------------------------
# XML: validation (xml.ts)
# ---------------------------------------------------------------------------------------------

_CANONICAL_INDEX_RE = re.compile(r"0|[1-9][0-9]*")


def _has_element_path(value: Any, path: list[str]) -> bool:
    current = value
    for key in path:
        if isinstance(current, list):
            if not _CANONICAL_INDEX_RE.fullmatch(key):
                return False
            index = int(key)
            if index >= len(current):
                return False
            current = current[index]
            continue
        if not isinstance(current, dict):
            return False
        if key not in current:
            return False
        current = current[key]
    return True


def _get_missing_elements(parsed: Any, required: list[str] | None) -> list[str]:
    if not required:
        return []
    return [el for el in required if not _has_element_path(parsed, el.split("."))]


def _validate_xml_candidate(xml: str, required: list[str] | None) -> tuple[bool, str, bool]:
    """Returns (isValid, reason, isWellFormed)."""
    try:
        parsed = _fxp_parse(xml)
    except (ValueError, TypeError) as err:
        return False, f"XML parsing failed: {err}", False
    missing = _get_missing_elements(parsed, required)
    if missing:
        return False, f"XML is missing required elements: {', '.join(missing)}", True
    return True, "XML is valid and contains all required elements", True


def _find_prolog_doctype(s: str) -> tuple[int, str] | None:
    """Locate a DOCTYPE in the prolog and return (start, saxes doctype text)."""
    i = 1 if s.startswith("\ufeff") else 0
    n = len(s)
    while i < n:
        i = _skip_xml_whitespace(s, i)
        if s.startswith("<?", i):
            end = s.find("?>", i + 2)
            if end == -1:
                return None
            i = end + 2
        elif s.startswith("<!--", i):
            end = s.find("-->", i + 4)
            if end == -1:
                return None
            i = end + 3
        elif s.startswith("<!DOCTYPE", i):
            start = i
            j = i + len("<!DOCTYPE")
            out: list[str] = []
            quote = ""
            in_dtd = False
            while j < n:
                c = s[j]
                if quote:
                    if c == quote:
                        quote = ""
                elif in_dtd:
                    if c == "]":
                        in_dtd = False
                    elif c in "\"'":
                        quote = c
                elif c == ">":
                    return start, "".join(out)
                elif c == "[":
                    in_dtd = True
                elif c in "\"'":
                    quote = c
                out.append(c)
                j += 1
            return None
        else:
            return None
    return None


_CJK_BASE = 0x4E00
_CJK_COUNT = 0x9FA5 - 0x4E00 + 1


def _expat_name_compat(s: str) -> str:
    """Map XML 5th-edition name characters expat doesn't know onto ones it does."""
    if s.isascii():
        return s
    mapping: dict[str, str] = {}
    out: list[str] = []
    for idx, c in enumerate(s):
        cp = ord(c)
        if cp < 0x80 or (idx == 0 and c == "\ufeff"):
            out.append(c)
        elif _is_xml_name_start_code_point(cp):
            m = mapping.get(c)
            if m is None:
                if len(mapping) >= _CJK_COUNT:
                    return s
                m = mapping[c] = chr(_CJK_BASE + len(mapping))
            out.append(m)
        elif _is_xml_name_only_code_point(cp):
            out.append(".")
        else:
            out.append(c)
    return "".join(out)


def _expat_parser() -> Any:
    parser = pyexpat.ParserCreate(namespace_separator=" ")
    # Never load external DTDs or entities; undeclared references inside a non-standalone
    # document with an external subset are skipped, as saxes' passthrough does.
    parser.SetParamEntityParsing(pyexpat.XML_PARAM_ENTITY_PARSING_NEVER)
    return parser


def _check_well_formed(xml: str) -> tuple[str | None, str | None]:
    """saxes stand-in. Returns (unsupportedReason, parseError); both None when well-formed."""
    mapped = _expat_name_compat(xml)
    doctype = _find_prolog_doctype(xml)
    try:
        if doctype is not None:
            start, text = doctype
            _expat_parser().Parse(_expat_name_compat(xml[:start]), False)
            if _has_doctype_internal_subset(text):
                return "DTD internal subsets are not supported by is-xml validation", None
            if not _is_well_formed_doctype(text):
                return None, "Malformed DOCTYPE declaration"
        _expat_parser().Parse(mapped, True)
    except pyexpat.ExpatError as err:
        return None, str(err)
    except (UnicodeError, ValueError) as err:
        return None, str(err)
    return None, None


def validate_xml(xml_string: str, required_elements: list[str] | None = None) -> dict[str, Any]:
    unsupported, parse_error = _check_well_formed(xml_string)
    if unsupported:
        return {"isValid": False, "reason": unsupported}
    if parse_error is not None:
        return {"isValid": False, "reason": f"XML parsing failed: {parse_error}"}
    is_valid, reason, _ = _validate_xml_candidate(xml_string, required_elements)
    return {"isValid": is_valid, "reason": reason}


def contains_xml(output_string: str, required_elements: list[str] | None = None) -> dict[str, Any]:
    candidates = _extract_xml_candidates(output_string)
    well_formed: list[tuple[int, int, int | None]] = []
    if not candidates:
        return {"isValid": False, "reason": "No XML content found in the output"}
    for cand in candidates:
        if any(cand[0] >= c[0] and cand[1] <= c[1] for c in well_formed):
            continue
        is_valid, reason, is_well_formed = _validate_xml_candidate(
            output_string[cand[0] : cand[1]], required_elements
        )
        if is_valid:
            return {"isValid": True, "reason": reason}
        if is_well_formed:
            well_formed.append(cand)
    return {"isValid": False, "reason": "No valid XML content found matching the requirements"}


def _to_string(el: Any) -> str:
    if el is None:
        raise TypeError("Cannot read properties of null (reading 'toString')")
    return js_string(el)


def handle_is_xml(params: AssertionParams) -> GradingResult:
    rendered = params.rendered_value
    required: list[str] | None = None
    if isinstance(rendered, str):
        required = [_js_trim(el) for el in rendered.split(",")]
    elif isinstance(rendered, list) and len(rendered) > 0:
        required = [_to_string(el) for el in rendered]
    elif isinstance(rendered, dict) and len(rendered) > 0:
        if "requiredElements" in rendered and isinstance(rendered["requiredElements"], list):
            required = [_to_string(el) for el in rendered["requiredElements"]]
        else:
            raise AssertionValueError(
                "xml assertion must contain a string, array value, or no value"
            )

    validate = validate_xml if params.base_type == "is-xml" else contains_xml
    res = validate(params.output_string, required)
    passed = res["isValid"] != params.inverse
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": "Assertion passed" if passed else res["reason"],
        "assertion": params.assertion,
    }


# ---------------------------------------------------------------------------------------------
# HTML: contains-html heuristics (html.ts)
# ---------------------------------------------------------------------------------------------

_HTML_PATTERNS = {
    "openingTag": re.compile(r"<[a-zA-Z][a-zA-Z0-9-]*(?:" + _JS_S + r"[^>]*)?>"),
    "closingTag": re.compile(r"</[a-zA-Z][a-zA-Z0-9-]*" + _JS_S + r"*>"),
    "selfClosingTag": re.compile(r"<[a-zA-Z][a-zA-Z0-9-]*(?:" + _JS_S + r"[^>]*)?/>"),
    "htmlEntity": re.compile(r"&(?:[a-zA-Z]+|#[0-9]+|#x[0-9a-fA-F]+);"),
    "doctype": re.compile(r"<!DOCTYPE" + _JS_S + r"+html", re.IGNORECASE),
    "htmlComment": re.compile(r"<!--[^-]*(?:-[^-]+)*-->"),
    "htmlAttribute": re.compile(_JS_S + r"""[a-zA-Z-]+=""" + _JS_S + r"""*["'][^"']*["']"""),
}
_COMMON_TAGS = re.compile(
    r"<(html|head|body|div|span|p|a|img|h[1-6]|ul|ol|li|table|tr|td|th|form|input|button|"
    r"script|style|link|meta|br|hr)\b",
    re.IGNORECASE | re.ASCII,
)
_MAX_INPUT_SIZE = 10 * 1024 * 1024


def _exceeds_max_size(text: str) -> bool:
    if len(text) * 2 <= _MAX_INPUT_SIZE:
        return False
    return _js_len(text) > _MAX_INPUT_SIZE


def _contains_html(text: str) -> bool:
    if _exceeds_max_size(text):
        return False
    indicators = 0
    has_opening = _HTML_PATTERNS["openingTag"].search(text) is not None
    has_closing = _HTML_PATTERNS["closingTag"].search(text) is not None
    if has_opening and has_closing:
        indicators += 2
    elif has_opening or has_closing:
        indicators += 1
    if _HTML_PATTERNS["selfClosingTag"].search(text):
        indicators += 1
    if _HTML_PATTERNS["htmlEntity"].search(text):
        indicators += 1
    if _HTML_PATTERNS["doctype"].search(text):
        indicators += 2
    if _HTML_PATTERNS["htmlComment"].search(text):
        indicators += 1
    if _HTML_PATTERNS["htmlAttribute"].search(text):
        indicators += 1
    if _COMMON_TAGS.search(text):
        indicators += 2
    return indicators >= 2


# ---------------------------------------------------------------------------------------------
# HTML: parse5 stand-in
# ---------------------------------------------------------------------------------------------

_HTML_NS, _SVG_NS, _MATH_NS = "html", "svg", "math"


class _Text:
    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value


class _Element:
    __slots__ = ("tag_name", "ns", "children", "has_location", "template_content")

    def __init__(self, tag_name: str, ns: str = _HTML_NS, has_location: bool = True) -> None:
        self.tag_name = tag_name
        self.ns = ns
        self.children: list[_Element | _Text] = []
        self.has_location = has_location
        # <template> children live in a separate fragment (not childNodes), as in parse5.
        self.template_content: list[_Element | _Text] | None = (
            [] if tag_name == "template" and ns == _HTML_NS else None
        )


_HTML_WS = "\t\n\f\r "
_VOID = {
    "area", "base", "basefont", "bgsound", "br", "col", "embed", "frame", "hr", "image",
    "img", "input", "keygen", "link", "meta", "param", "source", "track", "wbr",
}  # fmt: skip
_RAWTEXT = {"style", "xmp", "iframe", "noembed", "noframes", "noscript", "script"}
_RCDATA = {"title", "textarea"}
_HEAD_ELEMENTS = {
    "base", "basefont", "bgsound", "link", "meta", "noframes", "script", "style",
    "template", "title", "noscript",
}  # fmt: skip
_CLOSES_P = {
    "address", "article", "aside", "blockquote", "center", "details", "dialog", "dir", "div",
    "dl", "fieldset", "figcaption", "figure", "footer", "header", "hgroup", "main", "menu",
    "nav", "ol", "p", "search", "section", "summary", "ul", "pre", "listing", "form",
    "plaintext", "table", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "xmp",
}  # fmt: skip
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SPECIAL = {
    "address", "applet", "area", "article", "aside", "base", "basefont", "bgsound",
    "blockquote", "body", "br", "button", "caption", "center", "col", "colgroup", "dd",
    "details", "dir", "div", "dl", "dt", "embed", "fieldset", "figcaption", "figure", "footer",
    "form", "frame", "frameset", "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup",
    "hr", "html", "iframe", "img", "input", "keygen", "li", "link", "listing", "main",
    "marquee", "menu", "meta", "nav", "noembed", "noframes", "noscript", "object", "ol", "p",
    "param", "plaintext", "pre", "script", "search", "section", "select", "source", "style",
    "summary", "table", "tbody", "td", "template", "textarea", "tfoot", "th", "thead", "title",
    "tr", "track", "ul", "wbr", "xmp",
}  # fmt: skip
_SCOPE_BOUNDARY = {
    "applet", "caption", "html", "table", "td", "th", "marquee", "object", "template",
}  # fmt: skip
_TABLE_CONTEXT = {"table", "tbody", "thead", "tfoot", "tr"}
_IGNORED_IN_BODY = {
    "caption", "col", "colgroup", "frame", "head", "tbody", "td", "tfoot", "th", "thead", "tr",
}  # fmt: skip
_FOREIGN_BREAKOUT = {
    "b", "big", "blockquote", "body", "br", "center", "code", "dd", "div", "dl", "dt", "em",
    "embed", "h1", "h2", "h3", "h4", "h5", "h6", "head", "hr", "i", "img", "li", "listing",
    "menu", "meta", "nobr", "ol", "p", "pre", "ruby", "s", "small", "span", "strong", "strike",
    "sub", "sup", "table", "tt", "u", "ul", "var",
}  # fmt: skip
_FORMATTING = {
    "a", "b", "big", "code", "em", "font", "i", "nobr", "s", "small", "strike", "strong",
    "tt", "u",
}  # fmt: skip
_BLOCK_END = {
    "address", "article", "aside", "blockquote", "button", "center", "details", "dialog",
    "dir", "div", "dl", "fieldset", "figcaption", "figure", "footer", "header", "hgroup",
    "listing", "main", "menu", "nav", "ol", "pre", "search", "section", "summary", "ul",
    "li", "dd", "dt", "form", "applet", "marquee", "object",
    "h1", "h2", "h3", "h4", "h5", "h6",
}  # fmt: skip
_FRAMESET_NOT_OK = {
    "pre", "listing", "li", "dd", "dt", "button", "applet", "marquee", "object", "table",
    "area", "br", "embed", "img", "keygen", "wbr", "input", "hr", "textarea", "xmp",
    "iframe", "select", "image",
}  # fmt: skip
_MATHML_TEXT_INTEGRATION = {"mi", "mo", "mn", "ms", "mtext"}
_HTML_INTEGRATION = {"foreignObject", "desc", "title"}


class _Tokenizer:
    """HTML5 tokenizer subset. Tokens: ("start", name, attrs, self_closing), ("end", name),
    ("chars", text), ("comment",), ("doctype",). Incomplete tags at EOF are dropped."""

    def __init__(self, src: str) -> None:
        self.s = src
        self.i = 0
        self.raw_end_tag: str | None = None  # set by the tree builder after rawtext starts
        self.raw_rcdata = False
        self.plaintext = False
        self.allow_cdata = False

    def _read_until_tag_end(self, j: int) -> tuple[dict[str, str], bool, int] | None:
        """Parse attributes from j to '>'. Returns (attrs, self_closing, index_after)."""
        s, n = self.s, len(self.s)
        attrs: dict[str, str] = {}
        while True:
            while j < n and s[j] in _HTML_WS:
                j += 1
            if j >= n:
                return None
            c = s[j]
            if c == ">":
                return attrs, False, j + 1
            if c == "/":
                if j + 1 < n and s[j + 1] == ">":
                    return attrs, True, j + 2
                j += 1
                continue
            k = j + 1
            while k < n and s[k] not in _HTML_WS and s[k] not in "/>=":
                k += 1
            name = s[j:k].lower()
            j = k
            while j < n and s[j] in _HTML_WS:
                j += 1
            value = ""
            if j < n and s[j] == "=":
                j += 1
                while j < n and s[j] in _HTML_WS:
                    j += 1
                if j >= n:
                    return None
                if s[j] in "\"'":
                    end = s.find(s[j], j + 1)
                    if end == -1:
                        return None
                    value = s[j + 1 : end]
                    j = end + 1
                else:
                    k = j
                    while k < n and s[k] not in _HTML_WS and s[k] != ">":
                        k += 1
                    value = s[j:k]
                    j = k
            attrs.setdefault(name, value)

    def next(self) -> tuple[Any, ...] | None:
        s, n = self.s, len(self.s)
        i = self.i
        if i >= n:
            return None
        if self.plaintext:
            self.i = n
            return ("rawchars", s[i:])
        if self.raw_end_tag is not None:
            name = self.raw_end_tag
            j = i
            end_re = re.compile("</" + re.escape(name), re.IGNORECASE)
            while True:
                m = end_re.search(s, j)
                k = m.start() if m else -1
                if k == -1:
                    self.i = n
                    text = s[i:]
                    break
                after = k + 2 + len(name)
                if after >= n or s[after] in _HTML_WS + "/>":
                    self.i = k
                    text = s[i:k]
                    break
                j = k + 1
            self.raw_end_tag = None
            if text:
                return ("rawchars", _html.unescape(text) if self.raw_rcdata else text)
            return self.next()
        if s[i] != "<":
            k = s.find("<", i)
            k = n if k == -1 else k
            self.i = k
            return ("chars", _html.unescape(s[i:k]))
        c = s[i + 1] if i + 1 < n else ""
        if c.isascii() and c.isalpha():
            k = i + 1
            while k < n and s[k] not in _HTML_WS and s[k] not in "/>":
                k += 1
            name = s[i + 1 : k].lower()
            res = self._read_until_tag_end(k)
            if res is None:
                self.i = n
                return None
            self.i = res[2]
            return ("start", name, res[0], res[1])
        if c == "/":
            d = s[i + 2] if i + 2 < n else ""
            if d.isascii() and d.isalpha():
                k = i + 2
                while k < n and s[k] not in _HTML_WS and s[k] not in "/>":
                    k += 1
                name = s[i + 2 : k].lower()
                res = self._read_until_tag_end(k)
                if res is None:
                    self.i = n
                    return None
                self.i = res[2]
                return ("end", name)
            if d == ">":
                self.i = i + 3
                return self.next()
            if d == "":
                self.i = n
                return ("chars", "</")
            return self._bogus_comment(i + 2)
        if c == "!":
            if s.startswith("--", i + 2):
                if s.startswith(">", i + 4):
                    self.i = i + 5
                elif s.startswith("->", i + 4):
                    self.i = i + 6
                else:
                    end = s.find("-->", i + 4)
                    self.i = n if end == -1 else end + 3
                return ("comment",)
            if s[i + 2 : i + 9].upper() == "DOCTYPE":
                end = s.find(">", i + 9)
                self.i = n if end == -1 else end + 1
                body = s[i + 9 : n if end == -1 else end]
                return ("doctype", body.strip(_HTML_WS).lower())
            if self.allow_cdata and s.startswith("[CDATA[", i + 2):
                end = s.find("]]>", i + 9)
                self.i = n if end == -1 else end + 3
                return ("chars", s[i + 9 : n if end == -1 else end])
            return self._bogus_comment(i + 2)
        if c == "?":
            return self._bogus_comment(i + 1)
        self.i = i + 1
        return ("chars", "<")

    def _bogus_comment(self, j: int) -> tuple[Any, ...]:
        end = self.s.find(">", j)
        self.i = len(self.s) if end == -1 else end + 1
        return ("comment",)


class _TreeBuilder:
    def __init__(self, src: str) -> None:
        self.tok = _Tokenizer(src)
        self.document = _Element("#document", has_location=False)
        self.stack: list[_Element] = []
        self.head: _Element | None = None
        self.mode = "initial"
        self.reopen: list[str] = []
        self.frameset_ok = True
        self.quirks = True

    # -- insertion helpers --------------------------------------------------------------------

    @property
    def current(self) -> _Element:
        return self.stack[-1] if self.stack else self.document

    def _children_of(self, el: _Element) -> list[_Element | _Text]:
        return el.template_content if el.template_content is not None else el.children

    def _insert_node(self, node: _Element | _Text, foster: bool = False) -> None:
        target = self.current
        if foster and target.ns == _HTML_NS and target.tag_name in _TABLE_CONTEXT:
            table_idx = next(
                (
                    k
                    for k in range(len(self.stack) - 1, -1, -1)
                    if self.stack[k].tag_name == "table"
                ),
                None,
            )
            if table_idx is not None and table_idx > 0:
                table = self.stack[table_idx]
                parent = self._children_of(self.stack[table_idx - 1])
                pos = next((k for k, ch in enumerate(parent) if ch is table), len(parent))
                parent.insert(pos, node)
                return
        self._children_of(target).append(node)

    def _insert_text(self, text: str, foster: bool = False) -> None:
        if text:
            self._insert_node(_Text(text), foster)

    def _insert_element(
        self,
        name: str,
        ns: str = _HTML_NS,
        has_location: bool = True,
        push: bool = True,
        foster: bool = False,
    ) -> _Element:
        el = _Element(name, ns, has_location)
        self._insert_node(el, foster)
        if push:
            self.stack.append(el)
        return el

    def _in_scope(self, name: str, extra: frozenset[str] | set[str] = frozenset()) -> bool:
        for el in reversed(self.stack):
            if el.tag_name == name and el.ns == _HTML_NS:
                return True
            if el.ns != _HTML_NS or el.tag_name in _SCOPE_BOUNDARY or el.tag_name in extra:
                return False
        return False

    def _pop_until(self, name: str) -> None:
        while self.stack:
            el = self.stack.pop()
            if el.tag_name == name and el.ns == _HTML_NS:
                return

    def _close_p(self) -> None:
        if self._in_scope("p", {"button"}):
            self._pop_until("p")

    def _start_raw(self, name: str) -> None:
        if name == "plaintext":
            self.tok.plaintext = True
        elif name in _RAWTEXT or name in _RCDATA:
            self.tok.raw_end_tag = name
            self.tok.raw_rcdata = name in _RCDATA

    # -- driver -------------------------------------------------------------------------------

    def run(self) -> _Element:
        while True:
            node = self.current
            self.tok.allow_cdata = node.ns != _HTML_NS
            token = self.tok.next()
            if token is None:
                break
            self._process(token)
        # EOF: make sure html/head/body exist, as parse5 does.
        if self.mode in ("initial", "before_html"):
            self._process(("eof",))
        if self.mode == "before_head":
            self._process(("eof",))
        if self.mode in ("in_head", "after_head"):
            self._process(("eof",))
        return self.document

    def _process(self, token: tuple[Any, ...]) -> None:
        if token[0] == "rawchars":  # text inside a raw-text element goes to that element
            self._children_of(self.current).append(_Text(token[1]))
            return
        while True:
            handler = getattr(self, "_mode_" + self.mode)
            if not handler(token):
                return

    @staticmethod
    def _split_ws(token: tuple[Any, ...]) -> tuple[str, str]:
        text = token[1]
        stripped = text.lstrip(_HTML_WS)
        return text[: len(text) - len(stripped)], stripped

    # Each mode returns True to reprocess the token in the (new) current mode.

    def _mode_initial(self, token: tuple[Any, ...]) -> bool:
        kind = token[0]
        if kind == "comment":
            return False
        if kind == "chars":
            _, rest = self._split_ws(token)
            if not rest:
                return False
            token = ("chars", rest)
        if kind == "doctype":
            # Only `<!DOCTYPE html>` (no public/system id) selects no-quirks mode.
            self.quirks = token[1] != "html"
            self.mode = "before_html"
            return False
        self.mode = "before_html"
        return self._mode_before_html(token)

    def _mode_before_html(self, token: tuple[Any, ...]) -> bool:
        kind = token[0]
        if kind in ("comment", "doctype"):
            return False
        if kind == "chars":
            _, rest = self._split_ws(token)
            if not rest:
                return False
            token = ("chars", rest)
        if kind == "start" and token[1] == "html":
            self._insert_element("html")
            self.mode = "before_head"
            return False
        if kind == "end" and token[1] not in ("head", "body", "html", "br"):
            return False
        self._insert_element("html", has_location=False)
        self.mode = "before_head"
        return self._process_again(token)

    def _process_again(self, token: tuple[Any, ...]) -> bool:
        self._process(token)
        return False

    def _mode_before_head(self, token: tuple[Any, ...]) -> bool:
        kind = token[0]
        if kind in ("comment", "doctype"):
            return False
        if kind == "chars":
            _, rest = self._split_ws(token)
            if not rest:
                return False
            token = ("chars", rest)
        if kind == "start" and token[1] == "html":
            return False
        if kind == "start" and token[1] == "head":
            self.head = self._insert_element("head")
            self.mode = "in_head"
            return False
        if kind == "end" and token[1] not in ("head", "body", "html", "br"):
            return False
        self.head = self._insert_element("head", has_location=False)
        self.mode = "in_head"
        return self._process_again(token)

    def _mode_in_head(self, token: tuple[Any, ...]) -> bool:
        kind = token[0]
        if kind in ("comment", "doctype"):
            return False
        if kind == "chars":
            ws, rest = self._split_ws(token)
            self._insert_text(ws)
            if not rest:
                return False
            token = ("chars", rest)
        elif kind == "start":
            name = token[1]
            if name == "html":
                return False
            if name in _HEAD_ELEMENTS:
                self._insert_element(name, push=name not in _VOID)
                self._start_raw(name)
                return False
            if name == "head":
                return False
        elif kind == "end":
            name = token[1]
            if name in ("head", "noscript", "template"):
                if self.current.tag_name == name:
                    self.stack.pop()
                if name == "head":
                    self.mode = "after_head"
                return False
            if name in ("title", "style", "script", "noframes"):
                if self.current.tag_name == name:
                    self.stack.pop()
                return False
            if name not in ("body", "html", "br"):
                return False
        # anything else: pop head
        if self.current is self.head:
            self.stack.pop()
        self.mode = "after_head"
        return self._process_again(token) if kind != "eof" else self._mode_after_head(token)

    def _mode_after_head(self, token: tuple[Any, ...]) -> bool:
        kind = token[0]
        if kind in ("comment", "doctype"):
            return False
        if kind == "chars":
            ws, rest = self._split_ws(token)
            self._insert_text(ws)
            if not rest:
                return False
            token = ("chars", rest)
        elif kind == "start":
            name = token[1]
            if name == "html":
                return False
            if name == "body":
                self._insert_element(name)
                self.frameset_ok = False
                self.mode = "in_body"
                return False
            if name == "frameset":
                self._insert_element(name)
                self.mode = "in_frameset"
                return False
            if name in _HEAD_ELEMENTS and self.head is not None:
                self.stack.append(self.head)
                self._mode_in_head(token)
                if self.head in self.stack:
                    self.stack.remove(self.head)
                return False
            if name == "head":
                return False
        elif kind == "end" and token[1] not in ("body", "html", "br"):
            return False
        self._insert_element("body", has_location=False)
        self.mode = "in_body"
        return self._process_again(token) if kind != "eof" else False

    def _foreign_current(self) -> bool:
        node = self.current
        if node.ns == _HTML_NS:
            return False
        if node.ns == _SVG_NS and node.tag_name in _HTML_INTEGRATION:
            return False
        return not (node.ns == _MATH_NS and node.tag_name in _MATHML_TEXT_INTEGRATION)

    def _mode_in_body(self, token: tuple[Any, ...]) -> bool:
        kind = token[0]
        if kind in ("comment", "doctype", "eof"):
            return False
        if kind == "chars":
            text = token[1]
            in_table = self.current.ns == _HTML_NS and self.current.tag_name in _TABLE_CONTEXT
            if text.strip(_HTML_WS):
                self.frameset_ok = False
                if not in_table:
                    self._reconstruct_formatting()
            if in_table and text.strip(_HTML_WS):
                self._insert_text(text, foster=True)
            else:
                self._insert_text(text)
            return False
        name = token[1]
        if self._foreign_current():
            if kind == "start" and name not in _FOREIGN_BREAKOUT:
                self._insert_element(name, self.current.ns, push=not token[3])
                return False
            if kind == "end":
                for idx in range(len(self.stack) - 1, -1, -1):
                    el = self.stack[idx]
                    if el.ns == _HTML_NS:
                        break
                    if el.tag_name.lower() == name:
                        del self.stack[idx:]
                        return False
            if kind == "start":  # breakout: pop foreign content, reprocess as HTML
                while self.stack and self._foreign_current():
                    self.stack.pop()
        if kind == "start":
            self._start_tag_in_body(name, token[3])
        else:
            self._end_tag_in_body(name)
        return False

    def _start_tag_in_body(self, name: str, self_closing: bool) -> None:
        cur = self.current
        in_table = cur.ns == _HTML_NS and cur.tag_name in _TABLE_CONTEXT
        if in_table:
            if name == "caption" or name == "colgroup":
                self._insert_element(name)
                return
            if name == "col":
                self._insert_element(name, push=False)
                return
            if name in ("tbody", "thead", "tfoot"):
                while self.current.tag_name in ("tbody", "thead", "tfoot", "tr"):
                    self.stack.pop()
                self._insert_element(name)
                return
            if name == "tr":
                while self.current.tag_name == "tr":
                    self.stack.pop()
                if self.current.tag_name == "table":
                    self._insert_element("tbody", has_location=False)
                self._insert_element(name)
                return
            if name in ("td", "th"):
                if self.current.tag_name == "table":
                    self._insert_element("tbody", has_location=False)
                if self.current.tag_name != "tr":
                    self._insert_element("tr", has_location=False)
                self._insert_element(name)
                return
            if name in ("style", "script", "template"):
                self._insert_element(name)
                self._start_raw(name)
                return
            if name in ("input", "form"):
                self._insert_element(name, push=False)
                return
            if name == "table":
                self._pop_until("table")
                self._insert_element("table")
                return
        foster = in_table
        if name == "frameset":
            body = self.stack[1] if len(self.stack) > 1 else None
            in_template = any(e.tag_name == "template" for e in self.stack)
            if (
                self.frameset_ok
                and not in_template
                and body is not None
                and body.tag_name == "body"
            ):
                html_el = self.stack[0]
                html_el.children.remove(body)
                del self.stack[1:]
                self._insert_element(name)
                self.mode = "in_frameset"
            return
        if name in _FRAMESET_NOT_OK:
            self.frameset_ok = False
        if name == "body":
            self.frameset_ok = False
            return
        if name == "html":
            return
        if not foster:
            self._reconstruct_formatting()
        if name in _HEAD_ELEMENTS:
            self._insert_element(name, push=name not in _VOID, foster=foster)
            self._start_raw(name)
            return
        if name in _IGNORED_IN_BODY:
            return
        if name in _CLOSES_P and not foster and not (name == "table" and self.quirks):
            self._close_p()
            if name in _HEADINGS and self.current.tag_name in _HEADINGS:
                self.stack.pop()
        if name in ("li", "dd", "dt"):
            self._close_list_item(name)
        if name in ("option", "optgroup") and self.current.tag_name == "option":
            self.stack.pop()
        if name == "image":
            name = "img"
        if name in ("svg", "math"):
            self._insert_element(name, name, push=not self_closing, foster=foster)
            return
        self._insert_element(name, push=name not in _VOID, foster=foster)
        if name in _RAWTEXT or name in _RCDATA or name == "plaintext":
            self._start_raw(name)

    def _reconstruct_formatting(self) -> None:
        reopen, self.reopen = self.reopen, []
        for name in reopen:
            self._insert_element(name, has_location=False)

    def _close_list_item(self, name: str) -> None:
        targets = {"li"} if name == "li" else {"dd", "dt"}
        for idx in range(len(self.stack) - 1, -1, -1):
            el = self.stack[idx]
            if el.tag_name in targets and el.ns == _HTML_NS:
                del self.stack[idx:]
                return
            if el.tag_name in _SPECIAL and el.tag_name not in ("address", "div", "p"):
                return

    def _end_tag_in_body(self, name: str) -> None:
        if name in ("body", "html"):
            if self._in_scope("body"):
                self.mode = "after_body"
            return
        if name == "p" and not self._in_scope("p", {"button"}):
            self._insert_element("p", has_location=False)
        if name == "br":
            self._insert_element("br", has_location=False, push=False)
            return
        if name in self.reopen:
            self.reopen.remove(name)
            return
        if name in _BLOCK_END:
            if name in _HEADINGS:
                if any(self._in_scope(h) for h in _HEADINGS):
                    while self.stack:
                        el = self.stack.pop()
                        if el.tag_name in _HEADINGS and el.ns == _HTML_NS:
                            break
                return
            extra = {"ol", "ul"} if name == "li" else set()
            if self._in_scope(name, extra):
                self._pop_until(name)
            return
        # "any other end tag" (also used for table and formatting end tags). Closing a
        # formatting element re-opens the formatting elements nested inside it, a reduced
        # form of the adoption agency + "reconstruct the active formatting elements".
        for idx in range(len(self.stack) - 1, -1, -1):
            el = self.stack[idx]
            if el.tag_name == name and el.ns == _HTML_NS:
                if name in _FORMATTING:
                    self.reopen.extend(
                        e.tag_name
                        for e in self.stack[idx + 1 :]
                        if e.ns == _HTML_NS and e.tag_name in _FORMATTING
                    )
                del self.stack[idx:]
                return
            if el.tag_name in _SPECIAL and el.ns == _HTML_NS:
                return

    def _mode_in_frameset(self, token: tuple[Any, ...]) -> bool:
        kind = token[0]
        if kind == "chars":
            ws = "".join(c for c in token[1] if c in _HTML_WS)
            self._insert_text(ws)
        elif kind == "start" and token[1] == "frameset" and self.current.tag_name == "frameset":
            self._insert_element("frameset")
        elif kind == "start" and token[1] == "frame" and self.current.tag_name == "frameset":
            self._insert_element("frame", push=False)
        elif kind == "start" and token[1] == "noframes":
            self._insert_element("noframes")
            self._start_raw("noframes")
        elif kind == "end" and token[1] == "frameset" and self.current.tag_name == "frameset":
            if len(self.stack) > 1:
                self.stack.pop()
        return False

    def _mode_after_body(self, token: tuple[Any, ...]) -> bool:
        kind = token[0]
        if kind in ("comment", "doctype", "eof"):
            return False
        if kind == "chars" and not token[1].strip(_HTML_WS):
            return self._mode_in_body(token)
        if kind == "start" and token[1] == "html":
            return False
        if kind == "end" and token[1] == "html":
            return False
        self.mode = "in_body"
        return True


def _parse_html(src: str) -> _Element:
    return _TreeBuilder(src).run()


def _iter_children(node: _Element | _Text) -> list[_Element | _Text]:
    return node.children if isinstance(node, _Element) else []


def _find_first_element(root: _Element, predicate: Callable[[_Element], bool]) -> _Element | None:
    stack: list[_Element | _Text] = [root]
    while stack:
        current = stack.pop()
        if isinstance(current, _Element) and current is not root and predicate(current):
            return current
        stack.extend(reversed(_iter_children(current)))
    return None


_LITERAL_WRAPPER_PATTERNS = {
    "html": re.compile(r"<html(?=" + _JS_S + r"|[>/])"),
    "head": re.compile(r"<head(?=" + _JS_S + r"|[>/])"),
    "body": re.compile(r"<body(?=" + _JS_S + r"|[>/])"),
}

# fmt: off
_VALID_HTML_ELEMENTS = {
    # Document metadata
    "html", "head", "title", "base", "link", "meta", "style",
    # Content sectioning
    "body", "article", "section", "nav", "aside", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "footer", "address",
    # Text content
    "p", "hr", "pre", "blockquote", "ol", "ul", "li", "dl", "dt", "dd", "figure",
    "figcaption", "main", "div",
    # Inline text semantics
    "a", "em", "strong", "small", "s", "cite", "q", "dfn", "abbr", "ruby", "rt", "rp", "data",
    "time", "code", "var", "samp", "kbd", "sub", "sup", "i", "b", "u", "mark", "bdi", "bdo",
    "span", "br", "wbr",
    # Image and multimedia
    "img", "iframe", "embed", "object", "param", "picture", "source", "video", "audio",
    "track", "map", "area",
    # Embedded content
    "svg", "math",
    # Scripting
    "script", "noscript", "canvas",
    # Demarcating edits
    "ins", "del",
    # Table content
    "table", "caption", "colgroup", "col", "tbody", "thead", "tfoot", "tr", "td", "th",
    # Forms
    "form", "label", "input", "button", "select", "datalist", "optgroup", "option",
    "textarea", "output", "progress", "meter", "fieldset", "legend",
    # Interactive elements
    "details", "summary", "dialog", "menu",
    # Web Components
    "slot", "template",
}
# fmt: on


def _has_top_level_text(parent: _Element) -> bool:
    return any(isinstance(node, _Text) and _js_trim(node.value) for node in parent.children)


def _is_user_provided_element(element: _Element, input_lowercase: str) -> bool:
    tag_name = element.tag_name.lower()
    if tag_name in _LITERAL_WRAPPER_PATTERNS:
        return (
            _LITERAL_WRAPPER_PATTERNS[tag_name].search(input_lowercase) is not None
            and element.has_location
        )
    return tag_name in _VALID_HTML_ELEMENTS or "-" in tag_name


_XML_DECL_RE = re.compile(r"^" + _JS_S + r"*<\?xml", re.IGNORECASE)


def _validate_html(html_string: str) -> dict[str, Any]:
    trimmed = _js_trim(html_string)
    if not trimmed:
        return {"isValid": False, "reason": "Output is empty"}
    if _exceeds_max_size(trimmed):
        return {"isValid": False, "reason": "Output exceeds maximum size limit"}
    if _XML_DECL_RE.search(trimmed):
        return {"isValid": False, "reason": "Output appears to be XML, not HTML"}

    document = _parse_html(trimmed)
    input_lowercase = trimmed.lower()
    body = _find_first_element(document, lambda el: el.tag_name == "body")
    has_user_provided_body = (
        body is not None
        and _LITERAL_WRAPPER_PATTERNS["body"].search(input_lowercase) is not None
        and body.has_location
    )
    if not has_user_provided_body and body is not None and _has_top_level_text(body):
        return {"isValid": False, "reason": "Output must be wrapped in HTML tags"}

    found = _find_first_element(document, lambda el: _is_user_provided_element(el, input_lowercase))
    if found is None:
        return {"isValid": False, "reason": "Output does not contain recognized HTML elements"}
    return {"isValid": True, "reason": "Output is valid HTML"}


def handle_contains_html(params: AssertionParams) -> GradingResult:
    has_html = _contains_html(params.output_string)
    passed = has_html != params.inverse
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": "Assertion passed"
        if passed
        else f"Expected output to {'not ' if params.inverse else ''}contain HTML content",
        "assertion": params.assertion,
    }


def handle_is_html(params: AssertionParams) -> GradingResult:
    res = _validate_html(params.output_string)
    passed = res["isValid"] != params.inverse
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": "Assertion passed" if passed else res["reason"],
        "assertion": params.assertion,
    }


HANDLERS: dict[str, Handler] = {
    "is-xml": handle_is_xml,
    "contains-xml": handle_is_xml,
    "is-html": handle_is_html,
    "contains-html": handle_contains_html,
}
