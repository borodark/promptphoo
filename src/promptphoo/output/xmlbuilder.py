"""`new XMLBuilder(options).build(obj)` from fast-xml-parser 5.7.1 (fast-xml-builder 1.3.1).

Only the non-`preserveOrder` path is ported, with the options promptfoo uses: `format`,
`indentBy` and `ignoreAttributes: false` (keys prefixed `@_` become attributes and `#text` is the
text node). Entity escaping, empty-tag rendering, boolean-attribute suppression and the
`?xml` declaration handling follow the JS builder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from promptphoo.jsvalues import UNDEFINED, js_string
from promptphoo.output.jsyaml import v8_items

_ENTITIES = (("&", "&amp;"), (">", "&gt;"), ("<", "&lt;"), ("'", "&apos;"), ('"', "&quot;"))


def _val_to_str(value: Any) -> str:
    if isinstance(value, float) and value == 0 and math.copysign(1, value) < 0:
        return "-0"
    return js_string(value)


class XMLBuilder:
    def __init__(
        self,
        *,
        format: bool = False,  # noqa: A002 - fast-xml-parser option name
        indent_by: str = "  ",
        ignore_attributes: bool = True,
        attribute_name_prefix: str = "@_",
        text_node_name: str = "#text",
        suppress_empty_node: bool = False,
        suppress_boolean_attributes: bool = True,
        process_entities: bool = True,
        max_nested_tags: int = 100,
    ) -> None:
        self.format = format
        self.indent_by = indent_by
        self.ignore_attributes = ignore_attributes
        self.prefix = attribute_name_prefix
        self.text_node_name = text_node_name
        self.suppress_empty_node = suppress_empty_node
        self.suppress_boolean_attributes = suppress_boolean_attributes
        self.process_entities = process_entities
        self.max_nested_tags = max_nested_tags
        self.tag_end_char = ">\n" if format else ">"

    def build(self, obj: Any) -> str:
        return self._j2x(obj, 0)[1]

    # --- helpers --------------------------------------------------------------------------

    def _indent(self, level: int) -> str:
        return self.indent_by * level if self.format else ""

    def _attr_name(self, key: str) -> str | None:
        if self.ignore_attributes:
            return None
        if key.startswith(self.prefix) and key != self.text_node_name:
            return key[len(self.prefix) :]
        return None

    def _replace_entities(self, text: str) -> str:
        if text and self.process_entities:
            for raw, entity in _ENTITIES:
                text = text.replace(raw, entity)
        return text

    def _attr_pair(self, name: str, value: str) -> str:
        value = self._replace_entities(_val_to_str(value))
        if self.suppress_boolean_attributes and value == "true":
            return " " + name
        return f' {name}="' + value.replace('"', "&quot;").replace("'", "&apos;") + '"'

    def _close_tag(self, key: str) -> str:
        return "/" if self.suppress_empty_node else f"></{key}"

    @staticmethod
    def _items(obj: Any) -> list[tuple[str, Any]]:
        if isinstance(obj, Mapping):
            return v8_items(dict(obj))
        if isinstance(obj, list | tuple):
            return [(str(i), v) for i, v in enumerate(obj)]
        if isinstance(obj, str):
            return [(str(i), ch) for i, ch in enumerate(obj)]
        return []

    # --- fast-xml-builder j2x -------------------------------------------------------------

    def _j2x(self, obj: Any, level: int) -> tuple[str, str]:
        if self.max_nested_tags and level >= self.max_nested_tags:
            raise ValueError("Maximum nested tags exceeded")
        attr_str = ""
        val = ""
        for key, value in self._items(obj):
            attr = self._attr_name(key)
            if value is UNDEFINED:
                continue
            if value is None:
                if attr is not None:
                    continue
                if key.startswith("?"):
                    val += self._indent(level) + "<" + key + "?" + self.tag_end_char
                else:
                    val += self._indent(level) + "<" + key + "/" + self.tag_end_char
            elif not isinstance(value, Mapping | list | tuple):
                if attr is not None:
                    attr_str += self._attr_pair(attr, _val_to_str(value))
                elif key == self.text_node_name:
                    val += self._replace_entities(_val_to_str(value))
                else:
                    val += self._text_val_node(value, key, "", level)
            elif isinstance(value, list | tuple):
                for item in value:
                    if item is UNDEFINED:
                        continue
                    if item is None:
                        end = "?" if key.startswith("?") else "/"
                        val += self._indent(level) + "<" + key + end + self.tag_end_char
                    elif isinstance(item, Mapping | list | tuple):
                        val += self._text_or_obj_node(item, key, level)
                    else:
                        val += self._text_val_node(item, key, "", level)
            else:
                val += self._text_or_obj_node(value, key, level)
        return attr_str, val

    def _text_or_obj_node(self, obj: Any, key: str, level: int) -> str:
        attr_str, val = self._j2x(obj, level + 1)
        if key.startswith("?"):
            return self._text_val_node("", key, attr_str, level)
        if (
            isinstance(obj, Mapping)
            and obj.get(self.text_node_name, UNDEFINED) is not UNDEFINED
            and len(obj) == 1
        ):
            return self._text_val_node(obj[self.text_node_name], key, attr_str, level)
        return self._object_node(val, key, attr_str, level)

    def _object_node(self, val: str, key: str, attr_str: str, level: int) -> str:
        if val == "":
            if key.startswith("?"):
                return self._indent(level) + "<" + key + attr_str + "?" + self.tag_end_char
            return (
                self._indent(level)
                + "<"
                + key
                + attr_str
                + self._close_tag(key)
                + (self.tag_end_char)
            )
        if key.startswith("?"):
            return self._indent(level) + "<" + key + attr_str + "?" + self.tag_end_char
        tag_end = "</" + key + self.tag_end_char
        if "<" not in val:
            return self._indent(level) + "<" + key + attr_str + ">" + val + tag_end
        return (
            self._indent(level)
            + "<"
            + key
            + attr_str
            + self.tag_end_char
            + val
            + self._indent(level)
            + tag_end
        )

    def _text_val_node(self, value: Any, key: str, attr_str: str, level: int) -> str:
        if key.startswith("?"):
            return self._indent(level) + "<" + key + attr_str + "?" + self.tag_end_char
        text = _val_to_str(value)
        text = self._replace_entities(text)
        if text == "":
            return (
                self._indent(level)
                + "<"
                + key
                + attr_str
                + self._close_tag(key)
                + (self.tag_end_char)
            )
        return (
            self._indent(level)
            + "<"
            + key
            + attr_str
            + ">"
            + text
            + "</"
            + key
            + (self.tag_end_char)
        )
