"""Console results table.

Ports src/table.ts (`generateTable`, `wrapTable`) together with the parts of cli-table3 0.6.5 it
relies on: fixed column widths, word wrapping on word boundaries, ANSI-aware truncation and the
default box-drawing characters, plus @colors/colors styling of borders and the header row. Row
and column spans are not ported because promptfoo never uses them.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from promptphoo.jsvalues import js_string
from promptphoo.output.ansi import chalk, colors

_ANSI_RE = re.compile(r"\x1b\[(?:\d*;){0,5}\d*m")
_ANSI_CAPTURE_RE = re.compile(r"\x1b\[((?:\d*;){0,5}\d*)m")

CHARS = {
    "top": "─",
    "topMid": "┬",
    "topLeft": "┌",
    "topRight": "┐",
    "bottom": "─",
    "bottomMid": "┴",
    "bottomLeft": "└",
    "bottomRight": "┘",
    "left": "│",
    "leftMid": "├",
    "mid": "─",
    "midMid": "┼",
    "right": "│",
    "rightMid": "┤",
    "middle": "│",
}


def terminal_max_width() -> int:
    """TERMINAL_MAX_WIDTH: the TTY's columns minus 10, else 120."""
    try:
        if sys.stdout.isatty():
            columns = shutil.get_terminal_size((0, 0)).columns or os.get_terminal_size().columns
            if columns > 10:
                return columns - 10
    except (AttributeError, ValueError, OSError):
        pass
    return 120


def ellipsize(value: str, max_len: int) -> str:
    """util/text.ts ellipsize."""
    if len(value) > max_len:
        if max_len < 3:
            return value[: max(0, max_len)]
        return value[: max_len - 3] + "..."
    return value


# --- cli-table3 utils ---------------------------------------------------------------------


def _char_width(ch: str) -> int:
    if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf"):
        return 0
    code = ord(ch)
    if code < 32 or 0x7F <= code < 0xA0:
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _string_width(value: str) -> int:
    return sum(_char_width(ch) for ch in _ANSI_RE.sub("", value))


def strlen(value: str) -> int:
    stripped = _ANSI_RE.sub("", value)
    return max((_string_width(s) for s in stripped.split("\n")), default=0)


def _pad(value: str, length: int) -> str:
    current = strlen(value)
    if length + 1 >= current:
        value = value + " " * max(length - current, 0)
    return value


_CODE_CACHE: dict[str, dict[str, Any]] = {}


def _add_to_code_cache(name: str, on: int, off: int) -> None:
    on_s, off_s = f"\x1b[{on}m", f"\x1b[{off}m"
    _CODE_CACHE[on_s] = {"set": name, "to": True}
    _CODE_CACHE[off_s] = {"set": name, "to": False}
    _CODE_CACHE[name] = {"on": on_s, "off": off_s}


for _name, _on, _off in (
    ("bold", 1, 22),
    ("italics", 3, 23),
    ("underline", 4, 24),
    ("inverse", 7, 27),
    ("strikethrough", 9, 29),
):
    _add_to_code_cache(_name, _on, _off)


def _update_state(state: dict[str, Any], match: re.Match[str]) -> None:
    group = match.group(1)
    try:
        control = int(group.split(";")[0]) if group else 0
    except ValueError:
        control = 0
    if 30 <= control <= 39 or 90 <= control <= 97:
        state["lastForegroundAdded"] = match.group(0)
        return
    if 40 <= control <= 49 or 100 <= control <= 107:
        state["lastBackgroundAdded"] = match.group(0)
        return
    if control == 0:
        state.clear()
        return
    info = _CODE_CACHE.get(match.group(0))
    if info:
        state[info["set"]] = info["to"]


def _read_state(line: str) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for match in _ANSI_CAPTURE_RE.finditer(line):
        _update_state(state, match)
    return state


def _unwind_state(state: dict[str, Any], ret: str) -> str:
    bg = state.pop("lastBackgroundAdded", None)
    fg = state.pop("lastForegroundAdded", None)
    for key, value in state.items():
        if value:
            ret += _CODE_CACHE[key]["off"]
    if bg and bg != "\x1b[49m":
        ret += "\x1b[49m"
    if fg and fg != "\x1b[39m":
        ret += "\x1b[39m"
    return ret


def _rewind_state(state: dict[str, Any], ret: str) -> str:
    bg = state.pop("lastBackgroundAdded", None)
    fg = state.pop("lastForegroundAdded", None)
    for key, value in state.items():
        if value:
            ret = _CODE_CACHE[key]["on"] + ret
    if bg and bg != "\x1b[49m":
        ret = bg + ret
    if fg and fg != "\x1b[39m":
        ret = fg + ret
    return ret


def _truncate_width(value: str, desired: int) -> str:
    if len(value) == strlen(value):
        return value[:desired]
    while strlen(value) > desired:
        value = value[:-1]
    return value


def _truncate_width_with_ansi(value: str, desired: int) -> str:
    split = _ANSI_RE.split(value)
    codes = list(_ANSI_CAPTURE_RE.finditer(value))
    split_index = 0
    ret_len = 0
    ret = ""
    state: dict[str, Any] = {}
    while ret_len < desired:
        match = codes[split_index] if split_index < len(codes) else None
        to_add = split[split_index] if split_index < len(split) else ""
        split_index += 1
        if ret_len + strlen(to_add) > desired:
            to_add = _truncate_width(to_add, desired - ret_len)
        ret += to_add
        ret_len += strlen(to_add)
        if ret_len < desired:
            if not match:
                break
            ret += match.group(0)
            _update_state(state, match)
    return _unwind_state(state, ret)


def truncate(value: str, desired: int, truncate_char: str = "…") -> str:
    if strlen(value) <= desired:
        return value
    desired -= strlen(truncate_char)
    return _truncate_width_with_ansi(value, desired) + truncate_char


def _word_wrap(max_length: int, line_input: str) -> list[str]:
    lines: list[str] = []
    split = re.split(r"(\s+)", line_input)
    line: list[str] = []
    line_length = 0
    whitespace: str | None = None
    for i in range(0, len(split), 2):
        word = split[i]
        new_length = line_length + strlen(word)
        if line_length > 0 and whitespace:
            new_length += len(whitespace)
        if new_length > max_length:
            if line_length != 0:
                lines.append("".join(line))
            line = [word]
            line_length = strlen(word)
        else:
            line.extend([whitespace or "", word])
            line_length = new_length
        whitespace = split[i + 1] if i + 1 < len(split) else None
    if line_length:
        lines.append("".join(line))
    return lines


def _text_wrap(max_length: int, line_input: str) -> list[str]:
    lines: list[str] = []
    line = ""
    split = re.split(r"(\s+)", line_input)
    for i in range(0, len(split), 2):
        ws = split[i - 1] if i else ""
        if line and ws:
            line += ws
        line += split[i]
        while len(line) > max_length:
            lines.append(line[:max_length])
            line = line[max_length:]
    if line:
        lines.append(line)
    return lines


def multi_line_word_wrap(max_length: int, value: str, wrap_on_word_boundary: bool = True) -> list:
    handler = _word_wrap if wrap_on_word_boundary else _text_wrap
    out: list[str] = []
    for line in value.split("\n"):
        out.extend(handler(max_length, line))
    return out


def colorize_lines(lines: list[str]) -> list[str]:
    state: dict[str, Any] = {}
    out = []
    for raw in lines:
        line = _rewind_state(state, raw)
        state = _read_state(line)
        out.append(_unwind_state(dict(state), line))
    return out


# --- cli-table3 Table -----------------------------------------------------------------------


class _Cell:
    def __init__(self, content: Any, x: int, y: int) -> None:
        self.content = content if isinstance(content, str) else js_string(content)
        self.x = x
        self.y = y
        self.lines: list[str] = []
        self.width = 0
        self.height = 0
        self.draw_right = False


class CliTable:
    """A minimal cli-table3 `Table`: `CliTable(head=..., col_widths=...)`, `push(row)`, `str()`."""

    def __init__(
        self,
        head: Sequence[str] | None = None,
        col_widths: Sequence[int] | None = None,
        word_wrap: bool = False,
        wrap_on_word_boundary: bool = True,
        head_style: Sequence[str] = ("red",),
        border_style: Sequence[str] = ("grey",),
    ) -> None:
        self.head = list(head or [])
        self.col_widths = list(col_widths or [])
        self.word_wrap = word_wrap
        self.wrap_on_word_boundary = wrap_on_word_boundary
        self.head_style = list(head_style)
        self.border_style = list(border_style)
        self.rows: list[list[Any]] = []

    def push(self, row: Sequence[Any]) -> None:
        self.rows.append(list(row))

    def __str__(self) -> str:
        return self.to_string()

    def _border(self, text: str) -> str:
        return colors(text, self.border_style)

    def to_string(self) -> str:
        headers_present = bool(self.head)
        array = [self.head, *self.rows] if headers_present else list(self.rows)
        head_style = self.head_style if headers_present else []
        cells = [[_Cell(c, x, y) for x, c in enumerate(row)] for y, row in enumerate(array)]

        desired_widths: dict[int, int] = {}
        row_heights: list[int] = []
        for row in cells:
            height = 1
            for cell in row:
                fixed = self.col_widths[cell.x] if cell.x < len(self.col_widths) else None
                if fixed and self.word_wrap:
                    lines = multi_line_word_wrap(
                        fixed - 2, cell.content, self.wrap_on_word_boundary
                    )
                else:
                    lines = cell.content.split("\n")
                cell.lines = colorize_lines(lines)
                desired = strlen(cell.content) + 2
                desired_widths[cell.x] = max(desired_widths.get(cell.x, 0), desired, 1)
                height = max(height, len(cell.lines))
            row_heights.append(height)
        n_cols = max((len(r) for r in cells), default=0)
        widths = []
        for x in range(max(n_cols, len(self.col_widths))):
            fixed = self.col_widths[x] if x < len(self.col_widths) else None
            widths.append(max(1, fixed if isinstance(fixed, int) else desired_widths.get(x, 0)))
        for y, row in enumerate(cells):
            for cell in row:
                cell.width = widths[cell.x]
                cell.height = row_heights[y]
                cell.draw_right = cell.x + 1 == len(widths)

        result: list[str] = []
        for y, row in enumerate(cells):
            self._draw(row, "top", result, head_style)
            for line_num in range(row_heights[y]):
                self._draw(row, line_num, result, head_style)
            if y + 1 == len(cells):
                self._draw(row, "bottom", result, head_style)
        return "\n".join(result)

    def _draw(self, row: list[_Cell], line_num: Any, result: list[str], head: list[str]) -> None:
        text = "".join(self._draw_cell(cell, line_num, head) for cell in row)
        if text:
            result.append(text)

    def _draw_cell(self, cell: _Cell, line_num: Any, head: list[str]) -> str:
        if line_num == "top":
            if cell.y == 0:
                left = "topLeft" if cell.x == 0 else "topMid"
            else:
                left = "leftMid" if cell.x == 0 else "midMid"
            content = CHARS[left] + CHARS["top" if cell.y == 0 else "mid"] * cell.width
            if cell.draw_right:
                content += CHARS["topRight" if cell.y == 0 else "rightMid"]
            return self._border(content)
        if line_num == "bottom":
            left = CHARS["bottomLeft" if cell.x == 0 else "bottomMid"]
            right = CHARS["bottomRight"] if cell.draw_right else ""
            return self._border(left + CHARS["bottom"] * cell.width + right)
        left = CHARS["left" if cell.x == 0 else "middle"]
        right = CHARS["right"] if cell.draw_right else ""
        if line_num >= len(cell.lines):
            content = " " * cell.width
        else:
            line = cell.lines[line_num]
            length = cell.width - 2
            if len(cell.lines) > cell.height and line_num + 1 >= cell.height:
                line += "…"
            content = " " + _pad(truncate(line, length), length) + " "
        if cell.y == 0:
            content = colors(content, head)
        return self._border(left) + content + self._border(right)


# --- src/table.ts ---------------------------------------------------------------------------


def generate_table(
    evaluate_table: Mapping[str, Any], table_cell_max_length: int = 250, max_rows: int = 25
) -> str:
    """generateTable: the console table printed after `promptfoo eval`."""
    from promptphoo.evaluator.run import ResultFailureReason

    head = evaluate_table["head"]
    head_length = len(head["prompts"]) + len(head["vars"])
    width = terminal_max_width() // head_length if head_length else 0
    table = CliTable(
        head=[
            ellipsize(h, table_cell_max_length)
            for h in [
                *head["vars"],
                *(
                    f"[{js_string(p.get('provider'))}] {js_string(p.get('label'))}"
                    for p in head["prompts"]
                ),
            ]
        ],
        col_widths=[width] * head_length,
        word_wrap=True,
        wrap_on_word_boundary=True,
        head_style=["blue", "bold"],
    )
    for row in list(evaluate_table["body"])[:max_rows]:
        cells: list[str] = [ellipsize(js_string(v), table_cell_max_length) for v in row["vars"]]
        for output in row["outputs"]:
            text = ellipsize(js_string(output.get("text")), table_cell_max_length)
            if output.get("pass"):
                cells.append(chalk("[PASS] ", "green") + text)
                continue
            prefix = (
                "[FAIL] "
                if output.get("failureReason") == ResultFailureReason.ASSERT
                else "[ERROR] "
            )
            parts = text.split("---")
            colored = "---".join(
                chalk(c, "red", "bold") if i == 0 else c for i, c in enumerate(parts)
            )
            cells.append(chalk(prefix, "red") + colored)
        table.push(cells)
    return table.to_string()


def wrap_table(
    rows: Sequence[Mapping[str, Any]], column_widths: Mapping[str, int] | None = None
) -> str:
    """wrapTable: a plain word-wrapped table of dict rows (keys of the first row are columns)."""
    if not rows:
        return "No data to display"
    head = list(rows[0].keys())
    default_width = terminal_max_width() // len(head)
    col_widths = [(column_widths or {}).get(c) or default_width for c in head]
    table = CliTable(head=head, col_widths=col_widths, word_wrap=True, wrap_on_word_boundary=True)
    for row in rows:
        table.push(list(row.values()))
    return table.to_string()
