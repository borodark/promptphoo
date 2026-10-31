"""Terminal colour helpers that emit the same escape sequences as chalk and @colors/colors.

promptfoo colours CLI output with chalk (eval summary, table cells) and cli-table3 colours table
borders and headers with @colors/colors. They nest styles differently, so both are reproduced:

- `chalk(text, "red", "bold")` is `chalk.red.bold(text)`.
- `colors(text, ["blue", "bold"])` is cli-table3's `wrapWithStyleColors` for `style.head`.

Colour is on when the terminal supports it, using the supports-color rules both libraries share
(FORCE_COLOR, then a TTY whose TERM is not `dumb`).
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence

ANSI_STYLES: dict[str, tuple[int, int]] = {
    "bold": (1, 22),
    "dim": (2, 22),
    "italic": (3, 23),
    "underline": (4, 24),
    "inverse": (7, 27),
    "strikethrough": (9, 29),
    "black": (30, 39),
    "red": (31, 39),
    "green": (32, 39),
    "yellow": (33, 39),
    "blue": (34, 39),
    "magenta": (35, 39),
    "cyan": (36, 39),
    "white": (37, 39),
    "gray": (90, 39),
    "grey": (90, 39),
}


def _code(n: int) -> str:
    return f"\x1b[{n}m"


def color_enabled() -> bool:
    """supports-color: FORCE_COLOR wins, otherwise a non-dumb TTY on stdout."""
    force = os.environ.get("FORCE_COLOR")
    if force is not None:
        if force in ("", "true"):
            return True
        if force == "false":
            return False
        try:
            return int(force) > 0
        except ValueError:
            return True
    if "--no-color" in sys.argv or "--no-colors" in sys.argv:
        return False
    try:
        tty = sys.stdout.isatty()
    except (AttributeError, ValueError):
        tty = False
    return tty and os.environ.get("TERM") != "dumb"


def table_color_enabled() -> bool:
    """Table borders/headers (@colors/colors in promptfoo's CLI) are coloured even when stdout is
    piped: something in promptfoo's process enables @colors globally. Observed against 0.125.1,
    only FORCE_COLOR=0 (parseInt) or --no-color turns them off; NO_COLOR does not."""
    force = os.environ.get("FORCE_COLOR")
    if force is not None:
        # `FORCE_COLOR.length === 0 || parseInt(FORCE_COLOR, 10) !== 0` (NaN counts as on)
        m = re.match(r"\s*[+-]?\d+", force)
        return force == "" or m is None or int(m.group()) != 0
    return not any(flag in sys.argv for flag in ("--no-color", "--no-colors", "--color=false"))


def _encase_crlf(string: str, prefix: str, postfix: str) -> str:
    """chalk's stringEncaseCRLFWithFirstIndex: close styles before each line break."""
    out = []
    start = 0
    index = string.find("\n")
    while index != -1:
        got_cr = index > 0 and string[index - 1] == "\r"
        out.append(string[start : index - 1 if got_cr else index])
        out.append(prefix + ("\r\n" if got_cr else "\n") + postfix)
        start = index + 1
        index = string.find("\n", start)
    out.append(string[start:])
    return "".join(out)


def chalk(text: str, *styles: str, enabled: bool | None = None) -> str:
    """chalk.<styles...>(text), e.g. chalk(text, "red", "bold") == chalk.red.bold(text)."""
    if enabled is None:
        enabled = color_enabled()
    if not enabled or not text:
        return text
    open_all = "".join(_code(ANSI_STYLES[s][0]) for s in styles)
    close_all = "".join(_code(ANSI_STYLES[s][1]) for s in reversed(styles))
    if "\x1b" in text:
        for s in reversed(styles):
            o, c = ANSI_STYLES[s]
            text = text.replace(_code(c), _code(o))
    if "\n" in text:
        text = _encase_crlf(text, close_all, open_all)
    return open_all + text + close_all


def colors(text: str, styles: Sequence[str], enabled: bool | None = None) -> str:
    """@colors/colors applied the way cli-table3's wrapWithStyleColors builds the chain."""
    if not styles:
        return text
    if enabled is None:
        enabled = table_color_enabled()
    if not enabled or not text:
        return text
    # wrapWithStyleColors walks styles from last to first building `colors.a.b`, so the
    # builder's style list is reversed; applyStyle then wraps from the end of that list.
    newlines = "\n" in text
    for name in styles:
        o, c = (_code(n) for n in ANSI_STYLES[name])
        text = o + text.replace(c, o) + c
        if newlines:
            text = re.sub(r"\r?\n", lambda m, o=o, c=c: c + m.group(0) + o, text)
    return text
