"""Node golden output names promptfoo's commands; promptphoo names its own (D62).

`as_promptphoo` rewrites a Node golden into the spelling promptphoo prints, so the golden
tests still compare everything else byte for byte.
"""

from __future__ import annotations

import re

_PORTED = (
    "eval|init|list|show|delete|export|import|retry|cache|config|logs|debug|generate|feedback|"
    "validate|share"
)
_PORTED_RE = re.compile(rf"\b(?:npx )?promptfoo(?:@latest)? ({_PORTED})\b")
_NODE_ONLY_RE = re.compile(r"\b(?<!npx )promptfoo (view|redteam|auth)\b")


def as_promptphoo(text: str) -> str:
    text = _PORTED_RE.sub(r"promptphoo \1", text)
    return _NODE_ONLY_RE.sub(r"npx promptfoo@latest \1", text)
