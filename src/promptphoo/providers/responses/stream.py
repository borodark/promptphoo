"""Responses output-text extraction. Ports getResponsesOutputText from
src/providers/responses/stream.ts. The SSE reader (readResponsesStream) is not ported:
promptphoo does not stream Responses requests.
"""

from __future__ import annotations

from typing import Any


def get_responses_output_text(response: Any) -> str | None:
    """Visible assistant text of a (possibly partial) Responses payload, or None."""
    if not isinstance(response, dict):
        return None
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = response.get("output")
    if not isinstance(output, list):
        return None
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") not in ("message", "output_message"):
            continue
        role = item.get("role")
        if role and role != "assistant":
            continue
        content = item.get("content")
        text = (
            "".join(
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") in ("output_text", "text")
                and isinstance(part.get("text"), str)
            )
            if isinstance(content, list)
            else ""
        )
        if text.strip():
            texts.append(text)
    return "\n".join(texts) or None
