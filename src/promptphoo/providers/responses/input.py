"""Chat-format to Responses-format input translation. Ports src/providers/responses/input.ts."""

from __future__ import annotations

from typing import Any

# Content-part types the Responses API accepts; parts already using one pass through untouched.
RESPONSES_PART_TYPES = frozenset(
    ["input_text", "output_text", "input_image", "input_audio", "input_file", "refusal",
     "computer_screenshot", "summary_text", "encrypted_content"]
)  # fmt: skip


def _normalize_content_part(part: Any, role: Any) -> Any:
    """Text parts become `input_text` (`output_text` on assistant turns); chat image parts
    become `input_image` with a flat string `image_url`."""
    if not isinstance(part, dict):
        return part
    part_type = part.get("type")
    if not isinstance(part_type, str) or part_type in RESPONSES_PART_TYPES:
        return part
    if part_type == "text" and isinstance(part.get("text"), str):
        return {**part, "type": "output_text" if role == "assistant" else "input_text"}
    if part_type == "image_url":
        image_url = part.get("image_url")
        nested = image_url if isinstance(image_url, dict) else None
        url = image_url if isinstance(image_url, str) else (nested or {}).get("url")
        if isinstance(url, str):
            rest = {k: v for k, v in part.items() if k != "image_url"}
            detail = (nested or {}).get("detail")
            if detail is None:
                detail = part.get("detail")
            return {
                **rest,
                "type": "input_image",
                "image_url": url,
                **({"detail": detail} if isinstance(detail, str) else {}),
            }
    return part


def normalize_responses_input(input: Any) -> Any:
    """Normalize a Responses `input` array authored in the chat-completions format.

    Only role-bearing message objects (untyped, or `type: "message"`) with array content are
    touched; function calls, tool outputs and reasoning items keep their shape. Non-array input
    is returned unchanged.
    """
    if not isinstance(input, list):
        return input
    out: list[Any] = []
    for item in input:
        if (
            not isinstance(item, dict)
            or ("type" in item and item["type"] != "message")
            or "role" not in item
            or not isinstance(item.get("content"), list)
        ):
            out.append(item)
            continue
        role = item["role"]
        out.append({**item, "content": [_normalize_content_part(p, role) for p in item["content"]]})
    return out
