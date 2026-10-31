"""Context resolution for the context-* assertions. Ports src/assertions/contextUtils.ts."""

from __future__ import annotations

from typing import Any

from promptphoo.assertions.base import AssertionValueError
from promptphoo.jsvalues import is_truthy
from promptphoo.transform import get_transform_label, transform


def js_typeof(value: Any) -> str:
    if value is None:
        return "object"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if callable(value):
        return "function"
    return "object"


def get_transform_error_message(error: BaseException) -> str:
    """getTransformErrorMessage (src/util/transform.ts): unwrap transform()'s own
    `Transform failed (label): ...` wrapper and return the underlying message."""
    cause = error.__cause__
    raw = cause if isinstance(cause, Exception) else error
    return str(raw)


def invariant(condition: Any, message: str) -> None:
    """util/invariant: Node prefixes the message with 'Invariant failed: '."""
    if not condition:
        raise AssertionValueError(f"Invariant failed: {message}")


async def resolve_context(
    assertion: dict[str, Any],
    test: dict[str, Any],
    output: Any,
    prompt: str | None = None,
    fallback_context: str | None = None,
    provider_response: dict[str, Any] | None = None,
) -> str | list[str]:
    context_value: Any = None
    vars_ = test.get("vars") or {}
    ctx = vars_.get("context")
    if is_truthy(ctx) and ctx is not None:
        if isinstance(ctx, str):
            context_value = ctx
        elif isinstance(ctx, list):
            for idx, v in enumerate(ctx):
                if not isinstance(v, str):
                    invariant(
                        False,
                        "Invalid context: expected an array of strings, but found "
                        f"{js_typeof(v)} at index {idx}",
                    )
            context_value = ctx
    elif fallback_context:
        context_value = fallback_context

    context_transform = assertion.get("contextTransform")
    if context_transform:
        try:
            response = provider_response or {}
            output_for_transform = response.get("providerTransformedOutput")
            if output_for_transform is None:
                output_for_transform = output
            transform_context: dict[str, Any] = {
                "vars": test.get("vars"),
                "prompt": {"label": prompt},
            }
            if provider_response and provider_response.get("metadata"):
                transform_context["metadata"] = provider_response["metadata"]
            transformed = await transform(
                context_transform, output_for_transform, transform_context
            )
            if not (
                isinstance(transformed, str)
                or (isinstance(transformed, list) and all(isinstance(i, str) for i in transformed))
            ):
                invariant(
                    False,
                    "contextTransform must return a string or array of strings. Got "
                    f"{js_typeof(transformed)}. Check your transform expression: "
                    f"{get_transform_label(context_transform)}",
                )
            context_value = transformed
        except Exception as error:
            raise ValueError(
                "Failed to transform context using expression "
                f"'{get_transform_label(context_transform)}': {get_transform_error_message(error)}"
            ) from error

    invariant(
        (isinstance(context_value, str) and len(context_value) > 0)
        or (
            isinstance(context_value, list)
            and len(context_value) > 0
            and all(isinstance(i, str) and len(i) > 0 for i in context_value)
        ),
        "Context is required for context-based assertions. "
        'Provide either a "context" variable (string or array of strings) in your test case or '
        'use "contextTransform" to extract context from the provider response.',
    )
    return context_value


def serialize_context(context: str | list[str]) -> str:
    return "\n\n".join(context) if isinstance(context, list) else context
