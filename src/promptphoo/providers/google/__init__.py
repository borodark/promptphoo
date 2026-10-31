"""Google AI Studio (`google:`) and Vertex AI (`vertex:`) providers. The factory ports
src/providers/families/google.ts; Live, Interactions, Imagen, Veo and Gemini image models are
not ported and fail with a clear error.
"""

from __future__ import annotations

from typing import Any

from promptphoo.providers.base import ApiProvider

GEMINI_ROBOTICS_STANDARD_MODEL = "gemini-robotics-er-2-preview"
_INTERACTIONS_MODELS = (
    "gemini-omni-flash-preview",
    "gemini-omni-1.1-flash",
    GEMINI_ROBOTICS_STANDARD_MODEL,
)


def _unsupported(what: str, provider_path: str) -> ValueError:
    return ValueError(f"{what} is not supported by promptphoo ({provider_path}).")


def is_live_only_model(model_name: str) -> bool:
    return model_name == "gemini-3.5-live-translate-preview" or model_name.startswith(
        "gemini-robotics-er-2-streaming-"
    )


def validate_google_model_route(
    model_name: str, service_type: str | None = None, allow_native_interactions: bool = False
) -> None:
    if is_live_only_model(model_name) and service_type != "live":
        raise ValueError(
            f'Model "{model_name}" requires the Gemini Live API. Use google:live:{model_name}.'
        )
    if (
        model_name == GEMINI_ROBOTICS_STANDARD_MODEL
        and service_type is not None
        and service_type != "chat"
        and not (allow_native_interactions and service_type == "interactions")
    ):
        raise ValueError(
            f'Model "{model_name}" uses the standard Gemini Interactions API. Use '
            f"google:{model_name} or vertex:{model_name}."
        )


def _create_vertex(provider_path: str, options: dict[str, Any]) -> ApiProvider:
    from promptphoo.providers.google.vertex import VertexChatProvider, VertexEmbeddingProvider

    splits = provider_path.split(":")
    first_part = splits[1] if len(splits) > 1 else ""
    if first_part == "live":
        model_name = ":".join(splits[2:])
        if not model_name:
            raise ValueError("Missing model name. Use vertex:live:<model>.")
        validate_google_model_route(model_name, first_part)
        raise _unsupported("The Vertex Live API", provider_path)
    if first_part == "image":
        validate_google_model_route(":".join(splits[2:]), first_part)
        raise ValueError(
            "The promptfoo vertex: adapter does not implement Vertex image generation. "
            "Use google:image: only with a model supported by the native Gemini image API and "
            "native credentials."
        )
    explicit = (
        first_part
        if first_part in ("chat", "interactions", "video", "embedding", "embeddings")
        else None
    )
    routed = ":".join(splits[2:]) if explicit else ":".join(splits[1:])
    if explicit == "interactions" and routed == GEMINI_ROBOTICS_STANDARD_MODEL:
        raise ValueError(
            "The Vertex Interactions provider does not support gemini-robotics-er-2-preview. "
            "For the documented Gemini API route, use "
            "google:interactions:gemini-robotics-er-2-preview with vertexai: false and Gemini "
            "API credentials."
        )
    validate_google_model_route(routed, explicit)
    if explicit == "interactions" and routed not in (
        "gemini-omni-flash-preview",
        "gemini-omni-1.1-flash-preview",
    ):
        raise ValueError(f'The Vertex Interactions provider does not support model "{routed}".')
    model_name = (
        ":".join(splits[2:]) if first_part in ("chat", "interactions") else ":".join(splits[1:])
    )
    if model_name in (
        "gemini-omni-flash-preview",
        "gemini-omni-1.1-flash-preview",
        GEMINI_ROBOTICS_STANDARD_MODEL,
    ):
        raise _unsupported("The Gemini Interactions API", provider_path)
    if first_part == "video":
        raise _unsupported("Veo video generation", provider_path)
    if first_part == "chat":
        return VertexChatProvider(":".join(splits[2:]), options)
    if first_part in ("embedding", "embeddings"):
        return VertexEmbeddingProvider(":".join(splits[2:]), options)
    return VertexChatProvider(":".join(splits[1:]), options)


def _create_google(provider_path: str, options: dict[str, Any]) -> ApiProvider:
    from promptphoo.providers.google.ai_studio import (
        AIStudioChatProvider,
        AIStudioEmbeddingProvider,
    )

    splits = provider_path.split(":")
    if len(splits) >= 3:
        service_type = splits[1]
        model_name = ":".join(splits[2:])
        if model_name == GEMINI_ROBOTICS_STANDARD_MODEL and service_type == "chat":
            raise ValueError(
                f'Model "{model_name}" uses the standard Gemini Interactions API and does not '
                f"support {splits[0]}:chat:. Use {splits[0]}:{model_name}."
            )
        validate_google_model_route(model_name, service_type, True)
        if service_type == "interactions":
            if model_name not in _INTERACTIONS_MODELS:
                raise ValueError(
                    f'The Gemini Interactions provider does not support model "{model_name}".'
                )
            raise _unsupported("The Gemini Interactions API", provider_path)
        if service_type == "live":
            raise _unsupported("The Gemini Live API", provider_path)
        if service_type == "image":
            raise _unsupported("Imagen image generation", provider_path)
        if service_type == "video":
            raise _unsupported("Veo video generation", provider_path)
        if service_type in ("embedding", "embeddings"):
            if not model_name:
                raise ValueError(
                    f"Missing model name for {provider_path}. Use e.g. "
                    "google:embedding:gemini-embedding-001."
                )
            return AIStudioEmbeddingProvider(model_name, options)

    # Default to regular Google API
    model_name = splits[1] if len(splits) > 1 else ""
    validate_google_model_route(model_name)
    if model_name in _INTERACTIONS_MODELS:
        raise _unsupported("The Gemini Interactions API", provider_path)
    if "-image" in model_name or model_name == "gemini-nano-banana-2.1":
        raise _unsupported("Gemini native image generation", provider_path)
    return AIStudioChatProvider(model_name, options)


def create_provider(provider_path: str, options: dict[str, Any]) -> ApiProvider:
    if provider_path.startswith("vertex:"):
        return _create_vertex(provider_path, options)
    return _create_google(provider_path, options)
