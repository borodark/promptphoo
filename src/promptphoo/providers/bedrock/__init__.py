"""Amazon Bedrock provider family factory. Ports the `bedrock:` branch of
src/providers/families/aws.ts.

Ported routes: Converse (`bedrock:converse:<model>`), InvokeModel (`bedrock:<model>` and
`bedrock:completion:<model>`) and embeddings. Nova Sonic, the Anthropic Messages (Mantle/Runtime
messages), OpenAI Responses and Mantle chat providers, video (Luma Ray, Nova Reel), agents and
knowledge bases raise a "not supported by promptphoo" error at the point upstream would build them.
"""

from __future__ import annotations

import re
from typing import Any

from promptphoo.providers.bedrock.converse import AwsBedrockConverseProvider
from promptphoo.providers.bedrock.index import (
    AwsBedrockCompletionProvider,
    AwsBedrockEmbeddingProvider,
    assert_bedrock_model_is_available,
)
from promptphoo.providers.bedrock.routing import (
    get_bedrock_text_route,
    is_bedrock_anthropic_messages_model,
    is_rejected_prefixed_grok_id,
    is_rejected_prefixed_mythos_id,
    requires_bedrock_anthropic_messages_model,
)

NOVA_SONIC_MODEL_IDS = {"amazon.nova-sonic-v1:0", "amazon.nova-2-sonic-v1:0"}


def _is_unsupported_nova_sonic_geo_id(model_name: str) -> bool:
    return re.match(r"^[^.]+\.amazon\.nova(?:-2)?-sonic-v1:0$", model_name) is not None


def _unsupported(what: str, provider_path: str) -> ValueError:
    return ValueError(
        f"The Bedrock {what} provider is not supported by promptphoo ({provider_path})."
    )


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    splits = provider_path.split(":")
    model_type = splits[1] if len(splits) > 1 else ""
    model_name = ":".join(splits[2:])
    text_route = get_bedrock_text_route(provider_path)

    is_legacy_type = model_type in ("converse", "completion")
    bare_model_name = ":".join(splits[1:])
    nova_sonic_subtype = (
        {"expectedModel": "amazon.nova-sonic-v1:0", "name": "nova-sonic"}
        if model_type == "nova-sonic"
        else {"expectedModel": "amazon.nova-2-sonic-v1:0", "name": "nova-2-sonic"}
        if model_type == "nova-2-sonic"
        else None
    )
    if nova_sonic_subtype and model_name:
        name, expected = nova_sonic_subtype["name"], nova_sonic_subtype["expectedModel"]
        if model_name not in NOVA_SONIC_MODEL_IDS:
            raise ValueError(
                f'Amazon Bedrock model "{model_name}" is an unsupported Nova Sonic model ID. '
                f'Use "bedrock:{name}:{expected}".'
            )
        if model_name != expected:
            raise ValueError(
                f'Amazon Bedrock model "{model_name}" does not match provider subtype "{name}". '
                f'Use "bedrock:{name}:{expected}".'
            )
    requested_nova_sonic = (
        nova_sonic_subtype["expectedModel"]
        if nova_sonic_subtype and not model_name
        else bare_model_name
        if bare_model_name in NOVA_SONIC_MODEL_IDS
        else model_name
        if model_type in ("nova-sonic", "nova-2-sonic") and model_name in NOVA_SONIC_MODEL_IDS
        else None
    )
    if _is_unsupported_nova_sonic_geo_id(bare_model_name) or _is_unsupported_nova_sonic_geo_id(
        model_name
    ):
        rejected = (
            bare_model_name if _is_unsupported_nova_sonic_geo_id(bare_model_name) else model_name
        )
        unprefixed = re.sub(r"^[^.]+\.", "", rejected, count=1)
        raise ValueError(
            f'Amazon Bedrock model "{rejected}" does not support geo inference IDs. '
            f'Use the bare "bedrock:{unprefixed}" model ID in a supported region.'
        )
    if is_legacy_type and model_name in NOVA_SONIC_MODEL_IDS:
        raise ValueError(
            f'Amazon Bedrock model "{model_name}" supports only '
            "InvokeModelWithBidirectionalStream, "
            "not Converse or InvokeModel. Use its bare bedrock model ID."
        )
    if requested_nova_sonic:
        raise _unsupported("Nova Sonic", provider_path)

    anthropic_model = (
        model_name
        if model_type == "messages"
        else splits[1]
        if len(splits) == 2
        else model_name
        if is_legacy_type
        else None
    )
    if anthropic_model and is_rejected_prefixed_mythos_id(anthropic_model):
        raise ValueError(
            f'Amazon Bedrock model "{anthropic_model}" is not a valid Mythos model ID. Use '
            '"bedrock:anthropic.claude-mythos-5"; Mythos does not support geo or global '
            "inference IDs."
        )
    if anthropic_model and re.match(r"^(?:(?:us|global)\.)?anthropic\.claude-", anthropic_model):
        if requires_bedrock_anthropic_messages_model(anthropic_model) and is_legacy_type:
            api = "Converse" if model_type == "converse" else "InvokeModel"
            raise ValueError(
                f'Amazon Bedrock model "{anthropic_model}" uses the Anthropic Messages API, not '
                f'{api}. Use "bedrock:{anthropic_model}" or "bedrock:messages:{anthropic_model}".'
            )
        if (
            is_bedrock_anthropic_messages_model(anthropic_model)
            and text_route
            and text_route["apiMode"] == "messages"
        ):
            raise _unsupported("Anthropic Messages", provider_path)
    if model_type == "messages":
        raise ValueError(
            f'Amazon Bedrock model "{model_name}" is not supported by the Anthropic Messages '
            "provider. Use a us. or global. inference profile for Fable/Mythos 5.1. Mantle "
            "supports anthropic.claude-fable-5, anthropic.claude-mythos-5, and "
            "anthropic.claude-fable-5-1 (GovCloud West)."
        )

    if model_type == "responses":
        if not model_name:
            raise ValueError(
                "Amazon Bedrock Responses providers require a model id. Use "
                "bedrock:responses:<model-id>, for example bedrock:responses:openai.gpt-oss-120b."
            )
        raise _unsupported("OpenAI Responses", provider_path)

    candidate_responses_model = (
        model_name if is_legacy_type else splits[1] if len(splits) == 2 else None
    )
    routed_grok_model = model_name if model_type == "mantle" else candidate_responses_model
    if routed_grok_model and is_rejected_prefixed_grok_id(
        routed_grok_model, model_type == "mantle"
    ):
        raise ValueError(
            f'Amazon Bedrock model "{routed_grok_model}" is not a valid Grok mantle id. Use the '
            'bare "bedrock:xai.grok-4.3" id for mantle-served Grok models, or an inference '
            'profile such as "bedrock:converse:us.xai.grok-4.6" for Grok models Bedrock serves '
            "natively."
        )
    if text_route and text_route["apiMode"] == "responses":
        raise _unsupported("OpenAI Responses", provider_path)
    if model_type == "mantle":
        raise _unsupported("Mantle chat", provider_path)

    if model_type == "converse":
        assert_bedrock_model_is_available(model_name)
        return AwsBedrockConverseProvider(model_name, options)

    if "luma.ray" in model_type or "luma.ray" in model_name:
        raise _unsupported("Luma Ray video", provider_path)
    if "amazon.nova-reel" in model_type or (
        model_type == "video" and ("amazon.nova-reel" in model_name or model_name == "")
    ):
        raise _unsupported("Nova Reel video", provider_path)
    if model_type == "agents":
        raise _unsupported("Agents", provider_path)
    if model_type == "completion":
        return AwsBedrockCompletionProvider(model_name, options)
    if model_type in ("embeddings", "embedding"):
        return AwsBedrockEmbeddingProvider(model_name, options)
    if model_type in ("kb", "knowledge-base"):
        raise _unsupported("Knowledge Base", provider_path)
    return AwsBedrockCompletionProvider(":".join(splits[1:]), options)
