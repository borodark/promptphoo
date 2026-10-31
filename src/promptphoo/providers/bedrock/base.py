"""Base class and helpers for Amazon Bedrock providers. Ports src/providers/bedrock/base.ts and
src/providers/bedrock/util.ts.

Upstream uses `@aws-sdk/client-bedrock-runtime`; promptphoo uses a boto3 `bedrock-runtime` client
whose blocking calls run in a worker thread (`asyncio.to_thread`). Credential priority is the
same: explicit keys, then a Bedrock API key (config `apiKey` or AWS_BEARER_TOKEN_BEDROCK, sent
as `Authorization: Bearer`), then a named profile, then the default chain.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
from typing import Any

from promptphoo import envars
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.base import ApiProvider

logger = logging.getLogger(__name__)

BEDROCK_CACHE_KEY_HMAC_KEY = "promptfoo:bedrock:cache-key:v1"
REQUEST_TIMEOUT_MS = 300_000  # 5 minutes

# The geo/global prefix of a system-defined inference profile ID, e.g. `us.` in
# `us.anthropic.claude-sonnet-4-6`.
INFERENCE_PROFILE_PREFIX = re.compile(r"^(?:us|us-gov|eu|apac|global|jp|au|ca|in)\.")


def to_js_json(value: Any) -> Any:
    """Shape a value as JSON.stringify sees it: bytes become Node's Buffer JSON."""
    if isinstance(value, bytes | bytearray | memoryview):
        return {"type": "Buffer", "data": list(bytes(value))}
    if isinstance(value, dict):
        return {k: to_js_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_js_json(v) for v in value]
    return value


def _hash_bedrock_cache_value(value: Any) -> str:
    return hmac.new(
        BEDROCK_CACHE_KEY_HMAC_KEY.encode(),
        json_stringify(to_js_json(value)).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _non_empty(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _fingerprint_auth_value(auth_source: str, value: str, index: int) -> str:
    return hmac.new(
        value.encode("utf-8"),
        f"{BEDROCK_CACHE_KEY_HMAC_KEY}:{auth_source}:{index}".encode(),
        hashlib.sha256,
    ).hexdigest()


def _auth_cache_namespace(auth_source: str, values: list[str | None]) -> str:
    return _hash_bedrock_cache_value(
        [
            auth_source,
            *(
                _fingerprint_auth_value(auth_source, v, i) if v else None
                for i, v in enumerate(values)
            ),
        ]
    )


def _auth_cache_metadata(config: dict[str, Any]) -> dict[str, Any]:
    bearer_config = _non_empty(config.get("apiKey"))
    bearer_env = _non_empty(envars.get_env_string("AWS_BEARER_TOKEN_BEDROCK"))
    access_key_id = _non_empty(config.get("accessKeyId"))
    secret_access_key = _non_empty(config.get("secretAccessKey"))
    session_token = _non_empty(config.get("sessionToken"))
    profile = _non_empty(config.get("profile"))
    explicit = bool(access_key_id and secret_access_key)
    auth_source = (
        "explicit-credentials"
        if explicit
        else "bearer-config"
        if bearer_config
        else "bearer-env"
        if bearer_env
        else "profile"
        if profile
        else "default"
    )
    namespace = {
        "bearer-config": lambda: _auth_cache_namespace(auth_source, [bearer_config]),
        "bearer-env": lambda: _auth_cache_namespace(auth_source, [bearer_env]),
        "explicit-credentials": lambda: _auth_cache_namespace(
            auth_source, [access_key_id, secret_access_key, session_token]
        ),
        "profile": lambda: _auth_cache_namespace(auth_source, [profile]),
    }.get(auth_source)
    metadata: dict[str, Any] = {"authSource": auth_source}
    if namespace:
        metadata["credentialNamespace"] = namespace()
    if config.get("endpoint") is not None:
        metadata["endpoint"] = config["endpoint"]
    metadata["hasExplicitCredentials"] = explicit
    metadata["hasSessionToken"] = explicit and bool(session_token)
    return metadata


def create_bedrock_cache_key_hash(config: dict[str, Any], params: Any, region: str) -> str:
    auth_fingerprint = _hash_bedrock_cache_value(_auth_cache_metadata(config))
    return f"{auth_fingerprint}:{_hash_bedrock_cache_value({'params': params, 'region': region})}"


def aws_error_message(err: BaseException) -> str:
    """`err.message` of an AWS SDK error."""
    response = getattr(err, "response", None)
    if isinstance(response, dict) and isinstance(response.get("Error"), dict):
        return str(response["Error"].get("Message") or "")
    return str(err)


def aws_error_string(err: BaseException) -> str:
    """`String(err)` of an AWS SDK error: `<Name>: <message>`."""
    response = getattr(err, "response", None)
    if isinstance(response, dict) and isinstance(response.get("Error"), dict):
        name = response["Error"].get("Code") or type(err).__name__
        return f"{name}: {aws_error_message(err)}"
    return f"{type(err).__name__}: {err}" if str(err) else type(err).__name__


class AwsBedrockGenericProvider(ApiProvider):
    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        super().__init__(options)
        self.model_name = model_name
        self.config = options.get("config") or {}
        self.bedrock: Any = None

    def default_id(self) -> str:
        return f"bedrock:{self.model_name}"

    def __str__(self) -> str:
        return f"[Amazon Bedrock Provider {self.model_name}]"

    def requires_api_key(self) -> bool:
        return False

    def get_api_key(self) -> str | None:
        return self.config.get("apiKey") or envars.get_env_string("AWS_BEARER_TOKEN_BEDROCK")

    def get_credentials(self) -> dict[str, Any] | None:
        """Explicit credentials, or None when another mechanism supplies them."""
        if self.config.get("accessKeyId") and self.config.get("secretAccessKey"):
            logger.debug("Using credentials from config file")
            return {
                "aws_access_key_id": self.config["accessKeyId"],
                "aws_secret_access_key": self.config["secretAccessKey"],
                "aws_session_token": self.config.get("sessionToken"),
            }
        if self.get_api_key():
            logger.debug("Using Bedrock API key authentication")
            return None
        if self.config.get("profile"):
            logger.debug("Using SSO profile: %s", self.config["profile"])
            return None
        logger.debug("No explicit credentials in config, falling back to AWS default chain")
        return None

    def get_region(self) -> str:
        return (
            self.config.get("region")
            or (self.env or {}).get("AWS_BEDROCK_REGION")
            or envars.get_env_string("AWS_BEDROCK_REGION")
            or "us-east-1"
        )

    def get_bedrock_instance(self) -> Any:
        """A boto3 bedrock-runtime client (created once per provider)."""
        if self.bedrock is None:
            try:
                import boto3
                from botocore import UNSIGNED
                from botocore.config import Config
            except ImportError as err:
                raise ImportError(
                    "The boto3 package is required for Amazon Bedrock providers. Install it with "
                    "`pip install 'promptphoo[bedrock]'`."
                ) from err
            api_key = self.get_api_key()
            client_config = Config(
                retries={
                    "total_max_attempts": envars.get_env_int("AWS_BEDROCK_MAX_RETRIES", 10),
                    "mode": "adaptive",
                },
                read_timeout=REQUEST_TIMEOUT_MS / 1000,
                **({"signature_version": UNSIGNED} if api_key else {}),
            )
            credentials = self.get_credentials()
            session = (
                boto3.Session(profile_name=self.config["profile"])
                if self.config.get("profile") and not credentials and not api_key
                else boto3.Session()
            )
            client = session.client(
                "bedrock-runtime",
                region_name=self.get_region(),
                config=client_config,
                **({k: v for k, v in credentials.items() if v} if credentials else {}),
                **(
                    {"endpoint_url": self.config["endpoint"]} if self.config.get("endpoint") else {}
                ),
            )
            if api_key:

                def add_bearer(request: Any, **_: Any) -> None:
                    request.headers["Authorization"] = f"Bearer {api_key}"

                client.meta.events.register_first("before-send.bedrock-runtime.*", add_bearer)
            self.bedrock = client
        return self.bedrock


def load_bedrock_image_data(image_path: str) -> dict[str, str]:
    """Load image data from a file:// path, or return already-encoded data."""
    import base64

    if image_path.startswith("file://"):
        file_path = image_path[7:]
        resolved = os.path.abspath(file_path)
        if ".." in file_path and resolved != os.path.abspath(os.path.normpath(file_path)):
            return {"error": f"Invalid image path (path traversal detected): {file_path}"}
        if not os.path.exists(resolved):
            return {"error": f"Image file not found: {resolved}"}
        with open(resolved, "rb") as f:
            return {"data": base64.b64encode(f.read()).decode()}
    return {"data": image_path}


def nova_output_from_message(response: dict[str, Any]) -> str | None:
    content = ((response.get("output") or {}).get("message") or {}).get("content")
    if content is None:
        return None
    has_tool_use = any((block.get("toolUse") or {}).get("toolUseId") for block in content)
    if has_tool_use:
        parts = [
            None
            if is_truthy(block.get("text")) or "toolUse" not in block
            else json_stringify(block["toolUse"])
            for block in content
        ]
        return "\n\n".join(p for p in parts if p)
    return "\n\n".join("" if block.get("text") is None else block["text"] for block in content)


def nova_parse_messages(messages: str) -> dict[str, Any]:
    """Split a prompt into {system?, extractedMessages} in Nova's message format."""
    try:
        parsed = json.loads(messages)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        try:
            system_message = next((m for m in parsed if m.get("role") == "system"), None)
            result: dict[str, Any] = {
                "extractedMessages": [
                    {
                        "role": m.get("role"),
                        "content": m["content"]
                        if isinstance(m.get("content"), list)
                        else [{"text": m.get("content")}],
                    }
                    for m in parsed
                    if m.get("role") != "system"
                ]
            }
            if system_message is not None:
                content = system_message.get("content")
                result["system"] = content if isinstance(content, list) else [{"text": content}]
            return result
        except (AttributeError, TypeError):
            pass

    lines = [line.strip() for line in messages.split("\n") if line.strip()]
    system: list[dict[str, Any]] | None = None
    extracted: list[dict[str, Any]] = []
    current_role: str | None = None
    current_content: list[str] = []

    def push_message() -> None:
        nonlocal current_content
        if current_role and current_content:
            extracted.append(
                {"role": current_role, "content": [{"text": "\n".join(current_content)}]}
            )
            current_content = []

    for line in lines:
        if line.startswith("system:"):
            system = [{"text": line[7:].strip()}]
        elif line.startswith(("user:", "assistant:")):
            push_message()
            current_role = "user" if line.startswith("user:") else "assistant"
            current_content.append(line[line.index(":") + 1 :].strip())
        elif current_role:
            current_content.append(line)
        else:
            current_role = "user"
            current_content.append(line)
    push_message()
    if not extracted and not system:
        extracted.append({"role": "user", "content": [{"text": messages.strip()}]})
    out: dict[str, Any] = {"extractedMessages": extracted}
    if system is not None:
        out["system"] = system
    return out
