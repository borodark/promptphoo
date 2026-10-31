"""HttpProviderConfigSchema.parse: strip unknown keys, apply defaults, order keys like zod.
Ports the schema half of src/providers/http.ts and src/contracts/providerConfig/http*.ts.
"""

from __future__ import annotations

import json
from typing import Any

from promptphoo.jsvalues import UNDEFINED
from promptphoo.providers.http.signature import preprocess_signature_auth_config
from promptphoo.providers.http.tls import validate_tls_config

# HttpProviderConfigFieldsSchema shape order (zod output follows the shape, not the input).
CONFIG_KEYS = (
    "body", "headers", "maxRetries", "method", "multipart", "queryParams", "request", "tools",
    "tool_choice", "transformToolsFormat", "useHttps", "session", "sessionParser",
    "sessionSource", "stateful", "transformRequest", "transformResponse", "url",
    "validateStatus", "responseParser", "tokenEstimation", "auth", "signatureAuth", "tls",
)  # fmt: skip

_AUTH_KEYS = {
    "bearer": ("token", "type"),
    "basic": ("password", "type", "username"),
    "api_key": ("type", "value", "placement", "keyName"),
    "file": ("type", "path"),
}
_OAUTH_KEYS = ("type", "tokenUrl", "scopes", "clientId", "clientSecret", "grantType")
_OAUTH_PASSWORD_KEYS = (*_OAUTH_KEYS, "username", "password")
_SESSION_KEYS = ("url", "method", "headers", "body", "responseParser")
_SIGNATURE_KEYS = (
    "signatureValidityMs", "signatureDataTemplate", "signatureAlgorithm",
    "signatureRefreshBufferMs", "privateKeyPath", "privateKey", "keystorePath",
    "keystorePassword", "keyAlias", "keyPassword", "pfxPath", "pfxPassword", "certPath",
    "keyPath",
)  # fmt: skip


def _js_type(value: Any) -> str:
    if value is UNDEFINED:
        return "undefined"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def zod_error(issues: list[dict[str, Any]]) -> ValueError:
    """A ZodError's message: the issues as pretty-printed JSON."""
    return ValueError(json.dumps(issues, indent=2))


def _pick(obj: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: obj[k] for k in keys if k in obj and obj[k] is not UNDEFINED}


def _parse_auth(auth: Any) -> Any:
    if not isinstance(auth, dict):
        return auth
    auth_type = auth.get("type")
    if auth_type == "oauth":
        keys = _OAUTH_PASSWORD_KEYS if auth.get("grantType") == "password" else _OAUTH_KEYS
        return _pick(auth, keys)
    if auth_type in _AUTH_KEYS:
        return _pick(auth, _AUTH_KEYS[auth_type])
    raise zod_error([{"code": "invalid_union", "path": ["auth"], "message": "Invalid input"}])


def _parse_signature_auth(signature_auth: Any) -> Any:
    if not isinstance(signature_auth, dict):
        return signature_auth
    parsed: dict[str, Any] = {
        "signatureValidityMs": 300000,
        "signatureDataTemplate": "{{signatureTimestamp}}",
        "signatureAlgorithm": "SHA256",
    }
    parsed.update(_pick(signature_auth, _SIGNATURE_KEYS))
    for key, value in signature_auth.items():
        if key not in parsed and value is not UNDEFINED:
            parsed[key] = value
    return preprocess_signature_auth_config(parsed)


def parse_http_provider_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        received = _js_type(config)
        raise zod_error(
            [
                {
                    "expected": "object",
                    "code": "invalid_type",
                    "path": [],
                    "message": f"Invalid input: expected object, received {received}",
                }
            ]
        )
    max_retries = config.get("maxRetries")
    if isinstance(max_retries, int | float) and not isinstance(max_retries, bool):
        if max_retries < 0:
            raise zod_error(
                [
                    {
                        "origin": "number",
                        "code": "too_small",
                        "minimum": 0,
                        "inclusive": True,
                        "path": ["maxRetries"],
                        "message": "Too small: expected number to be >=0",
                    }
                ]
            )
    parsed = _pick(config, CONFIG_KEYS)
    if isinstance(parsed.get("session"), dict):
        session = {"method": "POST", **parsed["session"]}
        parsed["session"] = _pick(session, _SESSION_KEYS)
    if "tokenEstimation" in parsed and isinstance(parsed["tokenEstimation"], dict):
        estimation = parsed["tokenEstimation"]
        parsed["tokenEstimation"] = {
            "enabled": estimation.get("enabled", False),
            "multiplier": estimation.get("multiplier", 1.3),
        }
    if "auth" in parsed:
        parsed["auth"] = _parse_auth(parsed["auth"])
    if "signatureAuth" in parsed:
        parsed["signatureAuth"] = _parse_signature_auth(parsed["signatureAuth"])
    if isinstance(parsed.get("tls"), dict):
        tls = dict(parsed["tls"])
        tls.setdefault("rejectUnauthorized", True)
        validate_tls_config(tls)
        parsed["tls"] = tls
    return parsed


def validate_multipart_config(config: dict[str, Any]) -> None:
    if not config.get("multipart"):
        return
    if config.get("request"):
        raise ValueError("HTTP provider config cannot include both request and multipart")
    if config.get("body") is not None:
        raise ValueError("HTTP provider config cannot include both body and multipart")
