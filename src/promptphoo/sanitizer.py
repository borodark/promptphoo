"""Generic sanitizing of objects so secrets and credentials are not logged or persisted.

Ports src/util/sanitizer.ts (plus `safeJsonStringify` from src/util/json.ts and the grading
provider-map guard from src/util/gradingProvider.ts). URL handling goes through the WHATWG
parser in `whatwg_url.py` so redacted URLs serialize exactly as Node's `URL` would.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
import sys
from collections.abc import Callable, Iterable
from typing import Any

from promptphoo.jsvalues import UNDEFINED, json_stringify
from promptphoo.providers.base import is_api_provider
from promptphoo.whatwg_url import URL

MAX_DEPTH = 4
DUMMY_BASE = "http://placeholder"
URL_REFERENCE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|/[^?#]*[?#])", re.I)

REDACTED = "[REDACTED]"
_ENCODED_REDACTED = "%5BREDACTED%5D"

OPAQUE_CREDENTIAL_PATH_SEGMENT = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32,}"
    r"|(?:token|key|secret|credential|auth)[-_][a-z0-9._-]{8,}"
    r"|eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)\Z",
    re.I,
)

GRADING_PROVIDER_TYPE_KEYS = ("text", "embedding", "classification", "moderation")

# ---------------------------------------------------------------------------------------------
# JS primitives
# ---------------------------------------------------------------------------------------------

_JS_WS = "\t\n\v\f\r                  　﻿"  # noqa: E501


def _js_trim(value: str) -> str:
    return value.strip(_JS_WS)


class _URIError(ValueError):
    pass


_HEX = "0123456789abcdefABCDEF"


def decode_uri_component(value: str) -> str:
    """JS decodeURIComponent: raises on malformed escapes or invalid UTF-8."""
    out: list[str] = []
    i = 0
    n = len(value)
    while i < n:
        c = value[i]
        if c != "%":
            out.append(c)
            i += 1
            continue
        buf = bytearray()
        while i < n and value[i] == "%":
            h = value[i + 1 : i + 3]
            if len(h) != 2 or h[0] not in _HEX or h[1] not in _HEX:
                raise _URIError("URI malformed")
            buf.append(int(h, 16))
            i += 3
        try:
            out.append(buf.decode("utf-8"))
        except UnicodeDecodeError:
            raise _URIError("URI malformed") from None
    return "".join(out)


_URI_COMPONENT_SAFE = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.!~*'()"
)


def encode_uri_component(value: str) -> str:
    """JS encodeURIComponent."""
    return "".join(
        c
        if c in _URI_COMPONENT_SAFE
        else "".join(f"%{b:02X}" for b in c.encode("utf-8", "surrogatepass"))
        for c in value
    )


def _reject_constant(name: str) -> Any:
    raise ValueError(f"Unexpected token {name}")


def _json_parse(text: str) -> Any:
    """JSON.parse (rejects NaN/Infinity, which Python's json accepts)."""
    return json.loads(text, parse_constant=_reject_constant)


_ARRAY_INDEX = re.compile(r"^(?:0|[1-9][0-9]*)\Z")


def _js_key_order(d: dict[str, Any]) -> dict[str, Any]:
    """Objects enumerate integer-like keys first, ascending (V8 property order)."""
    index_keys = [k for k in d if _ARRAY_INDEX.match(k) and int(k) < 2**32 - 1]
    if not index_keys:
        return d
    out = {k: d[k] for k in sorted(index_keys, key=int)}
    for k, v in d.items():
        if k not in out:
            out[k] = v
    return out


def _js_key(key: Any) -> str:
    from promptphoo.jsvalues import js_string

    return key if isinstance(key, str) else js_string(key)


def _iso_date(value: _dt.datetime | _dt.date) -> str:
    if isinstance(value, _dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(_dt.UTC).replace(tzinfo=None)
        return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"
    return value.isoformat() + "T00:00:00.000Z"


class _SerializerError(Exception):
    """A provider `to_json` (JS toJSON) raised while serializing."""


# fast-safe-stringify's output when JSON.stringify throws (e.g. a toJSON that raises).
UNABLE_TO_SERIALIZE = "[unable to serialize, circular reference is too complex to analyze]"


def _json_round_trip(value: Any, *, circular: Any, error_fields: bool) -> Any:
    """JSON.parse(JSON.stringify(value)) with a circular-reference replacement.

    `circular` is the value substituted for a reference to an ancestor (fast-safe-stringify's
    '[Circular]', or UNDEFINED to drop it like util/json.ts's safeJsonStringify).
    `error_fields` serializes exceptions as {name, message} (sanitizeObject's replacer).
    Returns UNDEFINED where JSON.stringify would produce undefined.
    """
    ancestors: list[int] = []

    def convert(v: Any) -> Any:
        if v is UNDEFINED or (callable(v) and not hasattr(v, "to_json")):
            return UNDEFINED
        if v is None or isinstance(v, bool | str):
            return v
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return None if math.isnan(v) or math.isinf(v) else v
        if isinstance(v, BaseException):
            if error_fields:
                return {"name": type(v).__name__, "message": str(v)}
            return {}
        if isinstance(v, _dt.datetime | _dt.date):
            return _iso_date(v)
        to_json = getattr(v, "to_json", None)
        if callable(to_json):
            try:
                serialized = to_json()
            except Exception as error:
                raise _SerializerError(str(error)) from error
            return convert(serialized)
        model_dump = getattr(v, "model_dump", None)
        if callable(model_dump):
            return convert(model_dump(by_alias=True))
        if isinstance(v, bytes | bytearray):
            return {"type": "Buffer", "data": list(v)}
        if isinstance(v, set | frozenset | re.Pattern):
            return {}
        if isinstance(v, dict | list | tuple) or hasattr(v, "__dict__"):
            if id(v) in ancestors:
                return circular
            ancestors.append(id(v))
            try:
                if isinstance(v, list | tuple):
                    items = [convert(item) for item in v]
                    return [None if item is UNDEFINED else item for item in items]
                source = (
                    v
                    if isinstance(v, dict)
                    else {k: a for k, a in vars(v).items() if not k.startswith("_")}
                )
                out: dict[str, Any] = {}
                for k, item in source.items():
                    converted = convert(item)
                    if converted is not UNDEFINED:
                        out[_js_key(k)] = converted
                return _js_key_order(out)
            finally:
                ancestors.pop()
        return str(v)

    return convert(value)


def safe_json_stringify(value: Any, pretty_print: bool = False) -> str | None:
    """Ports safeJsonStringify (src/util/json.ts): drops circular references, None on failure."""
    try:
        converted = _json_round_trip(value, circular=UNDEFINED, error_fields=False)
    except Exception:
        return None
    if converted is UNDEFINED:
        return None
    return json_stringify(converted, 2 if pretty_print else None)


# ---------------------------------------------------------------------------------------------
# Grading provider maps (src/util/gradingProvider.ts)
# ---------------------------------------------------------------------------------------------


def _is_typed_provider_value(value: Any) -> bool:
    if isinstance(value, str):
        return len(value) > 0
    if is_api_provider(value):
        return True
    if not isinstance(value, dict):
        return False
    pid = value.get("id")
    return isinstance(pid, str) and len(pid) > 0


def is_provider_type_map(provider: Any) -> bool:
    return (
        isinstance(provider, dict)
        and bool(provider)
        and "id" not in provider
        and any(
            t in provider and _is_typed_provider_value(provider[t])
            for t in GRADING_PROVIDER_TYPE_KEYS
        )
    )


# ---------------------------------------------------------------------------------------------
# Secret detection
# ---------------------------------------------------------------------------------------------


def _has_url_userinfo(url: str) -> bool:
    scheme_index = url.find("://")
    if scheme_index == -1:
        return False
    authority_start = scheme_index + 3
    authority_end = len(url)
    for i in range(authority_start, len(url)):
        if url[i] in "/?#":
            authority_end = i
            break
    authority = url[authority_start:authority_end]
    return authority.find("@") > 0


_NON_SECRET_TOKEN_NAME = re.compile(
    r"^(?:eos|bos|pad|unk|mask|sep|cls|stop|start|end|next|prev|page|nextpage|continuation"
    r"|resume|cursor|max|min)tokens?\Z"
)
_SPLIT_WORDS = re.compile(r"[-_\s=]+")


def _is_secret_parameter_name(name: str) -> bool:
    normalized = normalize_field_name(name)
    if normalized == "key":
        return True
    if _NON_SECRET_TOKEN_NAME.search(normalized) or re.search(
        r"(?:version|type|enabled)\Z", normalized
    ):
        return False
    if is_secret_field(name) or any(is_secret_field(w) for w in _SPLIT_WORDS.split(name)):
        return True

    words = re.sub(r"(value|hash|encrypted)\Z", r"_\1", name, count=1, flags=re.I)
    words = re.sub(r"v?\d+", "_", words, flags=re.I)
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words)
    prefix = ""
    for word in _SPLIT_WORDS.split(words):
        prefix += word.lower()
        if any(
            prefix == secret or (len(secret) > 3 and prefix.endswith(secret))
            for secret in SECRET_PARAMETER_NAMES
        ):
            return True
    return False


_BOOLEAN_CONTROL_NAME = re.compile(
    r"(?:^|[.\[])(?:include|require|use|with|enable|disable)(?:[a-z]|[_-])[^.\[\]]*\]?\Z", re.I
)


def _is_secret_parameter(name: str, value: str | None) -> bool:
    if re.match(r"^(?:true|false)\Z", value or "", re.I) and _BOOLEAN_CONTROL_NAME.search(name):
        return False
    return any(_is_secret_parameter_name(part) for part in re.split(r"[.\[\]]+", name))


def _has_secret_form_segment(text: str) -> bool:
    for segment in re.split(r"[?&;#]", text):
        equals_index = segment.find("=")
        if equals_index <= 0:
            continue
        raw_value = segment[equals_index + 1 :]
        if not raw_value:
            continue
        decoded_value = _decode_form_component(raw_value)
        if looks_like_secret(raw_value) or (
            decoded_value is not None and looks_like_secret(decoded_value)
        ):
            return True
        raw_key = segment[:equals_index]
        key = _decode_form_component(raw_key)
        if key is None:
            key = raw_key
        if _is_secret_parameter(key, decoded_value):
            return True
    return False


def _unparseable_url_might_leak_secret(url: str) -> bool:
    return (
        _has_url_userinfo(url) or looks_like_secret(_js_trim(url)) or _has_secret_form_segment(url)
    )


# Keys are stored normalized (lowercase, no hyphens/underscores).
SECRET_FIELD_NAMES: frozenset[str] = frozenset(
    [
        # Password variants
        "password", "passwd", "pwd",
        # Secret variants
        "secret", "secrets", "secretkey", "credentials",
        # API keys and tokens
        "apikey", "apisecret", "token", "accesstoken", "refreshtoken", "idtoken", "bearertoken",
        "authtoken", "clientsecret", "webhooksecret", "anthropicapikey", "awsbearertokenbedrock",
        # AWS SigV4 credentials
        "secretaccesskey", "awssecretaccesskey", "sessiontoken", "awssessiontoken",
        "accesskeyid", "awsaccesskeyid", "authorization", "auth", "bearer", "apikeyenvar",
        # Header-specific patterns
        "xapikey", "xauthtoken", "xaccesstoken", "xauth", "xsecret", "xcsrftoken",
        "portkeyapikey", "portkeyvirtualkey", "xportkeyapikey", "xportkeyvirtualkey",
        "xportkeyawsaccesskeyid", "xportkeyawssecretaccesskey", "xportkeyawssessiontoken",
        "xsessiondata", "csrftoken", "sessionid", "session", "cookie", "setcookie",
        # Certificate and encryption
        "certificatepassword", "keystorepassword", "pfxpassword", "privatekey", "certkey",
        "encryptionkey", "signingkey", "signature", "sig", "passphrase", "certificatecontent",
        "keystorecontent", "pfx", "pfxcontent", "keycontent", "certcontent",
    ]
)  # fmt: skip
# Deterministic iteration (only membership/any() semantics depend on these lists).
_SECRET_FIELD_NAMES_ORDERED = sorted(SECRET_FIELD_NAMES)

# Ambiguous names need a complete segment match: oauth/useSession/sameSiteCookie are settings.
SECRET_PARAMETER_NAMES = [
    name
    for name in [*_SECRET_FIELD_NAMES_ORDERED, "subscriptionkey"]
    if name not in ("auth", "session", "cookie", "setcookie")
]


def normalize_field_name(field_name: str) -> str:
    return re.sub(r"[-_\s=]", "", field_name.lower())


def is_secret_field(field_name: str) -> bool:
    return normalize_field_name(field_name) in SECRET_FIELD_NAMES


ENV_SECRET_WHOLE_WORDS = frozenset(["KEY"])
ENV_SECRET_SUFFIX_WORDS = [
    w.upper()
    for w in [
        "PASSWORD",
        "PASSWD",
        "PASSPHRASE",
        "CREDENTIALS",
        "CREDENTIAL",
        "SECRET",
        "TOKEN",
        "PWD",
        "DSN",
        *[n for n in _SECRET_FIELD_NAMES_ORDERED if n.endswith("key") and len(n) > 3],
        "accesskey",
    ]
]
ENV_SECRET_TERMINAL_WORDS = frozenset(["AUTH"])
ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*\Z")


def is_secret_env_var_name(name: str) -> bool:
    if is_secret_field(name):
        return True
    normalized = name.upper()
    if not ENV_VAR_NAME_RE.match(normalized):
        return False
    words = normalized.split("_")
    if words[-1] in ENV_SECRET_TERMINAL_WORDS:
        return True
    return any(
        word in ENV_SECRET_WHOLE_WORDS or any(word.endswith(s) for s in ENV_SECRET_SUFFIX_WORDS)
        for word in words
    )


def sanitize_runtime_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only JSON-serializable runtime options (drops abortSignal and callables)."""
    if not options:
        return None
    sanitized = dict(options)
    sanitized.pop("abortSignal", None)
    for key, value in list(sanitized.items()):
        if callable(value):
            del sanitized[key]
    return sanitized


_NOT_LINE_TERM = r"[^\n\r  ]"
_SECRET_VALUE_PATTERNS = [
    re.compile(r"^sk-[a-zA-Z0-9\-_]{20,}"),
    re.compile(r"^sk-proj-[a-zA-Z0-9\-_]{20,}"),
    re.compile(r"^sk-ant-[a-zA-Z0-9\-_]{20,}"),
    re.compile(r"^key-[a-zA-Z0-9]{20,}"),
    re.compile(r"^Bearer\s+" + _NOT_LINE_TERM + "{20,}", re.I),
    re.compile(r"^Basic\s+" + _NOT_LINE_TERM + "{20,}", re.I),
    re.compile(r"^AKIA[A-Z0-9]{16}"),
    re.compile(r"^AIza[a-zA-Z0-9_-]{35}"),
]
_NON_BASE64 = re.compile(r"[^a-zA-Z0-9+/=_-]")


def looks_like_secret(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if any(p.search(value) for p in _SECRET_VALUE_PATTERNS):
        return True
    # Long base64-like strings (likely tokens/keys) - 64+ chars.
    return len(value) >= 64 and not _NON_BASE64.search(value)


NON_CREDENTIAL_HEADERS = frozenset(
    [
        "accept",
        "content-type",
        "openai-beta",
        "openai-organization",
        "openai-project",
        "user-agent",
        "x-openai-originator",
    ]
)


def is_non_credential_header(name: str) -> bool:
    return name.lower() in NON_CREDENTIAL_HEADERS


SAFE_TRACING_CREDENTIAL_TEMPLATE = re.compile(
    r"^(?:(?:bearer|basic|token|api[-_]?key)\s+)?\{\{\s*env(?:\.[A-Za-z_][A-Za-z0-9_]*"
    r"|\[['\"][A-Za-z_][A-Za-z0-9_]*['\"]\])+\s*(?:\|\s*(?:trim|urlencode)\s*)*\}\}\Z",
    re.I,
)
SAFE_TRACING_PROVIDER_HEADERS = frozenset(
    ["accept", "content-type", "x-org-id", "x-organization-id", "x-scope-orgid", "x-tenant-id"]
)

# WeakMap<provider, references> upstream. Dicts are not weak-referenceable, so this keys by
# id() and holds the provider object to guard against id reuse.
_tracing_credential_references: dict[int, tuple[Any, dict[str, dict[str, dict[str, str]]]]] = {}


def _get_tracing_references(provider: Any) -> dict[str, dict[str, dict[str, str]]] | None:
    entry = _tracing_credential_references.get(id(provider))
    if entry is None or entry[0] is not provider:
        return None
    return entry[1]


def _is_safe_tracing_credential_template(value: Any) -> bool:
    return isinstance(value, str) and bool(SAFE_TRACING_CREDENTIAL_TEMPLATE.search(_js_trim(value)))


_PUBLIC_KEY_ROLE = re.compile(
    r"^x?(?:(?:idempotency|cache|routing|partition|public)key|secwebsocketkey)\Z"
)
_CREDENTIAL_HEADER_WORD = re.compile(
    r"(?:api|access|subscription)key|auth|token|password|passwd|secret|credential|cookie|jwt"
)


def is_credential_header(name: str, value: str) -> bool:
    normalized_name = re.sub(r"[-_\s]", "", name).lower()
    public_key_role = bool(_PUBLIC_KEY_ROLE.search(normalized_name))
    trimmed = _js_trim(value)
    return (
        is_secret_field(normalized_name)
        or (not public_key_role and normalized_name.endswith("key"))
        or bool(re.search(r"session(?:access|id)?\Z", normalized_name))
        or bool(_CREDENTIAL_HEADER_WORD.search(normalized_name))
        or normalized_name == "xhoneycombteam"
        or bool(re.match(r"^(?:bearer|basic|token|api[-_]?key)\s+\S+", trimmed, re.I))
        or looks_like_secret(trimmed)
    )


def _is_non_sensitive_tracing_header(name: str, value: str) -> bool:
    return name.lower() in SAFE_TRACING_PROVIDER_HEADERS and not is_credential_header(name, value)


_TEMPLATE_ENV_VAR = re.compile(
    r"\benv(?:\.([A-Za-z_][A-Za-z0-9_]*)|\[['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\])"
)


def _get_tracing_template_environment_variable(template: Any) -> str | None:
    if not _is_safe_tracing_credential_template(template):
        return None
    match = _TEMPLATE_ENV_VAR.search(template)
    if not match:
        return None
    return match.group(1) if match.group(1) is not None else match.group(2)


def _tracing_provider(config: dict[str, Any] | None) -> dict[str, Any] | None:
    tracing = (config or {}).get("tracing")
    if not isinstance(tracing, dict):
        return None
    provider = tracing.get("provider")
    return provider if isinstance(provider, dict) else None


def preserve_tracing_credential_references(
    source_config: dict[str, Any], rendered_config: dict[str, Any]
) -> None:
    """Remember the safe `{{ env.X }}` templates a rendered tracing provider came from."""
    source_provider = _tracing_provider(source_config)
    rendered_provider = _tracing_provider(rendered_config)
    if not source_provider or not rendered_provider:
        return

    references: dict[str, dict[str, dict[str, str]]] = {"auth": {}, "headers": {}, "env": {}}

    rendered_auth = rendered_provider.get("auth") or {}
    for key, template in (source_provider.get("auth") or {}).items():
        if key not in ("token", "password"):
            continue
        rendered_value = rendered_auth.get(key)
        if _is_safe_tracing_credential_template(template) and isinstance(rendered_value, str):
            references["auth"][key] = {"template": template, "renderedValue": rendered_value}

    rendered_headers = rendered_provider.get("headers") or {}
    for name, template in (source_provider.get("headers") or {}).items():
        rendered_value = rendered_headers.get(name)
        if _is_safe_tracing_credential_template(template) and isinstance(rendered_value, str):
            references["headers"][name] = {"template": template, "renderedValue": rendered_value}

    credential_templates = [
        *references["auth"].values(),
        *(
            ref
            for name, ref in references["headers"].items()
            if not _is_non_sensitive_tracing_header(name, ref["renderedValue"])
        ),
    ]
    source_env = source_config.get("env") if isinstance(source_config.get("env"), dict) else {}
    rendered_env = (
        rendered_config.get("env") if isinstance(rendered_config.get("env"), dict) else {}
    )
    assert isinstance(source_env, dict) and isinstance(rendered_env, dict)
    visited: set[str] = set()
    for ref in credential_templates:
        name = _get_tracing_template_environment_variable(ref["template"])
        while name and name not in visited:
            visited.add(name)
            source_value = source_env.get(name)
            rendered_value = rendered_env.get(name)
            if not _is_safe_tracing_credential_template(source_value) or not isinstance(
                rendered_value, str
            ):
                break
            references["env"][name] = {
                "template": str(source_value),
                "renderedValue": rendered_value,
            }
            name = _get_tracing_template_environment_variable(source_value)

    if references["auth"] or references["headers"]:
        _tracing_credential_references[id(rendered_provider)] = (rendered_provider, references)


def _get_referenced_tracing_credential_environment_variables(
    auth: dict[str, Any] | None,
    headers: dict[str, Any] | None,
    env: dict[str, Any] | None,
    references: dict[str, dict[str, dict[str, str]]] | None,
) -> list[str]:
    variables: list[str] = []

    def add(v: str) -> None:
        if v not in variables:
            variables.append(v)

    for name, value in (auth or {}).items():
        if name in ("token", "password") and isinstance(value, str):
            variable = _get_tracing_template_environment_variable(value)
            if variable:
                add(variable)
    for name, value in (headers or {}).items():
        if not _is_non_sensitive_tracing_header(name, value if isinstance(value, str) else ""):
            variable = _get_tracing_template_environment_variable(value)
            if variable:
                add(variable)

    # A JS Set iterates entries added during iteration, so this walks reference chains.
    i = 0
    while i < len(variables):
        name = variables[i]
        i += 1
        value = (env or {}).get(name, UNDEFINED)
        reference = references["env"].get(name) if references else None
        if _is_safe_tracing_credential_template(value):
            template = value
        elif reference and reference["renderedValue"] == value:
            template = reference["template"]
        else:
            template = None
        variable = _get_tracing_template_environment_variable(template) if template else None
        if variable:
            add(variable)
    return variables


def sanitize_tracing_config_for_persistence(config: dict[str, Any]) -> dict[str, Any]:
    """Keep runtime trace-provider credentials out of persisted and exported eval configs."""
    provider = _tracing_provider(config)
    if not provider:
        return config

    raw_endpoint = provider.get("endpoint")
    sanitized_endpoint = raw_endpoint
    try:
        if not isinstance(raw_endpoint, str):
            raise TypeError("Invalid URL")
        endpoint = URL(raw_endpoint)
        if endpoint.username or endpoint.password or endpoint.search or endpoint.hash:
            endpoint.set_username("")
            endpoint.set_password("")
            endpoint.search = ""
            endpoint.hash = ""
            sanitized_endpoint = endpoint.href
        safe_endpoint = sanitize_url_for_logging(endpoint.href)
        if safe_endpoint != endpoint.href:
            sanitized_endpoint = safe_endpoint
    except Exception:
        sanitized_endpoint = sanitize_url(raw_endpoint)  # type: ignore[arg-type]

    references = _get_tracing_references(provider)
    auth = provider.get("auth")
    sanitized_auth: dict[str, Any] | None = None
    if auth:
        sanitized_auth = {}
        for key, value in auth.items():
            if key not in ("token", "password") or _is_safe_tracing_credential_template(value):
                sanitized_auth[key] = value
                continue
            reference = references["auth"].get(key) if references else None
            if reference and reference["renderedValue"] == value:
                sanitized_auth[key] = reference["template"]
    headers = provider.get("headers")
    sanitized_headers: dict[str, Any] | None = None
    if headers:
        sanitized_headers = {}
        for name, value in headers.items():
            if not isinstance(value, str):
                continue
            reference = references["headers"].get(name) if references else None
            if reference and reference["renderedValue"] == value:
                sanitized_headers[name] = reference["template"]
            elif _is_safe_tracing_credential_template(value) or _is_non_sensitive_tracing_header(
                name, value
            ):
                sanitized_headers[name] = value
    env = config.get("env")
    referenced = _get_referenced_tracing_credential_environment_variables(
        sanitized_auth, sanitized_headers, env if isinstance(env, dict) else None, references
    )
    sanitized_env: dict[str, Any] | None = None
    if env:
        sanitized_env = {}
        for name, value in env.items():
            if name not in referenced or _is_safe_tracing_credential_template(value):
                sanitized_env[name] = value
                continue
            reference = references["env"].get(name) if references else None
            if reference and reference["renderedValue"] == value:
                sanitized_env[name] = reference["template"]

    sanitized_provider = dict(provider)
    if "endpoint" in provider or sanitized_endpoint is not None:
        sanitized_provider["endpoint"] = sanitized_endpoint
    if auth:
        sanitized_provider["auth"] = sanitized_auth
    if headers:
        sanitized_provider["headers"] = sanitized_headers

    out = dict(config)
    if env:
        out["env"] = sanitized_env
    out["tracing"] = {**config["tracing"], "provider": sanitized_provider}
    return out


# ---------------------------------------------------------------------------------------------
# Provider prompt selectors and test provider refs
# ---------------------------------------------------------------------------------------------


def _is_live_provider(value: Any) -> bool:
    if is_api_provider(value):
        return True
    if isinstance(value, dict):
        return callable(value.get("id")) or callable(value.get("toJSON"))
    return hasattr(value, "to_json") and callable(value.to_json)


_PROVIDER_OPTION_KEYS = ("id", "label", "config", "env", "transform", "delay", "inputs", "prompts")


def strip_provider_prompt_selectors(providers: Any) -> Any:
    """Project providers and provider/type maps without their `prompts` selectors."""
    if isinstance(providers, list):
        return [strip_provider_prompt_selectors(p) for p in providers]
    if not providers or not (isinstance(providers, dict) or _is_live_provider(providers)):
        return providers

    def omit_selectors(provider: Any) -> Any:
        if not provider or not (isinstance(provider, dict) or _is_live_provider(provider)):
            return provider
        projected = (
            sanitize_object(
                provider,
                context="provider output",
                sanitize_urls=True,
                max_depth=math.inf,
                throw_on_error=True,
            )
            if _is_live_provider(provider)
            else provider
        )
        if not isinstance(projected, dict):
            return projected
        return {k: v for k, v in projected.items() if k != "prompts"}

    if is_provider_type_map(providers):
        projected_map = dict(providers)
        for t in GRADING_PROVIDER_TYPE_KEYS:
            if t in providers:
                projected_map[t] = omit_selectors(providers[t])
        return projected_map
    is_options = not isinstance(providers, dict) or any(
        k in providers for k in _PROVIDER_OPTION_KEYS
    )
    if is_options:
        return omit_selectors(providers)
    return {key: omit_selectors(p) for key, p in providers.items()}


def map_test_provider_refs(test: Any, map_provider: Callable[[Any], Any]) -> Any:
    """Map only provider slots in a test or assertion, preserving shared assertion sets."""
    visited: dict[int, dict[str, Any]] = {}

    def project(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        previous = visited.get(id(value))
        if previous is not None:
            return previous
        projected = dict(value)
        visited[id(value)] = projected
        options = value.get("options")
        if "provider" in value:
            projected["provider"] = map_provider(value["provider"])
        if isinstance(options, dict) and options.get("provider", UNDEFINED) is not UNDEFINED:
            projected["options"] = {**options, "provider": map_provider(options["provider"])}
        if isinstance(value.get("assert"), list):
            projected["assert"] = [project(a) for a in value["assert"]]
        return projected

    return project(test)


def strip_test_provider_prompt_selectors(test: Any) -> Any:
    return map_test_provider_refs(test, strip_provider_prompt_selectors)


_ABSOLUTE_PATH = re.compile(r"^(?:/|[a-zA-Z]:[\\/]|\\\\)")


def _collect_tests(value: dict[str, Any]) -> list[Any]:
    tests = value.get("tests")
    out: list[Any] = list(tests) if isinstance(tests, list) else []
    out.append(value.get("defaultTest"))
    for scenario in value.get("scenarios") or []:
        if isinstance(scenario, dict):
            out.extend(scenario.get("config") or [])
            if isinstance(scenario.get("tests"), list):
                out.extend(scenario["tests"])
    return out


def sanitize_config_for_output(
    config: dict[str, Any],
    *,
    should_strip_prompt_text: bool = False,
    should_strip_test_vars: bool = False,
    should_strip_metadata: bool = False,
    should_strip_response_output: bool = False,
) -> dict[str, Any]:
    """Sanitize exported/shared configuration while preserving safe tracing env references."""
    safe = sanitize_tracing_config_for_persistence(config)
    output_config = {k: v for k, v in safe.items() if k != "basePath"}
    if should_strip_prompt_text:
        if "providers" in output_config:
            output_config["providers"] = strip_provider_prompt_selectors(output_config["providers"])
        if "defaultTest" in output_config:
            output_config["defaultTest"] = strip_test_provider_prompt_selectors(
                output_config["defaultTest"]
            )
        if isinstance(output_config.get("tests"), list):
            output_config["tests"] = [
                strip_test_provider_prompt_selectors(t) for t in output_config["tests"]
            ]
        if output_config.get("scenarios") is not None:
            scenarios = []
            for scenario in output_config["scenarios"]:
                if isinstance(scenario, dict):
                    s = dict(scenario)
                    if scenario.get("config") is not None:
                        s["config"] = [
                            strip_test_provider_prompt_selectors(c) for c in scenario["config"]
                        ]
                    if isinstance(scenario.get("tests"), list):
                        s["tests"] = [
                            strip_test_provider_prompt_selectors(t) for t in scenario["tests"]
                        ]
                    scenarios.append(s)
                else:
                    scenarios.append(scenario)
            output_config["scenarios"] = scenarios
    sanitized = sanitize_object(
        output_config,
        context="output config",
        sanitize_urls=True,
        throw_on_error=True,
        max_depth=math.inf,
    )
    if "basePath" in safe and safe["basePath"] is not UNDEFINED:
        sanitized["basePath"] = safe["basePath"]
    if should_strip_prompt_text:
        sanitized.pop("prompts", None)
        sanitized.pop("providerPromptMap", None)

    source_tests = _collect_tests(safe)
    for index, test in enumerate(_collect_tests(sanitized)):
        if not isinstance(test, dict):
            continue
        source_test = source_tests[index] if index < len(source_tests) else None
        source_metadata = None
        if isinstance(source_test, dict) and "metadata" in source_test:
            src_md = source_test.get("metadata")
            source_metadata = src_md.get("__promptfoo") if isinstance(src_md, dict) else None
        if not isinstance(source_metadata, dict):
            source_metadata = None
        provider_base_path = source_metadata.get("providerBasePath") if source_metadata else None
        if (
            "metadata" in test
            and not (source_metadata and source_metadata.get("remote") is True)
            and isinstance(provider_base_path, str)
            and _ABSOLUTE_PATH.match(provider_base_path)
        ):
            md_raw = test.get("metadata")
            md: dict[str, Any] = md_raw if isinstance(md_raw, dict) else {}
            pf_raw = md.get("__promptfoo")
            pf: dict[str, Any] = pf_raw if isinstance(pf_raw, dict) else {}
            test["metadata"] = {
                **md,
                "__promptfoo": {**pf, "providerBasePath": provider_base_path},
            }
        if should_strip_test_vars and "vars" in test:
            del test["vars"]
        if should_strip_metadata and "metadata" in test:
            cur_md = test.get("metadata")
            cur_pf = cur_md.get("__promptfoo") if isinstance(cur_md, dict) else None
            if isinstance(cur_pf, dict) and cur_pf.get("remote") is True:
                test["metadata"] = {"__promptfoo": {"remote": True}}
            elif isinstance(cur_pf, dict) and isinstance(cur_pf.get("providerBasePath"), str):
                test["metadata"] = {"__promptfoo": {"providerBasePath": cur_pf["providerBasePath"]}}
            else:
                del test["metadata"]
        if should_strip_response_output and "providerOutput" in test:
            del test["providerOutput"]

    provider = _tracing_provider(safe)
    sanitized_provider = _tracing_provider(sanitized)
    if provider and sanitized_provider is not None:
        safe_env = safe.get("env") if isinstance(safe.get("env"), dict) else {}
        assert isinstance(safe_env, dict)
        referenced_env = _get_referenced_tracing_credential_environment_variables(
            provider.get("auth"), provider.get("headers"), safe_env, None
        )
        for name, value in safe_env.items():
            if (
                sanitized.get("env")
                and name in referenced_env
                and _is_safe_tracing_credential_template(value)
            ):
                sanitized["env"][name] = value
        for field in ("auth", "headers"):
            values = provider.get(field)
            if not values:
                continue
            sanitized_values = sanitize_object(values)
            for key, value in values.items():
                if _is_safe_tracing_credential_template(value):
                    sanitized_values[key] = value
            sanitized_provider[field] = sanitized_values
    return sanitized


# ---------------------------------------------------------------------------------------------
# Azure blob SAS tokens
# ---------------------------------------------------------------------------------------------


def _is_class_instance(obj: Any) -> bool:
    return not isinstance(obj, dict | list | tuple | str | int | float | bool) and obj is not None


def _redact_azure_blob_sas_token(value: str) -> str:
    if not re.match(r"^az://", value, re.I):
        return value
    query_start = value.find("?")
    if query_start == -1:
        return value
    fragment_start = value.find("#", query_start)
    query_end = len(value) if fragment_start == -1 else fragment_start
    redacted = False
    parameters = []
    for parameter in value[query_start + 1 : query_end].split("&"):
        equals_index = parameter.find("=")
        encoded_name = parameter if equals_index == -1 else parameter[:equals_index]
        try:
            name = decode_uri_component(encoded_name.replace("+", " "))
        except _URIError:
            parameters.append(parameter)
            continue
        if name.lower() != "sig":
            parameters.append(parameter)
            continue
        redacted = True
        parameters.append(f"{encoded_name}={_ENCODED_REDACTED}")
    if not redacted:
        return value
    return f"{value[: query_start + 1]}{'&'.join(parameters)}{value[query_end:]}"


def redact_azure_blob_sas_tokens(value: Any) -> Any:
    """Redact SAS credentials embedded in Azure Blob (`az://`) config references."""
    if isinstance(value, str):
        return _redact_azure_blob_sas_token(value)
    if isinstance(value, list):
        return [redact_azure_blob_sas_tokens(item) for item in value]
    if not value or not isinstance(value, dict):
        return value
    return {key: redact_azure_blob_sas_tokens(item) for key, item in value.items()}


# ---------------------------------------------------------------------------------------------
# Strings: JSON, form data, URLs
# ---------------------------------------------------------------------------------------------


def _sanitize_json_string(s: str, depth: int, max_depth: float, sanitize_urls: bool = False) -> str:
    redacted_azure_blob_uri = _redact_azure_blob_sas_token(s)
    if redacted_azure_blob_uri != s:
        return redacted_azure_blob_uri
    try:
        parsed = _json_parse(s)
    except (ValueError, RecursionError):
        if _looks_like_url_encoded_form_data(s):
            sanitized_url_encoded = sanitize_url_encoded_string(s)
            if sanitized_url_encoded != s:
                return sanitized_url_encoded
        if looks_like_secret(s):
            return REDACTED
        return s
    if isinstance(parsed, dict | list):
        return json_stringify(_recursive_sanitize(parsed, depth, max_depth, sanitize_urls))
    if isinstance(parsed, str):
        if depth >= max_depth or depth >= 64:
            return json_stringify(REDACTED)
        return json_stringify(_sanitize_json_string(parsed, depth + 1, max_depth, sanitize_urls))
    return s


URL_ENCODED_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._~+%\[\]-]+=[^&;]*\Z")


def _looks_like_url_encoded_form_data(value: str) -> bool:
    if "=" not in value or re.search(r"[\r\n\t]", value):
        return False
    segments = [s for s in re.split(r"[&;]", value) if s]
    if not segments:
        return False
    return all(URL_ENCODED_SEGMENT_RE.match(s) for s in segments)


URL_ENCODED_PAIR_RE = re.compile(r"(^|[&;])([^=&;]+)=([^&;]*)")


def _decode_form_component(component: str) -> str | None:
    try:
        return decode_uri_component(component.replace("+", " "))
    except _URIError:
        return None


def _redact_nested_json_value(decoded: str | None) -> str | None:
    if decoded is None:
        return None
    trimmed = _js_trim(decoded)
    if not trimmed or trimmed[0] not in "{[":
        return None
    try:
        parsed = _json_parse(trimmed)
    except (ValueError, RecursionError):
        return None
    if not isinstance(parsed, dict | list):
        return None
    sanitized = sanitize_object(parsed, sanitize_urls=True, max_depth=math.inf)
    original_serialized = json_stringify(parsed)
    sanitized_serialized = json_stringify(sanitized)
    return None if sanitized_serialized == original_serialized else sanitized_serialized


NUNJUCKS_PLACEHOLDER = re.compile(r"\{\{[^{}]*\}\}")


def _is_pure_template_value(value: str) -> bool:
    return "{{" in value and _js_trim(NUNJUCKS_PLACEHOLDER.sub("", value)) == ""


def sanitize_url_encoded_string(value: str) -> str:
    if "=" not in value:
        return value
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        separator, raw_key, raw_value = match.group(1), match.group(2), match.group(3)
        if len(raw_value) == 0:
            return match.group(0)
        if _is_pure_template_value(raw_value):
            return match.group(0)
        decoded_key = _decode_form_component(raw_key)
        decoded_value = _decode_form_component(raw_value)
        key_is_secret = decoded_key is not None and _is_secret_parameter(decoded_key, decoded_value)
        if key_is_secret:
            changed = True
            return f"{separator}{raw_key}={_ENCODED_REDACTED}"
        nested_json = _redact_nested_json_value(decoded_value)
        if nested_json is not None:
            changed = True
            return f"{separator}{raw_key}={encode_uri_component(nested_json)}"
        if looks_like_secret(raw_value) or (
            decoded_value is not None and looks_like_secret(decoded_value)
        ):
            changed = True
            return f"{separator}{raw_key}={_ENCODED_REDACTED}"
        return match.group(0)

    result = URL_ENCODED_PAIR_RE.sub(replace, value)
    return result if changed else value


# ---------------------------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------------------------

_SCHEME_PREFIX = re.compile(r"^[a-z][a-z\d+.-]*://", re.I)
_URLISH_KEY = re.compile(r"(?:url|uri|host|endpoint|proxy)\Z", re.I)
_OPENAI_BASE_URL = re.compile(r"^OPENAI_(?:API_)?BASE_URL\Z", re.I)


def _entries(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        return value.items()
    if isinstance(value, list | tuple):
        return ((str(i), v) for i, v in enumerate(value))
    if isinstance(value, str):
        return ((str(i), c) for i, c in enumerate(value))
    return ()


def _sanitize_plain_object(
    obj: dict[str, Any], depth: int, max_depth: float, sanitize_urls: bool, is_env_map: bool
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    key_suffix = 0
    is_secret_key = is_secret_env_var_name if is_env_map else is_secret_field
    for raw_key, value in obj.items():
        redacted_key = (
            sanitize_url(raw_key) if sanitize_urls and URL_REFERENCE.search(raw_key) else raw_key
        )
        key = redacted_key
        while key in sanitized or (key != raw_key and key in obj):
            key_suffix += 1
            key = f"{redacted_key}#{key_suffix}"
        if is_secret_key(key):
            sanitized[key] = REDACTED
        elif key.lower() == "headers" and value and isinstance(value, dict | list | tuple):
            sanitized[key] = {
                name: item
                if _is_safe_tracing_credential_template(item)
                or (
                    isinstance(item, str)
                    and not is_credential_header(name, item)
                    and (
                        is_non_credential_header(name)
                        or name.lower() in SAFE_TRACING_PROVIDER_HEADERS
                    )
                )
                else REDACTED
                for name, item in _entries(value)
            }
        elif isinstance(value, str) and (
            key == "apiHost" or (is_env_map and key.upper().endswith("_HOST"))
        ):
            has_scheme = bool(_SCHEME_PREFIX.search(value))
            endpoint = sanitize_url_for_logging(value if has_scheme else f"https://{value}")
            host = endpoint if has_scheme else re.sub(r"^https://", "", endpoint, count=1)
            has_path = (
                "/" in re.split(r"[?#]", _SCHEME_PREFIX.sub("", value, count=1), maxsplit=1)[0]
            )
            sanitized[key] = host if has_path else re.sub(r"/(?=[?#]|\Z)", "", host, count=1)
        elif isinstance(value, str) and (
            key in ("url", "apiBaseUrl", "server_url")
            or (is_env_map and key.upper().endswith("_URL"))
        ):
            sanitized[key] = (
                sanitize_url(value)
                if key == "url"
                or (
                    is_env_map and key.upper().endswith("_URL") and not _OPENAI_BASE_URL.search(key)
                )
                else sanitize_url_for_logging(value)
            )
        elif isinstance(value, str) and looks_like_secret(value):
            sanitized[key] = REDACTED
        else:
            sanitized_value = _recursive_sanitize(
                value, depth + 1, max_depth, sanitize_urls, key == "env"
            )
            sanitized[key] = (
                sanitize_url(sanitized_value)
                if isinstance(sanitized_value, str)
                and (key.lower() == "url" or (sanitize_urls and _URLISH_KEY.search(key)))
                else sanitized_value
            )
    return _js_key_order(sanitized)


def _recursive_sanitize(
    obj: Any,
    depth: int = 0,
    max_depth: float = MAX_DEPTH,
    sanitize_urls: bool = False,
    is_env_map: bool = False,
) -> Any:
    if callable(obj) and not isinstance(obj, type) and not hasattr(obj, "to_json"):
        return f"[Function] {getattr(obj, '__name__', '')}"
    if isinstance(obj, str):
        return (
            sanitize_url(obj)
            if sanitize_urls and URL_REFERENCE.search(obj)
            else _sanitize_json_string(obj, depth, max_depth, sanitize_urls)
        )
    if obj is None or obj is UNDEFINED or isinstance(obj, bool | int | float):
        return obj
    if depth > max_depth:
        return "[...]"
    if isinstance(obj, list | tuple):
        return [_recursive_sanitize(item, depth + 1, max_depth, sanitize_urls) for item in obj]
    if not isinstance(obj, dict):
        return f"[{type(obj).__name__ or 'Object'} Instance]"
    return _sanitize_plain_object(obj, depth, max_depth, sanitize_urls, is_env_map)


def sanitize_object(
    obj: Any,
    *,
    context: str = "object",
    sanitize_urls: bool = False,
    max_depth: float = MAX_DEPTH,
    throw_on_error: bool = False,
) -> Any:
    """A sanitized copy of `obj` with secret fields, credential values and URL secrets redacted.

    Pass `max_depth=math.inf` where upstream passes `Number.POSITIVE_INFINITY`.
    """
    try:
        if obj is None or obj is UNDEFINED:
            return obj
        if isinstance(obj, str):
            return _recursive_sanitize(obj, 0, max_depth, sanitize_urls)
        if isinstance(obj, bool | int | float):
            return obj
        if callable(obj) and not hasattr(obj, "to_json"):
            return obj
        try:
            safe_obj = _json_round_trip(obj, circular="[Circular]", error_fields=True)
        except _SerializerError:
            safe_obj = UNABLE_TO_SERIALIZE
        return _recursive_sanitize(safe_obj, 0, max_depth, sanitize_urls)
    except Exception as error:
        if throw_on_error:
            raise
        print(f"Error sanitizing {context}:", error, file=sys.stderr)
        return REDACTED


# Legacy aliases
sanitize_body = sanitize_object
sanitize_headers = sanitize_object
sanitize_query_params = sanitize_object


def _get_secret_looking_raw_query_keys(search: str) -> set[str]:
    secret_keys: set[str] = set()
    for segment in search[1:].split("&"):
        separator_index = segment.find("=")
        if separator_index < 0:
            continue
        raw_value = segment[separator_index + 1 :]
        if not looks_like_secret(raw_value):
            continue
        decoded_key = _decode_form_component(segment[:separator_index])
        if decoded_key is not None:
            secret_keys.add(decoded_key)
    return secret_keys


def _sanitize_templated_url(url: str) -> str:
    if _has_url_userinfo(url):
        return REDACTED
    hash_index = url.find("#")
    before_hash = url if hash_index == -1 else url[:hash_index]
    hash_ = "" if hash_index == -1 else url[hash_index + 1 :]
    query_index = before_hash.find("?")
    before_query = before_hash if query_index == -1 else before_hash[:query_index]
    query = "" if query_index == -1 else before_hash[query_index + 1 :]
    sanitized_query = sanitize_url_encoded_string(query) if query else query
    sanitized_hash = sanitize_url_encoded_string(hash_) if hash_ else hash_
    return (
        before_query
        + ("" if query_index == -1 else f"?{sanitized_query}")
        + ("" if hash_index == -1 else f"#{sanitized_hash}")
    )


def sanitize_url(url: str) -> str:
    """Redact credentials in a URL's userinfo, query and fragment."""
    try:
        if not isinstance(url, str) or not _js_trim(url):
            return url
        if "{{" in url and "}}" in url:
            return _sanitize_templated_url(url)
        nested_json = _redact_nested_json_value(url)
        if nested_json is not None:
            return nested_json

        is_path_only = url.startswith("/") and not url.startswith("//")
        parsed_url = URL(url, DUMMY_BASE) if is_path_only else URL(url)
        sanitized_url = URL(parsed_url.href)

        if sanitized_url.username or sanitized_url.password:
            sanitized_url.set_username("***")
            sanitized_url.set_password("***")

        raw_secret_param_keys = _get_secret_looking_raw_query_keys(parsed_url.search)
        try:
            for key, value in sanitized_url.search_params_entries():
                if (
                    _is_secret_parameter(key, value)
                    or key in raw_secret_param_keys
                    or looks_like_secret(value)
                    or (";" in value and _has_secret_form_segment(value))
                ):
                    sanitized_url.search_params_set(key, "[REDACTED]")
                else:
                    nested = _redact_nested_json_value(value)
                    if nested is not None:
                        sanitized_url.search_params_set(key, nested)
        except Exception as param_error:
            print(f"Failed to sanitize URL parameters: {param_error}", file=sys.stderr)
            return REDACTED

        if len(sanitized_url.hash) > 1:
            sanitized_hash = sanitize_url_encoded_string(sanitized_url.hash[1:])
            sanitized_url.hash = f"#{sanitized_hash}" if sanitized_hash else ""

        if sanitized_url.href == parsed_url.href:
            return url
        if is_path_only:
            return sanitized_url.pathname + sanitized_url.search + sanitized_url.hash
        return sanitized_url.href
    except Exception:
        return REDACTED if _unparseable_url_might_leak_secret(url) else url


_OPAQUE_VALUE = re.compile(r"^[a-z0-9._~+-]{12,}\Z", re.I)


def sanitize_url_for_logging(url: str) -> str:
    """sanitize_url plus redaction of opaque credential-looking path segments."""
    sanitized = sanitize_url(url)
    try:
        is_path_only = sanitized.startswith("/") and not sanitized.startswith("//")
        parsed = URL(sanitized, DUMMY_BASE) if is_path_only else URL(sanitized)
        segments = parsed.pathname.split("/")
        out_segments = []
        for index, segment in enumerate(segments):
            previous = _decode_form_component(segments[index - 1] if index > 0 else "") or ""
            try:
                decoded = decode_uri_component(segment)
            except _URIError:
                out_segments.append(_ENCODED_REDACTED if is_secret_field(previous) else segment)
                continue
            opaque_value = (
                is_secret_field(previous)
                and bool(_OPAQUE_VALUE.match(decoded))
                and bool(re.search(r"[a-z]", decoded, re.I))
                and bool(re.search(r"[0-9]", decoded))
            )
            out_segments.append(
                _ENCODED_REDACTED
                if opaque_value
                or OPAQUE_CREDENTIAL_PATH_SEGMENT.match(decoded)
                or looks_like_secret(decoded)
                else segment
            )
        sanitized_pathname = "/".join(out_segments)
        if sanitized_pathname == parsed.pathname:
            return sanitized
        parsed.pathname = sanitized_pathname
        return parsed.pathname + parsed.search + parsed.hash if is_path_only else parsed.href
    except Exception:
        if not isinstance(url, str):
            return sanitized
        has_opaque_path = any(
            OPAQUE_CREDENTIAL_PATH_SEGMENT.match(
                d if (d := _decode_form_component(segment)) is not None else segment
            )
            for segment in re.split(r"[/?#]", url)
        )
        return REDACTED if _unparseable_url_might_leak_secret(url) or has_opaque_path else sanitized
