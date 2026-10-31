"""Client TLS options (`config.tls`). Ports createHttpsAgent from src/providers/http.ts.

Node builds an undici Agent from https.AgentOptions. Here the options become an
`ssl.SSLContext` and a dedicated `httpx.AsyncClient`, which the provider hands to
`fetch_with_cache` as `options["dispatcher"]`.

Differences: `servername` (SNI override) is not applied, `secureProtocol` only understands the
`TLSv1_*_method` names, and JKS keystores are not supported.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import ssl
import tempfile
from typing import Any

import httpx

from promptphoo import cli_state
from promptphoo.providers.http.signature import read_pkcs12_private_key, safe_resolve

logger = logging.getLogger(__name__)

_TLS_VERSIONS = {
    "TLSv1": ssl.TLSVersion.TLSv1,
    "TLSv1.1": ssl.TLSVersion.TLSv1_1,
    "TLSv1.2": ssl.TLSVersion.TLSv1_2,
    "TLSv1.3": ssl.TLSVersion.TLSv1_3,
}
_SECURE_PROTOCOLS = {
    "TLSv1_method": ssl.TLSVersion.TLSv1,
    "TLSv1_1_method": ssl.TLSVersion.TLSv1_1,
    "TLSv1_2_method": ssl.TLSVersion.TLSv1_2,
    "TLSv1_3_method": ssl.TLSVersion.TLSv1_3,
}


def is_base64(value: str) -> bool:
    """Likely base64: length divisible by 4, base64 alphabet, longer than 100 characters."""
    return (
        len(value) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", value) is not None
    ) and len(value) > 100


def validate_tls_config(tls: dict[str, Any]) -> None:
    """The TlsCertificateSchema refinement: cert and key travel together unless using PFX."""
    has_cert = tls.get("cert") or tls.get("certPath")
    has_key = tls.get("key") or tls.get("keyPath")
    has_pfx = tls.get("pfx") or tls.get("pfxPath")
    if has_pfx:
        return
    if (has_cert or has_key) and not (has_cert and has_key):
        raise ValueError(
            "Both certificate and key must be provided for client certificate authentication "
            "(unless using PFX)"
        )


def _read(path: str, binary: bool = False) -> tuple[str, Any]:
    resolved = safe_resolve(cli_state.base_path() or "", path)
    with open(resolved, "rb") as f:
        data = f.read()
    return resolved, data if binary else data.decode("utf-8")


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list):
        return "\n".join(_as_text(v) for v in value)
    return str(value)


def _load_cert_chain(ctx: ssl.SSLContext, cert: str, key: str, password: str | None) -> None:
    """SSLContext.load_cert_chain only reads files, so write the PEM text to temp files."""
    with tempfile.TemporaryDirectory(prefix="promptphoo-tls-") as tmp:
        cert_file = os.path.join(tmp, "cert.pem")
        key_file = os.path.join(tmp, "key.pem")
        with open(cert_file, "w", encoding="utf-8") as f:
            f.write(cert)
        with open(key_file, "w", encoding="utf-8") as f:
            f.write(key)
        ctx.load_cert_chain(cert_file, key_file, password=password)


def _pfx_to_pem(data: bytes, passphrase: str | None) -> tuple[str, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.serialization import pkcs12

    key = read_pkcs12_private_key(data, passphrase or "")
    _key, cert, extra = pkcs12.load_key_and_certificates(data, (passphrase or "").encode() or None)
    chain = [c for c in [cert, *(extra or [])] if c is not None]
    cert_pem = "".join(c.public_bytes(serialization.Encoding.PEM).decode() for c in chain)
    return cert_pem, key


def create_ssl_context(tls: dict[str, Any]) -> ssl.SSLContext:
    """Build an SSLContext equivalent to the https.AgentOptions Node would use."""
    using_jks = bool(tls.get("jksPath") or tls.get("jksContent"))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    ca: Any = tls.get("ca")
    if not ca and tls.get("caPath"):
        resolved, ca = _read(tls["caPath"])
        logger.debug("[HTTP Provider] Loaded CA certificate from %s", resolved)
    if ca:
        # Node's `ca` replaces the default trust store.
        ctx.load_verify_locations(cadata=_as_text(ca))
    else:
        ctx.load_default_certs()

    passphrase = tls.get("passphrase")
    if using_jks:
        try:
            raise ValueError(
                "JKS keystores are not supported by promptphoo. Convert the keystore to PEM or "
                "PKCS#12 and use cert/key or pfx"
            )
        except Exception as err:
            logger.error("[HTTP Provider] Failed to load JKS certificate for TLS: Error: %s", err)
            raise ValueError(f"Failed to load JKS certificate: Error: {err}") from err

    cert: Any = tls.get("cert")
    if not cert and tls.get("certPath"):
        resolved, cert = _read(tls["certPath"])
        logger.debug("[HTTP Provider] Loaded client certificate from %s", resolved)
    key: Any = tls.get("key")
    if not key and tls.get("keyPath"):
        resolved, key = _read(tls["keyPath"])
        logger.debug("[HTTP Provider] Loaded private key from %s", resolved)

    pfx: Any = tls.get("pfx")
    if pfx:
        if isinstance(pfx, str):
            pfx = base64.b64decode(pfx) if is_base64(pfx) else pfx.encode("latin-1")
    elif tls.get("pfxPath"):
        resolved, pfx = _read(tls["pfxPath"], binary=True)
        logger.debug("[HTTP Provider] Loaded PFX certificate from %s", resolved)

    if cert and key:
        _load_cert_chain(ctx, _as_text(cert), _as_text(key), passphrase)
    if pfx:
        cert_pem, key_pem = _pfx_to_pem(pfx, passphrase)
        _load_cert_chain(ctx, cert_pem, key_pem, None)

    if tls.get("rejectUnauthorized") is False:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    if tls.get("servername"):
        logger.debug("[HTTP Provider] tls.servername is not applied by promptphoo")
    if tls.get("ciphers"):
        ctx.set_ciphers(tls["ciphers"])
    if tls.get("secureProtocol") in _SECURE_PROTOCOLS:
        version = _SECURE_PROTOCOLS[tls["secureProtocol"]]
        ctx.minimum_version = version
        ctx.maximum_version = version
    if tls.get("minVersion") in _TLS_VERSIONS:
        ctx.minimum_version = _TLS_VERSIONS[tls["minVersion"]]
    if tls.get("maxVersion") in _TLS_VERSIONS:
        ctx.maximum_version = _TLS_VERSIONS[tls["maxVersion"]]
    return ctx


def create_https_agent(tls: dict[str, Any]) -> httpx.AsyncClient:
    logger.debug("[HTTP Provider] Creating HTTPS agent with TLS configuration")
    return httpx.AsyncClient(verify=create_ssl_context(tls), follow_redirects=True)
