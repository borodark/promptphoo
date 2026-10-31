"""Request signing (`config.signatureAuth`). Ports preprocessSignatureAuthConfig,
diagnosePrivateKeyMaterial, generateSignature and needsSignatureRefresh from
src/providers/http.ts.

PEM keys are signed with the `cryptography` package (imported lazily). PFX bundles are read
with `cryptography`'s PKCS#12 loader (Node shells out to openssl through the `pem` package).
JKS keystores are not supported.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import time
from typing import Any

from promptphoo import cli_state, envars
from promptphoo.files import path_exists
from promptphoo.templating import get_engine

logger = logging.getLogger(__name__)


def safe_resolve(base: str, target: str) -> str:
    """path.resolve(base, target) (src/util/pathUtils.ts safeResolve)."""
    if os.path.isabs(target):
        return os.path.normpath(target)
    return os.path.abspath(os.path.join(base or os.getcwd(), target))


def preprocess_signature_auth_config(signature_auth: Any) -> Any:
    """Map generic certificate fields (certificateContent/Password/Filename) to typed fields."""
    if not signature_auth:
        return signature_auth
    certificate_content = signature_auth.get("certificateContent")
    certificate_password = signature_auth.get("certificatePassword")
    certificate_filename = signature_auth.get("certificateFilename")
    detected_type = signature_auth.get("type")
    rest = {
        k: v
        for k, v in signature_auth.items()
        if k not in ("certificateContent", "certificatePassword", "certificateFilename", "type")
    }

    if not detected_type:
        if certificate_filename:
            ext = certificate_filename.lower()
            if ext.endswith((".pfx", ".p12")):
                detected_type = "pfx"
            elif ext.endswith(".jks"):
                detected_type = "jks"
            elif ext.endswith((".pem", ".key")):
                detected_type = "pem"
        if not detected_type:
            detected_type = _detect_legacy_type(signature_auth)

    has_generic_fields = bool(certificate_content or certificate_password or certificate_filename)
    if not has_generic_fields and not detected_type:
        return signature_auth

    processed = dict(rest)
    if detected_type:
        processed["type"] = detected_type

    if detected_type:
        if detected_type == "pfx":
            if certificate_content and not processed.get("pfxContent"):
                processed["pfxContent"] = certificate_content
            if certificate_password and not processed.get("pfxPassword"):
                processed["pfxPassword"] = certificate_password
            if certificate_filename and not processed.get("pfxPath"):
                processed["certificateFilename"] = certificate_filename
        elif detected_type == "jks":
            if certificate_content and not processed.get("keystoreContent"):
                processed["keystoreContent"] = certificate_content
            if certificate_password and not processed.get("keystorePassword"):
                processed["keystorePassword"] = certificate_password
            if certificate_filename and not processed.get("keystorePath"):
                processed["certificateFilename"] = certificate_filename
        elif detected_type == "pem":
            if certificate_content and not processed.get("privateKey"):
                processed["privateKey"] = base64.b64decode(certificate_content).decode(
                    "utf-8", "replace"
                )
            if certificate_password:
                processed["certificatePassword"] = certificate_password
            if certificate_filename:
                processed["certificateFilename"] = certificate_filename
        else:
            raise ValueError(f"[Http Provider] Unknown certificate type: {detected_type}")
    elif has_generic_fields:
        raise ValueError(
            "[Http Provider] Cannot determine certificate type from filename: "
            f"{certificate_filename or 'no filename provided'}"
        )
    return processed


def _detect_legacy_type(signature_auth: dict[str, Any]) -> str | None:
    if signature_auth.get("privateKeyPath") or signature_auth.get("privateKey"):
        return "pem"
    if signature_auth.get("keystorePath") or signature_auth.get("keystoreContent"):
        return "jks"
    if (
        signature_auth.get("pfxPath")
        or signature_auth.get("pfxContent")
        or (signature_auth.get("certPath") and signature_auth.get("keyPath"))
    ):
        return "pfx"
    return None


def diagnose_private_key_material(material: Any) -> str | None:
    """Why PEM key material cannot be used for signing, or None when it looks sound."""
    text = material.strip() if isinstance(material, str) else ""
    if not text:
        return "it is empty"
    if "BEGIN CERTIFICATE" in text:
        return "it is a certificate, not a private key"
    if "PUBLIC KEY-----" in text:
        return "it is a public key, not a private key"
    if "ENCRYPTED PRIVATE KEY" in text or "Proc-Type: 4,ENCRYPTED" in text:
        return "it is passphrase-protected; decrypt it first or supply an unencrypted key"
    if "PRIVATE KEY-----" not in text:
        return 'it is not PEM-encoded (no "-----BEGIN ... PRIVATE KEY-----" header)'
    if "-----END" not in text:
        return 'it is truncated (no "-----END ... PRIVATE KEY-----" line)'
    return None


def _require_cryptography(feature: str) -> Any:
    try:
        import cryptography  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f'{feature} requires the "cryptography" Python package, which is not installed. '
            "Install it with: pip install cryptography"
        ) from e


def read_pkcs12_private_key(data: bytes, password: str) -> str:
    """The private key of a PFX/PKCS#12 bundle as PEM text."""
    _require_cryptography("PFX certificate support")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.serialization import pkcs12

    key, _cert, _extra = pkcs12.load_key_and_certificates(data, password.encode())
    if key is None:
        return ""
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def _hash_for_algorithm(algorithm: str) -> Any:
    from cryptography.hazmat.primitives import hashes

    name = algorithm.lower()
    for prefix in ("rsa-", "ecdsa-with-", "ecdsa-"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    table = {
        "sha1": hashes.SHA1,
        "sha224": hashes.SHA224,
        "sha256": hashes.SHA256,
        "sha384": hashes.SHA384,
        "sha512": hashes.SHA512,
        "sha3-256": hashes.SHA3_256,
        "sha3-384": hashes.SHA3_384,
        "sha3-512": hashes.SHA3_512,
        "md5": hashes.MD5,
    }
    if name not in table:
        raise ValueError("Invalid digest")
    return table[name]()


def sign_with_private_key(private_key_pem: str, algorithm: str, data: str) -> bytes:
    """crypto.createSign(algorithm).update(data).sign(privateKey): PKCS#1 v1.5 for RSA keys,
    DER-encoded ECDSA for EC keys."""
    _require_cryptography("Signature authentication")
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    digest = _hash_for_algorithm(algorithm)
    key = load_pem_private_key(private_key_pem.encode(), password=None)
    if isinstance(key, rsa.RSAPrivateKey):
        return key.sign(data.encode(), padding.PKCS1v15(), digest)
    if isinstance(key, ec.EllipticCurvePrivateKey):
        return key.sign(data.encode(), ec.ECDSA(digest))
    raise ValueError(f"Unsupported key type for signing: {type(key).__name__}")


async def generate_signature(signature_auth: dict[str, Any], signature_timestamp: int) -> str:
    """Load the configured private key and sign the rendered signatureDataTemplate (base64)."""
    try:
        base_path = cli_state.base_path() or ""
        auth_type = signature_auth.get("type") or _detect_legacy_type(signature_auth)
        private_key: str
        key_source: str
        if auth_type == "pem":
            if signature_auth.get("privateKeyPath"):
                resolved = safe_resolve(base_path, signature_auth["privateKeyPath"])
                with open(resolved, encoding="utf-8") as f:
                    private_key = f.read()
                key_source = f"privateKeyPath {resolved}"
            elif signature_auth.get("privateKey"):
                private_key = signature_auth["privateKey"]
                key_source = "the configured privateKey"
            elif signature_auth.get("certificateContent"):
                private_key = base64.b64decode(signature_auth["certificateContent"]).decode(
                    "utf-8", "replace"
                )
                key_source = "certificateContent"
            else:
                raise ValueError(
                    "PEM private key is required. Provide privateKey, privateKeyPath, or "
                    "certificateContent"
                )
        elif auth_type == "jks":
            password = (
                signature_auth.get("keystorePassword")
                or signature_auth.get("certificatePassword")
                or envars.get_env_string("PROMPTFOO_JKS_PASSWORD")
            )
            if not password:
                raise ValueError(
                    "JKS keystore password is required. Provide it via config "
                    "keystorePassword/certificatePassword or PROMPTFOO_JKS_PASSWORD environment "
                    "variable"
                )
            raise ValueError(
                "JKS keystores are not supported by promptphoo. Convert the keystore to PEM or "
                "PKCS#12 (keytool -importkeystore -deststoretype PKCS12) and use type pem or pfx"
            )
        elif auth_type == "pfx":
            private_key, key_source = _load_pfx_key(signature_auth, base_path)
        else:
            raise ValueError(f"Unsupported signature auth type: {_js(signature_auth.get('type'))}")

        problem = diagnose_private_key_material(private_key)
        if problem:
            raise ValueError(f"Private key from {key_source} cannot be used: {problem}")

        data = (
            get_engine()
            .render_string(
                signature_auth.get("signatureDataTemplate") or "{{signatureTimestamp}}",
                {"signatureTimestamp": signature_timestamp},
            )
            .replace("\\n", "\n")
        )
        algorithm = signature_auth.get("signatureAlgorithm") or "SHA256"
        logger.debug(
            "[Signature Auth] Preparing to sign with algorithm=%s, dataLength=%d, keyProvided=%s",
            algorithm,
            len(data),
            bool(private_key),
        )
        try:
            return base64.b64encode(sign_with_private_key(private_key, algorithm, data)).decode()
        except Exception as e:
            logger.error(
                "[Signature Auth] Signing failed: %s; keyLength=%d, algorithm=%s",
                e,
                len(private_key),
                algorithm,
            )
            raise
    except Exception as err:
        logger.error("Error generating signature: %s", err)
        raise ValueError(f"Failed to generate signature: {err}") from err


def _js(value: Any) -> str:
    return "undefined" if value is None else str(value)


def _load_pfx_key(signature_auth: dict[str, Any], base_path: str) -> tuple[str, str]:
    has_pfx_content = signature_auth.get("pfxContent") or signature_auth.get("certificateContent")
    has_pfx_path = signature_auth.get("pfxPath")
    has_cert_and_key = (signature_auth.get("certPath") and signature_auth.get("keyPath")) or (
        signature_auth.get("certContent") and signature_auth.get("keyContent")
    )
    if has_pfx_path or has_pfx_content:
        password = (
            signature_auth.get("pfxPassword")
            or signature_auth.get("certificatePassword")
            or envars.get_env_string("PROMPTFOO_PFX_PASSWORD")
        )
        if not password:
            raise ValueError(
                "PFX certificate password is required. Provide it via config "
                "pfxPassword/certificatePassword or PROMPTFOO_PFX_PASSWORD environment variable"
            )
        try:
            if has_pfx_content:
                data = base64.b64decode(has_pfx_content)
            else:
                resolved = safe_resolve(base_path, signature_auth["pfxPath"])
                try:
                    with open(resolved, "rb") as f:
                        data = f.read()
                except FileNotFoundError as e:
                    raise RuntimeError(
                        f"ENOENT: no such file or directory, open '{resolved}'"
                    ) from e
            key = read_pkcs12_private_key(data, password)
            if not key:
                logger.error("[Signature Auth][PFX] No private key extracted from PFX")
                raise ValueError("No private key found in PFX file")
            source = (
                "pfxContent"
                if has_pfx_content
                else f"pfxPath {safe_resolve(base_path, signature_auth['pfxPath'])}"
            )
            return key, source
        except Exception as err:
            message = str(err)
            if "ENOENT" in message and signature_auth.get("pfxPath"):
                resolved = safe_resolve(base_path, signature_auth["pfxPath"])
                raise ValueError(f"PFX file not found: {resolved}") from err
            if "invalid" in message or "decrypt" in message:
                raise ValueError(f"Invalid PFX file format or wrong password: {message}") from err
            logger.error("Error loading PFX certificate: %s", err)
            what = "content is valid" if has_pfx_content else "file exists"
            raise ValueError(
                f"Failed to load PFX certificate. Make sure the {what} and the password is "
                f"correct: Error: {message}"
            ) from err
    if has_cert_and_key:
        try:
            if signature_auth.get("keyContent"):
                key = base64.b64decode(signature_auth["keyContent"]).decode("utf-8", "replace")
                return key, "keyContent"
            cert_path = safe_resolve(base_path, signature_auth["certPath"])
            key_path = safe_resolve(base_path, signature_auth["keyPath"])
            if not path_exists(cert_path):
                raise ValueError(f"Certificate file not found: {cert_path}")
            try:
                with open(key_path, encoding="utf-8") as f:
                    key = f.read()
            except FileNotFoundError as e:
                raise ValueError(f"Key file not found: {key_path}") from e
            return key, f"keyPath {key_path}"
        except Exception as err:
            logger.error("Error loading certificate/key files: %s", err)
            raise ValueError(
                "Failed to load certificate/key files. Make sure both files exist and are "
                f"readable: Error: {err}"
            ) from err
    raise ValueError(
        "PFX type requires either pfxPath, pfxContent, both certPath and keyPath, or both "
        "certContent and keyContent"
    )


def now_ms() -> int:
    return int(time.time() * 1000)


def needs_signature_refresh(
    timestamp: int, validity_ms: float, buffer_ms: float | None = None
) -> bool:
    elapsed = now_ms() - timestamp
    effective_buffer = buffer_ms if buffer_ms is not None else math.floor(validity_ms * 0.1)
    return elapsed + effective_buffer >= validity_ms
