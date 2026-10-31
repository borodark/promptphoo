"""Entra ID token credentials for the Azure providers (see D24).

Upstream builds `ClientSecretCredential` or `AzureCliCredential` from `@azure/identity`. This
module reimplements the two over plain HTTP and a subprocess, with the same constructor
arguments, the same `get_token(scope)` result shape (`{"token", "expiresOnTimestamp"}`) and the
same user-facing error messages. Other credential types are not supported.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from promptphoo import fetch

DEFAULT_AUTHORITY_HOST = "https://login.microsoftonline.com"
# MSAL hands back a cached token until it is within five minutes of expiry.
TOKEN_REFRESH_WINDOW_MS = 5 * 60 * 1000

AZURE_CLI_NOT_INSTALLED = (
    "Azure CLI could not be found. Please visit https://aka.ms/azure-cli for installation "
    "instructions and then, once installed, authenticate to your Azure account using 'az login'."
)
AZURE_CLI_NOT_LOGGED_IN = (
    "Please run 'az login' from a command prompt to authenticate before using this credential."
)

_TENANT_ID = re.compile(r"^[0-9a-zA-Z.-]+$")
_SCOPE = re.compile(r"^[0-9a-zA-Z\-_.:/]+$")


class CredentialUnavailableError(RuntimeError):
    """The credential cannot be used in this environment (no CLI, not logged in)."""

    name = "CredentialUnavailableError"


class AuthenticationError(RuntimeError):
    """The token endpoint rejected the request."""

    name = "AuthenticationError"


def now_ms() -> float:
    return time.time() * 1000


def _check_tenant_id(tenant_id: str) -> None:
    if not _TENANT_ID.match(tenant_id):
        raise ValueError(
            "Invalid tenant id provided. You can locate your tenant id by following the "
            "instructions listed here: https://learn.microsoft.com/partner-center/"
            "find-ids-and-domain-names"
        )


def _check_scope(scope: str) -> None:
    if not _SCOPE.match(scope):
        raise ValueError(
            "Invalid scope was specified by the user or calling client. Scope must only contain "
            "alphanumeric characters, dashes, underscores, periods, colons and slashes."
        )


class ClientSecretCredential:
    """OAuth 2.0 client credentials grant against `<authorityHost>/<tenant>/oauth2/v2.0/token`.

    Tokens are cached per scope and reused until they are within five minutes of expiry.
    """

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        options: dict[str, Any] | None = None,
    ) -> None:
        _check_tenant_id(tenant_id)
        self.tenant_id = tenant_id
        self.client_id = client_id
        self._client_secret = client_secret
        authority = (options or {}).get("authorityHost") or DEFAULT_AUTHORITY_HOST
        self.authority_host = authority.rstrip("/")
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def token_url(self) -> str:
        return f"{self.authority_host}/{self.tenant_id}/oauth2/v2.0/token"

    async def get_token(self, scope: str) -> dict[str, Any] | None:
        cached = self._cache.get(scope)
        if cached and now_ms() < cached["expiresOnTimestamp"] - TOKEN_REFRESH_WINDOW_MS:
            return cached
        form = urlencode(
            {
                "client_id": self.client_id,
                "scope": scope,
                "grant_type": "client_credentials",
                "client_secret": self._client_secret,
            }
        )
        response = await fetch.fetch_with_timeout(
            self.token_url,
            {
                "method": "POST",
                "headers": {"Content-Type": "application/x-www-form-urlencoded"},
                "body": form,
            },
            fetch.get_request_timeout_ms(),
        )
        try:
            data = response.json()
        except ValueError:
            data = None
        if not response.ok or not isinstance(data, dict) or not data.get("access_token"):
            detail = (
                (data.get("error_description") or data.get("error"))
                if isinstance(data, dict)
                else None
            ) or f"HTTP {response.status} {response.status_text}"
            raise AuthenticationError(f"ClientSecretCredential authentication failed: {detail}")
        expires_in: Any = data.get("expires_in")
        try:
            seconds = float(expires_in)
        except (TypeError, ValueError):
            seconds = math.nan
        token: dict[str, Any] = {"token": data["access_token"]}
        if math.isfinite(seconds):
            token["expiresOnTimestamp"] = now_ms() + seconds * 1000
            self._cache[scope] = token
        return token


async def run_az(args: list[str]) -> tuple[int, str, str]:
    """Run the Azure CLI; returns (exit code, stdout, stderr). Tests replace this."""
    executable = shutil.which("az")
    if executable is None:
        raise FileNotFoundError("az")
    process = await asyncio.create_subprocess_exec(
        executable,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return (
        process.returncode if process.returncode is not None else 1,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _parse_cli_expiry(response: dict[str, Any]) -> float:
    expires_on = response.get("expires_on")
    try:
        if expires_on is not None and not isinstance(expires_on, bool):
            return float(expires_on) * 1000
    except (TypeError, ValueError):
        pass
    value = response.get("expiresOn")
    try:
        # `expiresOn` is a local-time "YYYY-MM-DD HH:MM:SS.ffffff" string.
        return datetime.fromisoformat(str(value)).timestamp() * 1000
    except ValueError as err:
        raise AuthenticationError(
            'Unexpected response from Azure CLI when getting token. Expected "expiresOn" to be '
            f"a RFC3339 date string. Got: {value}"
        ) from err


class AzureCliCredential:
    """`az account get-access-token --output json --resource <resource>`."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        self.tenant_id: str | None = options.get("tenantId")
        if self.tenant_id:
            _check_tenant_id(self.tenant_id)

    async def get_token(self, scope: str) -> dict[str, Any] | None:
        _check_scope(scope)
        resource = scope[: -len("/.default")] if scope.endswith("/.default") else scope
        args = ["account", "get-access-token", "--output", "json", "--resource", resource]
        if self.tenant_id:
            args += ["--tenant", self.tenant_id]
        try:
            code, stdout, stderr = await run_az(args)
        except FileNotFoundError as err:
            raise CredentialUnavailableError(AZURE_CLI_NOT_INSTALLED) from err
        if code != 0:
            not_installed = (
                "'az' is not recognized" in stderr
                or "az: command not found" in stderr
                or "az: not found" in stderr
            )
            if not_installed:
                raise CredentialUnavailableError(AZURE_CLI_NOT_INSTALLED)
            if "az login" in stderr or "az account set" in stderr:
                raise CredentialUnavailableError(AZURE_CLI_NOT_LOGGED_IN)
            raise CredentialUnavailableError(stderr.strip() or "Failed to get token from Azure CLI")
        try:
            response = json.loads(stdout)
        except ValueError as err:
            raise CredentialUnavailableError(stdout) from err
        if not isinstance(response, dict) or not response.get("accessToken"):
            raise CredentialUnavailableError(stdout)
        return {
            "token": response["accessToken"],
            "expiresOnTimestamp": _parse_cli_expiry(response),
        }
