"""Shared Azure provider state and authentication. Ports src/providers/azure/generic.ts.

Authentication runs on the first request rather than eagerly in the constructor (a Python
constructor has no running event loop to start it on); concurrent requests still share one
in-flight attempt, and a failed attempt is retried by the next request (see D24).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from promptphoo import envars
from promptphoo.providers import credentials
from promptphoo.providers.azure import identity
from promptphoo.providers.azure.util import throw_configuration_error
from promptphoo.providers.base import ApiProvider, ProviderResponse

logger = logging.getLogger(__name__)

# Upstream's message keeps the trailing space after "failed." and the trailing indent.
AZURE_AUTHENTICATION_FAILED = (
    "Azure Authentication failed. \n"
    "Please choose one of the following options:\n"
    "  1. Set an API key via the AZURE_API_KEY environment variable.\n"
    "  2. Provide client credentials (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID).\n"
    "  3. Authenticate with Azure CLI using az login.\n"
    "    "
)

_ENDPOINT_ENVARS = [
    "AZURE_API_BASE_URL",
    "AZURE_OPENAI_API_BASE_URL",
    "AZURE_OPENAI_BASE_URL",
    "AZURE_API_HOST",
    "AZURE_OPENAI_API_HOST",
]


class AzureAuthenticationError(RuntimeError):
    pass


class AzureGenericProvider(ApiProvider):
    # Refresh the bearer token when it is within this window of expiring.
    TOKEN_REFRESH_WINDOW_MS = 5 * 60 * 1000

    def __init__(self, deployment_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        super().__init__(options)
        self.deployment_name = deployment_name
        config = options.get("config") or {}
        self.config: dict[str, Any] = config
        self.env = options.get("env")
        self.api_host: str | None = None
        self.api_base_url: str | None = None
        if config.get("apiBaseUrl") or config.get("apiHost"):
            self.api_base_url = config.get("apiBaseUrl")
            self.api_host = config.get("apiHost")
        else:
            # Keep the legacy Azure aliases within the scope that selected the endpoint.
            endpoint = credentials.resolve_provider_env(self.env, _ENDPOINT_ENVARS)
            if endpoint and endpoint["name"].endswith("_HOST"):
                self.api_host = endpoint["value"]
            else:
                self.api_base_url = endpoint["value"] if endpoint else None

        self.auth_headers: dict[str, str] | None = None
        # Expiry of the cached Entra ID bearer token (ms epoch), if token auth is in use.
        self.auth_token_expires_on_timestamp: float | None = None
        self._authentication: asyncio.Future[None] | None = None
        self._cached_credential: Any = None
        self._warned_partial_service_principal = False

    @property
    def deploymentName(self) -> str:  # noqa: N802 - upstream attribute name
        return self.deployment_name

    # --- authentication --------------------------------------------------------------------

    def initialize(self) -> asyncio.Future[None]:
        """Fetch auth headers once; concurrent callers share the in-flight attempt."""
        if self._authentication is None:

            async def authenticate() -> None:
                try:
                    self.auth_headers = await self.get_auth_headers()
                finally:
                    self._authentication = None

            self._authentication = asyncio.ensure_future(authenticate())
        return self._authentication

    async def ensure_initialized(self) -> None:
        if not self.auth_headers:
            await self.initialize()
        await self._refresh_auth_token_if_needed()

    async def _refresh_auth_token_if_needed(self) -> None:
        """Re-mint the Entra ID bearer token at or near expiry. No-op for api-key auth and for
        tokens whose expiry is unknown."""
        if not (self.auth_headers or {}).get("Authorization"):
            return
        expires_at = self.auth_token_expires_on_timestamp
        if expires_at is None:
            return
        if identity.now_ms() < expires_at - self.TOKEN_REFRESH_WINDOW_MS:
            return
        await self.initialize()

    def get_api_key(self) -> str | None:
        return credentials.resolve_provider_api_key(
            self.config, self.env, ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY"]
        )

    def get_api_key_or_throw(self) -> str:
        api_key = self.get_api_key()
        if not api_key:
            throw_configuration_error("Azure API key must be set.")
        return api_key

    def _env_value(self, config_key: str, envar: str) -> str | None:
        value = self.config.get(config_key)
        if value:
            return value
        provider_value = envars.get_provider_env_string(self.env, envar)
        return provider_value if provider_value is not None else envars.get_env_string(envar)

    async def get_azure_token_credential(self) -> Any:
        """One credential per provider; a failed creation is evicted so the next call retries."""
        # Creation is synchronous here (no dynamic import), so concurrent callers cannot race.
        if self._cached_credential is None:
            self._cached_credential = self._create_token_credential()
        return self._cached_credential

    def _create_token_credential(self) -> Any:
        client_secret = self._env_value("azureClientSecret", "AZURE_CLIENT_SECRET")
        client_id = self._env_value("azureClientId", "AZURE_CLIENT_ID")
        tenant_id = self._env_value("azureTenantId", "AZURE_TENANT_ID")
        authority_host = self._env_value("azureAuthorityHost", "AZURE_AUTHORITY_HOST")

        try:
            if client_secret and client_id and tenant_id:
                logger.debug("[Azure] Using service principal credentials")
                return identity.ClientSecretCredential(
                    tenant_id,
                    client_id,
                    client_secret,
                    {"authorityHost": authority_host or identity.DEFAULT_AUTHORITY_HOST},
                )

            if not client_id and not client_secret and not tenant_id:
                logger.debug("[Azure] Using Azure CLI credentials")
            elif not self._warned_partial_service_principal:
                self._warned_partial_service_principal = True
                missing = [
                    label
                    for label, present in (
                        ("azureClientId (AZURE_CLIENT_ID)", client_id),
                        ("azureClientSecret (AZURE_CLIENT_SECRET)", client_secret),
                        ("azureTenantId (AZURE_TENANT_ID)", tenant_id),
                    )
                    if not present
                ]
                logger.warning(
                    "[Azure] Service principal configuration is incomplete, missing %s. "
                    "Falling back to Azure CLI credentials.",
                    ", ".join(missing),
                    extra={"missing": missing},
                )

            return identity.AzureCliCredential()
        except Exception as err:
            logger.error("Error creating Azure credential: %s", err)
            raise

    async def get_access_token(self) -> str:
        credential = await self.get_azure_token_credential()
        token_scope = self._env_value("azureTokenScope", "AZURE_TOKEN_SCOPE")
        token_response = await credential.get_token(
            token_scope or "https://cognitiveservices.azure.com/.default"
        )
        if not token_response:
            throw_configuration_error("Failed to retrieve access token.")
        self.auth_token_expires_on_timestamp = token_response.get("expiresOnTimestamp")
        return token_response["token"]

    async def get_auth_headers(self) -> dict[str, str]:
        api_key = self.get_api_key()
        if api_key:
            return {"api-key": api_key}
        try:
            token = await self.get_access_token()
            return {"Authorization": "Bearer " + token}
        except Exception as err:
            logger.info("Azure Authentication failed. Please check your credentials: %s", err)
            raise AzureAuthenticationError(AZURE_AUTHENTICATION_FAILED) from err

    # --- endpoint --------------------------------------------------------------------------

    def get_api_base_url(self) -> str | None:
        if self.api_base_url:
            return re.sub(r"/$", "", self.api_base_url)
        host = (
            re.sub(r"/$", "", re.sub(r"^(https?://)", "", self.api_host)) if self.api_host else ""
        )
        if not host:
            return None
        return f"https://{host}"

    def default_id(self) -> str:
        return f"azure:{self.deployment_name}"

    def __str__(self) -> str:
        return f"[Azure Provider {self.deployment_name}]"

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        raise NotImplementedError("Not implemented")
