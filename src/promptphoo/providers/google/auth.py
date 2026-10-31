"""Authentication for the Google AI Studio and Vertex AI providers. Ports
src/providers/google/auth.ts.

google-auth-library is replaced by the `google-auth` Python package (the `vertex` extra). Its
OAuth client is wrapped in `GoogleAuthClient`, whose `request` mirrors gaxios: it resolves with
`{data, status, headers}` and raises `GaxiosError` (carrying `.response`) on a non-2xx status.

Environment variable priority is aligned with the Python SDK:
1. config.apiKey (explicit)
2. VERTEX_API_KEY (Vertex mode only)
3. GOOGLE_API_KEY (primary)
4. GEMINI_API_KEY (secondary)
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any

from promptphoo import envars
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.shared import resolve_provider_env

logger = logging.getLogger(__name__)

DEFAULT_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

_MISSING_LIBRARY = (
    "The google-auth-library package is required for Vertex AI. "
    "Please install it: pip install 'promptphoo[vertex]'"
)


class GaxiosError(Exception):
    """A failed OAuth-client request; `response` has `status`, `statusText` and `data`."""

    def __init__(self, message: str, response: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.response = response


def _import_google_auth() -> Any:
    try:
        import google.auth  # type: ignore[import-not-found]
        import google.auth.transport  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(_MISSING_LIBRARY) from e
    return google.auth


def _transport_request() -> Any:
    """A google.auth.transport.Request backed by httpx (google-auth's own needs `requests`)."""
    import google.auth.transport  # type: ignore[import-not-found]
    import httpx

    class _Response(google.auth.transport.Response):
        def __init__(self, response: httpx.Response) -> None:
            self._response = response

        @property
        def status(self) -> int:
            return self._response.status_code

        @property
        def headers(self) -> Mapping[str, str]:
            return self._response.headers

        @property
        def data(self) -> bytes:
            return self._response.content

    class _Request(google.auth.transport.Request):
        def __call__(self, url, method="GET", body=None, headers=None, timeout=None, **kwargs):
            try:
                response = httpx.request(
                    method, url, content=body, headers=headers, timeout=timeout or 120
                )
            except httpx.HTTPError as e:
                import google.auth.exceptions  # type: ignore[import-not-found]

                raise google.auth.exceptions.TransportError(e) from e
            return _Response(response)

    return _Request()


class GoogleAuthClient:
    """The subset of google-auth-library's AuthClient used by the providers."""

    def __init__(self, credentials: Any) -> None:
        self.credentials = credentials

    async def get_access_token(self) -> str | None:
        creds = self.credentials
        if not getattr(creds, "valid", False):
            await asyncio.to_thread(creds.refresh, _transport_request())
        return getattr(creds, "token", None)

    async def request(
        self,
        url: str,
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
        data: Any = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        from promptphoo.fetch import fetch_with_proxy

        token = await self.get_access_token()
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        if token:
            request_headers["Authorization"] = f"Bearer {token}"
        quota_project = getattr(self.credentials, "quota_project_id", None)
        if quota_project:
            request_headers.setdefault("x-goog-user-project", quota_project)
        body = None if data is None else data if isinstance(data, str) else json_stringify(data)
        response = await fetch_with_proxy(
            url,
            {
                "method": method,
                "headers": request_headers,
                "body": body,
                **({"timeout": timeout / 1000} if timeout else {}),
            },
        )
        text = response.text()
        try:
            parsed: Any = js_json_parse(text) if text else ""
        except ValueError:
            parsed = text
        result = {
            "data": parsed,
            "status": response.status,
            "statusText": response.status_text,
            "headers": response.headers,
        }
        if not response.ok:
            raise GaxiosError(f"Request failed with status code {response.status}", result)
        return result


class GoogleAuthManager:
    """Centralized authentication manager for Google AI providers."""

    _cached_has_default_credentials: bool | None = None

    @staticmethod
    def get_api_key(
        config: Mapping[str, Any], env: Mapping[str, Any] | None = None, is_vertex_mode=False
    ) -> dict[str, Any]:
        """Get the API key with priority config > VERTEX_API_KEY (Vertex only) >
        GOOGLE_API_KEY > GEMINI_API_KEY > PALM_API_KEY. Scope wins over alias order."""
        if config.get("apiKey"):
            return {"apiKey": config["apiKey"], "source": "config"}

        keys = (
            ("VERTEX_API_KEY", "GOOGLE_API_KEY")
            if is_vertex_mode
            else ("GOOGLE_API_KEY", "GEMINI_API_KEY", "PALM_API_KEY")
        )
        masked: set[str] = set()
        layers = [
            env,
            envars.get_env_overrides("provider"),
            envars.get_env_overrides("suite"),
            envars.get_env_overrides("file"),
            {key: envars.get_env_string(key) for key in keys},
        ]
        for layer in layers:
            for source in keys:
                api_key = layer.get(source) if layer else None
                if source in masked or api_key is None:
                    continue
                masked.add(source)
                if not api_key:
                    continue
                if source == "VERTEX_API_KEY":
                    logger.warning(
                        "[Google] VERTEX_API_KEY is not a standard SDK env var. "
                        "Use GOOGLE_API_KEY instead."
                    )
                elif source == "PALM_API_KEY":
                    logger.warning(
                        "[Google] PALM_API_KEY is deprecated. Use GOOGLE_API_KEY instead."
                    )
                elif source == "GEMINI_API_KEY":
                    logger.debug(
                        "[Google] GEMINI_API_KEY is not a standard SDK env var. "
                        "Consider using GOOGLE_API_KEY."
                    )
                elif not is_vertex_mode and layer and layer.get("GEMINI_API_KEY"):
                    logger.debug(
                        "[Google] Both GOOGLE_API_KEY and GEMINI_API_KEY are set. "
                        "Using GOOGLE_API_KEY."
                    )
                return {"apiKey": str(api_key), "source": source}
        return {"apiKey": None, "source": "none"}

    @staticmethod
    def validate_and_warn(config: Mapping[str, Any], env: Mapping[str, Any] | None = None) -> None:
        api_key = config.get("apiKey")
        credentials = config.get("credentials")
        project_id = config.get("projectId")
        region = config.get("region")
        vertexai = config.get("vertexai")
        is_strict = config.get("strictMutualExclusivity") is True

        use_vertex_env = envars.get_provider_env_string(env, "GOOGLE_GENAI_USE_VERTEXAI")
        if use_vertex_env is None:
            use_vertex_env = envars.get_env_string("GOOGLE_GENAI_USE_VERTEXAI")
        cloud_project = (env or {}).get("GOOGLE_CLOUD_PROJECT")
        if cloud_project is None:
            cloud_project = envars.get_env_string("GOOGLE_CLOUD_PROJECT")
        has_project_environment = bool(
            (
                resolve_provider_env(
                    env, ["VERTEX_PROJECT_ID", "GOOGLE_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"]
                )
                or {}
            ).get("value")
        )

        if (project_id or region) and api_key:
            message = (
                "[Google] Project/location and API key are mutually exclusive in the client "
                "initializer. Use either apiKey for express mode OR projectId/region for OAuth "
                "mode, not both."
            )
            if is_strict:
                raise ValueError(message)
            logger.warning(message)

        if use_vertex_env and vertexai is False:
            logger.warning(
                "[Google] GOOGLE_GENAI_USE_VERTEXAI is set but vertexai: false was specified in "
                "config. Config takes precedence."
            )

        if cloud_project and project_id and cloud_project != project_id:
            logger.warning(
                "[Google] Both GOOGLE_CLOUD_PROJECT and config.projectId are set with different "
                "values. Using config.projectId."
            )

        if api_key and credentials:
            logger.debug(
                "[Google] Both apiKey and credentials are set. Using API key (express mode). "
                "Set expressMode: false to use OAuth/ADC instead."
            )

        if (
            vertexai
            and not api_key
            and not project_id
            and not has_project_environment
            and not (credentials)
        ):
            adc = (env or {}).get("GOOGLE_APPLICATION_CREDENTIALS")
            if adc is None:
                adc = envars.get_env_string("GOOGLE_APPLICATION_CREDENTIALS")
            if not adc:
                logger.debug(
                    "[Google] Vertex AI mode enabled but no projectId, credentials, or ADC "
                    "detected. Authentication may fail."
                )

    @staticmethod
    def determine_vertex_mode(config: Mapping[str, Any], env: Mapping[str, Any] | None = None):
        if config.get("vertexai") is not None:
            return config["vertexai"]

        use_vertex_env = envars.get_provider_env_string(env, "GOOGLE_GENAI_USE_VERTEXAI")
        if use_vertex_env is None:
            use_vertex_env = envars.get_env_string("GOOGLE_GENAI_USE_VERTEXAI")
        if use_vertex_env in ("true", "1"):
            logger.debug("[Google] Vertex AI mode enabled via GOOGLE_GENAI_USE_VERTEXAI")
            return True
        if use_vertex_env in ("false", "0"):
            return False

        has_project_id = bool(
            config.get("projectId")
            or (
                resolve_provider_env(
                    env, ["VERTEX_PROJECT_ID", "GOOGLE_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"]
                )
                or {}
            ).get("value")
        )
        if has_project_id or config.get("credentials"):
            logger.debug(
                "[Google] Auto-detected Vertex AI mode from projectId/credentials. "
                "Set vertexai: true/false explicitly to suppress this message."
            )
            return True
        return False

    @staticmethod
    def load_credentials(credentials: Any = None) -> str | None:
        """Load credentials from a file:// reference, a JSON string or a parsed object."""
        if not credentials:
            return None
        if isinstance(credentials, dict | list):
            return json_stringify(credentials)
        if credentials.startswith("file://"):
            from promptphoo.files import maybe_load_from_external_file

            try:
                loaded = maybe_load_from_external_file(credentials)
            except Exception as error:
                raise ValueError(f"Failed to load credentials from file: Error: {error}") from error
            if isinstance(loaded, dict | list):
                return json_stringify(loaded)
            return loaded
        return credentials

    @classmethod
    async def get_oauth_client(
        cls, options: Mapping[str, Any] | str | None = None
    ) -> tuple[GoogleAuthClient, str | None]:
        """Create an OAuth client and detect the project ID."""
        opts: Mapping[str, Any] = (
            {"credentials": options} if isinstance(options, str) else (options or {})
        )
        google_auth_options = dict(opts.get("googleAuthOptions") or {})
        scopes = opts.get("scopes")
        if scopes is None:
            scopes = google_auth_options.get("scopes")
        if scopes is None:
            scopes = DEFAULT_SCOPE
        scope_list = [scopes] if isinstance(scopes, str) else list(scopes)
        key_filename = google_auth_options.get("keyFilename") or opts.get("keyFilename")

        google_auth = _import_google_auth()
        processed = cls.load_credentials(opts.get("credentials"))

        def load() -> tuple[Any, str | None]:
            if processed:
                try:
                    info = json.loads(processed)
                except json.JSONDecodeError as parse_error:
                    raise ValueError(
                        f"[Google] Invalid credentials JSON format: {parse_error}"
                    ) from parse_error
                try:
                    return google_auth.load_credentials_from_dict(info, scopes=scope_list)
                except Exception as error:
                    logger.error("[Google] Could not load credentials: %s", error)
                    raise ValueError(f"[Google] Could not load credentials: {error}") from error
            if isinstance(google_auth_options.get("credentials"), dict):
                return google_auth.load_credentials_from_dict(
                    google_auth_options["credentials"], scopes=scope_list
                )
            if key_filename:
                return google_auth.load_credentials_from_file(key_filename, scopes=scope_list)
            return google_auth.default(scopes=scope_list)

        credentials, project_id = await asyncio.to_thread(load)
        project_id = google_auth_options.get("projectId") or project_id
        return GoogleAuthClient(credentials), project_id or None

    @classmethod
    async def resolve_project_id(
        cls, config: Mapping[str, Any], env: Mapping[str, Any] | None = None
    ) -> str:
        project = resolve_provider_env(
            env, ["VERTEX_PROJECT_ID", "GOOGLE_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"]
        )
        if project and project["name"] != "GOOGLE_CLOUD_PROJECT" and not config.get("projectId"):
            logger.debug(
                "[Google] %s is not a standard SDK env var. Consider using GOOGLE_CLOUD_PROJECT.",
                project["name"],
            )
        configured = config.get("projectId") or (project or {}).get("value")
        if configured:
            return configured
        _, auth_project_id = await cls.get_oauth_client({
            "credentials": config.get("credentials"),
            "googleAuthOptions": config.get("googleAuthOptions"),
            "keyFilename": config.get("keyFilename"),
            "scopes": config.get("scopes"),
        })  # fmt: skip
        return auth_project_id or ""

    @staticmethod
    def resolve_region(
        config: Mapping[str, Any],
        env: Mapping[str, Any] | None = None,
        has_api_key: bool | None = None,
        model_default_region: str | None = None,
    ) -> str:
        region = resolve_provider_env(env, ["VERTEX_REGION", "GOOGLE_CLOUD_LOCATION"])
        if region and region["name"] == "VERTEX_REGION" and not config.get("region"):
            logger.debug(
                "[Google] VERTEX_REGION is not a standard SDK env var. "
                "Consider using GOOGLE_CLOUD_LOCATION."
            )
        configured = config.get("region") or (region or {}).get("value")
        if configured:
            return configured
        if model_default_region:
            return model_default_region
        # SDK alignment: default to 'global' when Vertex AI mode without API key.
        if has_api_key is False:
            return "global"
        return "us-central1"

    @classmethod
    async def has_default_credentials(cls) -> bool:
        if cls._cached_has_default_credentials is not None:
            return cls._cached_has_default_credentials
        try:
            await cls.get_oauth_client()
            result = True
        except Exception:
            result = False
        cls._cached_has_default_credentials = result
        return result

    @classmethod
    def clear_cache(cls) -> None:
        cls._cached_has_default_credentials = None


load_credentials = GoogleAuthManager.load_credentials
get_google_client = GoogleAuthManager.get_oauth_client
resolve_project_id = GoogleAuthManager.resolve_project_id
get_google_api_key = GoogleAuthManager.get_api_key
determine_google_vertex_mode = GoogleAuthManager.determine_vertex_mode
has_google_default_credentials = GoogleAuthManager.has_default_credentials
clear_cached_auth = GoogleAuthManager.clear_cache
