"""
Secret provider abstraction.

On Databricks: reads from Databricks Secrets API (preferred) so that no
secret ever touches an environment variable or a file on disk.

Locally / CI: falls back to environment variables loaded from .env by
python-dotenv.  The .env file must never be committed (it is in .gitignore).

Usage
-----
    from configs.secrets import get_secret_provider
    sp = get_secret_provider()
    api_key = sp.require("qc-secrets", "MCP_API_KEY")

On Databricks the scope name defaults to "qc-secrets".  Create it once:
    databricks secrets create-scope qc-secrets
    databricks secrets put-secret qc-secrets MCP_API_KEY --string-value <hex>

Locally set the values in .env (see .env.example).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from configs.security_logging import log_secret_missing

logger = logging.getLogger(__name__)

# Default Databricks secrets scope for this application
DEFAULT_SCOPE = "qc-secrets"

# Canonical map: env-var name → (scope, key) on Databricks.
# Used by validate_all() to check every required secret at startup.
_SECRET_MAP: dict[str, tuple[str, str]] = {
    "MCP_API_KEY":        (DEFAULT_SCOPE, "MCP_API_KEY"),
    "DATABRICKS_HOST":    (DEFAULT_SCOPE, "DATABRICKS_HOST"),
    "DATABRICKS_TOKEN":   (DEFAULT_SCOPE, "DATABRICKS_TOKEN"),
}

_LOCAL_REQUIRED = ["DATABRICKS_HOST", "DATABRICKS_TOKEN", "MCP_API_KEY"]


class SecretProvider:
    """
    Unified secret loader.

    - On Databricks: uses Databricks Secrets API via the SDK.
    - Locally:       reads from environment variables (loaded by dotenv).

    Secrets are cached in memory after the first read to avoid repeated
    API calls.  The cache is process-scoped; secrets cannot be rotated
    without restarting the process.
    """

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}
        self._is_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, scope_or_env_var: str, key: Optional[str] = None) -> str:
        """
        Retrieve a secret value, or "" if not found.

        On Databricks:  get(scope, key)       → Databricks Secrets API
        Locally:        get(env_var_name)      → os.environ
        Either:         get(env_var, key=None) → os.environ fallback
        """
        cache_key = f"{scope_or_env_var}:{key}" if key else scope_or_env_var
        if cache_key in self._cache:
            return self._cache[cache_key]

        value = ""
        if self._is_databricks and key:
            value = self._databricks_secret(scope_or_env_var, key)

        # Always try env var as fallback (handles local dev and CI)
        if not value:
            env_var = key if key else scope_or_env_var
            value = os.environ.get(env_var, "")

        if value:
            self._cache[cache_key] = value
        return value

    def require(self, scope_or_env_var: str, key: Optional[str] = None) -> str:
        """Like get() but raises RuntimeError if the secret is absent."""
        value = self.get(scope_or_env_var, key)
        name = f"{scope_or_env_var}/{key}" if key else scope_or_env_var
        if not value:
            log_secret_missing(name=name)
            raise RuntimeError(
                f"Required secret {name!r} is not configured. "
                + (
                    f"On Databricks: databricks secrets put-secret {DEFAULT_SCOPE} {key or scope_or_env_var} --string-value <value>"
                    if self._is_databricks
                    else f"Locally: add {key or scope_or_env_var}=<value> to your .env file (see .env.example)"
                )
            )
        return value

    def validate_all(self) -> list[str]:
        """
        Check all required secrets are present.  Returns a list of missing
        secret names.  Call this once at application startup and fail fast
        if the list is non-empty.
        """
        missing: list[str] = []

        if self._is_databricks:
            for env_var, (scope, key) in _SECRET_MAP.items():
                if not self.get(scope, key):
                    missing.append(f"{scope}/{key}")
        else:
            for env_var in _LOCAL_REQUIRED:
                if not os.environ.get(env_var):
                    missing.append(env_var)

        for name in missing:
            log_secret_missing(name=name)

        return missing

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _databricks_secret(self, scope: str, key: str) -> str:
        """Read a secret from the Databricks Secrets API."""
        # Path 1: dbutils (available in interactive notebooks)
        try:
            import IPython
            ns = IPython.get_ipython()
            if ns is not None:
                dbutils = ns.user_ns.get("dbutils")
                if dbutils is not None:
                    return str(dbutils.secrets.get(scope=scope, key=key))
        except Exception:
            pass

        # Path 2: Databricks SDK (available in jobs, serverless)
        try:
            from databricks.sdk import WorkspaceClient
            client = WorkspaceClient()
            secret = client.secrets.get_secret(scope=scope, key=key)
            if secret and secret.value:
                return str(secret.value)
        except Exception:
            # Suppress details — the error may reference secret names
            logger.debug("Databricks Secrets API read failed for %s/%s", scope, key)

        return ""


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_provider: Optional[SecretProvider] = None


def get_secret_provider() -> SecretProvider:
    """Return the process-wide SecretProvider instance (lazy init)."""
    global _provider
    if _provider is None:
        _provider = SecretProvider()
    return _provider
