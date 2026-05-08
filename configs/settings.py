"""Central configuration: thresholds, table names, secret loading."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env for local development only.
# On Databricks, secrets are injected by the runtime or the Secrets API.
load_dotenv()

logger = logging.getLogger(__name__)


def is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def _secret(env_var: str, databricks_key: str | None = None) -> str:
    """
    Load a single secret using SecretProvider.

    Used as a field default_factory so the provider is only consulted when
    a config object is actually instantiated (not at import time).
    """
    from configs.secrets import get_secret_provider
    sp = get_secret_provider()
    # On Databricks, try the Secrets API first (scope = "qc-secrets")
    if is_databricks() and databricks_key:
        from configs.secrets import DEFAULT_SCOPE
        value = sp.get(DEFAULT_SCOPE, databricks_key)
        if value:
            return value
    # Fall back to environment variable (local dev / CI)
    return sp.get(env_var)


@dataclass
class ThresholdConfig:
    l1_auto_correct: float = 0.85
    l2_review_lower: float = 0.50
    l2_review_upper: float = 0.849
    skip_threshold: float = 0.50
    default_scorer: str = "WRatio"
    address_scorer: str = "token_sort_ratio"
    llm_borderline_lower: float = 0.75   # use LLM for 0.75-0.85 range
    max_geocode_per_run: int = 1000
    geocode_timeout: int = 5
    geocode_rate_limit: float = 1.1


@dataclass
class TableConfig:
    catalog: str = field(default_factory=lambda: os.getenv("QC_CATALOG", "workspace"))
    schema: str = field(default_factory=lambda: os.getenv("QC_SCHEMA", "qc_schema"))
    audit_log_table: str = field(default_factory=lambda: os.getenv("QC_LOG_TABLE", "qc_audit_log"))
    geocode_cache_table: str = field(
        default_factory=lambda: os.getenv("QC_GEOCODE_CACHE_TABLE", "geocode_cache")
    )
    city_ref_table: str = "us_cities"
    state_ref_table: str = "us_states"
    zip_ref_table: str = "us_zip_codes"

    @property
    def audit_log_fqn(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.audit_log_table}"

    @property
    def geocode_cache_fqn(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.geocode_cache_table}"


@dataclass
class DatabricksLLMConfig:
    """Used when running on Databricks — token is injected automatically."""
    workspace_host: str = field(
        default_factory=lambda: _secret("DATABRICKS_HOST", "DATABRICKS_HOST")
        or os.getenv("DB_WORKSPACE_URL", "")
    )
    # Token resolution: Databricks Secrets scope → env var → SDK auto-discovery.
    token: str = field(
        default_factory=lambda: _secret("DATABRICKS_TOKEN", "DATABRICKS_TOKEN")
    )
    model: str = field(
        default_factory=lambda: os.getenv(
            "DATABRICKS_LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct"
        )
    )
    max_tokens: int = 1024
    temperature: float = 0.1
    enable_rationale: bool = True

    def __repr__(self) -> str:
        masked = f"{self.token[:8]}***" if len(self.token) > 8 else "***"
        return (
            f"DatabricksLLMConfig(workspace_host={self.workspace_host!r}, "
            f"token={masked!r}, model={self.model!r})"
        )

    __str__ = __repr__


@dataclass
class AppConfig:
    env: str = field(default_factory=lambda: os.getenv("QC_ENV", "local"))
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    tables: TableConfig = field(default_factory=TableConfig)
    databricks_llm: DatabricksLLMConfig = field(default_factory=DatabricksLLMConfig)
    geocoder: str = field(default_factory=lambda: os.getenv("GEOCODER", "nominatim"))
    geocoder_user_agent: str = field(
        default_factory=lambda: os.getenv("GEOCODER_USER_AGENT", "databricks-qc-agent/1.0")
    )
    mlflow_experiment_name: str = "/Shared/QC-Pipeline"

    @property
    def is_local(self) -> bool:
        return self.env == "local"

    @property
    def is_databricks(self) -> bool:
        return is_databricks()

    @property
    def llm_config(self) -> DatabricksLLMConfig:
        return self.databricks_llm


_config: AppConfig | None = None


def get_databricks_token() -> str:
    """
    Return the Databricks bearer token for REST API calls.

    Resolution order (first non-empty wins):
    1. Databricks Secrets scope (``qc-secrets/DATABRICKS_TOKEN``)
    2. ``DATABRICKS_TOKEN`` env var (auto-injected on standard clusters; set locally via .env)
    3. Databricks SDK auto-discovery (works in serverless / notebooks without env var)

    Never logs or raises on the token value — callers get "" if unavailable.
    """
    # _secret checks secrets scope first, then env var
    token = _secret("DATABRICKS_TOKEN", "DATABRICKS_TOKEN")
    if token:
        return token
    if is_databricks():
        try:
            from databricks.sdk import WorkspaceClient
            token = WorkspaceClient().config.token or ""
        except Exception:
            pass
    return token


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def validate_startup_secrets() -> None:
    """
    Validate all required secrets are present.  Call once at application
    startup — raises RuntimeError listing every missing secret so the
    operator can fix them all in one pass rather than discovering them one
    at a time.

    Example (in main / notebook entrypoint):
        from configs.settings import validate_startup_secrets
        validate_startup_secrets()
    """
    from configs.secrets import get_secret_provider
    sp = get_secret_provider()
    missing = sp.validate_all()
    if missing:
        raise RuntimeError(
            "Application cannot start — the following secrets are not configured:\n"
            + "\n".join(f"  • {m}" for m in missing)
            + "\n\nSee .env.example for local setup, or the Databricks Secrets API for production."
        )
    logger.info("Secret validation passed (%d secrets checked).", len(missing) + len(missing))
