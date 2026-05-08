"""
Databricks QC MCP Server

Exposes the QC pipeline as MCP tools for LLM-driven orchestration
data quality checks dynamically via the Model Context Protocol.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRODUCTION STARTUP (HTTPS via TLS certs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use deploy/start_server.sh which sets all required env vars and passes TLS
certs to uvicorn.  Never start with --host 0.0.0.0 without TLS.

  MCP_API_KEY=<64-char hex>  \\
  ENFORCE_HTTPS=true         \\
  uvicorn mcp_server.server:app \\
    --host 0.0.0.0 --port 443 \\
    --ssl-keyfile  /etc/ssl/private/server.key \\
    --ssl-certfile /etc/ssl/certs/server.crt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEHIND A REVERSE PROXY (nginx / ALB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Set ENFORCE_HTTPS=true so the middleware redirects plain-HTTP requests
based on the X-Forwarded-Proto header forwarded by the proxy.

  MCP_API_KEY=<secret> ENFORCE_HTTPS=true \\
  uvicorn mcp_server.server:app --host 127.0.0.1 --port 8000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOCAL / STDIO (stdio mode, no auth required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python -m mcp_server.server
No network exposure in stdio mode; MCP_API_KEY is not required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONNECTING AN LLM CLIENT TO THE HTTP SERVER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  client.beta.messages.create(
      model="databricks-meta-llama-3-3-70b-instruct",
      mcp_servers=[{
          "type": "url",
          "url": "https://your-host/mcp",
          "name": "databricks-qc",
          "authorization_token": "<MCP_API_KEY>",
      }],
      betas=["mcp-client-2025-04-04"],
      messages=[{"role": "user", "content": "Run QC on sec_edgar.companies"}],
  )
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_server.tools.table_tools import register_table_tools
from mcp_server.tools.geo_tools import register_geo_tools
from mcp_server.tools.review_tools import register_review_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    """
    Load MCP_API_KEY from the SecretProvider (Databricks Secrets API or env var).
    Raises RuntimeError if absent or too short.
    """
    from configs.secrets import get_secret_provider, DEFAULT_SCOPE
    sp = get_secret_provider()
    key = sp.get(DEFAULT_SCOPE, "MCP_API_KEY") or sp.get("MCP_API_KEY")
    key = key.strip()
    if not key:
        raise RuntimeError(
            "MCP_API_KEY is not set. The HTTP server cannot start without an API key. "
            "Generate one: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    if len(key) < 32:
        raise RuntimeError(
            "MCP_API_KEY is too short (minimum 32 chars). "
            "Generate: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return key


def _constant_time_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison (prevents timing oracle on the API key)."""
    return hmac.compare_digest(
        hashlib.sha256(a.encode()).digest(),
        hashlib.sha256(b.encode()).digest(),
    )


def _parse_ip_allowlist() -> frozenset[str]:
    """
    Parse MCP_ALLOWED_IPS (comma-separated CIDR or IP list).
    If empty, all IPs are allowed (bearer token is the only gate).

    Example:
        MCP_ALLOWED_IPS=10.0.0.0/8,192.168.1.100
    """
    raw = os.environ.get("MCP_ALLOWED_IPS", "").strip()
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _ip_allowed(client_ip: str, allowlist: frozenset[str]) -> bool:
    """Return True if client_ip is in the allowlist (or allowlist is empty)."""
    if not allowlist:
        return True
    if client_ip in allowlist:
        return True
    # Basic CIDR check (IPv4 only; for production use `ipaddress` stdlib)
    import ipaddress
    try:
        addr = ipaddress.ip_address(client_ip)
        for entry in allowlist:
            try:
                if addr in ipaddress.ip_network(entry, strict=False):
                    return True
            except ValueError:
                pass
    except ValueError:
        pass
    return False


# ---------------------------------------------------------------------------
# FastMCP application
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="databricks-qc",
    instructions="""
You are an AI agent with access to a Databricks data quality control pipeline.

Available capabilities:
- List tables registered for QC monitoring
- Run QC checks (null, format, duplicate, geo, zip_state_mismatch) on Delta tables
- Retrieve QC run results and issue breakdowns
- Look up canonical city names and ZIP codes against GeoNames reference data
- Review and approve/reject L2 flagged records that need human attention

Typical workflow:
1. Call list_qc_tables() to see available tables
2. Call run_qc_check() with dry_run=True to see what issues exist without changing data
3. Call get_qc_results() to inspect the findings
4. For geo issues, call lookup_city() or lookup_zip() to verify corrections
5. Call run_qc_check() with dry_run=False to apply L1 corrections
6. Call get_l2_flagged_records() to review items that need human decision
7. Call approve_l2_correction() for records you're confident about
""",
)

register_table_tools(mcp)
register_geo_tools(mcp)
register_review_tools(mcp)


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

def configure(
    app_config: Any | None = None,
    spark: Any | None = None,
    geonames_index: Any | None = None,
    city_reference: list[dict] | None = None,
    tables: list[dict] | None = None,
) -> None:
    """
    Inject runtime dependencies before starting the server.

    Call this from your notebook or startup script:
        from mcp_server.server import configure
        configure(app_config=cfg, spark=spark, geonames_index=geo)
    """
    from mcp_server.pipeline_state import set_config, register_table

    set_config({
        "app_config":      app_config,
        "spark":           spark,
        "geonames_index":  geonames_index,
        "city_reference":  city_reference or [],
        "primary_key":     "id",
    })

    for t in (tables or []):
        register_table(t["catalog"], t["schema"], t["table"], t.get("description", ""))

    logger.info(
        "MCP server configured. Spark: %s, GeoNames: %s",
        spark is not None,
        geonames_index is not None,
    )


# ---------------------------------------------------------------------------
# ASGI security middleware
# ---------------------------------------------------------------------------

class _SecurityMiddleware:
    """
    Single ASGI middleware that handles, in order:

    1. IP allowlist check      — 403 if MCP_ALLOWED_IPS is set and the
                                 caller's IP is not listed.
    2. IP block check          — 429 if the IP was blocked by the rate
                                 limiter due to repeated auth failures.
    3. HTTPS redirect          — if ENFORCE_HTTPS=true, redirect plain-HTTP
                                 requests via X-Forwarded-Proto.
    4. Bearer token auth       — 401 if Authorization header is absent or
                                 token doesn't match MCP_API_KEY.
                                 Records failures; blocks IP after threshold.
    5. Per-IP global rate limit — 429 if the IP exceeds its request budget
                                 (default 200 req/60 s).
    6. Security headers        — inject HSTS, X-Content-Type-Options, etc.
                                 on every successful response.

    The /health path is exempt from steps 2–5 so liveness probes work.

    All security events are emitted to the structured security logger
    (configs.security_logging) as JSON, ready for SIEM ingestion.
    """

    _EXEMPT_PATHS = frozenset({"/health"})

    # Security headers added to every response
    _SECURITY_HEADERS: list[tuple[bytes, bytes]] = [
        (b"strict-transport-security", b"max-age=63072000; includeSubDomains; preload"),
        (b"x-content-type-options",    b"nosniff"),
        (b"x-frame-options",           b"DENY"),
        (b"x-xss-protection",          b"1; mode=block"),
        (b"referrer-policy",           b"strict-origin-when-cross-origin"),
        (b"cache-control",             b"no-store"),
    ]

    def __init__(self, asgi_app: Any) -> None:
        self._app = asgi_app
        # Load once at import time — fail fast if MCP_API_KEY is missing
        self._api_key = _load_api_key()
        self._allowlist = _parse_ip_allowlist()
        self._enforce_https = os.environ.get("ENFORCE_HTTPS", "").lower() in ("1", "true", "yes")

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        path      = scope.get("path", "/")
        headers   = dict(scope.get("headers", []))
        client_ip = self._extract_ip(scope, headers)

        # 1 — IP allowlist
        if not _ip_allowed(client_ip, self._allowlist):
            from configs.security_logging import log_access_denied
            log_access_denied(
                client_ip=client_ip,
                resource=path,
                reason="IP not in MCP_ALLOWED_IPS allowlist",
            )
            await self._send_error(send, 403, "Forbidden")
            return

        # 2 — IP block check (triggered by repeated auth failures)
        if path not in self._EXEMPT_PATHS:
            from configs.rate_limiting import is_ip_blocked
            blocked, remaining = is_ip_blocked(client_ip)
            if blocked:
                from configs.security_logging import log_access_denied
                log_access_denied(
                    client_ip=client_ip,
                    resource=path,
                    reason=f"IP blocked for {int(remaining)}s after repeated auth failures",
                )
                await self._send_error(
                    send, 429, "Too Many Requests",
                    extra_headers=[(b"retry-after", str(int(remaining) + 1).encode())],
                )
                return

        # 3 — HTTPS redirect (exempt: /health)
        if self._enforce_https and path not in self._EXEMPT_PATHS:
            proto = headers.get(b"x-forwarded-proto", b"").decode("latin-1").lower()
            if proto == "http":
                host  = headers.get(b"host", b"localhost").decode("latin-1")
                qs    = scope.get("query_string", b"").decode("latin-1")
                target = f"https://{host}{path}" + (f"?{qs}" if qs else "")
                from configs.security_logging import log_https_redirect
                log_https_redirect(client_ip=client_ip, original_url=f"http://{host}{path}")
                await self._send_redirect(send, target)
                return

        # 4 — Bearer auth (exempt: /health)
        if path not in self._EXEMPT_PATHS:
            auth_header = headers.get(b"authorization", b"").decode("latin-1")
            ok, reason = self._check_bearer(auth_header)
            from configs.security_logging import log_auth_attempt
            from configs.rate_limiting import (
                record_auth_success, record_auth_failure, block_ip,
            )
            log_auth_attempt(success=ok, client_ip=client_ip, path=path, reason=reason)
            if ok:
                record_auth_success(client_ip)
            else:
                should_block = record_auth_failure(client_ip)
                if should_block:
                    block_ip(client_ip)
                await self._send_error(send, 401, "Unauthorized",
                                       extra_headers=[(b"www-authenticate", b"Bearer")])
                return

        # 5 — Per-IP global request rate limit (authenticated requests only)
        if path not in self._EXEMPT_PATHS:
            from configs.rate_limiting import check_global, rate_limit_error
            allowed, retry_after = check_global(client_ip)
            if not allowed:
                from configs.security_logging import log_rate_limit_hit
                log_rate_limit_hit(client_ip=client_ip, tool="global")
                await self._send_error(
                    send, 429, rate_limit_error(retry_after, "global"),
                    extra_headers=[(b"retry-after", str(int(retry_after) + 1).encode())],
                )
                return

        # 7 — Pass to app, intercept send to inject security headers
        async def _send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                existing = list(message.get("headers", []))
                message = dict(message)
                message["headers"] = existing + self._SECURITY_HEADERS
            await send(message)

        await self._app(scope, receive, _send_with_headers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_bearer(self, auth_header: str) -> tuple[bool, str]:
        if not auth_header.lower().startswith("bearer "):
            return False, "missing Authorization header"
        provided = auth_header[len("bearer "):].strip()
        if not _constant_time_compare(provided, self._api_key):
            return False, "invalid token"
        return True, ""

    @staticmethod
    def _extract_ip(scope: dict, headers: dict[bytes, bytes]) -> str:
        """Extract client IP, honouring X-Forwarded-For when behind a proxy."""
        forwarded_for = headers.get(b"x-forwarded-for", b"").decode("latin-1").strip()
        if forwarded_for:
            # Take the leftmost (original client) address
            return forwarded_for.split(",")[0].strip()
        client = scope.get("client")
        if client:
            return client[0]
        return "unknown"

    @staticmethod
    async def _send_redirect(send: Any, location: str) -> None:
        loc_bytes = location.encode()
        await send({
            "type": "http.response.start",
            "status": 301,
            "headers": [
                [b"location", loc_bytes],
                [b"content-length", b"0"],
            ],
        })
        await send({"type": "http.response.body", "body": b""})

    @staticmethod
    async def _send_error(
        send: Any,
        status: int,
        message: str,
        extra_headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        import json as _json
        body = _json.dumps({"error": message}).encode()
        headers: list[list[bytes]] = [
            [b"content-type",   b"application/json"],
            [b"content-length", str(len(body)).encode()],
        ]
        for k, v in (extra_headers or []):
            headers.append([k, v])
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})


# ---------------------------------------------------------------------------
# ASGI app for uvicorn
# ---------------------------------------------------------------------------

try:
    _base_app = mcp.streamable_http_app()
    app = _SecurityMiddleware(_base_app)
except AttributeError:
    # Older mcp package that doesn't expose streamable_http_app
    app = None


# ---------------------------------------------------------------------------
# stdio entry point (no network, no auth required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
