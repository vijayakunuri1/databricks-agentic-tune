#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Production startup for the Databricks QC MCP HTTP server.
#
# Usage (direct TLS — server owns the certs):
#   ./deploy/start_server.sh
#
# Usage (behind nginx / ALB — proxy handles TLS termination):
#   PROXY_MODE=true ./deploy/start_server.sh
#
# Required environment variables
# ─────────────────────────────
#   MCP_API_KEY          64-char hex bearer token.  Generate once with:
#                          python -c "import secrets; print(secrets.token_hex(32))"
#                        On Databricks, store in the 'qc-secrets' secrets scope:
#                          databricks secrets put-secret qc-secrets MCP_API_KEY --string-value <value>
#
#   SSL_CERT             Path to TLS certificate file  (PEM, full-chain)
#   SSL_KEY              Path to TLS private key file  (PEM, unencrypted)
#
#   These are NOT required when PROXY_MODE=true (the proxy handles TLS).
#
# Optional environment variables
# ──────────────────────────────
#   HOST                 Bind address          (default: 0.0.0.0)
#   PORT                 Bind port             (default: 443, or 8000 in proxy mode)
#   WORKERS              uvicorn worker count  (default: 1)
#   MCP_ALLOWED_IPS      Comma-separated CIDRs/IPs to allowlist  (default: all IPs)
#   ENFORCE_HTTPS        "true" to redirect plain-HTTP → HTTPS   (default: true)
#   QC_ENV               "prod" | "dev"        (default: prod)
#   LOG_LEVEL            uvicorn log level     (default: info)
#   SECURITY_LOG_FILE    Path for rotating JSON security events log
#                          (default: /var/log/qc-security.log)
#
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
HOST="${HOST:-0.0.0.0}"
PROXY_MODE="${PROXY_MODE:-false}"
LOG_LEVEL="${LOG_LEVEL:-info}"
WORKERS="${WORKERS:-1}"
QC_ENV="${QC_ENV:-prod}"
ENFORCE_HTTPS="${ENFORCE_HTTPS:-true}"
SECURITY_LOG_FILE="${SECURITY_LOG_FILE:-/var/log/qc-security.log}"

if [[ "${PROXY_MODE}" == "true" ]]; then
    PORT="${PORT:-8000}"
else
    PORT="${PORT:-443}"
fi

# ---------------------------------------------------------------------------
# Required secret: MCP_API_KEY
# On Databricks this is read from the Secrets API by SecretProvider.
# For bare-metal / VM deployments, export it in your systemd unit or wrapper.
# ---------------------------------------------------------------------------
if [[ -z "${MCP_API_KEY:-}" ]]; then
    echo "ERROR: MCP_API_KEY is not set." >&2
    echo "  Generate: python -c \"import secrets; print(secrets.token_hex(32))\"" >&2
    echo "  Then either:" >&2
    echo "    export MCP_API_KEY=<value>  (this shell)" >&2
    echo "    databricks secrets put-secret qc-secrets MCP_API_KEY --string-value <value>" >&2
    exit 1
fi

if [[ "${#MCP_API_KEY}" -lt 32 ]]; then
    echo "ERROR: MCP_API_KEY is too short (minimum 32 chars)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# TLS cert validation (direct mode only)
# ---------------------------------------------------------------------------
if [[ "${PROXY_MODE}" != "true" ]]; then
    SSL_CERT="${SSL_CERT:-/etc/ssl/certs/mcp-server.crt}"
    SSL_KEY="${SSL_KEY:-/etc/ssl/private/mcp-server.key}"

    if [[ ! -f "${SSL_CERT}" ]]; then
        echo "ERROR: TLS certificate not found at ${SSL_CERT}" >&2
        echo "  Set SSL_CERT to the path of your PEM certificate." >&2
        exit 1
    fi
    if [[ ! -f "${SSL_KEY}" ]]; then
        echo "ERROR: TLS private key not found at ${SSL_KEY}" >&2
        echo "  Set SSL_KEY to the path of your PEM private key." >&2
        exit 1
    fi
    if [[ "$(stat -c %a "${SSL_KEY}" 2>/dev/null || stat -f %Lp "${SSL_KEY}")" != "600" ]]; then
        echo "WARNING: ${SSL_KEY} should have mode 600 (current owner-only read)." >&2
    fi
fi

# ---------------------------------------------------------------------------
# Export runtime config consumed by the application
# ---------------------------------------------------------------------------
export QC_ENV="${QC_ENV}"
export ENFORCE_HTTPS="${ENFORCE_HTTPS}"
export MCP_API_KEY="${MCP_API_KEY}"

# ---------------------------------------------------------------------------
# Configure the security event log file (optional but recommended)
# ---------------------------------------------------------------------------
export SECURITY_LOG_FILE="${SECURITY_LOG_FILE}"

# Ensure log directory exists
LOG_DIR="$(dirname "${SECURITY_LOG_FILE}")"
if [[ ! -d "${LOG_DIR}" ]]; then
    echo "WARNING: Security log directory ${LOG_DIR} does not exist; skipping file logging." >&2
    unset SECURITY_LOG_FILE
fi

# ---------------------------------------------------------------------------
# Build uvicorn arguments
# ---------------------------------------------------------------------------
UVICORN_ARGS=(
    "mcp_server.server:app"
    "--host" "${HOST}"
    "--port" "${PORT}"
    "--workers" "${WORKERS}"
    "--log-level" "${LOG_LEVEL}"
    "--no-access-log"          # access logging is handled by the security middleware
    "--timeout-keep-alive" "30"
)

if [[ "${PROXY_MODE}" != "true" ]]; then
    UVICORN_ARGS+=(
        "--ssl-keyfile"  "${SSL_KEY}"
        "--ssl-certfile" "${SSL_CERT}"
    )
fi

# ---------------------------------------------------------------------------
# Register the security log file with the application at startup
# (done via a tiny Python init hook so we don't need a separate launcher)
# ---------------------------------------------------------------------------
INIT_PY=""
if [[ -n "${SECURITY_LOG_FILE:-}" ]]; then
    INIT_PY="
import logging, sys
from configs.security_logging import configure_security_log_file
configure_security_log_file('${SECURITY_LOG_FILE}')
logging.getLogger('security.events').info(
    '{\"event\":\"server_startup\",\"ts\":\"%s\",\"mode\":\"${PROXY_MODE}\",\"port\":${PORT}}'
    % __import__('datetime').datetime.utcnow().isoformat()
)
"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "Starting MCP server on ${HOST}:${PORT} (proxy_mode=${PROXY_MODE}, workers=${WORKERS})" >&2

if [[ -n "${INIT_PY}" ]]; then
    python -c "${INIT_PY}"
fi

exec uvicorn "${UVICORN_ARGS[@]}"
