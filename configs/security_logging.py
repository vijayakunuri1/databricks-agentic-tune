"""
Structured security event logger.

All security-relevant events — auth attempts, access denials, data writes,
rate-limit hits, and API errors — are emitted as newline-delimited JSON to
the ``security.events`` logger.  Route that logger to a file, CloudWatch,
Splunk, or any SIEM by configuring its handler in your deployment.

Events are machine-readable so they can be queried directly:

    grep '"event":"auth_failure"' security.log | jq .client_ip | sort | uniq -c

Abuse detection: the module tracks auth-failure counts per IP in a sliding
60-second window.  When an IP exceeds the threshold a ``brute_force_alert``
event is emitted and subsequent requests from that IP are flagged.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Literal

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

_sec_logger = logging.getLogger("security.events")

# Ensure the security logger is not swallowed by a root handler configured
# with level WARNING.  The caller's basicConfig / dictConfig can override this.
if not _sec_logger.handlers and not _sec_logger.propagate:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))   # raw JSON, no prefix
    _sec_logger.addHandler(_h)
_sec_logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

SecurityEventType = Literal[
    "auth_success",
    "auth_failure",
    "brute_force_alert",
    "access_denied",
    "rate_limit_hit",
    "data_read",
    "data_modification",
    "api_error",
    "pipeline_start",
    "pipeline_complete",
    "secret_missing",
    "https_redirect",
]

# ---------------------------------------------------------------------------
# Abuse / brute-force detection
# ---------------------------------------------------------------------------

_FAILURE_WINDOW_SECONDS = 60
_FAILURE_THRESHOLD = 5          # alert after this many failures in the window

# {client_ip: [(timestamp, ...), ...]}
_failure_windows: dict[str, list[float]] = defaultdict(list)
_failure_lock = threading.Lock()


def _record_failure(client_ip: str) -> bool:
    """
    Slide the failure window for client_ip.
    Returns True if the IP has exceeded the brute-force threshold.
    """
    now = time.monotonic()
    with _failure_lock:
        window = _failure_windows[client_ip]
        # Drop timestamps outside the window
        cutoff = now - _FAILURE_WINDOW_SECONDS
        _failure_windows[client_ip] = [t for t in window if t > cutoff]
        _failure_windows[client_ip].append(now)
        return len(_failure_windows[client_ip]) >= _FAILURE_THRESHOLD


def _clear_failures(client_ip: str) -> None:
    with _failure_lock:
        _failure_windows.pop(client_ip, None)


# ---------------------------------------------------------------------------
# Core emit helper
# ---------------------------------------------------------------------------

def _emit(event_type: SecurityEventType, **fields: object) -> None:
    event: dict[str, object] = {
        "event":   event_type,
        "ts":      datetime.now(timezone.utc).isoformat(),
    }
    event.update(fields)
    _sec_logger.info(json.dumps(event, default=str))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_auth_attempt(
    *,
    success: bool,
    client_ip: str,
    path: str = "",
    reason: str = "",
) -> None:
    """Log an MCP server authentication attempt."""
    if success:
        _clear_failures(client_ip)
        _emit("auth_success", client_ip=client_ip, path=path)
    else:
        _emit("auth_failure", client_ip=client_ip, path=path, reason=reason)
        if _record_failure(client_ip):
            _emit(
                "brute_force_alert",
                client_ip=client_ip,
                message=(
                    f"IP {client_ip!r} has exceeded {_FAILURE_THRESHOLD} "
                    f"auth failures in {_FAILURE_WINDOW_SECONDS}s — possible brute force."
                ),
            )


def log_access_denied(
    *,
    client_ip: str,
    resource: str,
    reason: str,
    tool: str = "",
) -> None:
    """Log an IDOR / allowlist rejection."""
    _emit(
        "access_denied",
        client_ip=client_ip,
        resource=resource,
        reason=reason,
        tool=tool,
    )


def log_rate_limit_hit(*, client_ip: str, tool: str) -> None:
    """Log when a caller is throttled by the rate limiter."""
    _emit("rate_limit_hit", client_ip=client_ip, tool=tool)


def log_data_modification(
    *,
    table_fqn: str,
    record_id: str,
    action: str,
    reviewer: str,
    dry_run: bool = False,
) -> None:
    """Log every human review decision written back to Delta."""
    _emit(
        "data_modification",
        table=table_fqn,
        record_id=record_id,
        action=action,
        reviewer=reviewer,
        dry_run=dry_run,
    )


def log_api_error(
    *,
    client_ip: str,
    path: str,
    status: int,
    error_type: str,
    detail: str = "",
) -> None:
    """Log an HTTP error response sent to a client."""
    _emit(
        "api_error",
        client_ip=client_ip,
        path=path,
        status=status,
        error_type=error_type,
        detail=detail,
    )


def log_pipeline_event(
    *,
    event: Literal["start", "complete"],
    table_fqn: str,
    run_id: str,
    dry_run: bool = False,
    **extra: object,
) -> None:
    """Log pipeline start/complete so runs are traceable in the security log."""
    _emit(
        "pipeline_start" if event == "start" else "pipeline_complete",
        table=table_fqn,
        run_id=run_id,
        dry_run=dry_run,
        **extra,
    )


def log_https_redirect(*, client_ip: str, original_url: str) -> None:
    """Log when a plain-HTTP request is redirected to HTTPS."""
    _emit("https_redirect", client_ip=client_ip, original_url=original_url)


def log_secret_missing(*, name: str) -> None:
    """Log when a required secret is absent at startup."""
    _emit("secret_missing", secret_name=name)


# ---------------------------------------------------------------------------
# Logging configuration helper
# ---------------------------------------------------------------------------

def configure_security_log_file(path: str) -> None:
    """
    Add a rotating file handler for security events.

    Call this once at startup if you want events written to a local file
    in addition to stdout:

        configure_security_log_file("/var/log/qc-security.log")
    """
    import logging.handlers
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=50 * 1024 * 1024,   # 50 MB per file
        backupCount=10,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    _sec_logger.addHandler(handler)
