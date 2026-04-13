"""
Sliding-window, per-key rate limiter.

Provides two distinct layers of abuse protection:

Layer 1 — IP-level (ASGI middleware)
    Every authenticated request consumes the caller IP's global budget.
    Auth failures are counted per-IP; when the threshold is hit the IP
    is blocked for BLOCK_DURATION seconds and all subsequent requests
    receive 429 immediately — no token comparison, no tool execution.

Layer 2 — Resource-level (MCP tools)
    Each tool type has its own global budget independent of caller IP.
    This catches runaway agent loops and credential-sharing abuse even
    when a single bearer token is shared across multiple callers.

Default limits (all overrideable via environment variables):

    Bucket          Env var               Default   Window
    ─────────────────────────────────────────────────────────
    auth (per IP)   RL_AUTH_MAX           10        60 s
    global (per IP) RL_GLOBAL_MAX        200        60 s
    qc_run          RL_QC_RUN_MAX         10        60 s   ← Spark + LLM
    read            RL_READ_MAX          120        60 s   ← list/get/results
    geo_lookup      RL_GEO_MAX            60        60 s   ← city/ZIP index
    llm_call        RL_LLM_MAX            10        60 s   ← external AI API
    l2_write        RL_L2_WRITE_MAX       10        60 s   ← Delta writes

IP blocking:
    Threshold       RL_BLOCK_THRESHOLD     5 auth failures in 60 s
    Duration        RL_BLOCK_DURATION    900 s (15 minutes)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


_WINDOW = 60.0  # seconds — shared by all buckets

# Per-IP limits
_AUTH_MAX    = _int_env("RL_AUTH_MAX",      10)
_GLOBAL_MAX  = _int_env("RL_GLOBAL_MAX",   200)

# Tool-level global limits
_QC_RUN_MAX   = _int_env("RL_QC_RUN_MAX",   10)
_READ_MAX     = _int_env("RL_READ_MAX",     120)
_GEO_MAX      = _int_env("RL_GEO_MAX",      60)
_LLM_MAX      = _int_env("RL_LLM_MAX",      10)
_L2_WRITE_MAX = _int_env("RL_L2_WRITE_MAX", 10)

# IP blocking
_BLOCK_THRESHOLD = _int_env("RL_BLOCK_THRESHOLD", 5)
_BLOCK_DURATION  = _int_env("RL_BLOCK_DURATION",  900)


# ---------------------------------------------------------------------------
# Sliding-window counter
# ---------------------------------------------------------------------------

class _SlidingWindow:
    """
    Thread-safe sliding-window rate counter.

    All timestamps use time.monotonic() to avoid clock-skew and DST issues.
    Each call within the window is stored as a timestamp so the window
    genuinely slides rather than resetting at fixed epoch boundaries.
    """

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(
        self,
        key: str,
        max_calls: int,
        window: float = _WINDOW,
    ) -> tuple[bool, float]:
        """
        Check and (if allowed) consume one call unit for `key`.

        Returns:
            (allowed, retry_after_seconds)
            retry_after is 0.0 when the call is allowed.
        """
        now = time.monotonic()
        cutoff = now - window
        with self._lock:
            ts = self._windows[key]
            # Drop timestamps outside the sliding window
            self._windows[key] = [t for t in ts if t > cutoff]
            if len(self._windows[key]) >= max_calls:
                # Oldest entry tells us when the next slot opens
                retry_after = self._windows[key][0] + window - now
                return False, max(0.0, retry_after)
            self._windows[key].append(now)
            return True, 0.0

    def count(self, key: str, window: float = _WINDOW) -> int:
        """Current number of calls for `key` within the sliding window."""
        now = time.monotonic()
        cutoff = now - window
        with self._lock:
            return sum(1 for t in self._windows.get(key, []) if t > cutoff)

    def clear(self, key: str) -> None:
        """Reset the counter for `key` (e.g. after successful authentication)."""
        with self._lock:
            self._windows.pop(key, None)


# Process-wide singleton — all threads in the server share this instance
_limiter = _SlidingWindow()


# ---------------------------------------------------------------------------
# IP blocking
# ---------------------------------------------------------------------------

_blocked_ips: dict[str, float] = {}  # ip → monotonic unblock-time
_block_lock = threading.Lock()


def block_ip(ip: str, duration: int = _BLOCK_DURATION) -> None:
    """Block `ip` for `duration` seconds (default: 15 minutes)."""
    unblock_at = time.monotonic() + duration
    with _block_lock:
        _blocked_ips[ip] = unblock_at
    logger.warning(
        "Rate limiter: IP %r blocked for %ds after repeated auth failures.", ip, duration
    )


def unblock_ip(ip: str) -> None:
    """Explicitly unblock `ip` (e.g. after admin action or successful auth)."""
    with _block_lock:
        _blocked_ips.pop(ip, None)


def is_ip_blocked(ip: str) -> tuple[bool, float]:
    """
    Check whether `ip` is currently blocked.

    Returns:
        (blocked, seconds_remaining)
    """
    with _block_lock:
        unblock_at = _blocked_ips.get(ip)
        if unblock_at is None:
            return False, 0.0
        remaining = unblock_at - time.monotonic()
        if remaining <= 0:
            del _blocked_ips[ip]
            return False, 0.0
        return True, remaining


# ---------------------------------------------------------------------------
# Layer 1 — Per-IP checks (used by ASGI middleware)
# ---------------------------------------------------------------------------

def check_global(ip: str) -> tuple[bool, float]:
    """
    Per-IP global request budget.

    Call this on every authenticated request.  Returns (allowed, retry_after).
    """
    return _limiter.is_allowed(f"global:{ip}", _GLOBAL_MAX)


def record_auth_failure(ip: str) -> bool:
    """
    Record one authentication failure for `ip`.

    Returns True when the failure count hits _BLOCK_THRESHOLD, signalling
    that the caller should block the IP.
    """
    # is_allowed also appends the timestamp; we just need the count
    _limiter.is_allowed(f"auth:{ip}", _AUTH_MAX)  # consume a slot
    count = _limiter.count(f"auth:{ip}")
    return count >= _BLOCK_THRESHOLD


def record_auth_success(ip: str) -> None:
    """Reset auth failure counter and unblock `ip` on successful authentication."""
    _limiter.clear(f"auth:{ip}")
    unblock_ip(ip)


# ---------------------------------------------------------------------------
# Layer 2 — Resource-type checks (used by MCP tools)
# ---------------------------------------------------------------------------

def check_qc_run() -> tuple[bool, float]:
    """
    Budget for run_qc_check calls.

    Conservative limit — each call spawns a Spark job and may trigger
    multiple LLM calls.
    """
    return _limiter.is_allowed("tool:qc_run", _QC_RUN_MAX)


def check_read() -> tuple[bool, float]:
    """
    Budget for read-only tool calls:
    list_qc_tables, get_qc_results, get_l2_flagged_records.
    """
    return _limiter.is_allowed("tool:read", _READ_MAX)


def check_geo_lookup() -> tuple[bool, float]:
    """
    Budget for geo/address lookup calls:
    lookup_city, lookup_zip, validate_city_zip.
    """
    return _limiter.is_allowed("tool:geo", _GEO_MAX)


def check_llm_call() -> tuple[bool, float]:
    """
    Budget for external LLM API calls (Claude / Databricks Model Serving).

    Prevents runaway agent loops from exhausting token quotas.
    """
    return _limiter.is_allowed("tool:llm", _LLM_MAX)


def check_l2_write() -> tuple[bool, float]:
    """
    Budget for L2 review write operations (approve_l2_correction).

    Most restrictive bucket — each call writes to a Delta table.
    """
    return _limiter.is_allowed("tool:l2_write", _L2_WRITE_MAX)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def rate_limit_error(retry_after: float, bucket: str = "") -> str:
    """Return a human-readable rate-limit message with a Retry-After hint."""
    secs = int(retry_after) + 1
    label = f" ({bucket})" if bucket else ""
    return (
        f"Rate limit exceeded{label}. "
        f"Please wait {secs}s before retrying. "
        "Limits reset on a 60-second sliding window."
    )
