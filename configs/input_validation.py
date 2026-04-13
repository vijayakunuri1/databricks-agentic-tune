"""
Centralised input validation for all user-supplied and message-bus data.

Every piece of external or semi-trusted input that eventually reaches a
spark.sql() call, a Delta table reference, or a filesystem path MUST be
validated here before use.  All functions either return the sanitised
value or raise ValueError with a descriptive message.

Design principles
─────────────────
- Allowlists over denylists: reject everything that isn't explicitly valid.
- Validate at the earliest boundary: validate when data arrives, not at
  the point of use deep inside an implementation.
- Never construct SQL identifiers from unvalidated strings.
- All string limits are strict: avoid unbounded inputs.

Functions exported
──────────────────
validate_identifier(s, label)      → safe Spark SQL identifier part
validate_fqn(s, label)             → safe 3-part FQN (catalog.schema.table)
validate_column_name(s, label)     → safe column / primary-key name
validate_qc_checks(checks, label)  → list of allowed check names
validate_batch_id(s, label)        → safe batch/run ID for temp-view names
validate_sample_fraction(v, label) → float in (0.0, 1.0]
validate_local_file(path, roots)   → resolved absolute path inside roots
sanitize_freetext(s, max_len)      → printable-ASCII string, no SQL metacharacters
sanitize_reviewer(s, label)        → reviewer name safe for SQL string literals
"""
from __future__ import annotations

import os
import re
from typing import Sequence

# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------

_ALLOWED_QC_CHECKS: frozenset[str] = frozenset(
    {"null", "format", "duplicate", "geo", "zip_state_mismatch"}
)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Spark SQL identifier: must start with letter or underscore; only
# alphanumeric, underscore, hyphen allowed (max 128 chars).
# Hyphens are permitted for Unity Catalog catalog names; we backtick-quote
# all identifiers before placing them in SQL so hyphens are safe.
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\-]{0,127}$")

# Column / primary-key names: no hyphens (hyphens in column names are rare
# and are not used in this project).
_COLUMN_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,127}$")

# Batch/run IDs: UUID-like, alphanumeric + hyphen/underscore (max 64 chars).
# Used as temp-view name suffixes after replacing hyphens with underscores.
_BATCH_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")

# Reviewer / human label: printable ASCII only, no single-quote or backslash.
_REVIEWER_RE = re.compile(r"^[a-zA-Z0-9 @_\-\.]{1,128}$")

# Free-text (city names, street values, etc.): printable ASCII + common
# address characters; no SQL metacharacters.
_FREETEXT_SAFE_RE = re.compile(r"^[\w\s\'\-\.,/#&()]{1,1024}$")

# Allowed file extensions for local data uploads
_ALLOWED_FILE_EXTENSIONS: frozenset[str] = frozenset({".csv", ".parquet"})

# Default roots for local data files (overridden by QC_LOCAL_DATA_DIR env var)
_DEFAULT_LOCAL_ROOTS: tuple[str, ...] = ("/tmp", "/dbfs/tmp")


# ---------------------------------------------------------------------------
# Identifier validation
# ---------------------------------------------------------------------------

def validate_identifier(value: str, label: str = "identifier") -> str:
    """
    Validate a single Spark SQL identifier part (catalog, schema, or table name).

    Returns the value unchanged if valid; raises ValueError otherwise.
    Use backtick_identifier() when interpolating into SQL.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {label}: must be a non-empty string.")
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(
            f"Invalid {label} {value!r}: only letters, digits, underscores, "
            "and hyphens are allowed (max 128 chars, must start with letter or underscore)."
        )
    return value


def validate_fqn(value: str, label: str = "table FQN") -> str:
    """
    Validate a fully-qualified table name: catalog.schema.table.

    Returns the FQN unchanged if all three parts are valid identifiers.
    Raises ValueError if the format is wrong or any part is invalid.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {label}: must be a non-empty string.")
    parts = value.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid {label} {value!r}: must be exactly catalog.schema.table "
            f"(got {len(parts)} parts)."
        )
    catalog, schema, table = parts
    validate_identifier(catalog, f"{label} catalog")
    validate_identifier(schema,  f"{label} schema")
    validate_identifier(table,   f"{label} table")
    return value


def validate_column_name(value: str, label: str = "column name") -> str:
    """
    Validate a column or primary-key name.

    Stricter than validate_identifier: no hyphens (column names in this
    project never contain hyphens).
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {label}: must be a non-empty string.")
    if not _COLUMN_RE.match(value):
        raise ValueError(
            f"Invalid {label} {value!r}: only letters, digits, and underscores "
            "are allowed (max 128 chars, must start with letter or underscore)."
        )
    return value


def backtick_identifier(value: str) -> str:
    """
    Wrap a validated identifier in backticks for safe Spark SQL interpolation.

    Only call this on values that have already been validated by
    validate_identifier() or validate_column_name().  Backtick characters
    within the value are escaped (doubled) as a defence-in-depth measure.
    """
    return f"`{value.replace('`', '``')}`"


def backtick_fqn(fqn: str) -> str:
    """
    Return a backtick-quoted FQN safe for Spark SQL interpolation.

    Input must already be validated by validate_fqn().
    """
    catalog, schema, table = fqn.split(".")
    return f"{backtick_identifier(catalog)}.{backtick_identifier(schema)}.{backtick_identifier(table)}"


# ---------------------------------------------------------------------------
# QC-specific validation
# ---------------------------------------------------------------------------

def validate_qc_checks(
    checks: list[str] | str,
    label: str = "qc_checks",
) -> list[str]:
    """
    Validate a list (or comma-separated string) of QC check names.

    Rejects any name not in the fixed allowlist.
    Returns a deduplicated list preserving order.
    """
    if isinstance(checks, str):
        parts = [c.strip().lower() for c in checks.split(",") if c.strip()]
    elif isinstance(checks, (list, tuple)):
        parts = [str(c).strip().lower() for c in checks if str(c).strip()]
    else:
        raise ValueError(f"Invalid {label}: expected list or comma-separated string.")

    if not parts:
        raise ValueError(f"Invalid {label}: at least one check type is required.")

    unknown = [c for c in parts if c not in _ALLOWED_QC_CHECKS]
    if unknown:
        raise ValueError(
            f"Invalid {label}: unknown check type(s) {unknown}. "
            f"Allowed: {sorted(_ALLOWED_QC_CHECKS)}"
        )
    # Deduplicate while preserving order
    seen: set[str] = set()
    return [c for c in parts if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]


def validate_batch_id(value: str, label: str = "batch_id") -> str:
    """
    Validate a batch or run ID for safe use as a Spark temp view name suffix.

    Returns the value unchanged if valid (hyphens are allowed and will be
    replaced with underscores when building the view name).
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {label}: must be a non-empty string.")
    if not _BATCH_ID_RE.match(value):
        raise ValueError(
            f"Invalid {label} {value!r}: only alphanumeric, hyphen, and "
            "underscore characters are allowed (max 64 chars)."
        )
    return value


def validate_sample_fraction(
    value: float | str | None,
    label: str = "sample_fraction",
) -> float | None:
    """
    Validate a Spark sample fraction.

    Returns None if value is None (no sampling).
    Otherwise returns a float in the range (0.0, 1.0].
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {label}: must be a number, got {value!r}.")
    if not (0.0 < f <= 1.0):
        raise ValueError(
            f"Invalid {label}: must be in the range (0.0, 1.0], got {f}."
        )
    return f


# ---------------------------------------------------------------------------
# File path validation
# ---------------------------------------------------------------------------

def validate_local_file(
    path: str,
    allowed_roots: Sequence[str] | None = None,
    label: str = "local_file",
) -> str:
    """
    Validate a local file path against an allowlist of root directories.

    - Resolves symlinks and `..` traversal (os.path.realpath).
    - Rejects paths outside the allowed roots.
    - Rejects file extensions not in _ALLOWED_FILE_EXTENSIONS.
    - Returns the resolved absolute path.

    allowed_roots defaults to QC_LOCAL_DATA_DIR env var, then
    _DEFAULT_LOCAL_ROOTS.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(f"Invalid {label}: must be a non-empty string.")

    # Determine roots
    if allowed_roots is None:
        env_roots = os.environ.get("QC_LOCAL_DATA_DIR", "")
        roots = (
            [r for r in env_roots.split(os.pathsep) if r]
            if env_roots
            else list(_DEFAULT_LOCAL_ROOTS)
        )
    else:
        roots = list(allowed_roots)

    # Extension check before resolving (avoids resolving a file that will be rejected anyway)
    _, ext = os.path.splitext(path)
    if ext.lower() not in _ALLOWED_FILE_EXTENSIONS:
        raise ValueError(
            f"Invalid {label}: file extension {ext!r} is not allowed. "
            f"Allowed: {sorted(_ALLOWED_FILE_EXTENSIONS)}"
        )

    # Resolve to catch symlinks and traversal
    resolved = os.path.realpath(path)

    # Check against allowed roots
    if not any(
        resolved.startswith(root + os.sep) or resolved == root
        for root in roots
    ):
        raise ValueError(
            f"Invalid {label}: path {path!r} resolves to {resolved!r}, "
            f"which is outside the allowed directories: {roots}. "
            "Set QC_LOCAL_DATA_DIR to override."
        )

    return resolved


# ---------------------------------------------------------------------------
# Free-text / human-label sanitisation
# ---------------------------------------------------------------------------

def sanitize_reviewer(value: str, label: str = "reviewer") -> str:
    """
    Sanitise a reviewer name for safe use in SQL string literals.

    Applies two layers:
    1. Character allowlist: only letters, digits, spaces, @, _, -, .
    2. SQL single-quote escaping (doubles any remaining single quotes).

    Raises ValueError if the value is empty after stripping.
    """
    if not isinstance(value, str):
        raise ValueError(f"Invalid {label}: must be a string.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"Invalid {label}: must not be empty.")
    if not _REVIEWER_RE.match(stripped):
        raise ValueError(
            f"Invalid {label} {value!r}: only letters, digits, spaces, "
            "@, _, -, . are allowed (max 128 chars)."
        )
    # Defence-in-depth: escape single quotes even though the regex
    # already excludes them.
    return stripped.replace("'", "''")


def sanitize_freetext(
    value: str,
    max_len: int = 1024,
    label: str = "value",
) -> str:
    """
    Sanitise a free-text value (city name, street, address) for logging
    and safe embedding in Spark string literals.

    Does NOT attempt to make the value safe for direct SQL interpolation —
    use parameterised queries / DataFrames instead.  Call this function
    before storing free-text in audit log fields or log messages.
    """
    if not isinstance(value, str):
        raise ValueError(f"Invalid {label}: must be a string.")
    # Strip NUL bytes (Spark rejects strings containing \x00)
    cleaned = value.replace("\x00", "").strip()
    if len(cleaned) > max_len:
        raise ValueError(
            f"Invalid {label}: exceeds maximum length of {max_len} characters "
            f"(got {len(cleaned)})."
        )
    return cleaned


# ---------------------------------------------------------------------------
# Composite helpers used by multiple callers
# ---------------------------------------------------------------------------

def validate_table_coordinates(
    catalog: str,
    schema: str,
    table: str,
) -> tuple[str, str, str]:
    """
    Validate and return (catalog, schema, table) as a tuple.

    Raises ValueError on the first invalid component.
    """
    return (
        validate_identifier(catalog, "catalog"),
        validate_identifier(schema,  "schema"),
        validate_identifier(table,   "table"),
    )


def build_safe_fqn(catalog: str, schema: str, table: str) -> str:
    """
    Validate table coordinates and return a plain FQN string (no backticks).

    Use backtick_fqn() when the result will be interpolated into SQL.
    """
    c, s, t = validate_table_coordinates(catalog, schema, table)
    return f"{c}.{s}.{t}"
