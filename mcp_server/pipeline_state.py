"""Singleton state store shared across MCP tool calls within a server process."""
from __future__ import annotations

from typing import Any
from schemas.qc_results import WorkflowResult

# Module-level state (single process, single server)
_LAST_RUN: dict[str, dict] = {}          # table_fqn → serialized WorkflowResult
_L2_RECORDS: dict[str, list[dict]] = {}  # table_fqn → list of CorrectionRecord dicts
_REGISTERED_TABLES: list[dict] = []       # [{catalog, schema, table, description}]
_PIPELINE_CONFIG: dict[str, Any] = {}    # runtime config injected at startup

# Frozenset of authorised FQNs for O(1) lookup — kept in sync with _REGISTERED_TABLES
_REGISTERED_FQNS: set[str] = set()


def register_table(catalog: str, schema: str, table: str, description: str = "") -> None:
    fqn = f"{catalog}.{schema}.{table}"
    entry = {"catalog": catalog, "schema": schema, "table": table,
             "fqn": fqn, "description": description}
    if entry not in _REGISTERED_TABLES:
        _REGISTERED_TABLES.append(entry)
        _REGISTERED_FQNS.add(fqn)


def list_tables() -> list[dict]:
    return list(_REGISTERED_TABLES)


def assert_table_registered(catalog: str, schema: str, table: str) -> str:
    """
    Verify that catalog.schema.table is in the registered allowlist.

    This is the central IDOR guard for all MCP tools that accept table
    coordinates as input. Every tool MUST call this before reading from
    or writing to any table.

    Returns the validated FQN string on success.
    Raises PermissionError if the table is not registered.
    """
    fqn = f"{catalog}.{schema}.{table}"
    if fqn not in _REGISTERED_FQNS:
        registered = sorted(_REGISTERED_FQNS) or ["(none — call configure() first)"]
        raise PermissionError(
            f"Access denied: table {fqn!r} is not registered for QC access. "
            f"Registered tables: {registered}"
        )
    return fqn


def store_run_result(table_fqn: str, result: WorkflowResult) -> None:
    # Only store results for registered tables
    if table_fqn in _REGISTERED_FQNS:
        _LAST_RUN[table_fqn] = result.model_dump(mode="json")


def get_last_run(table_fqn: str) -> dict | None:
    return _LAST_RUN.get(table_fqn)


def store_l2_records(table_fqn: str, records: list[dict]) -> None:
    if table_fqn in _REGISTERED_FQNS:
        _L2_RECORDS[table_fqn] = records


def get_l2_records(table_fqn: str) -> list[dict]:
    return _L2_RECORDS.get(table_fqn, [])


def set_config(config: dict[str, Any]) -> None:
    _PIPELINE_CONFIG.update(config)


def get_config() -> dict[str, Any]:
    return _PIPELINE_CONFIG
