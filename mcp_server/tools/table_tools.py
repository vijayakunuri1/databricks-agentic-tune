"""MCP tools: list tables, trigger QC run, fetch results."""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Allowed QC check names — reject anything outside this set
_ALLOWED_CHECKS = frozenset({"null", "format", "duplicate", "geo", "zip_state_mismatch"})

# Safe SQL identifier: alphanumeric + underscore (no dots in individual parts)
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_\-]{0,127}$")


def _validate_identifier(name: str, label: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid {label} {name!r}: only alphanumeric and underscore allowed (max 128 chars)."
        )
    return name


def _validate_checks(checks_str: str) -> list[str]:
    parts = [c.strip().lower() for c in checks_str.split(",") if c.strip()]
    unknown = [c for c in parts if c not in _ALLOWED_CHECKS]
    if unknown:
        raise ValueError(
            f"Unknown check type(s): {unknown}. "
            f"Allowed: {sorted(_ALLOWED_CHECKS)}"
        )
    if not parts:
        raise ValueError("At least one check type is required.")
    return parts


def register_table_tools(mcp) -> None:
    """Register table-related tools on the FastMCP app."""

    @mcp.tool()
    def list_qc_tables() -> str:
        """
        List all Delta tables registered for QC monitoring.
        Returns a JSON array of {catalog, schema, table, fqn, description}.
        """
        from configs.rate_limiting import check_read, rate_limit_error
        allowed, retry_after = check_read()
        if not allowed:
            from configs.security_logging import log_rate_limit_hit
            log_rate_limit_hit(client_ip="mcp-tool", tool="list_qc_tables")
            return json.dumps({"error": rate_limit_error(retry_after, "read")})

        from mcp_server.pipeline_state import list_tables
        tables = list_tables()
        if not tables:
            return json.dumps({"tables": [], "message": "No tables registered. Use register_table() at startup."})
        return json.dumps({"tables": tables, "count": len(tables)})

    @mcp.tool()
    def run_qc_check(
        catalog: str,
        schema: str,
        table: str,
        checks: str = "null,format,duplicate,geo,zip_state_mismatch",
        primary_key: str = "id",
        dry_run: bool = True,
    ) -> str:
        """
        Run the full QC pipeline on a Delta table.

        Args:
            catalog:     Unity Catalog catalog name (e.g. 'main')
            schema:      Schema name (e.g. 'sec_edgar')
            table:       Table name (e.g. 'companies')
            checks:      Comma-separated QC checks to run. Options:
                         null, format, duplicate, geo, zip_state_mismatch
            primary_key: Primary key column name (default: 'id')
            dry_run:     If true, detect issues but do NOT write corrections back
                         to the table. Safe default for exploration.

        Returns:
            JSON summary with total_records, total_issues, l1_corrected, l2_flagged.
        """
        # --- Input validation (format) ---
        try:
            catalog = _validate_identifier(catalog, "catalog")
            schema = _validate_identifier(schema, "schema")
            table = _validate_identifier(table, "table")
            primary_key = _validate_identifier(primary_key, "primary_key")
            checks_list = _validate_checks(checks)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        # --- IDOR check: table must be in the registered allowlist ---
        from mcp_server.pipeline_state import assert_table_registered
        try:
            table_fqn = assert_table_registered(catalog, schema, table)
        except PermissionError as exc:
            from configs.security_logging import log_access_denied
            log_access_denied(
                client_ip="mcp-tool",
                resource=f"{catalog}.{schema}.{table}",
                reason=str(exc),
                tool="run_qc_check",
            )
            logger.warning("run_qc_check access denied: %s", exc)
            return json.dumps({"error": str(exc)})

        # --- Rate limiting: expensive operation (Spark + LLM) ---
        from configs.rate_limiting import check_qc_run, rate_limit_error
        allowed, retry_after = check_qc_run()
        if not allowed:
            from configs.security_logging import log_rate_limit_hit
            log_rate_limit_hit(client_ip="mcp-tool", tool="run_qc_check")
            return json.dumps({"error": rate_limit_error(retry_after, "qc_run")})

        from mcp_server.pipeline_state import get_config, store_run_result
        from pipelines.full_qc_pipeline import run_pipeline

        cfg = get_config()

        logger.info("MCP: run_qc_check on %s checks=%s", table_fqn, checks_list)

        result = run_pipeline(
            table_catalog=catalog,
            table_schema=schema,
            table_name=table,
            qc_checks=checks_list,
            primary_key=primary_key,
            dry_run=dry_run,
            config=cfg.get("app_config"),
            spark=cfg.get("spark"),
            city_reference=cfg.get("city_reference"),
        )

        store_run_result(table_fqn, result)

        return json.dumps({
            "status": "completed",
            "table": table_fqn,
            "total_records": result.total_records,
            "total_issues": result.total_issues,
            "l1_auto_corrected": result.l1_corrected,
            "l2_flagged_for_review": result.l2_flagged,
            "skipped": result.skipped,
            "avg_confidence": round(result.avg_confidence, 3),
            "duration_seconds": round(result.duration_seconds, 1),
            "dry_run": result.dry_run,
            "check_type_breakdown": result.check_type_breakdown,
            "severity_breakdown": result.severity_breakdown,
            "run_id": result.run_id,
        })

    @mcp.tool()
    def get_qc_results(catalog: str, schema: str, table: str) -> str:
        """
        Retrieve the results of the most recent QC run for a table.

        Returns the full WorkflowResult as JSON, including issue breakdowns,
        confidence scores, and run metadata.
        """
        # --- Input validation (format) ---
        try:
            catalog = _validate_identifier(catalog, "catalog")
            schema = _validate_identifier(schema, "schema")
            table = _validate_identifier(table, "table")
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        # --- Rate limiting: read ---
        from configs.rate_limiting import check_read, rate_limit_error
        allowed, retry_after = check_read()
        if not allowed:
            from configs.security_logging import log_rate_limit_hit
            log_rate_limit_hit(client_ip="mcp-tool", tool="get_qc_results")
            return json.dumps({"error": rate_limit_error(retry_after, "read")})

        # --- IDOR check: table must be in the registered allowlist ---
        from mcp_server.pipeline_state import assert_table_registered
        try:
            table_fqn = assert_table_registered(catalog, schema, table)
        except PermissionError as exc:
            from configs.security_logging import log_access_denied
            log_access_denied(
                client_ip="mcp-tool",
                resource=f"{catalog}.{schema}.{table}",
                reason=str(exc),
                tool="get_qc_results",
            )
            logger.warning("get_qc_results access denied: %s", exc)
            return json.dumps({"error": str(exc)})

        from mcp_server.pipeline_state import get_last_run
        result = get_last_run(table_fqn)
        if result is None:
            return json.dumps({
                "error": f"No QC run found for {table_fqn}. Call run_qc_check first."
            })
        return json.dumps(result)
