"""MCP tools: L2 human review — fetch flagged records, approve corrections."""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Only allow safe identifier characters to prevent SQL injection
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_\-\.]{0,127}$")
# record_id may be a UUID, integer string, or short alphanumeric key
_RECORD_ID_RE = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")
# Reviewer name: printable ASCII, no SQL metacharacters
_REVIEWER_RE = re.compile(r"^[a-zA-Z0-9_\-\. @]{1,64}$")
# Manual value: strip SQL metacharacters; max length 1024
_MAX_VALUE_LEN = 1024

_VALID_DECISIONS = frozenset({"ACCEPT", "REJECT", "MANUAL_EDIT"})


def _validate_identifier(name: str, label: str) -> str:
    """Raise ValueError if name is not a safe SQL identifier."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid {label} {name!r}: only alphanumeric, underscore, hyphen, "
            "dot allowed (max 128 chars)."
        )
    return name


def _validate_record_id(value: str) -> str:
    if not _RECORD_ID_RE.match(value):
        raise ValueError(
            f"Invalid record_id {value!r}: only alphanumeric, hyphen, underscore, "
            "dot allowed (max 128 chars)."
        )
    return value


def _validate_reviewer(value: str) -> str:
    if not _REVIEWER_RE.match(value):
        raise ValueError(
            f"Invalid reviewer {value!r}: only alphanumeric, space, hyphen, "
            "underscore, dot, @ allowed (max 64 chars)."
        )
    return value


def _sanitize_manual_value(value: str) -> str:
    """Truncate and escape characters that cannot appear in a Spark SQL literal."""
    safe = value.replace("\\", "").replace("'", "\\'").replace("\x00", "")
    return safe[:_MAX_VALUE_LEN]


def register_review_tools(mcp) -> None:
    """Register L2 review tools on the FastMCP app."""

    @mcp.tool()
    def get_l2_flagged_records(
        catalog: str,
        schema: str,
        table: str,
        limit: int = 50,
        min_confidence: float = 0.0,
    ) -> str:
        """
        Retrieve records flagged for human review (L2 support level).

        These are records where the QC pipeline found an issue but confidence
        was too low for automatic correction (0.50–0.84). A human reviewer
        (or this AI session) should inspect and decide whether to accept,
        reject, or manually edit the suggestion.

        Args:
            catalog, schema, table: Table coordinates
            limit:          Max records to return (1–1000, default: 50)
            min_confidence: Only return records with confidence >= this value

        Returns:
            JSON with {count, records: [{record_id, column, original, suggested, confidence}]}
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
            log_rate_limit_hit(client_ip="mcp-tool", tool="get_l2_flagged_records")
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
                tool="get_l2_flagged_records",
            )
            logger.warning("get_l2_flagged_records access denied: %s", exc)
            return json.dumps({"error": str(exc)})

        # Bound inputs to prevent resource exhaustion
        limit = max(1, min(int(limit), 1000))
        min_confidence = max(0.0, min(float(min_confidence), 1.0))

        from mcp_server.pipeline_state import get_config, get_l2_records

        records = get_l2_records(table_fqn)

        if not records:
            # Try reading from Spark if available
            cfg = get_config()
            spark = cfg.get("spark")
            if spark:
                try:
                    rows = (
                        spark.table(table_fqn)
                        .filter(
                            "qc_support_level = 'L2' "
                            "AND (qc_human_reviewed IS NULL OR qc_human_reviewed = false)"
                        )
                        .select(
                            "id", "qc_flag", "qc_corrected_column",
                            "qc_original_value", "qc_corrected_value",
                            "qc_confidence_score", "qc_all_issues",
                        )
                        .limit(limit)
                        .toPandas()
                    )
                    records = rows.to_dict("records")
                except Exception as exc:
                    # Don't leak internal table structure in the error
                    logger.error("Spark read failed for %s: %s", table_fqn, exc, exc_info=True)
                    return json.dumps({"error": f"Could not read L2 records from {table_fqn}."})

        if not records:
            return json.dumps({
                "count": 0,
                "records": [],
                "message": f"No L2 flagged records for {table_fqn}.",
            })

        filtered = [
            r for r in records
            if float(r.get("qc_confidence_score", r.get("confidence_score", 0))) >= min_confidence
        ][:limit]

        return json.dumps({
            "table": table_fqn,
            "count": len(filtered),
            "records": filtered,
            "note": "Review each record. Use approve_l2_correction to accept a suggestion.",
        })

    @mcp.tool()
    def approve_l2_correction(
        catalog: str,
        schema: str,
        table: str,
        record_id: str,
        decision: str = "ACCEPT",
        manual_value: str = "",
        reviewer: str = "mcp-agent",
    ) -> str:
        """
        Record a human decision on an L2-flagged record.

        Args:
            catalog, schema, table: Table coordinates
            record_id:   Primary key value of the record to review
            decision:    One of: ACCEPT (use suggested value), REJECT (keep original),
                         MANUAL_EDIT (use manual_value instead)
            manual_value: The corrected value when decision=MANUAL_EDIT
            reviewer:    Name/identifier of the reviewer (default: 'mcp-agent')

        Returns:
            JSON confirmation of the decision applied.
        """
        # --- Input validation (format + injection prevention) ---
        try:
            catalog = _validate_identifier(catalog, "catalog")
            schema = _validate_identifier(schema, "schema")
            table = _validate_identifier(table, "table")
            record_id = _validate_record_id(record_id)
            reviewer = _validate_reviewer(reviewer)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        # --- Rate limiting: Delta write (most restrictive bucket) ---
        from configs.rate_limiting import check_l2_write, rate_limit_error
        allowed, retry_after = check_l2_write()
        if not allowed:
            from configs.security_logging import log_rate_limit_hit
            log_rate_limit_hit(client_ip="mcp-tool", tool="approve_l2_correction")
            return json.dumps({"error": rate_limit_error(retry_after, "l2_write")})

        # --- IDOR check 1: table must be in the registered allowlist ---
        from mcp_server.pipeline_state import assert_table_registered
        try:
            table_fqn = assert_table_registered(catalog, schema, table)
        except PermissionError as exc:
            from configs.security_logging import log_access_denied
            log_access_denied(
                client_ip="mcp-tool",
                resource=f"{catalog}.{schema}.{table}",
                reason=str(exc),
                tool="approve_l2_correction",
            )
            logger.warning("approve_l2_correction access denied: %s", exc)
            return json.dumps({"error": str(exc)})

        decision = decision.upper().strip()
        if decision not in _VALID_DECISIONS:
            return json.dumps({
                "error": f"Invalid decision {decision!r}. Use ACCEPT, REJECT, or MANUAL_EDIT."
            })

        safe_manual_value = _sanitize_manual_value(manual_value) if manual_value else ""

        from mcp_server.pipeline_state import get_config
        from datetime import datetime, timezone

        cfg = get_config()
        spark = cfg.get("spark")

        if not spark:
            return json.dumps({
                "error": "No Spark session available. Cannot process review decision."
            })

        try:
            from delta.tables import DeltaTable

            # --- IDOR check 2: verify the specific record is actually in the L2 queue ---
            # The record must exist, have qc_support_level='L2', and not yet be reviewed.
            # This prevents a caller from writing decisions to arbitrary rows that were
            # never flagged for review, even within a registered table.
            pk_col = cfg.get("primary_key", "id")
            ownership_check = (
                spark.table(table_fqn)
                .filter(
                    f"`{pk_col}` = '{record_id}' "
                    "AND qc_support_level = 'L2' "
                    "AND (qc_human_reviewed IS NULL OR qc_human_reviewed = false)"
                )
                .limit(1)
            )
            if ownership_check.count() == 0:
                logger.warning(
                    "approve_l2_correction: record %r not found in L2 queue of %s",
                    record_id, table_fqn,
                )
                return json.dumps({
                    "error": (
                        f"Record {record_id!r} is not in the L2 review queue for {table_fqn}. "
                        "It may have already been reviewed, may not exist, or may not be "
                        "flagged for human review."
                    )
                })

            # --- Apply the decision ---
            dt = DeltaTable.forName(spark, table_fqn)
            now_str = datetime.now(timezone.utc).isoformat()

            # record_id has been validated against _RECORD_ID_RE (alphanumeric/hyphen/dot only)
            # so it is safe to interpolate into the condition string.
            condition = f"`{pk_col}` = '{record_id}'"

            set_clause: dict = {
                "qc_human_reviewed":    "true",
                "qc_human_decision":    f"'{decision}'",
                "qc_human_reviewed_at": f"'{now_str}'",
                "qc_human_reviewer":    f"'{reviewer}'",
            }

            if decision == "ACCEPT":
                set_clause["qc_status"] = "'AUTO_CORRECTED'"
            elif decision == "MANUAL_EDIT" and safe_manual_value:
                set_clause["qc_corrected_value"] = f"'{safe_manual_value}'"
                set_clause["qc_status"] = "'AUTO_CORRECTED'"

            dt.update(condition=condition, set=set_clause)
            msg = f"Decision '{decision}' recorded for record {record_id} in {table_fqn}."
            logger.info(msg)

            from configs.security_logging import log_data_modification
            log_data_modification(
                table_fqn=table_fqn,
                record_id=record_id,
                action=decision,
                reviewer=reviewer,
                dry_run=False,
            )

        except Exception as exc:
            # Suppress internal details from the response — log them server-side only
            logger.error(
                "Delta update failed for record %s in %s: %s",
                record_id, table_fqn, exc, exc_info=True,
            )
            return json.dumps({"error": "Failed to record decision. See server logs for details."})

        return json.dumps({
            "status": "ok",
            "record_id": record_id,
            "decision": decision,
            "table": table_fqn,
            "reviewer": reviewer,
            "message": msg,
        })
