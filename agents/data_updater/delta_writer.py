"""Write QC audit columns back to Delta tables via MERGE."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from schemas.qc_results import CorrectionRecord

logger = logging.getLogger(__name__)

_QC_AUDIT_COLUMNS = [
    "qc_status", "qc_flag", "qc_corrected_column", "qc_original_value",
    "qc_corrected_value", "qc_confidence_score", "qc_correction_method",
    "qc_support_level", "qc_run_id", "qc_batch_id", "qc_checked_at",
    "qc_corrected_at", "qc_agent_version", "qc_all_issues",
    "qc_human_reviewed", "qc_human_decision", "qc_human_reviewed_at",
    "qc_human_reviewer",
]


class DeltaWriter:
    """
    Adds QC audit columns to the target table (idempotent ALTER TABLE)
    then MERGEs corrections using the Delta Lake Python API.
    """

    AGENT_VERSION = "data_updater_agent_v1.0.0"

    def __init__(self, spark: Any):
        self.spark = spark

    def merge_corrections(
        self,
        source_table: str,
        corrections: list[CorrectionRecord],
        primary_key: str,
        batch_id: str,
        run_id: str,
        dry_run: bool = False,
    ) -> int:
        """
        Merge QC corrections back to the source Delta table.
        Adds missing qc_* columns via ALTER TABLE first, then runs a standard MERGE.
        Returns the number of records merged.
        """
        from configs.input_validation import (
            validate_fqn, validate_column_name, validate_batch_id,
            backtick_fqn, backtick_identifier,
        )
        source_table = validate_fqn(source_table, "source_table")
        primary_key  = validate_column_name(primary_key, "primary_key")
        batch_id     = validate_batch_id(batch_id, "batch_id")

        if not corrections:
            return 0

        if dry_run:
            logger.info("[dry_run] Would merge %d corrections to %s", len(corrections), source_table)
            return len(corrections)

        import pandas as pd

        now = datetime.utcnow()
        rows = []
        for c in corrections:
            rows.append({
                primary_key: c.record_id,
                "qc_status": "AUTO_CORRECTED" if c.was_applied else "NEEDS_REVIEW",
                "qc_flag": c.qc_flag,
                "qc_corrected_column": c.column_name,
                "qc_original_value": c.original_value,
                "qc_corrected_value": c.corrected_value,
                "qc_confidence_score": float(c.confidence_score),
                "qc_correction_method": c.correction_method,
                "qc_support_level": "L1" if c.was_applied else "L2",
                "qc_run_id": run_id,
                "qc_batch_id": batch_id,
                "qc_checked_at": now,
                "qc_corrected_at": now if c.was_applied else None,
                "qc_agent_version": self.AGENT_VERSION,
                "qc_all_issues": json.dumps({"issue_id": c.issue_id, "reasoning": c.llm_reasoning}),
                "qc_human_reviewed": False,
                "qc_human_decision": None,
                "qc_human_reviewed_at": None,
                "qc_human_reviewer": None,
            })

        updates_df = self.spark.createDataFrame(pd.DataFrame(rows))
        # Build a safe temp view name: batch_id is already validated to contain
        # only [a-zA-Z0-9_\-]; replace hyphens so the name is a valid SQL identifier.
        safe_view_suffix = batch_id.replace("-", "_")
        view_name = f"_qc_updates_{safe_view_suffix}"
        updates_df.createOrReplaceTempView(view_name)

        # Ensure all qc_* columns exist on the target table before MERGE.
        self._ensure_qc_columns(source_table)

        # Build explicit SET clause so the MERGE only touches qc_* columns.
        # All column names in _QC_AUDIT_COLUMNS are internal constants — no
        # user input involved — but we backtick them for correctness.
        qc_cols = [c for c in _QC_AUDIT_COLUMNS]
        set_clause = ", ".join(f"target.`{c}` = source.`{c}`" for c in qc_cols)

        # Backtick-quote all identifiers that came from user/message input.
        safe_table  = backtick_fqn(source_table)
        safe_pk     = backtick_identifier(primary_key)

        self.spark.sql(f"""
            MERGE INTO {safe_table} AS target
            USING {view_name} AS source
            ON target.{safe_pk} = source.{safe_pk}
            WHEN MATCHED THEN UPDATE SET {set_clause}
        """)

        logger.info("Merged %d corrections to %s", len(corrections), source_table)
        return len(corrections)

    def _ensure_qc_columns(self, table_fqn: str) -> None:
        """
        Idempotently add qc_* audit columns to the target table.
        Checks existing columns first, then issues a single ALTER TABLE ADD COLUMNS
        statement for only the columns that are missing.
        Spark SQL syntax: ALTER TABLE t ADD COLUMNS (col TYPE, ...)
        """
        from configs.input_validation import validate_fqn, backtick_fqn
        table_fqn = validate_fqn(table_fqn, "table_fqn")

        _col_types = [
            ("qc_status",            "STRING"),
            ("qc_flag",              "STRING"),
            ("qc_corrected_column",  "STRING"),
            ("qc_original_value",    "STRING"),
            ("qc_corrected_value",   "STRING"),
            ("qc_confidence_score",  "DOUBLE"),
            ("qc_correction_method", "STRING"),
            ("qc_support_level",     "STRING"),
            ("qc_run_id",            "STRING"),
            ("qc_batch_id",          "STRING"),
            ("qc_checked_at",        "TIMESTAMP"),
            ("qc_corrected_at",      "TIMESTAMP"),
            ("qc_agent_version",     "STRING"),
            ("qc_all_issues",        "STRING"),
            ("qc_human_reviewed",    "BOOLEAN"),
            ("qc_human_decision",    "STRING"),
            ("qc_human_reviewed_at", "TIMESTAMP"),
            ("qc_human_reviewer",    "STRING"),
        ]
        existing = {c.lower() for c in self.spark.table(table_fqn).columns}
        missing = [(col, dtype) for col, dtype in _col_types if col not in existing]
        if not missing:
            return
        # _col_types is an internal constant — col names are safe — but
        # backtick them for correctness. dtype values are also constants.
        col_defs = ", ".join(f"`{col}` {dtype}" for col, dtype in missing)
        safe_table = backtick_fqn(table_fqn)
        self.spark.sql(f"ALTER TABLE {safe_table} ADD COLUMNS ({col_defs})")
        logger.info("Added %d qc_* columns to %s: %s",
                    len(missing), table_fqn, [c for c, _ in missing])

    def write_audit_log(
        self,
        audit_log_table: str,
        corrections: list[CorrectionRecord],
        batch_id: str,
        run_id: str,
        table_fqn: str,
    ) -> None:
        """Append one row per correction to the QC audit log Delta table."""
        if not corrections:
            return

        import pandas as pd

        now = datetime.utcnow()
        rows = [
            {
                "audit_id": c.issue_id,
                "run_id": run_id,
                "batch_id": batch_id,
                "table_fqn": table_fqn,
                "record_id": c.record_id,
                "column_name": c.column_name,
                "check_type": "unknown",
                "original_value": c.original_value,
                "corrected_value": c.corrected_value,
                "confidence_score": float(c.confidence_score),
                "support_level": "L1" if c.was_applied else "L2",
                "was_applied": c.was_applied,
                "correction_method": c.correction_method,
                "llm_reasoning": c.llm_reasoning,
                "created_at": now,
            }
            for c in corrections
        ]

        log_df = self.spark.createDataFrame(pd.DataFrame(rows))
        log_df.write.format("delta").mode("append").saveAsTable(audit_log_table)
        logger.info("Wrote %d rows to audit log %s", len(rows), audit_log_table)
