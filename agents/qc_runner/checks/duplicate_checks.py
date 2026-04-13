"""Duplicate record QC checks."""
from __future__ import annotations

import logging
from typing import Any

from schemas.qc_results import QCIssue

logger = logging.getLogger(__name__)

_DEFAULT_DEDUP_KEYS = ["customer_id", "address_line1", "zip_code"]


class DuplicateCheckRunner:
    def __init__(self, config: Any):
        cfg = getattr(config, "duplicate_checks", {}) or {}
        self.dedup_keys: list[str] = cfg.get("dedup_key_columns", _DEFAULT_DEDUP_KEYS)

    def run(self, df: Any, table_fqn: str, target_columns: list[str] | None = None) -> list[QCIssue]:
        try:
            from pyspark.sql import DataFrame as SparkDF
            if isinstance(df, SparkDF):
                return self._run_spark(df, table_fqn)
        except ImportError:
            pass
        return self._run_pandas(df, table_fqn)

    def _run_spark(self, df: Any, table_fqn: str) -> list[QCIssue]:
        import pyspark.sql.functions as F

        available_keys = [k for k in self.dedup_keys if k in df.columns]
        if not available_keys:
            logger.warning("No dedup key columns found in DataFrame, skipping duplicate check.")
            return []

        pk_col = self._detect_pk(df.columns)
        window = __import__("pyspark.sql.window", fromlist=["Window"]).Window
        w = window.partitionBy(*available_keys).orderBy(F.lit(1))

        dupes = (
            df.withColumn("_dup_count", F.count("*").over(w.rowsBetween(
                window.unboundedPreceding, window.unboundedFollowing
            )))
            .filter(F.col("_dup_count") > 1)
            .drop("_dup_count")
        )

        issues = []
        for row in dupes.limit(500).collect():
            record_id = str(getattr(row, pk_col, "unknown"))
            key_vals = {k: str(getattr(row, k, "")) for k in available_keys}
            issues.append(
                QCIssue(
                    record_id=record_id,
                    table_fqn=table_fqn,
                    column_name=",".join(available_keys),
                    check_type="duplicate",
                    original_value=str(key_vals),
                    suggested_value=None,
                    confidence_score=1.0,
                    support_level="L2",   # duplicates always L2 - never auto-delete
                    severity="warning",
                    issue_metadata={"dedup_keys": available_keys, "key_values": key_vals},
                )
            )
        return issues

    def _run_pandas(self, df: Any, table_fqn: str) -> list[QCIssue]:
        available_keys = [k for k in self.dedup_keys if k in df.columns]
        if not available_keys:
            return []

        pk_col = self._detect_pk(list(df.columns))
        dupes = df[df.duplicated(subset=available_keys, keep=False)]
        issues = []
        for _, row in dupes.iterrows():
            record_id = str(row.get(pk_col, row.name))
            key_vals = {k: str(row.get(k, "")) for k in available_keys}
            issues.append(
                QCIssue(
                    record_id=record_id,
                    table_fqn=table_fqn,
                    column_name=",".join(available_keys),
                    check_type="duplicate",
                    original_value=str(key_vals),
                    suggested_value=None,
                    confidence_score=1.0,
                    support_level="L2",
                    severity="warning",
                    issue_metadata={"dedup_keys": available_keys, "key_values": key_vals},
                )
            )
        return issues

    @staticmethod
    def _detect_pk(columns: list[str]) -> str:
        for name in ("id", "customer_id", "record_id", "pk"):
            if name in columns:
                return name
        return columns[0] if columns else "index"
