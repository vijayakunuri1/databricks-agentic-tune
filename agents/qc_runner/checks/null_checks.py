"""Null / missing value QC checks."""
from __future__ import annotations

import logging
from typing import Any

from schemas.qc_results import QCIssue

logger = logging.getLogger(__name__)

CRITICAL_COLUMNS_DEFAULT = {"customer_id", "id", "cik", "record_id", "address_line1"}


class NullCheckRunner:
    def __init__(self, config: Any):
        cfg = getattr(config, "null_checks", {}) or {}
        self.critical_columns: set[str] = set(
            cfg.get("critical_columns", list(CRITICAL_COLUMNS_DEFAULT))
        )

    def run(self, df: Any, table_fqn: str, target_columns: list[str] | None = None) -> list[QCIssue]:
        """
        Accepts either a Spark DataFrame or a pandas DataFrame.
        Returns one QCIssue per null/blank cell found.
        """
        import pandas as pd
        # Detect any Spark DataFrame variant by class name (covers both classic
        # pyspark.sql.DataFrame and Spark Connect's pyspark.sql.connect.dataframe.DataFrame
        # used in Databricks serverless)
        type_name = type(df).__name__
        module_name = type(df).__module__ or ""
        if not isinstance(df, pd.DataFrame) and ("DataFrame" in type_name or "pyspark" in module_name):
            df = df.toPandas()

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Unsupported DataFrame type: {type(df)}")

        columns = target_columns or list(df.columns)
        return self._run_pandas(df, table_fqn, columns)

    def _run_spark(self, df: Any, table_fqn: str, columns: list[str]) -> list[QCIssue]:
        import pyspark.sql.functions as F

        issues: list[QCIssue] = []
        # Collect null counts per column in one pass
        def _is_blank(c):
            return F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == "")

        null_counts = df.select(
            [F.sum(_is_blank(c).cast("int")).alias(c) for c in columns if c in df.columns]
        ).collect()[0]

        for col_name in columns:
            if col_name not in df.columns:
                continue
            count = getattr(null_counts, col_name, 0) or 0
            if count == 0:
                continue

            # For each null/blank row, create an issue (sample up to 1000)
            null_rows = df.filter(_is_blank(col_name)).limit(1000)
            # Detect primary key column heuristically
            pk_col = self._detect_pk(df.columns)

            for row in null_rows.collect():
                record_id = str(getattr(row, pk_col, "unknown"))
                severity = "critical" if col_name in self.critical_columns else "warning"
                issues.append(
                    QCIssue(
                        record_id=record_id,
                        table_fqn=table_fqn,
                        column_name=col_name,
                        check_type="null",
                        original_value=None,
                        suggested_value=None,
                        confidence_score=1.0,
                        support_level="L2",
                        severity=severity,
                    )
                )
        return issues

    def _run_pandas(self, df: Any, table_fqn: str, columns: list[str]) -> list[QCIssue]:

        issues: list[QCIssue] = []
        pk_col = self._detect_pk(list(df.columns))
        for col_name in columns:
            if col_name not in df.columns:
                continue
            null_rows = df[df[col_name].isnull() | (df[col_name].astype(str).str.strip() == "")]
            for _, row in null_rows.iterrows():
                record_id = str(row.get(pk_col, row.name))
                severity = "critical" if col_name in self.critical_columns else "warning"
                issues.append(
                    QCIssue(
                        record_id=record_id,
                        table_fqn=table_fqn,
                        column_name=col_name,
                        check_type="null",
                        original_value=None,
                        suggested_value=None,
                        confidence_score=1.0,
                        support_level="L2",
                        severity=severity,
                    )
                )
        return issues

    @staticmethod
    def _detect_pk(columns: list[str]) -> str:
        for name in ("id", "customer_id", "record_id", "pk"):
            if name in columns:
                return name
        return columns[0] if columns else "index"
