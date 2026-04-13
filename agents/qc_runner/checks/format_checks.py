"""Format validation QC checks (email, phone, zip, date, state)."""
from __future__ import annotations

import re
import logging
from typing import Any

from schemas.qc_results import QCIssue

logger = logging.getLogger(__name__)

_US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC","PR","VI","GU","AS","MP",
}

_RULES: list[dict] = [
    {
        "column_pattern": re.compile(r".*email.*", re.I),
        "format": "email",
        "regex": re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"),
    },
    {
        "column_pattern": re.compile(r".*phone.*|.*mobile.*|.*fax.*", re.I),
        "format": "phone_us",
        "regex": re.compile(
            r"^(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$"
        ),
    },
    {
        "column_pattern": re.compile(r".*zip.*|.*postal.*", re.I),
        "format": "zip_us",
        "regex": re.compile(r"^[0-9]{5}(-[0-9]{4})?$"),
    },
    {
        "column_pattern": re.compile(r".*date.*|.*_at$|.*_on$", re.I),
        "format": "iso_date",
        "regex": re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?$"),
    },
]


class FormatCheckRunner:
    def __init__(self, config: Any):
        self.rules = _RULES

    def run(self, df: Any, table_fqn: str, target_columns: list[str] | None = None) -> list[QCIssue]:
        import pandas as pd
        # Handle both classic pyspark.sql.DataFrame and Spark Connect variant (serverless)
        if not isinstance(df, pd.DataFrame) and "DataFrame" in type(df).__name__:
            df = df.toPandas()
        return self._run_pandas(df, table_fqn, target_columns)

    def _run_pandas(self, df: Any, table_fqn: str, target_columns: list[str] | None) -> list[QCIssue]:

        issues: list[QCIssue] = []
        columns = target_columns or list(df.columns)
        pk_col = self._detect_pk(list(df.columns))

        for col_name in columns:
            if col_name not in df.columns:
                continue
            rule = self._match_rule(col_name)
            if rule is None:
                # Special case: state column validation
                if re.match(r".*state.*", col_name, re.I):
                    issues.extend(self._check_state(df, col_name, table_fqn, pk_col))
                continue

            for _, row in df.iterrows():
                val = row.get(col_name)
                if val is None or (isinstance(val, float) and __import__("math").isnan(val)):
                    continue
                val_str = str(val).strip()
                if not rule["regex"].match(val_str):
                    record_id = str(row.get(pk_col, row.name))
                    issues.append(
                        QCIssue(
                            record_id=record_id,
                            table_fqn=table_fqn,
                            column_name=col_name,
                            check_type="format",
                            original_value=val_str,
                            suggested_value=None,
                            confidence_score=0.95,
                            support_level="L1",
                            severity="warning",
                            issue_metadata={"format": rule["format"]},
                        )
                    )
        return issues

    def _check_state(self, df: Any, col_name: str, table_fqn: str, pk_col: str) -> list[QCIssue]:
        issues = []
        for _, row in df.iterrows():
            val = row.get(col_name)
            if val is None:
                continue
            val_str = str(val).strip().upper()
            if val_str not in _US_STATES:
                record_id = str(row.get(pk_col, row.name))
                issues.append(
                    QCIssue(
                        record_id=record_id,
                        table_fqn=table_fqn,
                        column_name=col_name,
                        check_type="format",
                        original_value=str(val).strip(),
                        suggested_value=None,
                        confidence_score=0.90,
                        support_level="L2",
                        severity="warning",
                        issue_metadata={"format": "us_state", "valid_values": "US state codes"},
                    )
                )
        return issues

    def _match_rule(self, col_name: str) -> dict | None:
        for rule in self.rules:
            if rule["column_pattern"].match(col_name):
                return rule
        return None

    @staticmethod
    def _detect_pk(columns: list[str]) -> str:
        for name in ("id", "customer_id", "record_id", "pk"):
            if name in columns:
                return name
        return columns[0] if columns else "index"
