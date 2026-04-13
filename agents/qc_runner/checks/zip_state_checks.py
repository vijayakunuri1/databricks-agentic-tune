"""ZIP code vs state cross-validation check using GeoNames reference data."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from schemas.qc_results import QCIssue

if TYPE_CHECKING:
    from data_ingestion.geonames_fetcher import GeonamesIndex

logger = logging.getLogger(__name__)


class ZipStateMismatchChecker:
    """
    Detects records where the state code doesn't match what GeoNames
    expects for the given ZIP code.

    Example SEC EDGAR issue: company lists state='CA' but ZIP='75001' which
    is in Texas. This is a strong signal of a data entry error or stale address.

    Requires a GeonamesIndex to be provided at construction time.
    """

    def __init__(self, config: Any, geonames_index: "GeonamesIndex | None" = None):
        self.geonames = geonames_index
        self.zip_col = "zip_code"
        self.state_col = "state_or_country"   # SEC EDGAR column name
        self.city_col = "city"

    def run(self, df: Any, table_fqn: str, target_columns: list[str] | None = None) -> list[QCIssue]:
        if self.geonames is None:
            logger.warning("ZipStateMismatchChecker: no GeoNames index, skipping.")
            return []

        import pandas as pd
        # Handle both classic pyspark.sql.DataFrame and Spark Connect variant (serverless)
        if not isinstance(df, pd.DataFrame) and "DataFrame" in type(df).__name__:
            df = df.toPandas()
        return self._run_pandas(df, table_fqn)

    def _run_pandas(self, df: Any, table_fqn: str) -> list[QCIssue]:
        import pandas as pd

        issues: list[QCIssue] = []
        pk_col = self._detect_pk(list(df.columns))

        # Determine actual column names in this DataFrame
        zip_col = self._find_col(df.columns, ["zip_code", "zipCode", "postal_code", "zip"])
        state_col = self._find_col(df.columns, ["state_or_country", "stateOrCountry", "state", "state_code"])
        city_col = self._find_col(df.columns, ["city"])

        if zip_col is None or state_col is None:
            logger.info("ZipStateMismatchChecker: zip/state columns not found, skipping.")
            return []

        for _, row in df.iterrows():
            zip_val = str(row.get(zip_col, "") or "").strip()
            state_val = str(row.get(state_col, "") or "").strip().upper()
            city_val = str(row.get(city_col, "") or "").strip() if city_col else ""

            if not zip_val or not state_val:
                continue

            # Only check 5-digit US ZIPs
            zip5 = zip_val.zfill(5)[:5]
            if not zip5.isdigit():
                continue

            expected_state = self.geonames.state_for_zip(zip5)
            if expected_state is None:
                continue   # ZIP not in GeoNames (valid foreign ZIP, skip)

            if state_val != expected_state:
                record_id = str(row.get(pk_col, row.name))
                expected_city = self.geonames.canonical_city_for_zip(zip5)
                issues.append(
                    QCIssue(
                        record_id=record_id,
                        table_fqn=table_fqn,
                        column_name=state_col,
                        check_type="zip_state_mismatch",
                        original_value=state_val,
                        suggested_value=expected_state,
                        confidence_score=0.90,   # GeoNames is authoritative
                        support_level="L2",       # always L2 — state mismatch is business-significant
                        severity="warning",
                        issue_metadata={
                            "zip_code": zip5,
                            "recorded_state": state_val,
                            "expected_state_for_zip": expected_state,
                            "expected_city_for_zip": expected_city,
                            "recorded_city": city_val,
                        },
                    )
                )
        return issues

    @staticmethod
    def _find_col(columns: Any, candidates: list[str]) -> str | None:
        col_lower = {c.lower(): c for c in columns}
        for candidate in candidates:
            if candidate.lower() in col_lower:
                return col_lower[candidate.lower()]
        return None

    @staticmethod
    def _detect_pk(columns: list[str]) -> str:
        for name in ("id", "cik", "customer_id", "record_id", "pk"):
            if name in columns:
                return name
        return columns[0] if columns else "index"
