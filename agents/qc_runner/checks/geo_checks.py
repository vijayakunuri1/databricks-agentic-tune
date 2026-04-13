"""Geographic address QC checks: parse + geocode validation."""
from __future__ import annotations

import logging
import time
from typing import Any

from schemas.qc_results import QCIssue

logger = logging.getLogger(__name__)


class GeoCheckRunner:
    """
    Two-stage geo validation:
    1. Parse address components with usaddress (structural)
    2. Attempt geocode with Nominatim to confirm the address exists

    - geo_typo: address parses but geocoder returns no result (likely a typo)
    - geo_invalid: address cannot be parsed at all
    """

    def __init__(self, config: Any):
        self.config = config
        self.geocoder = self._init_geocoder()
        self._geocode_cache: dict[str, tuple[float, float] | None] = {}
        self._geocode_calls = 0
        self.max_geocode = config.thresholds.max_geocode_per_run
        self.rate_limit_delay = config.thresholds.geocode_rate_limit
        self.timeout = config.thresholds.geocode_timeout

        # Address column mapping
        qc_rules = self._load_qc_rules(config)
        geo_cfg = qc_rules.get("geo_checks", {})
        addr_cols = geo_cfg.get("address_columns", {})
        self.street_col = addr_cols.get("street", "address_line1")
        self.city_col = addr_cols.get("city", "city")
        self.state_col = addr_cols.get("state", "state")
        self.zip_col = addr_cols.get("zip", "zip_code")
        self.country_col = addr_cols.get("country", "country")

    def run(self, df: Any, table_fqn: str, target_columns: list[str] | None = None) -> list[QCIssue]:
        try:
            from pyspark.sql import DataFrame as SparkDF
            if isinstance(df, SparkDF):
                return self._run_pandas(df.toPandas(), table_fqn)
        except ImportError:
            pass
        return self._run_pandas(df, table_fqn)

    def _run_pandas(self, df: Any, table_fqn: str) -> list[QCIssue]:
        issues: list[QCIssue] = []
        pk_col = self._detect_pk(list(df.columns))

        for _, row in df.iterrows():
            if self._geocode_calls >= self.max_geocode:
                logger.warning("Max geocode limit (%d) reached, stopping geo checks.", self.max_geocode)
                break

            full_address = self._build_full_address(row)
            if not full_address.strip():
                continue

            record_id = str(row.get(pk_col, row.name))
            issue = self._check_address(full_address, record_id, table_fqn, row)
            if issue:
                issues.append(issue)

        return issues

    def _check_address(
        self, full_address: str, record_id: str, table_fqn: str, row: Any
    ) -> QCIssue | None:
        # Stage 1: Parse
        parsed, parse_type = self._parse_address(full_address)
        if parse_type == "Ambiguous" or not parsed:
            return QCIssue(
                record_id=record_id,
                table_fqn=table_fqn,
                column_name=self.street_col,
                check_type="geo_invalid",
                original_value=full_address,
                suggested_value=None,
                confidence_score=0.3,
                support_level="L2",
                severity="warning",
                issue_metadata={"full_address": full_address, "parse_type": parse_type},
            )

        # Stage 2: Geocode
        coords = self._geocode_with_cache(full_address)
        if coords is None:
            # Try to suggest a corrected city/state
            city_val = str(row.get(self.city_col, "")).strip()
            state_val = str(row.get(self.state_col, "")).strip()
            return QCIssue(
                record_id=record_id,
                table_fqn=table_fqn,
                column_name=self.city_col,
                check_type="geo_typo",
                original_value=full_address,
                suggested_value=None,
                confidence_score=0.6,
                support_level="L2",
                severity="warning",
                issue_metadata={
                    "full_address": full_address,
                    "city": city_val,
                    "state": state_val,
                    "parsed_components": parsed,
                },
            )
        return None

    def _parse_address(self, address: str) -> tuple[dict, str]:
        try:
            import usaddress
            tagged, address_type = usaddress.tag(address)
            return dict(tagged), address_type
        except Exception as exc:
            logger.debug("usaddress parse error for '%s': %s", address, exc)
            return {}, "Ambiguous"

    def _geocode_with_cache(self, address: str) -> tuple[float, float] | None:
        if address in self._geocode_cache:
            return self._geocode_cache[address]

        if self.geocoder is None:
            return None

        try:
            time.sleep(self.rate_limit_delay)  # respect Nominatim rate limit
            self._geocode_calls += 1
            location = self.geocoder.geocode(address, timeout=self.timeout)
            result = (location.latitude, location.longitude) if location else None
            self._geocode_cache[address] = result
            return result
        except Exception as exc:
            logger.warning("Geocode error for '%s': %s", address, exc)
            self._geocode_cache[address] = None
            return None

    def _build_full_address(self, row: Any) -> str:
        parts = []
        for col in (self.street_col, self.city_col, self.state_col, self.zip_col):
            val = row.get(col, "")
            if val and str(val).strip().lower() not in ("nan", "none", ""):
                parts.append(str(val).strip())
        country = row.get(self.country_col, "US")
        if country and str(country).strip().lower() not in ("nan", "none", ""):
            parts.append(str(country).strip())
        return ", ".join(parts)

    def _init_geocoder(self) -> Any:
        try:
            from geopy.geocoders import Nominatim
            return Nominatim(user_agent=self.config.geocoder_user_agent)
        except Exception as exc:
            logger.warning("Could not initialize geocoder: %s", exc)
            return None

    @staticmethod
    def _load_qc_rules(config: Any) -> dict:
        try:
            import yaml
            import os
            path = os.path.join(os.path.dirname(__file__), "../../../configs/qc_rules.yaml")
            with open(os.path.abspath(path)) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    @staticmethod
    def _detect_pk(columns: list[str]) -> str:
        for name in ("id", "customer_id", "record_id", "pk"):
            if name in columns:
                return name
        return columns[0] if columns else "index"
