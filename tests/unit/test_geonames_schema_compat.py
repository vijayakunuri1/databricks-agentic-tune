"""Smoke tests: GeoNames schema compatibility after overwriteSchema fix.

The GeoNames raw data (12-column schema from download.geonames.org) differs from
the legacy Delta table schema (8 columns). These tests verify that all downstream
consumers — GeonamesIndex, ZipStateMismatchChecker, and the full pipeline — work
correctly with both schemas.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data_ingestion.geonames_fetcher import GeonamesIndex


# ---------------------------------------------------------------------------
# Fixtures: two DataFrames representing old and new GeoNames schemas
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_geonames_df() -> pd.DataFrame:
    """12-column schema as returned by GeonamesIndex.download() (the raw data)."""
    return pd.DataFrame([
        {
            "country_code": "US", "postal_code": "10001", "place_name": "New York",
            "admin_name1": "New York", "admin_code1": "NY",
            "admin_name2": "New York", "admin_code2": "061",
            "admin_name3": "", "admin_code3": "",
            "latitude": 40.7484, "longitude": -73.9967, "accuracy": "4",
        },
        {
            "country_code": "US", "postal_code": "90001", "place_name": "Los Angeles",
            "admin_name1": "California", "admin_code1": "CA",
            "admin_name2": "Los Angeles", "admin_code2": "037",
            "admin_name3": "", "admin_code3": "",
            "latitude": 33.9425, "longitude": -118.2551, "accuracy": "4",
        },
        {
            "country_code": "US", "postal_code": "75001", "place_name": "Addison",
            "admin_name1": "Texas", "admin_code1": "TX",
            "admin_name2": "Dallas", "admin_code2": "113",
            "admin_name3": "", "admin_code3": "",
            "latitude": 32.9612, "longitude": -96.8292, "accuracy": "4",
        },
    ])


@pytest.fixture
def legacy_geonames_df() -> pd.DataFrame:
    """8-column schema as it existed in the old Delta table."""
    return pd.DataFrame([
        {
            "postal_code": "10001", "city": "New York", "state_code": "NY",
            "state_name": "New York", "county": "New York",
            "latitude": 40.7484, "longitude": -73.9967, "city_upper": "NEW YORK",
        },
        {
            "postal_code": "90001", "city": "Los Angeles", "state_code": "CA",
            "state_name": "California", "county": "Los Angeles",
            "latitude": 33.9425, "longitude": -118.2551, "city_upper": "LOS ANGELES",
        },
        {
            "postal_code": "75001", "city": "Addison", "state_code": "TX",
            "state_name": "Texas", "county": "Dallas",
            "latitude": 32.9612, "longitude": -96.8292, "city_upper": "ADDISON",
        },
    ])


# ---------------------------------------------------------------------------
# 1. GeonamesIndex works with BOTH schemas
# ---------------------------------------------------------------------------

class TestGeonamesIndexSchemaCompat:
    """GeonamesIndex._normalise_columns() must handle both schemas."""

    def test_raw_schema_state_lookup(self, raw_geonames_df):
        idx = GeonamesIndex(raw_geonames_df)
        assert idx.state_for_zip("10001") == "NY"
        assert idx.state_for_zip("90001") == "CA"
        assert idx.state_for_zip("75001") == "TX"

    def test_legacy_schema_state_lookup(self, legacy_geonames_df):
        idx = GeonamesIndex(legacy_geonames_df)
        assert idx.state_for_zip("10001") == "NY"
        assert idx.state_for_zip("90001") == "CA"
        assert idx.state_for_zip("75001") == "TX"

    def test_raw_schema_city_lookup(self, raw_geonames_df):
        idx = GeonamesIndex(raw_geonames_df)
        assert idx.canonical_city_for_zip("10001") == "New York"
        assert idx.canonical_city_for_zip("90001") == "Los Angeles"

    def test_legacy_schema_city_lookup(self, legacy_geonames_df):
        idx = GeonamesIndex(legacy_geonames_df)
        assert idx.canonical_city_for_zip("10001") == "New York"
        assert idx.canonical_city_for_zip("90001") == "Los Angeles"

    def test_raw_schema_validate_city_zip(self, raw_geonames_df):
        idx = GeonamesIndex(raw_geonames_df)
        result = idx.validate_city_zip("New York", "10001")
        assert result["valid"] is True

    def test_legacy_schema_validate_city_zip(self, legacy_geonames_df):
        idx = GeonamesIndex(legacy_geonames_df)
        result = idx.validate_city_zip("New York", "10001")
        assert result["valid"] is True

    def test_raw_schema_fuzzy_lookup(self, raw_geonames_df):
        idx = GeonamesIndex(raw_geonames_df)
        matches = idx.fuzzy_city_lookup("Los Angles", state_code="CA")
        assert len(matches) > 0
        assert matches[0]["city"] == "Los Angeles"

    def test_legacy_schema_fuzzy_lookup(self, legacy_geonames_df):
        idx = GeonamesIndex(legacy_geonames_df)
        matches = idx.fuzzy_city_lookup("Los Angles", state_code="CA")
        assert len(matches) > 0
        assert matches[0]["city"] == "Los Angeles"

    def test_raw_schema_lookup_zip(self, raw_geonames_df):
        idx = GeonamesIndex(raw_geonames_df)
        info = idx.lookup_zip("75001")
        assert info is not None
        assert info["state_code"] == "TX"
        assert info["canonical_city"] == "Addison"

    def test_legacy_schema_lookup_zip(self, legacy_geonames_df):
        idx = GeonamesIndex(legacy_geonames_df)
        info = idx.lookup_zip("75001")
        assert info is not None
        assert info["state_code"] == "TX"
        assert info["canonical_city"] == "Addison"


# ---------------------------------------------------------------------------
# 2. ZipStateMismatchChecker works with GeoNames raw schema
# ---------------------------------------------------------------------------

class TestZipStateMismatchWithGeoNames:
    """ZipStateMismatchChecker should detect mismatches using either schema."""

    def test_detects_mismatch_with_raw_schema(self, raw_geonames_df):
        from agents.qc_runner.checks.zip_state_checks import ZipStateMismatchChecker
        idx = GeonamesIndex(raw_geonames_df)
        checker = ZipStateMismatchChecker(config=None, geonames_index=idx)

        # ZIP 10001 is NY, but record says CA — should be flagged
        df = pd.DataFrame([
            {"id": "1", "city": "Los Angeles", "state_or_country": "CA", "zip_code": "10001"},
            {"id": "2", "city": "New York", "state_or_country": "NY", "zip_code": "10001"},
        ])
        issues = checker.run(df, "test.schema.table")
        assert len(issues) == 1
        assert issues[0].record_id == "1"
        assert issues[0].check_type == "zip_state_mismatch"

    def test_detects_mismatch_with_legacy_schema(self, legacy_geonames_df):
        from agents.qc_runner.checks.zip_state_checks import ZipStateMismatchChecker
        idx = GeonamesIndex(legacy_geonames_df)
        checker = ZipStateMismatchChecker(config=None, geonames_index=idx)

        df = pd.DataFrame([
            {"id": "1", "city": "Miami", "state_or_country": "FL", "zip_code": "75001"},
            {"id": "2", "city": "Addison", "state_or_country": "TX", "zip_code": "75001"},
        ])
        issues = checker.run(df, "test.schema.table")
        assert len(issues) == 1
        assert issues[0].record_id == "1"

    def test_no_false_positives(self, raw_geonames_df):
        from agents.qc_runner.checks.zip_state_checks import ZipStateMismatchChecker
        idx = GeonamesIndex(raw_geonames_df)
        checker = ZipStateMismatchChecker(config=None, geonames_index=idx)

        df = pd.DataFrame([
            {"id": "1", "city": "New York", "state_or_country": "NY", "zip_code": "10001"},
            {"id": "2", "city": "Los Angeles", "state_or_country": "CA", "zip_code": "90001"},
        ])
        issues = checker.run(df, "test.schema.table")
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# 3. Full pipeline dry-run with GeoNames raw data
# ---------------------------------------------------------------------------

class TestPipelineWithGeoNames:
    """Smoke test: the full pipeline can accept a GeonamesIndex built from raw data."""

    def test_pipeline_dry_run_with_raw_geonames(self, raw_geonames_df, tmp_path, mock_config):
        mock_config.claude.api_key = ""
        idx = GeonamesIndex(raw_geonames_df)

        # Create a sample CSV with a known zip/state mismatch
        sample = pd.DataFrame([
            {"id": "1", "name": "Good Corp", "city": "New York", "state_or_country": "NY", "zip_code": "10001"},
            {"id": "2", "name": "Bad Corp", "city": "LA", "state_or_country": "CA", "zip_code": "10001"},  # mismatch
            {"id": "3", "name": "Null Corp", "city": None, "state_or_country": "TX", "zip_code": "75001"},  # null
        ])
        csv_path = str(tmp_path / "test_companies.csv")
        sample.to_csv(csv_path, index=False)

        from pipelines.full_qc_pipeline import run_pipeline
        result = run_pipeline(
            table_catalog="local",
            table_schema="test",
            table_name="companies",
            qc_checks=["null", "format", "zip_state_mismatch"],
            dry_run=True,
            local_file=csv_path,
            config=mock_config,
            geonames_index=idx,
        )

        assert result.total_records == 3
        assert result.dry_run is True
        # Should find at least the null issue and the zip/state mismatch
        assert result.total_issues >= 2
        assert result.check_type_breakdown.get("zip_state_mismatch", 0) >= 1
        assert result.check_type_breakdown.get("null", 0) >= 1
