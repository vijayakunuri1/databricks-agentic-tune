"""Integration test: full pipeline run against a local CSV file.

Mocks out the Anthropic API and Delta writes so no external services are needed.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def customers_csv(sample_csv_path):
    return sample_csv_path


def test_full_pipeline_dry_run(customers_csv, mock_config):
    """Pipeline should detect issues and return a valid WorkflowResult without writing."""
    mock_config.claude.api_key = ""   # disable LLM calls

    from pipelines.full_qc_pipeline import run_pipeline

    result = run_pipeline(
        table_catalog="local",
        table_schema="test",
        table_name="customers",
        qc_checks=["null", "format"],   # skip geo to avoid geocoder calls
        dry_run=True,
        local_file=customers_csv,
        config=mock_config,
    )

    assert result.total_records == 10
    assert result.total_issues > 0       # fixture has known issues
    assert result.dry_run is True
    assert result.l1_corrected == 0      # dry_run = no writes


def test_pipeline_detects_null(customers_csv, mock_config):
    mock_config.claude.api_key = ""

    from pipelines.full_qc_pipeline import run_pipeline

    result = run_pipeline(
        table_catalog="local",
        table_schema="test",
        table_name="customers",
        qc_checks=["null"],
        dry_run=True,
        local_file=customers_csv,
        config=mock_config,
    )

    # Row 6 has name=None
    assert result.total_issues >= 1
    assert result.check_type_breakdown.get("null", 0) >= 1


def test_pipeline_detects_format_issues(customers_csv, mock_config):
    mock_config.claude.api_key = ""

    from pipelines.full_qc_pipeline import run_pipeline

    result = run_pipeline(
        table_catalog="local",
        table_schema="test",
        table_name="customers",
        qc_checks=["format"],
        dry_run=True,
        local_file=customers_csv,
        config=mock_config,
    )

    # Row 5 has invalid email, row 4 has invalid zip
    assert result.check_type_breakdown.get("format", 0) >= 2
