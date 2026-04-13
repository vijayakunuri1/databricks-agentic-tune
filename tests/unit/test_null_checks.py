"""Unit tests for NullCheckRunner."""
import pandas as pd
import pytest
from agents.qc_runner.checks.null_checks import NullCheckRunner
from configs.settings import AppConfig


@pytest.fixture
def runner():
    return NullCheckRunner(AppConfig())


def test_detects_null_in_city(runner):
    df = pd.DataFrame([
        {"id": "1", "city": None, "state": "CA"},
        {"id": "2", "city": "LA", "state": "CA"},
    ])
    issues = runner.run(df, "cat.sch.tbl")
    assert len(issues) == 1
    assert issues[0].column_name == "city"
    assert issues[0].record_id == "1"
    assert issues[0].check_type == "null"


def test_critical_column_severity(runner):
    df = pd.DataFrame([{"id": None, "city": "NY"}])
    issues = runner.run(df, "cat.sch.tbl")
    assert any(i.column_name == "id" and i.severity == "critical" for i in issues)


def test_non_critical_column_warning(runner):
    # 'name' is not in the critical columns list → should be warning
    df = pd.DataFrame([{"id": "1", "name": None, "customer_id": "c1"}])
    issues = runner.run(df, "cat.sch.tbl")
    name_issues = [i for i in issues if i.column_name == "name"]
    assert name_issues[0].severity == "warning"


def test_no_nulls_returns_empty(runner):
    df = pd.DataFrame([{"id": "1", "city": "NY"}])
    issues = runner.run(df, "cat.sch.tbl")
    assert issues == []
