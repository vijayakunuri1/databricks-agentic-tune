"""Unit tests for FormatCheckRunner."""
import pandas as pd
import pytest
from agents.qc_runner.checks.format_checks import FormatCheckRunner
from configs.settings import AppConfig


@pytest.fixture
def runner():
    return FormatCheckRunner(AppConfig())


def test_detects_invalid_email(runner):
    df = pd.DataFrame([{"id": "1", "email": "not-an-email"}])
    issues = runner.run(df, "t")
    assert len(issues) == 1
    assert issues[0].check_type == "format"
    assert issues[0].original_value == "not-an-email"


def test_valid_email_no_issue(runner):
    df = pd.DataFrame([{"id": "1", "email": "valid@example.com"}])
    issues = runner.run(df, "t")
    assert issues == []


def test_detects_invalid_zip(runner):
    df = pd.DataFrame([{"id": "1", "zip_code": "ABCDE"}])
    issues = runner.run(df, "t")
    assert any(i.column_name == "zip_code" for i in issues)


def test_valid_zip_9digit(runner):
    df = pd.DataFrame([{"id": "1", "zip_code": "10001-1234"}])
    issues = runner.run(df, "t")
    assert issues == []


def test_detects_invalid_state(runner):
    df = pd.DataFrame([{"id": "1", "state": "ZZ"}])
    issues = runner.run(df, "t")
    assert any(i.column_name == "state" for i in issues)


def test_valid_state_abbr(runner):
    df = pd.DataFrame([{"id": "1", "state": "CA"}])
    issues = runner.run(df, "t")
    assert issues == []
