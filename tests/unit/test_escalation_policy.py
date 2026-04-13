"""Unit tests for EscalationPolicy."""
import pytest
from agents.orchestrator.escalation_policy import EscalationPolicy
from configs.settings import ThresholdConfig
from schemas.qc_results import QCIssue


def make_issue(**kwargs) -> QCIssue:
    defaults = dict(
        record_id="r1",
        table_fqn="cat.sch.tbl",
        column_name="city",
        check_type="geo_typo",
        original_value="Los Angles",
        suggested_value="Los Angeles",
        confidence_score=0.90,
    )
    defaults.update(kwargs)
    return QCIssue(**defaults)


@pytest.fixture
def policy():
    return EscalationPolicy(ThresholdConfig())


def test_l1_high_confidence(policy):
    issue = make_issue(confidence_score=0.90)
    assert policy.classify(issue) == "L1"


def test_l2_medium_confidence(policy):
    issue = make_issue(confidence_score=0.70)
    assert policy.classify(issue) == "L2"


def test_skip_low_confidence(policy):
    issue = make_issue(confidence_score=0.30)
    assert policy.classify(issue) == "skip"


def test_duplicate_always_l2(policy):
    issue = make_issue(check_type="duplicate", confidence_score=0.99)
    assert policy.classify(issue) == "L2"


def test_critical_column_always_l2(policy):
    issue = make_issue(column_name="customer_id", confidence_score=0.99)
    assert policy.classify(issue) == "L2"


def test_borderline_needs_llm(policy):
    issue = make_issue(confidence_score=0.78)
    assert policy.needs_llm_review(issue) is True


def test_high_confidence_no_llm(policy):
    issue = make_issue(confidence_score=0.95)
    assert policy.needs_llm_review(issue) is False


def test_apply_to_issues(policy):
    issues = [
        make_issue(issue_id="a", confidence_score=0.90),
        make_issue(issue_id="b", confidence_score=0.60),
        make_issue(issue_id="c", confidence_score=0.20),
    ]
    result = policy.apply_to_issues(issues)
    levels = {i.issue_id: i.support_level for i in result}
    assert levels["a"] == "L1"
    assert levels["b"] == "L2"
    assert levels["c"] == "skip"
