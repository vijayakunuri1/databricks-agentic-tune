"""L1/L2 escalation routing logic."""
from __future__ import annotations

import logging
from typing import Literal

from configs.settings import ThresholdConfig
from schemas.qc_results import QCIssue, SupportLevel

logger = logging.getLogger(__name__)

# These columns are business-critical → always L2 regardless of confidence
CRITICAL_COLUMNS = frozenset({
    "customer_id", "id", "record_id", "pk",
    "ssn", "tax_id", "account_number",
})

# Duplicate issues are always L2 (never auto-delete)
# Null issues are always L2 — there is no suggested value to auto-correct with
ALWAYS_L2_CHECK_TYPES = frozenset({"duplicate", "null"})


class EscalationPolicy:
    """
    Rules-based first pass.

    Decision tree:
    1. If check_type in ALWAYS_L2 → L2
    2. If column is business-critical → L2
    3. If confidence >= L1_THRESHOLD → L1
    4. If L2_LOWER <= confidence < L1_THRESHOLD → L2
    5. Otherwise → skip
    """

    def __init__(self, thresholds: ThresholdConfig):
        self.t = thresholds

    def classify(self, issue: QCIssue) -> SupportLevel:
        # Rule 1: duplicate → always L2
        if issue.check_type in ALWAYS_L2_CHECK_TYPES:
            return "L2"

        # Rule 2: critical column → always L2
        col_lower = issue.column_name.lower()
        if any(col_lower == c or col_lower.startswith(c + "_") for c in CRITICAL_COLUMNS):
            return "L2"

        # Rule 3-5: confidence-based routing
        c = issue.confidence_score
        if c >= self.t.l1_auto_correct:
            return "L1"
        if c >= self.t.l2_review_lower:
            return "L2"
        return "skip"

    def needs_llm_review(self, issue: QCIssue) -> bool:
        """True for borderline confidence (0.75-0.85) — LLM can tip the scale."""
        return self.t.llm_borderline_lower <= issue.confidence_score < self.t.l1_auto_correct

    def apply_to_issues(self, issues: list[QCIssue]) -> list[QCIssue]:
        """Set support_level on each issue and return the list."""
        for issue in issues:
            issue.support_level = self.classify(issue)
        return issues
