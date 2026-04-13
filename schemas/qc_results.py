"""QC issue and correction result schemas."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SupportLevel = Literal["L1", "L2", "skip"]
Severity = Literal["critical", "warning", "info"]
CheckType = Literal["null", "format", "duplicate", "geo_invalid", "geo_typo", "zip_state_mismatch"]
QCFlag = Literal["AUTO_CORRECTED", "NEEDS_REVIEW", "NO_CHANGE", "SKIPPED"]
QCStatus = Literal["CLEAN", "AUTO_CORRECTED", "NEEDS_REVIEW", "FAILED"]


class QCIssue(BaseModel):
    issue_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    record_id: str
    table_fqn: str
    column_name: str
    check_type: CheckType
    original_value: str | None
    suggested_value: str | None
    confidence_score: float = 0.0
    support_level: SupportLevel = "L2"
    severity: Severity = "warning"
    issue_metadata: dict[str, Any] = Field(default_factory=dict)


class CorrectionRecord(BaseModel):
    issue_id: str
    record_id: str
    column_name: str
    original_value: str | None
    corrected_value: str | None
    confidence_score: float
    correction_method: Literal["fuzzy_match", "geocode", "rule_based", "llm", "none"]
    was_applied: bool
    qc_flag: QCFlag
    audited_at: datetime = Field(default_factory=datetime.utcnow)
    audited_by: str = "data_updater_agent_v1"
    llm_reasoning: str | None = None


class CorrectionResult(BaseModel):
    """Single-field correction result from AddressCorrector."""
    original_value: str | None
    corrected_value: str | None
    confidence: float
    method: str
    candidates: list[tuple[str, float, int]] = Field(default_factory=list)


class WorkflowResult(BaseModel):
    run_id: str
    batch_id: str
    table_fqn: str
    total_records: int
    total_issues: int
    l1_corrected: int
    l2_flagged: int
    skipped: int
    avg_confidence: float
    duration_seconds: float
    dry_run: bool
    severity_breakdown: dict[str, int] = Field(default_factory=dict)
    check_type_breakdown: dict[str, int] = Field(default_factory=dict)
