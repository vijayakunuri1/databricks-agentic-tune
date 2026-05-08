"""Data Updater Agent: applies fuzzy corrections and writes audit flags."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from agents.base_agent import BaseAgent
from agents.data_updater.address_corrector import AddressCorrector
from agents.data_updater.delta_writer import DeltaWriter
from configs.settings import AppConfig
from llm.prompts.updater_prompts import UPDATER_SYSTEM, build_rationale_prompt
from messaging.message_bus import MessageBus
from schemas.messages import AgentMessage
from schemas.qc_results import CorrectionRecord, QCIssue, QCFlag

logger = logging.getLogger(__name__)


class DataUpdaterAgent(BaseAgent):
    """
    Receives issues from the Orchestrator, applies corrections (L1)
    or flags for human review (L2), then writes back to Delta.
    """

    def __init__(
        self,
        agent_id: str,
        config: AppConfig,
        message_bus: MessageBus,
        spark: Any | None = None,
        city_reference: list[str] | None = None,
        geonames_index: Any | None = None,
    ):
        super().__init__(agent_id, "data_updater", config, message_bus, spark)
        self.corrector = AddressCorrector(
            city_reference=city_reference or self._load_city_reference(),
            scorer_name=config.thresholds.address_scorer,
            geonames_index=geonames_index,
        )
        self.delta_writer: DeltaWriter | None = DeltaWriter(spark) if spark else None

    def get_system_prompt(self) -> str:
        return UPDATER_SYSTEM

    def handle_message(self, message: AgentMessage) -> AgentMessage:
        if message.header.message_type != "CORRECTION_REQUEST":
            raise ValueError(f"DataUpdaterAgent cannot handle: {message.header.message_type}")

        req = message.payload
        start = datetime.utcnow()

        # --- Validate all fields that will reach Delta SQL or be stored ---
        from configs.input_validation import (
            validate_fqn, validate_column_name, validate_batch_id,
        )
        try:
            dry_run: bool = bool(req.get("dry_run", False))
            batch_id: str = validate_batch_id(req.get("batch_id", ""), "batch_id") if req.get("batch_id") else ""
            run_id: str   = validate_batch_id(req.get("run_id", ""), "run_id") if req.get("run_id") else ""
            raw_fqn = req.get("table_fqn", "")
            table_fqn: str = validate_fqn(raw_fqn, "table_fqn") if raw_fqn else ""
            primary_key: str = validate_column_name(req.get("primary_key", "id"), "primary_key")
        except ValueError as exc:
            raise ValueError(f"CORRECTION_REQUEST payload validation failed: {exc}") from exc

        issues = [QCIssue(**i) for i in req.get("issues", [])]
        l1_issues = [i for i in issues if i.support_level == "L1"]
        l2_issues = [i for i in issues if i.support_level == "L2"]

        self.logger.info(
            "Processing %d L1 (auto-fix) and %d L2 (flag) issues.", len(l1_issues), len(l2_issues)
        )

        l1_corrections = self._apply_l1_corrections(l1_issues)
        l2_corrections = self._flag_l2_records(l2_issues)
        all_corrections = l1_corrections + l2_corrections

        # Generate LLM rationale for corrections
        all_corrections = self._enrich_rationale(all_corrections)

        # Write to Delta
        if self.delta_writer and table_fqn and not dry_run:
            try:
                self.delta_writer.merge_corrections(
                    table_fqn, all_corrections, primary_key, batch_id, run_id, dry_run=False
                )
                self.delta_writer.write_audit_log(
                    self.config.tables.audit_log_fqn, all_corrections, batch_id, run_id, table_fqn
                )
            except Exception as exc:
                self.logger.error("Delta write failed: %s", exc, exc_info=True)
                raise
        elif dry_run:
            self.logger.info("[dry_run] Skipping Delta write.")

        duration = (datetime.utcnow() - start).total_seconds()
        return message.reply(
            message_type="CORRECTION_COMPLETE",
            payload={
                "batch_id": batch_id,
                "run_id": run_id,
                "records_updated": 0 if dry_run else len(l1_corrections),
                "records_flagged_l2": len(l2_corrections),
                "records_skipped": sum(1 for i in issues if i.support_level == "skip"),
                "corrections": [c.model_dump() for c in all_corrections],
                "dry_run": dry_run,
                "duration_seconds": duration,
            },
        )

    def _apply_l1_corrections(self, issues: list[QCIssue]) -> list[CorrectionRecord]:
        corrections = []
        for issue in issues:
            result = self._correct_issue(issue)
            if result is None:
                continue

            corrected_val = result.corrected_value
            applied = corrected_val is not None and result.confidence >= self.config.thresholds.l1_auto_correct
            flag: QCFlag = "AUTO_CORRECTED" if applied else "NEEDS_REVIEW"

            corrections.append(
                CorrectionRecord(
                    issue_id=issue.issue_id,
                    record_id=issue.record_id,
                    column_name=issue.column_name,
                    original_value=issue.original_value,
                    corrected_value=corrected_val if applied else None,
                    confidence_score=result.confidence,
                    correction_method=result.method,
                    was_applied=applied,
                    qc_flag=flag,
                )
            )
        return corrections

    def _flag_l2_records(self, issues: list[QCIssue]) -> list[CorrectionRecord]:
        corrections = []
        for issue in issues:
            result = self._correct_issue(issue)
            suggested = result.corrected_value if result else None
            corrections.append(
                CorrectionRecord(
                    issue_id=issue.issue_id,
                    record_id=issue.record_id,
                    column_name=issue.column_name,
                    original_value=issue.original_value,
                    corrected_value=suggested,   # suggestion only, not applied
                    confidence_score=result.confidence if result else 0.0,
                    correction_method=result.method if result else "none",
                    was_applied=False,
                    qc_flag="NEEDS_REVIEW",
                )
            )
        return corrections

    def _correct_issue(self, issue: QCIssue) -> Any | None:
        col = issue.column_name.lower()
        val = issue.original_value or ""

        if issue.check_type in ("geo_typo", "geo_invalid"):
            if "city" in col:
                return self.corrector.correct_city(val)
            if "state" in col:
                return self.corrector.correct_state(val)
            if "zip" in col or "postal" in col:
                return self.corrector.correct_zip(val)
            return self.corrector.correct_street(val)

        if issue.check_type == "format":
            meta = issue.issue_metadata or {}
            fmt = meta.get("format", "")
            if fmt == "us_state":
                return self.corrector.correct_state(val)
            if fmt == "zip_us":
                return self.corrector.correct_zip(val)
        return None

    def _enrich_rationale(self, corrections: list[CorrectionRecord]) -> list[CorrectionRecord]:
        to_explain = [
            c for c in corrections
            if c.was_applied and c.corrected_value and c.corrected_value != c.original_value
        ][:20]   # limit to 20 per LLM call

        if not to_explain:
            return corrections

        try:
            prompt = build_rationale_prompt([
                {
                    "issue_id": c.issue_id,
                    "column": c.column_name,
                    "original": c.original_value,
                    "corrected": c.corrected_value,
                    "method": c.correction_method,
                    "confidence": round(c.confidence_score, 2),
                }
                for c in to_explain
            ])
            results = json.loads(
                self.llm.complete(self.get_system_prompt(), prompt, force_json=True)
            )
            lookup = {r["issue_id"]: r["rationale"] for r in results if isinstance(results, list)}
            for c in corrections:
                if c.issue_id in lookup:
                    c.llm_reasoning = lookup[c.issue_id]
        except Exception as exc:
            self.logger.warning("Rationale generation failed: %s", exc)

        return corrections

    def _load_city_reference(self) -> list[str]:
        """Load a small built-in city list as fallback reference data."""
        return [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
            "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
            "Indianapolis", "San Francisco", "Seattle", "Denver", "Nashville",
            "Oklahoma City", "El Paso", "Washington", "Las Vegas", "Louisville",
            "Memphis", "Portland", "Baltimore", "Milwaukee", "Albuquerque",
            "Tucson", "Fresno", "Mesa", "Sacramento", "Atlanta",
            "Kansas City", "Omaha", "Colorado Springs", "Raleigh", "Miami",
            "Long Beach", "Virginia Beach", "Minneapolis", "Tampa", "New Orleans",
            "Arlington", "Bakersfield", "Honolulu", "Anaheim", "Aurora",
            "Santa Ana", "Corpus Christi", "Riverside", "St. Louis", "Pittsburgh",
            "Lexington", "Anchorage", "Stockton", "Cincinnati", "St. Paul",
            "Greensboro", "Toledo", "Newark", "Plano", "Henderson",
            "Lincoln", "Buffalo", "Fort Wayne", "Jersey City", "Chula Vista",
            "Orlando", "St. Petersburg", "Norfolk", "Chandler", "Laredo",
            "Madison", "Durham", "Lubbock", "Winston-Salem", "Garland",
            "Glendale", "Hialeah", "Reno", "Baton Rouge", "Irvine",
            "Chesapeake", "Irving", "Scottsdale", "North Las Vegas", "Fremont",
            "Gilbert", "San Bernardino", "Birmingham", "Boise", "Rochester",
        ]
