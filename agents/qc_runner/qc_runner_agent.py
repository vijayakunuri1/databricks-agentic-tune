"""QC Runner Agent: executes all data quality checks."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from agents.base_agent import BaseAgent
from agents.qc_runner.checks.null_checks import NullCheckRunner
from agents.qc_runner.checks.format_checks import FormatCheckRunner
from agents.qc_runner.checks.duplicate_checks import DuplicateCheckRunner
from agents.qc_runner.checks.geo_checks import GeoCheckRunner
from agents.qc_runner.checks.zip_state_checks import ZipStateMismatchChecker
from configs.settings import AppConfig
from llm.prompts.qc_runner_prompts import QC_RUNNER_SYSTEM, build_classify_prompt
from messaging.message_bus import MessageBus
from schemas.messages import AgentMessage
from schemas.qc_results import QCIssue

logger = logging.getLogger(__name__)


class QCRunnerAgent(BaseAgent):
    """
    Executes QC checks against a DataFrame / Delta table.
    Uses the LLM to classify ambiguous issues and assign severity.
    """

    def __init__(
        self,
        agent_id: str,
        config: AppConfig,
        message_bus: MessageBus,
        spark: Any | None = None,
        geonames_index: Any | None = None,
    ):
        super().__init__(agent_id, "qc_runner", config, message_bus, spark)
        self.checks: dict[str, Any] = {
            "null": NullCheckRunner(config),
            "format": FormatCheckRunner(config),
            "duplicate": DuplicateCheckRunner(config),
            "geo": GeoCheckRunner(config),
            "zip_state_mismatch": ZipStateMismatchChecker(config, geonames_index),
        }

    def get_system_prompt(self) -> str:
        return QC_RUNNER_SYSTEM

    def handle_message(self, message: AgentMessage) -> AgentMessage:
        if message.header.message_type != "QC_RUN_REQUEST":
            raise ValueError(f"QCRunnerAgent cannot handle: {message.header.message_type}")

        req = message.payload

        # --- Validate all user-controlled fields before touching any data ---
        from configs.input_validation import (
            build_safe_fqn, validate_qc_checks, validate_batch_id,
            validate_column_name, validate_sample_fraction,
        )
        try:
            catalog = req["table_catalog"]
            schema  = req["table_schema"]
            table   = req["table_name"]
            table_fqn = build_safe_fqn(catalog, schema, table)

            raw_checks = req.get("qc_checks", list(self.checks.keys()))
            checks_to_run = validate_qc_checks(raw_checks)

            raw_pk = req.get("primary_key", "id")
            if raw_pk:
                validate_column_name(raw_pk, "primary_key")

            raw_batch = req.get("batch_id", "")
            if raw_batch:
                validate_batch_id(raw_batch, "batch_id")

            raw_run = req.get("run_id", "")
            if raw_run:
                validate_batch_id(raw_run, "run_id")

            raw_fraction = req.get("sample_fraction")
            validate_sample_fraction(raw_fraction, "sample_fraction")

            raw_cols: list[str] | None = req.get("target_columns")
            if raw_cols is not None:
                for col in raw_cols:
                    validate_column_name(col, "target_columns entry")

        except (ValueError, KeyError) as exc:
            raise ValueError(f"QC_RUN_REQUEST payload validation failed: {exc}") from exc

        start = datetime.utcnow()
        self.logger.info("Starting QC run for %s", table_fqn)

        df = self._load_data(req)
        target_columns: list[str] | None = req.get("target_columns")

        issues = self._run_all_checks(df, table_fqn, checks_to_run, target_columns)
        issues = self._enrich_with_llm(issues)

        duration = (datetime.utcnow() - start).total_seconds()
        self.logger.info(
            "QC run complete: %d issues found in %.1fs", len(issues), duration
        )

        severity_breakdown = self._count_by(issues, "severity")
        check_type_breakdown = self._count_by(issues, "check_type")

        return message.reply(
            message_type="QC_RUN_COMPLETE",
            payload={
                "batch_id": req.get("batch_id", ""),
                "run_id": req.get("run_id", ""),
                "table_fqn": table_fqn,
                "total_records": self._count_records(df),
                "issues": [i.model_dump() for i in issues],
                "severity_breakdown": severity_breakdown,
                "check_type_breakdown": check_type_breakdown,
                "duration_seconds": duration,
            },
        )

    # Directories that local_file paths must be rooted under.
    # Override by setting QC_LOCAL_DATA_DIR environment variable.
    _LOCAL_DATA_ROOTS: tuple[str, ...] = ("/tmp", "/dbfs/tmp")

    def _load_data(self, req: dict) -> Any:

        catalog = req["table_catalog"]
        schema = req["table_schema"]
        table = req["table_name"]
        local_file = req.get("local_file")

        if local_file:
            from configs.input_validation import validate_local_file
            resolved = validate_local_file(local_file, label="local_file")
            import pandas as pd
            if resolved.endswith(".csv"):
                return pd.read_csv(resolved)
            return pd.read_parquet(resolved)

        if self.spark:
            df = self.spark.table(f"{catalog}.{schema}.{table}")
            sample = req.get("sample_fraction")
            if sample:
                df = df.sample(fraction=float(sample))
            return df

        raise RuntimeError("No Spark session and no local_file provided.")

    def _run_all_checks(
        self,
        df: Any,
        table_fqn: str,
        checks_to_run: list[str],
        target_columns: list[str] | None,
    ) -> list[QCIssue]:
        all_issues: list[QCIssue] = []
        for check_name in checks_to_run:
            runner = self.checks.get(check_name)
            if runner is None:
                self.logger.warning("Unknown check: %s, skipping.", check_name)
                continue
            try:
                found = runner.run(df, table_fqn, target_columns)
                self.logger.info("Check '%s': %d issues", check_name, len(found))
                all_issues.extend(found)
            except Exception as exc:
                self.logger.error("Check '%s' failed: %s", check_name, exc, exc_info=True)
        return all_issues

    def _enrich_with_llm(self, issues: list[QCIssue]) -> list[QCIssue]:
        """Use the LLM to verify ambiguous geo issues and assign severity."""
        if not issues:
            return issues

        # Only send geo and format issues to LLM (nulls and dupes are deterministic)
        ambiguous = [i for i in issues if i.check_type in ("geo_typo", "geo_invalid", "format")]
        if not ambiguous:
            return issues

        try:
            prompt = build_classify_prompt([i.model_dump(include={"issue_id", "check_type", "column_name", "original_value", "suggested_value"}) for i in ambiguous])
            results = json.loads(self.llm.complete(self.get_system_prompt(), prompt, force_json=True))

            # Build lookup by issue_id
            updates = {r["issue_id"]: r for r in results if isinstance(results, list)}

            for issue in issues:
                upd = updates.get(issue.issue_id)
                if upd:
                    if not upd.get("is_genuine_error", True):
                        issues.remove(issue)
                        continue
                    issue.severity = upd.get("severity", issue.severity)
        except Exception as exc:
            self.logger.warning("LLM enrichment failed, using raw issues: %s", exc)

        return issues

    @staticmethod
    def _count_records(df: Any) -> int:
        try:
            # Spark DataFrame: count() returns an int
            # pandas DataFrame: count() returns a Series — use len() instead
            result = df.count()
            if isinstance(result, int):
                return result
            return len(df)
        except Exception:
            return len(df)

    @staticmethod
    def _count_by(issues: list[QCIssue], field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in issues:
            val = getattr(issue, field, "unknown")
            counts[val] = counts.get(val, 0) + 1
        return counts
