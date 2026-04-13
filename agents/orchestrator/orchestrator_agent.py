"""Orchestrator Agent: coordinates QC workflow and L1/L2 escalation."""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from agents.base_agent import BaseAgent
from agents.orchestrator.escalation_policy import EscalationPolicy
from configs.settings import AppConfig
from llm.prompts.orchestrator_prompts import (
    ORCHESTRATOR_SYSTEM,
    build_summary_prompt,
    build_triage_prompt,
)
from messaging.message_bus import MessageBus
from schemas.messages import AgentMessage
from schemas.qc_results import QCIssue, WorkflowResult

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Master coordinator. Runs the full L1/L2 support workflow:

    1. Create MLflow run
    2. Send QC_RUN_REQUEST → QCRunnerAgent
    3. Receive QC_RUN_COMPLETE, apply EscalationPolicy
    4. Use Claude for borderline cases
    5. Send CORRECTION_REQUEST → DataUpdaterAgent
    6. Receive CORRECTION_COMPLETE
    7. Log metrics to MLflow, return WorkflowResult
    """

    def __init__(
        self,
        agent_id: str,
        config: AppConfig,
        message_bus: MessageBus,
        spark: Any | None = None,
        mlflow_run_id: str | None = None,
    ):
        super().__init__(agent_id, "orchestrator", config, message_bus, spark)
        self.policy = EscalationPolicy(config.thresholds)
        self._mlflow_run_id = mlflow_run_id

    def get_system_prompt(self) -> str:
        return ORCHESTRATOR_SYSTEM

    def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        # The orchestrator typically drives the workflow imperatively via run_workflow().
        # This handler exists for future async/event-driven use.
        logger.warning("OrchestratorAgent.handle_message called — use run_workflow() instead.")
        return None

    def run_workflow(
        self,
        table_catalog: str,
        table_schema: str,
        table_name: str,
        qc_checks: list[str] | None = None,
        target_columns: list[str] | None = None,
        primary_key: str = "id",
        dry_run: bool = False,
        local_file: str | None = None,
        sample_fraction: float | None = None,
    ) -> WorkflowResult:
        """
        Drive the full QC pipeline and return a WorkflowResult.
        """
        batch_id = str(uuid.uuid4())
        run_id = self._mlflow_run_id or str(uuid.uuid4())
        table_fqn = f"{table_catalog}.{table_schema}.{table_name}"
        start_ts = time.time()
        checks = qc_checks or ["null", "format", "duplicate", "geo"]

        self.logger.info("=" * 60)
        self.logger.info("Orchestrator starting QC workflow")
        self.logger.info("  Table:    %s", table_fqn)
        self.logger.info("  Checks:   %s", checks)
        self.logger.info("  Batch ID: %s", batch_id)
        self.logger.info("  Run ID:   %s", run_id)
        self.logger.info("  Dry Run:  %s", dry_run)
        self.logger.info("=" * 60)

        # ── Step 1: Run QC checks ─────────────────────────────────────
        qc_response = self.send(
            recipient="qc_runner",
            message_type="QC_RUN_REQUEST",
            payload={
                "table_catalog": table_catalog,
                "table_schema": table_schema,
                "table_name": table_name,
                "qc_checks": checks,
                "target_columns": target_columns,
                "batch_id": batch_id,
                "run_id": run_id,
                "local_file": local_file,
                "sample_fraction": sample_fraction,
            },
        )

        if qc_response is None or qc_response.header.message_type != "QC_RUN_COMPLETE":
            raise RuntimeError("QC Runner did not return QC_RUN_COMPLETE")

        qc_payload = qc_response.payload
        total_records: int = qc_payload.get("total_records", 0)
        issues = [QCIssue(**i) for i in qc_payload.get("issues", [])]

        self.logger.info("QC complete: %d records, %d issues found.", total_records, len(issues))

        if not issues:
            duration = time.time() - start_ts
            return self._build_clean_result(run_id, batch_id, table_fqn, total_records, duration, dry_run)

        # ── Step 2: Escalation policy ─────────────────────────────────
        issues = self.policy.apply_to_issues(issues)

        # Use LLM for borderline decisions
        borderline = [i for i in issues if self.policy.needs_llm_review(i)]
        if borderline and self.config.claude.api_key:
            issues = self._llm_triage(issues, borderline)

        l1_count = sum(1 for i in issues if i.support_level == "L1")
        l2_count = sum(1 for i in issues if i.support_level == "L2")
        skip_count = sum(1 for i in issues if i.support_level == "skip")

        self.logger.info(
            "Escalation: L1=%d (auto-fix), L2=%d (review), skip=%d", l1_count, l2_count, skip_count
        )

        # Only send actionable issues to the updater
        actionable = [i for i in issues if i.support_level in ("L1", "L2")]

        # ── Step 3: Apply corrections ─────────────────────────────────
        upd_response = self.send(
            recipient="data_updater",
            message_type="CORRECTION_REQUEST",
            payload={
                "issues": [i.model_dump() for i in actionable],
                "batch_id": batch_id,
                "run_id": run_id,
                "table_fqn": table_fqn,
                "primary_key": primary_key,
                "dry_run": dry_run,
            },
        )

        if upd_response is None or upd_response.header.message_type != "CORRECTION_COMPLETE":
            raise RuntimeError("Data Updater did not return CORRECTION_COMPLETE")

        upd_payload = upd_response.payload
        duration = time.time() - start_ts

        # ── Step 4: Generate summary via Claude ───────────────────────
        summary = self._generate_summary(qc_payload, upd_payload)
        self.logger.info("Workflow summary: %s", summary)

        # ── Step 5: Log to MLflow ─────────────────────────────────────
        self._log_mlflow(
            run_id=run_id,
            batch_id=batch_id,
            table_fqn=table_fqn,
            total_records=total_records,
            issues=issues,
            upd_payload=upd_payload,
            duration=duration,
        )

        avg_confidence = (
            sum(i.confidence_score for i in issues) / len(issues) if issues else 0.0
        )
        severity_breakdown: dict[str, int] = {}
        check_type_breakdown: dict[str, int] = {}
        for i in issues:
            severity_breakdown[i.severity] = severity_breakdown.get(i.severity, 0) + 1
            check_type_breakdown[i.check_type] = check_type_breakdown.get(i.check_type, 0) + 1

        return WorkflowResult(
            run_id=run_id,
            batch_id=batch_id,
            table_fqn=table_fqn,
            total_records=total_records,
            total_issues=len(issues),
            l1_corrected=upd_payload.get("records_updated", 0),
            l2_flagged=upd_payload.get("records_flagged_l2", 0),
            skipped=skip_count,
            avg_confidence=avg_confidence,
            duration_seconds=duration,
            dry_run=dry_run,
            severity_breakdown=severity_breakdown,
            check_type_breakdown=check_type_breakdown,
        )

    def _llm_triage(self, all_issues: list[QCIssue], borderline: list[QCIssue]) -> list[QCIssue]:
        """Call Claude to decide L1/L2 for borderline-confidence issues."""
        try:
            prompt = build_triage_prompt(
                [i.model_dump(include={"issue_id","check_type","column_name","original_value","suggested_value","confidence_score"}) for i in borderline],
                borderline_only=True,
            )
            results = json.loads(
                self.llm.complete(self.get_system_prompt(), prompt, force_json=True)
            )
            if isinstance(results, list):
                overrides = {r["issue_id"]: r["support_level"] for r in results}
                for issue in all_issues:
                    if issue.issue_id in overrides:
                        new_level = overrides[issue.issue_id]
                        if new_level in ("L1", "L2", "skip"):
                            issue.support_level = new_level
        except Exception as exc:
            self.logger.warning("LLM triage failed, keeping rule-based decisions: %s", exc)
        return all_issues

    def _generate_summary(self, qc_payload: dict, upd_payload: dict) -> str:
        if not self.config.claude.api_key:
            return ""
        try:
            stats = {
                "total_records": qc_payload.get("total_records"),
                "total_issues": len(qc_payload.get("issues", [])),
                "severity_breakdown": qc_payload.get("severity_breakdown"),
                "check_type_breakdown": qc_payload.get("check_type_breakdown"),
                "auto_corrected": upd_payload.get("records_updated"),
                "flagged_for_review": upd_payload.get("records_flagged_l2"),
            }
            result = json.loads(
                self.llm.complete(self.get_system_prompt(), build_summary_prompt(stats), force_json=True)
            )
            return result.get("summary", "")
        except Exception as exc:
            self.logger.warning("Summary generation failed: %s", exc)
            return ""

    def _log_mlflow(
        self,
        run_id: str,
        batch_id: str,
        table_fqn: str,
        total_records: int,
        issues: list[QCIssue],
        upd_payload: dict,
        duration: float,
    ) -> None:
        try:
            import os
            import time
            import requests as _req
            from configs.settings import get_databricks_token
            host  = os.getenv("DATABRICKS_HOST", "").rstrip("/")
            token = get_databricks_token()
            exp_id = os.getenv("MLFLOW_EXPERIMENT_ID", "")
            if host and token and exp_id:
                # Use REST API directly — mlflow Python client fails in serverless
                hdrs = {"Authorization": f"Bearer {token}"}
                ts_ms = int(time.time() * 1000)
                r = _req.post(f"{host}/api/2.0/mlflow/runs/create",
                              headers=hdrs,
                              json={"experiment_id": exp_id,
                                    "run_name": f"qc-{batch_id[:8]}",
                                    "start_time": ts_ms},
                              timeout=10)
                if not r.ok:
                    self.logger.warning("MLflow run create failed: %s", r.text)
                    return
                mlflow_run_id = r.json()["run"]["info"]["run_id"]

                metrics = [
                    ("total_records_scanned",  total_records),
                    ("total_issues_found",      len(issues)),
                    ("l1_auto_corrected",       upd_payload.get("records_updated", 0)),
                    ("l2_flagged",              upd_payload.get("records_flagged_l2", 0)),
                    ("pipeline_duration_seconds", round(duration, 2)),
                ]
                if issues:
                    avg = sum(i.confidence_score for i in issues) / len(issues)
                    metrics.append(("avg_confidence_score", round(avg, 4)))
                for check_type in ("null", "format", "duplicate", "geo_typo", "geo_invalid", "zip_state_mismatch"):
                    count = sum(1 for i in issues if i.check_type == check_type)
                    metrics.append((f"issues_{check_type}", count))
                llm_stats = self.llm.stats
                metrics.append(("llm_calls", llm_stats["call_count"]))
                metrics.append(("llm_total_tokens",
                                llm_stats["total_input_tokens"] + llm_stats["total_output_tokens"]))

                for key, val in metrics:
                    _req.post(f"{host}/api/2.0/mlflow/runs/log-metric",
                              headers=hdrs,
                              json={"run_id": mlflow_run_id, "key": key,
                                    "value": float(val), "timestamp": ts_ms, "step": 0},
                              timeout=5)
                for tag_key, tag_val in [("source_table", table_fqn), ("batch_id", batch_id)]:
                    _req.post(f"{host}/api/2.0/mlflow/runs/set-tag",
                              headers=hdrs,
                              json={"run_id": mlflow_run_id, "key": tag_key, "value": tag_val},
                              timeout=5)
                _req.post(f"{host}/api/2.0/mlflow/runs/update",
                          headers=hdrs,
                          json={"run_id": mlflow_run_id, "status": "FINISHED",
                                "end_time": int(time.time() * 1000)},
                          timeout=5)
                self.logger.info("MLflow run logged: %s", mlflow_run_id)
            else:
                # Fallback: standard mlflow Python client (local / non-serverless)
                import mlflow
                with mlflow.start_run(run_name=f"qc-{batch_id[:8]}"):
                    mlflow.set_tag("source_table", table_fqn)
                    mlflow.set_tag("batch_id", batch_id)
                    mlflow.log_metric("total_records_scanned", total_records)
                    mlflow.log_metric("total_issues_found", len(issues))
                    mlflow.log_metric("l1_auto_corrected", upd_payload.get("records_updated", 0))
                    mlflow.log_metric("l2_flagged", upd_payload.get("records_flagged_l2", 0))
                    mlflow.log_metric("pipeline_duration_seconds", duration)
                    if issues:
                        avg = sum(i.confidence_score for i in issues) / len(issues)
                        mlflow.log_metric("avg_confidence_score", round(avg, 4))
                    for check_type in ("null", "format", "duplicate", "geo_typo", "geo_invalid", "zip_state_mismatch"):
                        count = sum(1 for i in issues if i.check_type == check_type)
                        mlflow.log_metric(f"issues_{check_type}", count)
                    llm_stats = self.llm.stats
                    mlflow.log_metric("llm_calls", llm_stats["call_count"])
                    mlflow.log_metric("llm_total_tokens",
                                      llm_stats["total_input_tokens"] + llm_stats["total_output_tokens"])
        except Exception as exc:
            self.logger.warning("MLflow logging failed: %s", exc)

    def _build_clean_result(
        self, run_id: str, batch_id: str, table_fqn: str, total_records: int, duration: float, dry_run: bool
    ) -> WorkflowResult:
        return WorkflowResult(
            run_id=run_id,
            batch_id=batch_id,
            table_fqn=table_fqn,
            total_records=total_records,
            total_issues=0,
            l1_corrected=0,
            l2_flagged=0,
            skipped=0,
            avg_confidence=1.0,
            duration_seconds=duration,
            dry_run=dry_run,
        )
