"""Full QC pipeline entry point: wires all three agents together."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

from configs.settings import AppConfig, get_config
from messaging.message_bus import MessageBus
from schemas.qc_results import WorkflowResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("pipeline")


def run_pipeline(
    table_catalog: str,
    table_schema: str,
    table_name: str,
    qc_checks: list[str] | None = None,
    target_columns: list[str] | None = None,
    primary_key: str = "id",
    mlflow_experiment_name: str | None = None,
    dry_run: bool = False,
    local_file: str | None = None,
    sample_fraction: float | None = None,
    config: AppConfig | None = None,
    spark: Any | None = None,
    city_reference: list[str] | None = None,
    geonames_index: Any | None = None,
) -> WorkflowResult:
    """
    Wire up all three agents and run the full QC workflow.

    Parameters
    ----------
    table_catalog, table_schema, table_name
        Fully-qualified table coordinates.
    qc_checks
        Subset of ["null", "format", "duplicate", "geo"]. Default: all four.
    primary_key
        Column name used as the record ID for MERGE operations.
    dry_run
        If True, runs all checks and corrections but does NOT write to Delta.
    local_file
        Path to a local CSV/Parquet file for testing without Databricks.
    city_reference
        Custom city reference list for the fuzzy matcher.
    """
    cfg = config or get_config()

    # Setup MLflow experiment
    if mlflow_experiment_name or cfg.mlflow_experiment_name:
        try:
            import mlflow
            exp_name = mlflow_experiment_name or cfg.mlflow_experiment_name

            if cfg.is_databricks:
                # In Databricks serverless (client: "2"), set_experiment() and
                # set_tracking_uri("databricks") both raise CONFIG_NOT_AVAILABLE.
                # Use the REST API to create/get the experiment, then set
                # MLFLOW_EXPERIMENT_ID so mlflow.start_run() picks it up automatically.
                try:
                    import requests as _req
                    from configs.settings import get_databricks_token
                    host  = os.getenv("DATABRICKS_HOST", "").rstrip("/")
                    token = get_databricks_token()
                    if not (host and token):
                        raise ValueError("DATABRICKS_HOST/TOKEN not set")
                    hdrs = {"Authorization": f"Bearer {token}"}
                    # Resolve /Shared/ → /Users/{user}/
                    if exp_name.startswith("/Shared/"):
                        short = exp_name.split("/")[-1]
                        me = _req.get(f"{host}/api/2.0/preview/scim/v2/Me",
                                      headers=hdrs, timeout=5)
                        user_name = me.json().get("userName", "") if me.ok else ""
                        if user_name:
                            exp_name = f"/Users/{user_name}/{short}"
                    # Create or fetch experiment
                    r = _req.post(f"{host}/api/2.0/mlflow/experiments/create",
                                  headers=hdrs, json={"name": exp_name}, timeout=10)
                    if r.status_code in (200, 201):
                        exp_id = r.json()["experiment_id"]
                    elif "RESOURCE_ALREADY_EXISTS" in r.text:
                        r2 = _req.get(f"{host}/api/2.0/mlflow/experiments/get-by-name",
                                      headers=hdrs,
                                      params={"experiment_name": exp_name}, timeout=10)
                        exp_id = r2.json()["experiment"]["experiment_id"]
                    else:
                        exp_id = None
                        logger.warning("MLflow experiment create failed: %s", r.text)
                    if exp_id:
                        os.environ["MLFLOW_EXPERIMENT_ID"] = exp_id
                        logger.info("MLflow experiment: %s (id=%s)", exp_name, exp_id)
                except Exception as _e:
                    logger.warning("MLflow REST setup failed: %s", _e)
            else:
                mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
                mlflow.set_experiment(exp_name)
                logger.info("MLflow experiment: %s", exp_name)
        except Exception as exc:
            logger.warning("MLflow setup failed: %s", exc)

    # Build Spark session if not provided
    if spark is None and not local_file:
        spark = _get_spark(cfg)

    # Message bus
    bus = MessageBus(mode="in_process")

    # Lazy imports keep startup fast when not all packages are installed
    from agents.qc_runner.qc_runner_agent import QCRunnerAgent
    from agents.data_updater.data_updater_agent import DataUpdaterAgent
    from agents.orchestrator.orchestrator_agent import OrchestratorAgent

    qc_runner = QCRunnerAgent(
        agent_id="qc_runner_1",
        config=cfg,
        message_bus=bus,
        spark=spark,
        geonames_index=geonames_index,
    )

    data_updater = DataUpdaterAgent(
        agent_id="data_updater_1",
        config=cfg,
        message_bus=bus,
        spark=spark,
        city_reference=city_reference,
        geonames_index=geonames_index,
    )

    orchestrator = OrchestratorAgent(
        agent_id="orchestrator_1",
        config=cfg,
        message_bus=bus,
        spark=spark,
    )

    result = orchestrator.run_workflow(
        table_catalog=table_catalog,
        table_schema=table_schema,
        table_name=table_name,
        qc_checks=qc_checks,
        target_columns=target_columns,
        primary_key=primary_key,
        dry_run=dry_run,
        local_file=local_file,
        sample_fraction=sample_fraction,
    )

    _print_result(result)
    return result


def _get_spark(cfg: AppConfig) -> Any:
    try:
        from pyspark.sql import SparkSession
        if cfg.is_databricks:
            return SparkSession.getActiveSession()
        builder = (
            SparkSession.builder.master("local[*]")
            .appName("databricks-agentic-qc-local")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )
        return builder.getOrCreate()
    except ImportError:
        logger.warning("PySpark not available. Use local_file for local testing.")
        return None


def _print_result(result: WorkflowResult) -> None:
    logger.info("")
    logger.info("━━━━━━━━━━━━ QC PIPELINE RESULT ━━━━━━━━━━━━")
    logger.info("  Table:           %s", result.table_fqn)
    logger.info("  Run ID:          %s", result.run_id)
    logger.info("  Total records:   %d", result.total_records)
    logger.info("  Total issues:    %d", result.total_issues)
    logger.info("  L1 auto-fixed:   %d", result.l1_corrected)
    logger.info("  L2 flagged:      %d", result.l2_flagged)
    logger.info("  Skipped:         %d", result.skipped)
    logger.info("  Avg confidence:  %.2f", result.avg_confidence)
    logger.info("  Duration:        %.1fs", result.duration_seconds)
    logger.info("  Dry run:         %s", result.dry_run)
    if result.check_type_breakdown:
        logger.info("  Issues by type:  %s", result.check_type_breakdown)
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def main() -> None:
    parser = argparse.ArgumentParser(description="Databricks Agentic QC Pipeline")
    parser.add_argument("--table-catalog", default="local")
    parser.add_argument("--table-schema", default="default")
    parser.add_argument("--table-name", default="customers")
    parser.add_argument("--qc-checks", default="null,format,duplicate,geo")
    parser.add_argument("--primary-key", default="id")
    parser.add_argument("--mlflow-experiment-name", default="/Shared/QC-Pipeline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-file", default=None)
    parser.add_argument("--sample-fraction", type=float, default=None)

    args = parser.parse_args()

    from configs.input_validation import (
        build_safe_fqn, validate_qc_checks, validate_column_name,
        validate_sample_fraction, validate_local_file,
    )
    try:
        build_safe_fqn(args.table_catalog, args.table_schema, args.table_name)
        checks = validate_qc_checks(args.qc_checks)
        validate_column_name(args.primary_key, "primary_key")
        validate_sample_fraction(args.sample_fraction, "sample_fraction")
        if args.local_file:
            args.local_file = validate_local_file(args.local_file, label="local_file")
    except ValueError as exc:
        logger.error("Invalid argument: %s", exc)
        sys.exit(2)

    result = run_pipeline(
        table_catalog=args.table_catalog,
        table_schema=args.table_schema,
        table_name=args.table_name,
        qc_checks=checks,
        primary_key=args.primary_key,
        mlflow_experiment_name=args.mlflow_experiment_name,
        dry_run=args.dry_run,
        local_file=args.local_file,
        sample_fraction=args.sample_fraction,
    )
    sys.exit(0 if result.total_issues == 0 or result.l2_flagged == 0 else 1)


if __name__ == "__main__":
    main()
