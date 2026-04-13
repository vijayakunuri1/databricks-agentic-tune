# Databricks notebook source
# MAGIC %md
# MAGIC ## 05 · SEC EDGAR Address QC Pipeline
# MAGIC
# MAGIC Runs the 3-agent QC pipeline on public SEC EDGAR company addresses using
# MAGIC GeoNames as the authoritative address reference.
# MAGIC
# MAGIC **QC checks performed:**
# MAGIC | Check | What it finds |
# MAGIC |-------|--------------|
# MAGIC | `null` | Companies with no address, city, state, or ZIP |
# MAGIC | `format` | Invalid ZIP format, bad state codes |
# MAGIC | `zip_state_mismatch` | ZIP code belongs to a different state (e.g. `state=CA`, `zip=75001` TX) |
# MAGIC | `geo` | Addresses that can't be geocoded (optional — slow) |
# MAGIC
# MAGIC **Output:** Companies table gets `qc_*` audit columns merged back.
# MAGIC **MLflow:** All metrics logged to the experiment.

# COMMAND ----------
import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + os.path.dirname(os.path.dirname(_nb_path))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Inject the workspace host so DatabricksLLMClient can locate the serving endpoint.
# The SDK auto-discovers the bearer token from the cluster context — never store it
# in an env var or print it.
os.environ.setdefault("DATABRICKS_HOST", "https://" + spark.conf.get("spark.databricks.workspaceUrl"))

# COMMAND ----------
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("edgar_schema", "sec_edgar", "SEC EDGAR Schema")
dbutils.widgets.text("reference_schema", "reference", "Reference Schema")
dbutils.widgets.text("qc_schema", "qc_schema", "QC Schema")
dbutils.widgets.dropdown("dry_run", "true", ["true", "false"], "Dry Run")
dbutils.widgets.text("mlflow_experiment", "/Shared/SEC-EDGAR-QC", "MLflow Experiment")

catalog       = dbutils.widgets.get("catalog")
edgar_schema  = dbutils.widgets.get("edgar_schema")
ref_schema    = dbutils.widgets.get("reference_schema")
qc_schema     = dbutils.widgets.get("qc_schema")
dry_run       = dbutils.widgets.get("dry_run") == "true"
mlflow_exp    = dbutils.widgets.get("mlflow_experiment")

# COMMAND ----------
# MAGIC %md ### Step 1 — Load GeoNames index into memory

# COMMAND ----------
from data_ingestion.geonames_fetcher import GeonamesIndex

print("Loading GeoNames reference index...")
geo_df = spark.table(f"{catalog}.{ref_schema}.us_postal_codes").toPandas()
geonames_index = GeonamesIndex(geo_df)
print(f"GeoNames ready: {len(geonames_index._zip_to_primary):,} ZIPs indexed across {len(geonames_index._state_to_cities)} states")

# COMMAND ----------
# MAGIC %md ### Step 2 — Run QC Pipeline

# COMMAND ----------
import os, requests as _req
_user    = spark.sql("SELECT current_user()").collect()[0][0]
_exp_path = f"/Users/{_user}/SEC-EDGAR-QC"
_host    = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
# Fetch token inline for REST calls only — do NOT store in env var or print.
_token   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_hdrs    = {"Authorization": f"Bearer {_token}"}

# Create experiment via REST API (works in serverless; set_experiment() does not)
_r = _req.post(f"{_host}/api/2.0/mlflow/experiments/create",
               headers=_hdrs, json={"name": _exp_path}, timeout=10)
if _r.status_code in (200, 201):
    _exp_id = _r.json()["experiment_id"]
elif "RESOURCE_ALREADY_EXISTS" in _r.text:
    _r2 = _req.get(f"{_host}/api/2.0/mlflow/experiments/get-by-name",
                   headers=_hdrs, params={"experiment_name": _exp_path}, timeout=10)
    _exp_id = _r2.json()["experiment"]["experiment_id"]
else:
    _exp_id = None
    print(f"MLflow experiment create failed: {_r.text}")

if _exp_id:
    os.environ["MLFLOW_EXPERIMENT_ID"] = _exp_id
    print(f"MLflow experiment: {_exp_path}  (id={_exp_id})")

from pipelines.full_qc_pipeline import run_pipeline
from configs.settings import get_config
import copy

cfg = copy.deepcopy(get_config())
cfg.tables.catalog = catalog
cfg.tables.schema = qc_schema

result = run_pipeline(
    table_catalog=catalog,
    table_schema=edgar_schema,
    table_name="companies",
    qc_checks=["null", "format", "zip_state_mismatch"],
    primary_key="id",
    mlflow_experiment_name=mlflow_exp,
    dry_run=dry_run,
    spark=spark,
    geonames_index=geonames_index,
    config=cfg,
)

# COMMAND ----------
import json, time
print(json.dumps(result.model_dump(mode="json"), indent=2))

# Log result to MLflow via REST API (works in serverless where Python client fails)
if _exp_id and _host and _token:
    _ts = int(time.time() * 1000)
    _r_create = _req.post(f"{_host}/api/2.0/mlflow/runs/create",
        headers=_hdrs,
        json={"experiment_id": _exp_id, "run_name": f"qc-{result.run_id[:8]}", "start_time": _ts},
        timeout=10)
    if _r_create.ok:
        _mlflow_run_id = _r_create.json()["run"]["info"]["run_id"]
        _metrics = [
            ("total_records",        result.total_records),
            ("total_issues",         result.total_issues),
            ("l1_corrected",         result.l1_corrected),
            ("l2_flagged",           result.l2_flagged),
            ("avg_confidence",       round(result.avg_confidence, 4)),
            ("duration_seconds",     round(result.duration_seconds, 2)),
        ]
        for _k, _ct in result.check_type_breakdown.items():
            _metrics.append((f"issues_{_k}", _ct))
        for _key, _val in _metrics:
            _req.post(f"{_host}/api/2.0/mlflow/runs/log-metric", headers=_hdrs,
                json={"run_id": _mlflow_run_id, "key": _key, "value": float(_val),
                      "timestamp": _ts, "step": 0}, timeout=5)
        _req.post(f"{_host}/api/2.0/mlflow/runs/set-tag", headers=_hdrs,
            json={"run_id": _mlflow_run_id, "key": "table", "value": result.table_fqn}, timeout=5)
        _req.post(f"{_host}/api/2.0/mlflow/runs/set-tag", headers=_hdrs,
            json={"run_id": _mlflow_run_id, "key": "dry_run", "value": str(dry_run)}, timeout=5)
        _req.post(f"{_host}/api/2.0/mlflow/runs/update", headers=_hdrs,
            json={"run_id": _mlflow_run_id, "status": "FINISHED",
                  "end_time": int(time.time() * 1000)}, timeout=5)
        print(f"MLflow run logged: {_mlflow_run_id}")
    else:
        print(f"MLflow run create failed: {_r_create.status_code} {_r_create.text}")

# COMMAND ----------
# MAGIC %md ### Step 3 — Explore QC Results

# COMMAND ----------

companies = spark.table(f"{catalog}.{edgar_schema}.companies")
_has_qc_cols = "qc_status" in companies.columns

# Issue breakdown by check type
if not dry_run and _has_qc_cols:
    display(
        companies.filter("qc_status IS NOT NULL")
        .groupBy("qc_flag", "qc_support_level", "qc_correction_method")
        .count()
        .orderBy("count", ascending=False)
    )
elif dry_run:
    print("dry_run=true — skipping QC results display.")
else:
    print("QC columns not yet present (no corrections were applied).")

# COMMAND ----------
# MAGIC %md ### Step 4 — ZIP/State Mismatches (most interesting findings)

# COMMAND ----------
# Show companies where the state code doesn't match their ZIP code
if _has_qc_cols:
    zip_state_issues = (
        companies
        .filter("qc_flag = 'NEEDS_REVIEW'")
        .select("name", "city", "state_or_country", "zip_code",
                "qc_original_value", "qc_corrected_value",
                "qc_confidence_score", "qc_all_issues")
        .orderBy("qc_confidence_score", ascending=False)
        .limit(50)
    )
    display(zip_state_issues)
else:
    print("No QC columns on table yet — skipping mismatch display.")

# COMMAND ----------
# MAGIC %md ### Step 5 — Quick GeoNames spot-checks

# COMMAND ----------
# Verify a specific city typo against GeoNames
test_lookups = [("Los Angles", "CA"), ("Chcago", "IL"), ("Atlnata", "GA"), ("Pheonix", "AZ")]

for city, state in test_lookups:
    matches = geonames_index.fuzzy_city_lookup(city, state_code=state, limit=1)
    print(f"  '{city}' ({state}) → {matches[0] if matches else 'NO MATCH'}")

# COMMAND ----------
# MAGIC %md ### Step 6 — Verify a ZIP code

# COMMAND ----------
test_zips = ["90001", "10001", "75001", "60601", "99999"]
for z in test_zips:
    info = geonames_index.lookup_zip(z)
    print(f"  ZIP {z} → {info}")

# COMMAND ----------
print("QC complete. Run notebook 06 for the MCP interactive session.")
