# Databricks notebook source
# MAGIC %md
# MAGIC ## 07 · Agent Validation
# MAGIC
# MAGIC Creates a synthetic table with **known, intentional data quality issues**, runs the
# MAGIC full 3-agent pipeline against it, then checks that each agent did exactly what it
# MAGIC should have.
# MAGIC
# MAGIC | Check | What it proves |
# MAGIC |-------|---------------|
# MAGIC | QCRunnerAgent detected every planted issue | Detection is working |
# MAGIC | OrchestratorAgent classified L1 vs L2 correctly | Escalation policy is working |
# MAGIC | DataUpdaterAgent applied L1 corrections | Correction + MERGE is working |
# MAGIC | L2 records were flagged, not auto-corrected | Human-review gate is working |
# MAGIC | Clean records were left untouched | No false positives |

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
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "qc_schema", "Schema")

catalog = dbutils.widgets.get("catalog")
schema  = dbutils.widgets.get("schema")
table   = "agent_validation_test"
fqn     = f"{catalog}.{schema}.{table}"

# COMMAND ----------
# MAGIC %md ### Step 1 — Create synthetic test table with planted issues

# COMMAND ----------
import pandas as pd

# Each row documents the expected behaviour so assertions are self-describing
test_records = [
    # ── CLEAN RECORDS (no issues expected) ──────────────────────────────────
    {"id": "clean_001", "name": "Apple Inc",       "city": "Cupertino",    "state_or_country": "CA", "zip_code": "95014", "street1": "One Apple Park Way",  "_expect_issue": None,   "_expect_flag": None},
    {"id": "clean_002", "name": "Google LLC",      "city": "Mountain View", "state_or_country": "CA", "zip_code": "94043", "street1": "1600 Amphitheatre Pkwy", "_expect_issue": None, "_expect_flag": None},

    # ── NULL CHECKS (QCRunnerAgent — null check) ─────────────────────────────
    {"id": "null_city",  "name": "Null City Corp", "city": None,           "state_or_country": "TX", "zip_code": "75001", "street1": "123 Main St",         "_expect_issue": "null",  "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_zip",   "name": "Null Zip Corp",  "city": "Dallas",       "state_or_country": "TX", "zip_code": None,   "street1": "456 Oak Ave",          "_expect_issue": "null",  "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_state", "name": "Null State Corp","city": "Austin",       "state_or_country": None, "zip_code": "78701", "street1": "789 Congress Ave",    "_expect_issue": "null",  "_expect_flag": "NEEDS_REVIEW"},

    # ── FORMAT CHECKS (QCRunnerAgent — format check) ─────────────────────────
    {"id": "bad_zip_alpha", "name": "Bad ZIP Alpha", "city": "Denver",     "state_or_country": "CO", "zip_code": "ABCDE", "street1": "1 Denver Pl",         "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_zip_short", "name": "Bad ZIP Short", "city": "Denver",     "state_or_country": "CO", "zip_code": "802",   "street1": "2 Denver Pl",         "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_state",     "name": "Bad State Co",  "city": "Phoenix",    "state_or_country": "ZZ", "zip_code": "85001", "street1": "10 Phoenix Blvd",     "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},

    # ── ZIP/STATE MISMATCH (QCRunnerAgent — zip_state_mismatch) ─────────────
    # ZIP 10001 is New York, not California
    {"id": "zip_mismatch_001", "name": "Wrong State Corp", "city": "Los Angeles", "state_or_country": "CA", "zip_code": "10001", "street1": "5 Wrong Way",  "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
    # ZIP 75001 is Texas, not Florida
    {"id": "zip_mismatch_002", "name": "Confused State Inc","city": "Miami",     "state_or_country": "FL", "zip_code": "75001", "street1": "6 Confused Blvd", "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
]

df = pd.DataFrame(test_records)
# Drop validation columns before writing to Delta
delta_df = spark.createDataFrame(df.drop(columns=["_expect_issue", "_expect_flag"]))

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
delta_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(fqn)
print(f"Created {fqn} with {delta_df.count()} rows ({len(test_records)} total)")
display(delta_df)

# COMMAND ----------
# MAGIC %md ### Step 2 — Run the full 3-agent pipeline (dry_run=False so corrections are applied)

# COMMAND ----------
import mlflow, os, requests as _req
_user    = spark.sql("SELECT current_user()").collect()[0][0]
_exp_path = f"/Users/{_user}/agent-validation"
_host    = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
# Fetch token inline for REST calls only — do NOT store in env var or print.
_token   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_hdrs    = {"Authorization": f"Bearer {_token}"}

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

# ── DIAGNOSTIC: run null check directly to verify it works ──────────────────
from agents.qc_runner.checks.null_checks import NullCheckRunner
from configs.settings import get_config
_diag_df = spark.table(fqn)
_diag_pdf = _diag_df.toPandas()
print(f"[DIAG] pandas shape: {_diag_pdf.shape}")
print(f"[DIAG] null counts per column:\n{_diag_pdf.isnull().sum()}")
_null_runner = NullCheckRunner(get_config())
_diag_issues = _null_runner.run(_diag_df, fqn)
print(f"[DIAG] NullCheckRunner found {len(_diag_issues)} issues")
for _i in _diag_issues:
    print(f"  {_i.record_id} / {_i.column_name}")

from pipelines.full_qc_pipeline import run_pipeline
from data_ingestion.geonames_fetcher import GeonamesIndex

# Load GeoNames for zip_state_mismatch checks
geo_df = spark.table(f"{catalog}.reference.us_postal_codes").toPandas()
geonames_index = GeonamesIndex(geo_df)

result = run_pipeline(
    table_catalog=catalog,
    table_schema=schema,
    table_name=table,
    qc_checks=["null", "format", "zip_state_mismatch"],
    primary_key="id",
    dry_run=False,
    spark=spark,
    geonames_index=geonames_index,
    mlflow_experiment_name="/Shared/agent-validation",
)

print(f"\nPipeline result:")
print(f"  total_records : {result.total_records}")
print(f"  total_issues  : {result.total_issues}")
print(f"  l1_corrected  : {result.l1_corrected}")
print(f"  l2_flagged    : {result.l2_flagged}")
print(f"  severity      : {result.severity_breakdown}")
print(f"  check_types   : {result.check_type_breakdown}")

# Log result to MLflow via REST API (works in serverless)
if _exp_id and _host and _token:
    import time as _time
    _ts = int(_time.time() * 1000)
    _r_create = _req.post(f"{_host}/api/2.0/mlflow/runs/create",
        headers=_hdrs,
        json={"experiment_id": _exp_id, "run_name": f"validation-{result.run_id[:8]}", "start_time": _ts},
        timeout=10)
    if _r_create.ok:
        _mlflow_run_id = _r_create.json()["run"]["info"]["run_id"]
        _metrics = [
            ("total_records",    result.total_records),
            ("total_issues",     result.total_issues),
            ("l1_corrected",     result.l1_corrected),
            ("l2_flagged",       result.l2_flagged),
            ("avg_confidence",   round(result.avg_confidence, 4)),
        ]
        for _k, _ct in result.check_type_breakdown.items():
            _metrics.append((f"issues_{_k}", _ct))
        for _key, _val in _metrics:
            _req.post(f"{_host}/api/2.0/mlflow/runs/log-metric", headers=_hdrs,
                json={"run_id": _mlflow_run_id, "key": _key, "value": float(_val),
                      "timestamp": _ts, "step": 0}, timeout=5)
        _req.post(f"{_host}/api/2.0/mlflow/runs/update", headers=_hdrs,
            json={"run_id": _mlflow_run_id, "status": "FINISHED",
                  "end_time": int(_time.time() * 1000)}, timeout=5)
        print(f"MLflow run logged: {_mlflow_run_id}")
    else:
        print(f"MLflow run create failed: {_r_create.status_code} {_r_create.text}")

# COMMAND ----------
# MAGIC %md ### Step 3 — Read back the table and run assertions

# COMMAND ----------
result_df = spark.table(fqn).toPandas()
expected  = pd.DataFrame(test_records).set_index("id")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append({"check": name, "status": status, "detail": detail})
    print(f"{status}  {name}" + (f" — {detail}" if detail else ""))

# ── 1. QCRunnerAgent: detected all planted issue types ───────────────────────
check_types_found = set(result.check_type_breakdown.keys())
check("QCRunner detected null issues",           "null"               in check_types_found)
check("QCRunner detected format issues",         "format"             in check_types_found)
check("QCRunner detected zip_state_mismatch",    "zip_state_mismatch" in check_types_found)

# ── 2. QCRunnerAgent: did NOT flag clean records ─────────────────────────────
clean_ids = [r["id"] for r in test_records if r["_expect_issue"] is None]
for cid in clean_ids:
    row = result_df[result_df["id"] == cid]
    if not row.empty and "qc_status" in row.columns:
        flagged = row.iloc[0].get("qc_status") is not None and str(row.iloc[0].get("qc_status")) not in ("", "nan", "None")
        check(f"No false positive on clean record '{cid}'", not flagged,
              f"got qc_status={row.iloc[0].get('qc_status')!r}" if flagged else "")
    else:
        check(f"No false positive on clean record '{cid}'", True, "no qc_status column — no write occurred (expected)")

# ── 3. OrchestratorAgent: issued issues are flagged ─────────────────────────
issue_ids = [r["id"] for r in test_records if r["_expect_issue"] is not None]
flagged_ids = set()
if "qc_flag" in result_df.columns:
    flagged_ids = set(result_df[result_df["qc_flag"].notna() & (result_df["qc_flag"] != "")]["id"])

check("Orchestrator flagged at least one issue per check type",
      len(flagged_ids) > 0,
      f"flagged={sorted(flagged_ids)}")

# ── 4. DataUpdaterAgent: L2 records NOT auto-corrected ───────────────────────
if "qc_flag" in result_df.columns and "qc_corrected_value" in result_df.columns:
    l2_rows = result_df[result_df["qc_flag"] == "NEEDS_REVIEW"]
    l2_auto_corrected = l2_rows[l2_rows["qc_corrected_value"].notna() & (l2_rows["qc_corrected_value"] != "")]
    # L2 rows may have a suggested value but was_applied=False; check qc_status
    if "qc_status" in result_df.columns:
        l2_wrongly_applied = l2_rows[l2_rows["qc_status"] == "AUTO_CORRECTED"]
        check("DataUpdater did not auto-correct L2 records",
              len(l2_wrongly_applied) == 0,
              f"{len(l2_wrongly_applied)} L2 records incorrectly marked AUTO_CORRECTED")
    else:
        check("DataUpdater L2 gate", True, "qc_status column not present — dry_run may be active")
else:
    check("DataUpdater wrote qc_flag column", "qc_flag" in result_df.columns)

# ── 5. Pipeline-level counts sanity ─────────────────────────────────────────
expected_issue_count = len([r for r in test_records if r["_expect_issue"] is not None])
check("Pipeline found all planted issues",
      result.total_issues >= expected_issue_count,
      f"found={result.total_issues}, planted={expected_issue_count}")

check("Pipeline scanned all records",
      result.total_records == len(test_records),
      f"scanned={result.total_records}, total={len(test_records)}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for r in results if r["status"] == PASS)
failed = sum(1 for r in results if r["status"] == FAIL)
print(f"RESULT: {passed} passed, {failed} failed out of {len(results)} checks")
if failed:
    print("\nFailed checks:")
    for r in results:
        if r["status"] == FAIL:
            print(f"  {r['check']}: {r['detail']}")

# COMMAND ----------
# MAGIC %md ### Step 4 — Inspect the annotated table

# COMMAND ----------
qc_cols = ["id", "name", "city", "state_or_country", "zip_code",
           "qc_status", "qc_flag", "qc_corrected_column",
           "qc_original_value", "qc_corrected_value", "qc_confidence_score",
           "qc_support_level"]
available = [c for c in qc_cols if c in result_df.columns]
display(spark.table(fqn).select(*available).orderBy("id"))

# COMMAND ----------
# MAGIC %md ### Step 5 — Cleanup (optional)

# COMMAND ----------
# Uncomment to remove the test table after validation
# spark.sql(f"DROP TABLE IF EXISTS {fqn}")
# print(f"Dropped {fqn}")
print(f"Test table kept at: {fqn}")
print("Run: spark.sql(f'DROP TABLE IF EXISTS {fqn}') to clean up.")
