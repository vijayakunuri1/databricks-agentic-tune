# Databricks notebook source
# MAGIC %md
# MAGIC ## 02 · Run QC Pipeline
# MAGIC
# MAGIC Runs the full 3-agent QC pipeline against a Delta table.
# MAGIC
# MAGIC **Widgets:**
# MAGIC - `catalog` / `schema` / `table_name` — source table coordinates
# MAGIC - `qc_checks` — comma-separated list: `null,format,duplicate,geo`
# MAGIC - `primary_key` — primary key column name
# MAGIC - `dry_run` — `true` = run checks but do NOT write corrections back

# COMMAND ----------
import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + os.path.dirname(os.path.dirname(_nb_path))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# COMMAND ----------
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "default", "Schema")
dbutils.widgets.text("table_name", "customers", "Table Name")
dbutils.widgets.text("qc_checks", "null,format,duplicate,geo", "QC Checks")
dbutils.widgets.text("primary_key", "id", "Primary Key Column")
dbutils.widgets.dropdown("dry_run", "false", ["true", "false"], "Dry Run")
dbutils.widgets.text("mlflow_experiment", "/Shared/QC-Pipeline", "MLflow Experiment")

# COMMAND ----------
catalog     = dbutils.widgets.get("catalog")
schema      = dbutils.widgets.get("schema")
table_name  = dbutils.widgets.get("table_name")
qc_checks   = [c.strip() for c in dbutils.widgets.get("qc_checks").split(",")]
primary_key = dbutils.widgets.get("primary_key")
dry_run     = dbutils.widgets.get("dry_run").lower() == "true"
mlflow_exp  = dbutils.widgets.get("mlflow_experiment")

# COMMAND ----------
# Install wheel if running interactively (otherwise pre-installed via job libraries)
# %pip install databricks-agentic-tune  # uncomment if running from notebook without the wheel

# COMMAND ----------
from pipelines.full_qc_pipeline import run_pipeline

result = run_pipeline(
    table_catalog=catalog,
    table_schema=schema,
    table_name=table_name,
    qc_checks=qc_checks,
    primary_key=primary_key,
    mlflow_experiment_name=mlflow_exp,
    dry_run=dry_run,
    spark=spark,
)

# COMMAND ----------
# Display result summary
import json
print(json.dumps(result.model_dump(), indent=2, default=str))

# COMMAND ----------
# Show L2 records that need human review
display(
    spark.table(f"{catalog}.{schema}.{table_name}")
    .filter("qc_support_level = 'L2' AND qc_human_reviewed = false")
    .select("*")
    .limit(200)
)
