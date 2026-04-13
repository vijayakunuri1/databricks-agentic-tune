# Databricks notebook source
# MAGIC %md
# MAGIC ## 03 · L2 Human Review
# MAGIC
# MAGIC Use this notebook to review records flagged for human attention (L2 support level).
# MAGIC For each record you can: **ACCEPT** the suggested correction, **REJECT** it, or **MANUAL_EDIT**.

# COMMAND ----------
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "default", "Schema")
dbutils.widgets.text("table_name", "customers", "Table Name")
dbutils.widgets.text("primary_key", "id", "Primary Key")
dbutils.widgets.text("reviewer", "", "Your Username (auto-filled if blank)")

# COMMAND ----------
from delta.tables import DeltaTable
from datetime import datetime
from configs.input_validation import (
    build_safe_fqn, validate_column_name, sanitize_reviewer,
)

catalog    = dbutils.widgets.get("catalog")
schema     = dbutils.widgets.get("schema")
table_name = dbutils.widgets.get("table_name")
pk_col     = dbutils.widgets.get("primary_key")
reviewer   = dbutils.widgets.get("reviewer") or dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# Validate all widget inputs before use
try:
    table_fqn = build_safe_fqn(catalog, schema, table_name)
    pk_col    = validate_column_name(pk_col, "primary_key")
    reviewer  = sanitize_reviewer(reviewer, "reviewer")
except ValueError as _e:
    raise ValueError(f"Invalid widget input: {_e}") from _e

# COMMAND ----------
# View pending L2 records
pending = (
    spark.table(table_fqn)
    .filter("qc_support_level = 'L2' AND (qc_human_reviewed IS NULL OR qc_human_reviewed = false)")
    .select(pk_col, "qc_flag", "qc_corrected_column", "qc_original_value",
            "qc_corrected_value", "qc_confidence_score", "qc_all_issues")
)
display(pending)
print(f"Total pending L2 records: {pending.count()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Bulk Accept high-confidence L2 suggestions (>= 0.80)
# MAGIC Edit the threshold or record IDs as needed.

# COMMAND ----------
ACCEPT_THRESHOLD = 0.80

dt = DeltaTable.forName(spark, table_fqn)
(
    dt.update(
        condition=f"qc_support_level = 'L2' AND qc_confidence_score >= {ACCEPT_THRESHOLD} AND (qc_human_reviewed IS NULL OR qc_human_reviewed = false)",
        set={
            "qc_human_reviewed":    "true",
            "qc_human_decision":    "'ACCEPT'",
            "qc_human_reviewed_at": f"'{datetime.utcnow().isoformat()}'",
            "qc_human_reviewer":    f"'{reviewer}'",
            # Apply the suggested correction
            "qc_corrected_column":  "qc_corrected_column",
        }
    )
)
print("Bulk accept complete.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### View accepted vs rejected breakdown

# COMMAND ----------
display(
    spark.table(table_fqn)
    .filter("qc_support_level = 'L2'")
    .groupBy("qc_human_decision")
    .count()
    .orderBy("count", ascending=False)
)
