# Databricks notebook source
# MAGIC %md
# MAGIC ## 01 · Setup QC Tables
# MAGIC
# MAGIC Creates the `qc_audit_log` and `geocode_cache` Delta tables in the target catalog/schema.
# MAGIC
# MAGIC **Run this once before the first pipeline execution.**

# COMMAND ----------
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "qc_schema", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------
# QC Audit Log table
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.qc_audit_log (
    audit_id        STRING NOT NULL,
    run_id          STRING NOT NULL,
    batch_id        STRING NOT NULL,
    table_fqn       STRING NOT NULL,
    record_id       STRING NOT NULL,
    column_name     STRING,
    check_type      STRING NOT NULL,
    original_value  STRING,
    corrected_value STRING,
    confidence_score FLOAT,
    support_level   STRING,
    was_applied     BOOLEAN,
    correction_method STRING,
    llm_reasoning   STRING,
    created_at      TIMESTAMP NOT NULL
)
USING DELTA
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact'   = 'true'
)
""")

# COMMAND ----------
# Geocode cache table (avoids re-hitting Nominatim for the same addresses)
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.geocode_cache (
    address_key     STRING NOT NULL,
    full_address    STRING NOT NULL,
    latitude        DOUBLE,
    longitude       DOUBLE,
    geocoded_at     TIMESTAMP NOT NULL
)
USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
""")

print(f"Tables created in {catalog}.{schema}")
print(f"  - {catalog}.{schema}.qc_audit_log")
print(f"  - {catalog}.{schema}.geocode_cache")
