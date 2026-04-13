# Databricks notebook source
# MAGIC %md
# MAGIC ## 04 · Ingest Public Reference Data
# MAGIC
# MAGIC Fetches two free public datasets and loads them into Delta:
# MAGIC
# MAGIC | Dataset | Source | What it gives us |
# MAGIC |---------|--------|-----------------|
# MAGIC | **SEC EDGAR** | `data.sec.gov` (free, no key) | ~10,000 public company registered addresses |
# MAGIC | **GeoNames** | `download.geonames.org` (CC-BY 4.0) | 42,000 US ZIP → canonical city + state |
# MAGIC
# MAGIC **Run this notebook once** before running notebook 05.

# COMMAND ----------
import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = "/Workspace" + os.path.dirname(os.path.dirname(_nb_path))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# COMMAND ----------
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("sec_edgar_schema", "sec_edgar", "SEC EDGAR Schema")
dbutils.widgets.text("reference_schema", "reference", "Reference Schema")
dbutils.widgets.text("company_limit", "500", "Companies to Fetch (start small, max ~10000)")

catalog         = dbutils.widgets.get("catalog")
edgar_schema    = dbutils.widgets.get("sec_edgar_schema")
ref_schema      = dbutils.widgets.get("reference_schema")
company_limit   = int(dbutils.widgets.get("company_limit"))

# COMMAND ----------
# %pip install rapidfuzz geopy usaddress anthropic pydantic databricks-agentic-tune

# COMMAND ----------
# MAGIC %md ### Step 1 — Create schemas

# COMMAND ----------
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{edgar_schema}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{ref_schema}")
print(f"Schemas ready: {catalog}.{edgar_schema}, {catalog}.{ref_schema}")

# COMMAND ----------
# MAGIC %md ### Step 2 — GeoNames US Postal Codes (42k ZIP codes, free)

# COMMAND ----------
from data_ingestion.geonames_fetcher import GeonamesIndex

print("Downloading GeoNames US postal codes...")
geo_index = GeonamesIndex.download(cache_path="/tmp/geonames_us.parquet")

geo_sdf = spark.createDataFrame(geo_index._df)
geo_sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{ref_schema}.us_postal_codes")

print(f"GeoNames loaded: {geo_index._df.shape[0]:,} records → {catalog}.{ref_schema}.us_postal_codes")
display(geo_sdf.limit(10))

# COMMAND ----------
# MAGIC %md ### Step 3 — SEC EDGAR Company Registrations (free public data)
# MAGIC
# MAGIC The SEC requires a User-Agent header but no API key.
# MAGIC Rate limit: 10 req/sec — the fetcher respects this automatically.

# COMMAND ----------
from data_ingestion.sec_edgar_fetcher import SECEdgarFetcher

print(f"Fetching {company_limit} companies from SEC EDGAR...")
fetcher = SECEdgarFetcher()
companies_df = fetcher.fetch_companies(
    limit=company_limit,
    us_only=True,
    cache_path="/tmp/sec_edgar_companies.parquet",
)

# Write to Delta
companies_sdf = spark.createDataFrame(companies_df.astype(str).fillna(""))
companies_sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{edgar_schema}.companies")

print(f"SEC EDGAR loaded: {len(companies_df):,} companies → {catalog}.{edgar_schema}.companies")
display(companies_sdf.select("name", "city", "state_or_country", "zip_code", "sic_description").limit(20))

# COMMAND ----------
# MAGIC %md ### Step 4 — Preview: known address quality issues in SEC EDGAR

# COMMAND ----------
# Companies where city or zip might be blank
issues_preview = companies_sdf.filter(
    (companies_sdf.city == "") | (companies_sdf.zip_code == "") | (companies_sdf.state_or_country == "")
)
print(f"Companies with blank address fields: {issues_preview.count()}")
display(issues_preview.select("name", "city", "state_or_country", "zip_code").limit(20))

# COMMAND ----------
print("✓ Ingestion complete. Run notebook 05 to start QC.")
print(f"  GeoNames: {catalog}.{ref_schema}.us_postal_codes")
print(f"  Companies: {catalog}.{edgar_schema}.companies")
