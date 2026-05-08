# Databricks notebook source
# MAGIC %md
# MAGIC ## 06 · MCP Interactive QC Session
# MAGIC
# MAGIC Demonstrates two MCP integration modes for letting Claude autonomously
# MAGIC drive the QC pipeline on SEC EDGAR data.
# MAGIC
# MAGIC | Mode | How it works | When to use |
# MAGIC |------|-------------|-------------|
# MAGIC | **Tool-Use** | Claude calls Python functions directly as tools | Local / notebook / no server needed |
# MAGIC | **MCP Server** | Claude connects to a running HTTP MCP server | Production / Databricks Jobs |
# MAGIC
# MAGIC **Prerequisites:** Run notebooks 04 and 05 first.

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
dbutils.widgets.text("edgar_schema", "sec_edgar", "SEC EDGAR Schema")
dbutils.widgets.text("reference_schema", "reference", "Reference Schema")
dbutils.widgets.dropdown("mode", "tool_use", ["tool_use", "mcp_server"], "MCP Mode")
dbutils.widgets.text("mcp_server_url", "http://localhost:8000/mcp", "MCP Server URL (server mode only)")

catalog      = dbutils.widgets.get("catalog")
edgar_schema = dbutils.widgets.get("edgar_schema")
ref_schema   = dbutils.widgets.get("reference_schema")
mode         = dbutils.widgets.get("mode")
mcp_url      = dbutils.widgets.get("mcp_server_url")

# COMMAND ----------
# Ensure schemas exist when this notebook runs standalone
# (the pipeline may write QC audit data to the default qc_schema).
from configs.settings import get_config as _get_cfg
_qc_schema = _get_cfg().tables.schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{_qc_schema}")

# COMMAND ----------
# MAGIC %md ### Load GeoNames Index

# COMMAND ----------
from data_ingestion.geonames_fetcher import GeonamesIndex
from mcp_server.pipeline_state import register_table, set_config

geo_df = spark.table(f"{catalog}.{ref_schema}.us_postal_codes").toPandas()
geonames_index = GeonamesIndex(geo_df)

# Register the companies table so Claude can discover it
register_table(catalog, edgar_schema, "companies",
               description="SEC EDGAR public company registrations — registered business addresses")

# Inject Spark + GeoNames into the MCP pipeline state
set_config({
    "spark": spark,
    "geonames_index": geonames_index,
    "primary_key": "id",
})

print("GeoNames and Spark injected into MCP pipeline state.")

# COMMAND ----------
# MAGIC %md ### Mode A: Tool-Use (Claude calls Python tools directly)

# COMMAND ----------
if mode == "tool_use":
    from agents.orchestrator.mcp_orchestrator import MCPOrchestratorAgent

    orchestrator = MCPOrchestratorAgent(
        config=None,        # uses get_config() from env vars
        spark=spark,
        geonames_index=geonames_index,
    )

    USER_REQUEST = f"""
    Analyze the data quality of the SEC EDGAR companies table ({catalog}.{edgar_schema}.companies).

    I specifically want to understand:
    1. How many companies have blank or null address fields?
    2. How many have ZIP codes that don't match their registered state?
    3. What are the top 5 most common city name typos you can identify?
    4. Use lookup_zip and lookup_city tools to investigate at least 3 specific suspicious records.
    5. Recommend which corrections should be auto-applied (L1) vs reviewed by a human (L2).

    Start with dry_run=True so we don't modify data yet.
    Be specific — name actual companies and address values from the data.
    """

    print("Starting Claude tool-use session...")
    print("Claude will call tools autonomously to analyze the data.\n")

    result = orchestrator.run_interactive_session(
        user_request=USER_REQUEST,
        tables=[{"catalog": catalog, "schema": edgar_schema, "table": "companies"}],
        max_iterations=25,
    )

    print("\n" + "=" * 70)
    print("CLAUDE'S ANALYSIS")
    print("=" * 70)
    print(result["final_answer"])
    print(f"\nStats: {result['iterations']} iterations, {len(result['tool_calls'])} tool calls")

    # Show all tool calls for transparency
    print("\n" + "-" * 40)
    print("Tools called by Claude:")
    for i, tc in enumerate(result["tool_calls"], 1):
        print(f"  {i}. {tc['tool']}({list(tc['input'].keys())})")

# COMMAND ----------
# MAGIC %md ### Mode B: MCP Server Mode

# COMMAND ----------
if mode == "mcp_server":
    # NOTE: Start the MCP server first: uvicorn mcp_server.server:app --host 0.0.0.0 --port 8000
    from agents.orchestrator.mcp_orchestrator import MCPOrchestratorAgent
    from configs.settings import get_config

    cfg = get_config()
    orchestrator = MCPOrchestratorAgent(config=cfg, spark=spark, geonames_index=geonames_index)

    USER_REQUEST = (
        f"Run a complete data quality check on the SEC EDGAR companies table "
        f"({catalog}.{edgar_schema}.companies). "
        "Find address typos, ZIP/state mismatches, and null fields. "
        "Apply safe L1 corrections and summarize what needs human review."
    )

    result = orchestrator.run_interactive_session(user_request=USER_REQUEST)
    answer = result.get("final_answer", "")

    print("\n" + "=" * 70)
    print("LLM ANALYSIS")
    print("=" * 70)
    print(answer)

# COMMAND ----------
# MAGIC %md
# MAGIC ### How MCP Connects to Databricks
# MAGIC
# MAGIC ```
# MAGIC ┌────────────────────────────────────────────────────────────────┐
# MAGIC │               Databricks Workspace                             │
# MAGIC │                                                                │
# MAGIC │  ┌─────────────────┐    MCP tools     ┌──────────────────┐   │
# MAGIC │  │  Claude (via    │ ◄──────────────► │  mcp_server/     │   │
# MAGIC │  │  Anthropic API) │   JSON over HTTP  │  server.py       │   │
# MAGIC │  └─────────────────┘                  │  (FastMCP)       │   │
# MAGIC │                                        └────────┬─────────┘   │
# MAGIC │                                                 │             │
# MAGIC │                                    ┌────────────▼──────────┐  │
# MAGIC │                                    │  3-Agent Pipeline     │  │
# MAGIC │                                    │  OrchestratorAgent    │  │
# MAGIC │                                    │  QCRunnerAgent        │  │
# MAGIC │                                    │  DataUpdaterAgent     │  │
# MAGIC │                                    └────────────┬──────────┘  │
# MAGIC │                                                 │             │
# MAGIC │                                    ┌────────────▼──────────┐  │
# MAGIC │                                    │  Delta Lake           │  │
# MAGIC │                                    │  sec_edgar.companies  │  │
# MAGIC │                                    │  reference.us_postal  │  │
# MAGIC │                                    └───────────────────────┘  │
# MAGIC └────────────────────────────────────────────────────────────────┘
# MAGIC ```
