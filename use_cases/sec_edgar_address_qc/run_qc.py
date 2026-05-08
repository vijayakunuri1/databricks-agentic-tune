"""
SEC EDGAR Address QC — main runner.

Three modes:
  1. Standard pipeline (classic 3-agent, no MCP)
     python -m use_cases.sec_edgar_address_qc.run_qc

  2. MCP tool-use (LLM drives the pipeline as tools, no server needed)
     python -m use_cases.sec_edgar_address_qc.run_qc --mode mcp-tools

  3. MCP server mode (LLM connects to running mcp_server.server)
     uvicorn mcp_server.server:app --port 8000 &
     python -m use_cases.sec_edgar_address_qc.run_qc --mode mcp-server --mcp-url http://localhost:8000/mcp
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

DATA_DIR = Path("./data")
GEO_CACHE = str(DATA_DIR / "geonames_us_postal.parquet")
EDGAR_CACHE = str(DATA_DIR / "sec_edgar_companies.parquet")


def load_geonames():
    from data_ingestion.geonames_fetcher import GeonamesIndex
    if not Path(GEO_CACHE).exists():
        raise FileNotFoundError(
            f"GeoNames cache not found at {GEO_CACHE}. Run: python -m use_cases.sec_edgar_address_qc.ingest"
        )
    logger.info("Loading GeoNames index from %s...", GEO_CACHE)
    return GeonamesIndex.from_file(GEO_CACHE)


def run_standard_pipeline(dry_run: bool = False):
    """Mode 1: classic 3-agent pipeline."""
    from pipelines.full_qc_pipeline import run_pipeline

    if not Path(EDGAR_CACHE).exists():
        raise FileNotFoundError("SEC EDGAR cache not found. Run: python -m use_cases.sec_edgar_address_qc.ingest")

    geonames = load_geonames()

    result = run_pipeline(
        table_catalog="local",
        table_schema="sec_edgar",
        table_name="companies",
        qc_checks=["null", "format", "zip_state_mismatch"],   # skip geocoder for speed
        primary_key="id",
        dry_run=dry_run,
        local_file=EDGAR_CACHE,
        geonames_index=geonames,
    )

    print("\n" + "=" * 60)
    print("SEC EDGAR Address QC Results")
    print("=" * 60)
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return result


def run_mcp_tool_use(user_request: str | None = None):
    """Mode 2: LLM drives via tool-use, no MCP server required."""
    from agents.orchestrator.mcp_orchestrator import MCPOrchestratorAgent
    from mcp_server.pipeline_state import register_table

    geonames = load_geonames()

    # Register the SEC EDGAR table for the LLM to discover
    register_table("local", "sec_edgar", "companies",
                   description="SEC EDGAR public company registrations with business addresses")

    orchestrator = MCPOrchestratorAgent(geonames_index=geonames)

    request = user_request or (
        "I need you to run a data quality check on the SEC EDGAR companies table (local.sec_edgar.companies). "
        "Focus on address issues: city name typos, ZIP/state mismatches, and null address fields. "
        "Start with a dry run to understand the scope, then tell me:\n"
        "1. How many companies have address issues?\n"
        "2. What are the most common city name typos?\n"
        "3. Which ZIP/state combinations are mismatched?\n"
        "4. What's your recommended action plan for L1 vs L2 corrections?\n"
        "Use lookup_city and lookup_zip tools to investigate specific suspicious values."
    )

    logger.info("Starting MCP tool-use session...")
    result = orchestrator.run_interactive_session(
        user_request=request,
        tables=[{"catalog": "local", "schema": "sec_edgar", "table": "companies"}],
    )

    print("\n" + "=" * 60)
    print("LLM Analysis")
    print("=" * 60)
    print(result["final_answer"])
    print(f"\nTotal tool calls: {len(result['tool_calls'])}, Iterations: {result['iterations']}")
    return result


def run_mcp_server_mode(mcp_url: str, user_request: str | None = None):
    """Mode 3: LLM connects to running MCP HTTP server."""
    from agents.orchestrator.mcp_orchestrator import MCPOrchestratorAgent

    geonames = load_geonames()
    orchestrator = MCPOrchestratorAgent(geonames_index=geonames)

    request = user_request or (
        "Run a complete data quality analysis on the SEC EDGAR companies table. "
        "Check for address typos, ZIP/state mismatches, and null fields. "
        "Apply L1 corrections automatically and provide a summary of L2 items needing review."
    )

    result = orchestrator.run_interactive_session(user_request=request)
    answer = result.get("final_answer", "")

    print("\n" + "=" * 60)
    print("LLM Analysis")
    print("=" * 60)
    print(answer)
    return answer


def main():
    parser = argparse.ArgumentParser(description="SEC EDGAR Address QC Pipeline")
    parser.add_argument("--mode", choices=["standard", "mcp-tools", "mcp-server"], default="standard")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--mcp-url", default="http://localhost:8000/mcp")
    parser.add_argument("--request", default=None, help="Custom request for the LLM (MCP modes)")
    args = parser.parse_args()

    if args.mode == "standard":
        run_standard_pipeline(dry_run=args.dry_run)
    elif args.mode == "mcp-tools":
        run_mcp_tool_use(user_request=args.request)
    elif args.mode == "mcp-server":
        run_mcp_server_mode(mcp_url=args.mcp_url, user_request=args.request)


if __name__ == "__main__":
    main()
