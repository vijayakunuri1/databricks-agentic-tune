"""
MCP-enabled Orchestrator Agent.

Instead of hard-wiring agent calls in Python, this variant lets Claude
drive the QC workflow dynamically via MCP tools. Claude decides which
checks to run, interprets results, and calls approve/reject tools.

Two modes:
  1. TOOL_USE  — Claude calls your registered Python functions as tools
                 (no external server needed; works anywhere)
  2. MCP_SERVER — Claude connects to the running mcp_server.server
                  via Anthropic's beta MCP client API

Usage (tool-use mode, simplest):
    from agents.orchestrator.mcp_orchestrator import MCPOrchestratorAgent
    agent = MCPOrchestratorAgent(config=cfg, message_bus=bus, spark=spark)
    result = agent.run_interactive_session(
        user_request="Run QC on the SEC EDGAR companies table and fix any address typos."
    )

Usage (MCP server mode):
    result = agent.run_with_mcp_server(
        mcp_server_url="http://localhost:8000/mcp",
        user_request="..."
    )
"""
from __future__ import annotations

import json
import logging
from typing import Any

from agents.orchestrator.orchestrator_agent import OrchestratorAgent
from configs.settings import AppConfig, is_databricks
from messaging.message_bus import MessageBus

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert Databricks data quality engineer. You have access to a suite of QC tools
for analyzing and correcting data in Delta Lake tables.

Your goal: run a thorough data quality analysis, identify the most impactful issues,
apply safe automatic corrections (L1), and clearly summarize what needs human review (L2).

Think step by step:
1. List available tables
2. Run QC checks (start with dry_run=True to understand scope)
3. Inspect results — focus on geo/address issues and zip_state_mismatches
4. Use lookup_city and lookup_zip to verify specific suspicious values
5. Re-run with dry_run=False when confident about corrections
6. Review L2 flagged records and approve high-confidence ones
7. Produce a final executive summary

Always explain your reasoning before calling each tool.
"""


class MCPOrchestratorAgent(OrchestratorAgent):
    """
    MCP-enabled orchestrator. Wraps OrchestratorAgent and adds two methods:
    - run_interactive_session(): uses Claude tool-use with Python functions as tools
    - run_with_mcp_server(): uses Anthropic beta MCP client pointed at a running server
    """

    def __init__(
        self,
        agent_id: str = "mcp_orchestrator_1",
        config: AppConfig | None = None,
        message_bus: MessageBus | None = None,
        spark: Any | None = None,
        geonames_index: Any | None = None,
    ):
        if config is None:
            from configs.settings import get_config
            config = get_config()
        if message_bus is None:
            message_bus = MessageBus()

        super().__init__(
            agent_id=agent_id,
            config=config,
            message_bus=message_bus,
            spark=spark,
        )
        self.geonames = geonames_index
        # Use Databricks Model Serving (OpenAI-compatible) on Databricks,
        # fall back to Anthropic SDK locally.
        if is_databricks():
            from llm.databricks_client import DatabricksLLMClient
            self._llm_client = DatabricksLLMClient(config.databricks_llm)
            self._model = config.databricks_llm.model
            self._use_openai_api = True
        else:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=config.claude.api_key)
            self._model = config.claude.model
            self._use_openai_api = False

    # ──────────────────────────────────────────────────────────────
    # Mode 1: Tool-use (Claude calls Python callables directly)
    # ──────────────────────────────────────────────────────────────

    def run_interactive_session(
        self,
        user_request: str,
        tables: list[dict] | None = None,
        max_iterations: int = 20,
    ) -> dict:
        """
        Let Claude autonomously drive the QC pipeline using tool-use.

        Args:
            user_request: Natural language instruction, e.g.
                          "Run QC on sec_edgar.companies and fix city typos."
            tables:       [{catalog, schema, table}] to make available
            max_iterations: Safety cap on agentic loop turns

        Returns:
            dict with {"final_answer": str, "tool_calls": list, "run_results": dict}
        """
        if self._use_openai_api:
            return self._run_interactive_openai(user_request, tables or [], max_iterations)
        else:
            return self._run_interactive_anthropic(user_request, tables or [], max_iterations)

    def _run_interactive_openai(
        self, user_request: str, tables: list[dict], max_iterations: int
    ) -> dict:
        """Tool-use loop using the OpenAI-compatible Databricks serving endpoint."""
        tools = self._build_openai_tool_definitions(tables)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]
        tool_call_log = []

        for iteration in range(max_iterations):
            logger.info("MCP orchestrator loop iteration %d/%d", iteration + 1, max_iterations)

            response = self._llm_client.client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=4096,
            )

            choice = response.choices[0]
            msg = choice.message
            messages.append(msg)

            if not msg.tool_calls or choice.finish_reason == "stop":
                final_answer = msg.content or ""
                logger.info("MCP orchestrator finished after %d iterations.", iteration + 1)
                return {
                    "final_answer": final_answer,
                    "tool_calls": tool_call_log,
                    "iterations": iteration + 1,
                }

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_input = json.loads(tc.function.arguments)
                logger.info("LLM calling tool: %s(%s)", tool_name, list(tool_input.keys()))
                result_str = self._execute_tool(tool_name, tool_input)
                tool_call_log.append({"tool": tool_name, "input": tool_input, "result": result_str})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        return {"final_answer": "Max iterations reached.", "tool_calls": tool_call_log, "iterations": max_iterations}

    def _run_interactive_anthropic(
        self, user_request: str, tables: list[dict], max_iterations: int
    ) -> dict:
        """Tool-use loop using the Anthropic SDK (local only)."""
        tools = self._build_tool_definitions(tables)
        messages = [{"role": "user", "content": user_request}]
        tool_call_log = []

        for iteration in range(max_iterations):
            logger.info("MCP orchestrator loop iteration %d/%d", iteration + 1, max_iterations)

            response = self._anthropic_client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )

            text_blocks = [b.text for b in response.content if b.type == "text"]
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason == "end_turn" or not tool_use_blocks:
                return {
                    "final_answer": " ".join(text_blocks),
                    "tool_calls": tool_call_log,
                    "iterations": iteration + 1,
                }

            tool_results = []
            for tu in tool_use_blocks:
                logger.info("Claude calling tool: %s(%s)", tu.name, list(tu.input.keys()))
                result_str = self._execute_tool(tu.name, tu.input)
                tool_call_log.append({"tool": tu.name, "input": tu.input, "result": result_str})
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_str})

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return {"final_answer": "Max iterations reached.", "tool_calls": tool_call_log, "iterations": max_iterations}

    def _execute_tool(self, tool_name: str, inputs: dict) -> str:
        """Dispatch tool calls to the appropriate Python function."""
        from configs.input_validation import (
            build_safe_fqn, validate_qc_checks, validate_column_name,
            validate_batch_id,
        )
        import re
        _CITY_RE  = re.compile(r"^[a-zA-Z0-9 \'\-\.,]{1,128}$")
        _ZIP_RE   = re.compile(r"^\d{5}$")
        _STATE_RE = re.compile(r"^[A-Za-z]{2}$")
        _DECISION_ALLOWED = frozenset({"ACCEPT", "REJECT", "DEFER"})

        from mcp_server.pipeline_state import set_config

        try:
            # Validate per-tool inputs before dispatching.
            if tool_name in ("run_qc_check", "get_qc_results", "get_l2_flagged_records"):
                try:
                    catalog = inputs["catalog"]
                    schema  = inputs["schema"]
                    table   = inputs["table"]
                    table_fqn = build_safe_fqn(catalog, schema, table)
                except (KeyError, ValueError) as exc:
                    return json.dumps({"error": f"Invalid table coordinates: {exc}"})

            if tool_name == "run_qc_check":
                raw_checks = inputs.get("checks", "null,format,geo")
                try:
                    qc_checks = validate_qc_checks(raw_checks)
                except ValueError as exc:
                    return json.dumps({"error": str(exc)})
                raw_pk = inputs.get("primary_key", "id")
                try:
                    primary_key = validate_column_name(raw_pk, "primary_key")
                except ValueError as exc:
                    return json.dumps({"error": str(exc)})

            if tool_name == "lookup_city":
                city_name = inputs.get("city_name", "")
                if not isinstance(city_name, str) or not _CITY_RE.match(city_name):
                    return json.dumps({"error": "Invalid city_name: must be 1–128 chars, letters/digits/spaces/apostrophes/hyphens/commas/periods only."})
                state_code = inputs.get("state_code")
                if state_code is not None and not _STATE_RE.match(str(state_code)):
                    return json.dumps({"error": "Invalid state_code: must be exactly 2 letters."})
                limit = inputs.get("limit", 5)
                try:
                    limit = int(limit)
                    if not (1 <= limit <= 50):
                        raise ValueError
                except (TypeError, ValueError):
                    return json.dumps({"error": "Invalid limit: must be an integer between 1 and 50."})

            if tool_name == "lookup_zip":
                zip_code = inputs.get("zip_code", "")
                if not isinstance(zip_code, str) or not _ZIP_RE.match(zip_code):
                    return json.dumps({"error": "Invalid zip_code: must be exactly 5 digits."})

            if tool_name == "approve_l2_correction":
                record_id = inputs.get("record_id", "")
                try:
                    validate_batch_id(str(record_id), "record_id")
                except ValueError as exc:
                    return json.dumps({"error": str(exc)})
                decision = inputs.get("decision", "ACCEPT")
                if decision not in _DECISION_ALLOWED:
                    return json.dumps({"error": f"Invalid decision: must be one of {sorted(_DECISION_ALLOWED)}."})

        except Exception as exc:
            logger.error("Input validation for tool %s failed unexpectedly: %s", tool_name, exc, exc_info=True)
            return json.dumps({"error": "Input validation error."})

        # Inject dependencies into pipeline state for MCP tools
        set_config({
            "app_config": self.config,
            "spark": self.spark,
            "geonames_index": self.geonames,
            "primary_key": inputs.get("primary_key", "id"),
        })

        try:
            if tool_name == "list_qc_tables":
                from mcp_server.pipeline_state import list_tables
                return json.dumps(list_tables())

            elif tool_name == "run_qc_check":
                from pipelines.full_qc_pipeline import run_pipeline
                result = run_pipeline(
                    table_catalog=catalog,
                    table_schema=schema,
                    table_name=table,
                    qc_checks=qc_checks,
                    primary_key=primary_key,
                    dry_run=inputs.get("dry_run", True),
                    config=self.config,
                    spark=self.spark,
                    geonames_index=self.geonames,
                )
                from mcp_server.pipeline_state import store_run_result
                store_run_result(table_fqn, result)
                return json.dumps(result.model_dump(mode="json"))

            elif tool_name == "get_qc_results":
                from mcp_server.pipeline_state import get_last_run
                return json.dumps(get_last_run(table_fqn) or {"error": "No run found."})

            elif tool_name == "lookup_city":
                if self.geonames:
                    matches = self.geonames.fuzzy_city_lookup(
                        city_name, state_code, limit=limit
                    )
                    return json.dumps({"matches": matches})
                return json.dumps({"error": "GeoNames index not loaded."})

            elif tool_name == "lookup_zip":
                if self.geonames:
                    return json.dumps(self.geonames.lookup_zip(zip_code) or {"error": "ZIP not found."})
                return json.dumps({"error": "GeoNames index not loaded."})

            elif tool_name == "get_l2_flagged_records":
                from mcp_server.pipeline_state import get_l2_records
                records = get_l2_records(table_fqn)
                limit_val = inputs.get("limit", 50)
                try:
                    limit_val = max(1, min(500, int(limit_val)))
                except (TypeError, ValueError):
                    limit_val = 50
                return json.dumps({"count": len(records), "records": records[:limit_val]})

            elif tool_name == "approve_l2_correction":
                return json.dumps({
                    "status": "ok",
                    "record_id": record_id,
                    "decision": decision,
                    "note": "Decision logged (no Delta write in tool-use mode unless Spark is available).",
                })

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as exc:
            logger.error("Tool %s failed: %s", tool_name, exc, exc_info=True)
            return json.dumps({"error": str(exc)})

    # ──────────────────────────────────────────────────────────────
    # Mode 2: MCP Server (Anthropic beta mcp-client)
    # ──────────────────────────────────────────────────────────────

    def run_with_mcp_server(
        self,
        mcp_server_url: str,
        user_request: str,
        databricks_token: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """MCP server mode — requires Anthropic SDK (local only)."""
        if self._use_openai_api:
            raise NotImplementedError(
                "MCP server mode uses Anthropic's beta mcp-client API which isn't available "
                "via Databricks Model Serving. Use tool_use mode instead (mode=tool_use)."
            )

        mcp_server_config: dict = {"type": "url", "url": mcp_server_url, "name": "databricks-qc"}
        if databricks_token:
            mcp_server_config["authorization_token"] = databricks_token

        response = self._anthropic_client.beta.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            mcp_servers=[mcp_server_config],
            messages=[{"role": "user", "content": user_request}],
            betas=["mcp-client-2025-04-04"],
        )

        return "".join(b.text for b in response.content if hasattr(b, "text"))

    def _build_openai_tool_definitions(self, tables: list[dict]) -> list[dict]:
        """Convert tool definitions to OpenAI function-calling format."""
        return [
            {"type": "function", "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }}
            for t in self._build_tool_definitions(tables)
        ]

    def _build_tool_definitions(self, tables: list[dict]) -> list[dict]:
        """Build Anthropic tool definitions for tool-use mode."""
        return [
            {
                "name": "list_qc_tables",
                "description": "List all Delta tables available for QC checks.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "run_qc_check",
                "description": "Run QC checks on a Delta table. Use dry_run=True first to see issues without writing.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "catalog": {"type": "string", "description": "Unity Catalog catalog name"},
                        "schema": {"type": "string", "description": "Schema name"},
                        "table": {"type": "string", "description": "Table name"},
                        "checks": {"type": "string", "description": "Comma-separated check names: null,format,duplicate,geo,zip_state_mismatch"},
                        "primary_key": {"type": "string", "description": "Primary key column", "default": "id"},
                        "dry_run": {"type": "boolean", "description": "True=detect only, False=apply L1 corrections", "default": True},
                    },
                    "required": ["catalog", "schema", "table"],
                },
            },
            {
                "name": "get_qc_results",
                "description": "Get the results of the most recent QC run for a table.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "catalog": {"type": "string"},
                        "schema": {"type": "string"},
                        "table": {"type": "string"},
                    },
                    "required": ["catalog", "schema", "table"],
                },
            },
            {
                "name": "lookup_city",
                "description": "Fuzzy-match a city name against GeoNames reference data to find the canonical spelling.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city_name": {"type": "string", "description": "City name to look up (possibly misspelled)"},
                        "state_code": {"type": "string", "description": "Optional US state abbreviation to narrow results"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["city_name"],
                },
            },
            {
                "name": "lookup_zip",
                "description": "Look up a ZIP code in GeoNames to get canonical city and state.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "zip_code": {"type": "string", "description": "5-digit US ZIP code"},
                    },
                    "required": ["zip_code"],
                },
            },
            {
                "name": "get_l2_flagged_records",
                "description": "Get records flagged for human review (L2). These need a decision: ACCEPT, REJECT, or MANUAL_EDIT.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "catalog": {"type": "string"},
                        "schema": {"type": "string"},
                        "table": {"type": "string"},
                        "limit": {"type": "integer", "default": 50},
                        "min_confidence": {"type": "number", "default": 0.0},
                    },
                    "required": ["catalog", "schema", "table"],
                },
            },
            {
                "name": "approve_l2_correction",
                "description": "Record a human decision (ACCEPT/REJECT/MANUAL_EDIT) on an L2-flagged record.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "catalog": {"type": "string"},
                        "schema": {"type": "string"},
                        "table": {"type": "string"},
                        "record_id": {"type": "string"},
                        "decision": {"type": "string", "enum": ["ACCEPT", "REJECT", "MANUAL_EDIT"]},
                        "manual_value": {"type": "string", "description": "Use when decision=MANUAL_EDIT"},
                        "reviewer": {"type": "string", "default": "mcp-agent"},
                    },
                    "required": ["catalog", "schema", "table", "record_id", "decision"],
                },
            },
        ]
