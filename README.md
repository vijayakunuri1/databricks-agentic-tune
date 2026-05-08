# Databricks Agentic QC Pipeline

A 3-agent AI system for automated data quality control on Databricks Delta Lake tables.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR AGENT                          │
│  • Coordinates full L1/L2 support workflow                      │
│  • Uses Databricks LLM for borderline escalation (0.75-0.85)   │
│  • Logs all metrics to Databricks MLflow                        │
└──────────────┬────────────────────────────┬────────────────────┘
               │ QC_RUN_REQUEST             │ CORRECTION_REQUEST
               ▼                            ▼
┌──────────────────────────┐  ┌────────────────────────────────────┐
│   QC RUNNER AGENT        │  │      DATA UPDATER AGENT            │
│  • Null checks           │  │  • Fuzzy match (rapidfuzz)         │
│  • Format checks         │  │  • Address correction pipeline     │
│    (email/phone/zip/date)│  │  • L1: auto-apply corrections      │
│  • Duplicate detection   │  │  • L2: flag for human review       │
│  • Geo/address validation│  │  • Delta MERGE audit columns back  │
│    (usaddress + Nominatim│  │  • Write to qc_audit_log table     │
│  • LLM anomaly triage    │  │  • LLM correction rationale        │
└──────────────────────────┘  └────────────────────────────────────┘
```

### L1 vs L2 Support

| Level | Trigger | Action |
|-------|---------|--------|
| **L1** (auto) | confidence ≥ 0.85 | Correction applied, `qc_flag = AUTO_CORRECTED` |
| **L2** (review) | 0.50 ≤ confidence < 0.85 | Flagged with suggestion, `qc_flag = NEEDS_REVIEW` |
| **Skip** | confidence < 0.50 | No action taken |

Business-critical columns (`customer_id`, `id`, etc.) and **duplicates** always go to L2.

### LLM Usage

All three agents use **Databricks Model Serving** exclusively (`databricks-meta-llama-3-3-70b-instruct` by default via the OpenAI-compatible endpoint). No external LLM API keys are required — authentication uses the workspace token injected automatically by the Databricks runtime.

The LLM is invoked for:
- **Borderline triage** (0.75–0.85 confidence): decides L1 vs L2 when rules alone can't
- **Geo/format anomaly enrichment**: verifies ambiguous issues and assigns severity
- **Correction rationale**: explains why each L1 correction was applied
- **Run summary**: produces a human-readable QC summary after each pipeline run

## QC Audit Columns Added to Source Table

After a pipeline run, these columns are merged onto every affected row:

| Column | Description |
|--------|-------------|
| `qc_status` | `CLEAN` / `AUTO_CORRECTED` / `NEEDS_REVIEW` |
| `qc_flag` | `AUTO_CORRECTED` / `NEEDS_REVIEW` / `GEO_TYPO` / … |
| `qc_original_value` | Value before correction |
| `qc_corrected_value` | Corrected or suggested value |
| `qc_confidence_score` | 0.0–1.0 fuzzy match confidence |
| `qc_correction_method` | `fuzzy_match` / `rule_based` / `geocode` / `llm` |
| `qc_support_level` | `L1` / `L2` |
| `qc_run_id` | MLflow run ID for full traceability |
| `qc_human_reviewed` | Set to `true` after human review in notebook 03 |

## Project Structure

```
databricks-agentic-tune/
├── agents/
│   ├── base_agent.py              Abstract base (all agents)
│   ├── orchestrator/
│   │   ├── orchestrator_agent.py  Master coordinator
│   │   └── escalation_policy.py  L1/L2 routing rules
│   ├── qc_runner/
│   │   ├── qc_runner_agent.py     Runs all QC checks
│   │   └── checks/
│   │       ├── null_checks.py
│   │       ├── format_checks.py
│   │       ├── duplicate_checks.py
│   │       └── geo_checks.py      usaddress + Nominatim geocoder
│   └── data_updater/
│       ├── data_updater_agent.py  Applies corrections
│       ├── fuzzy_matcher.py       rapidfuzz wrapper
│       ├── address_corrector.py   City/state/zip/street correction
│       └── delta_writer.py        Delta MERGE + audit log
├── configs/
│   ├── settings.py                AppConfig (thresholds, tables, LLM)
│   └── qc_rules.yaml              Declarative QC rules
├── llm/
│   ├── databricks_client.py       Databricks Model Serving wrapper (OpenAI-compatible)
│   └── prompts/                   System + user prompts per agent
├── messaging/
│   └── message_bus.py             In-process agent message delivery
├── schemas/
│   ├── messages.py                Inter-agent message schemas (Pydantic v2)
│   ├── qc_results.py              QCIssue, CorrectionRecord, WorkflowResult
│   └── delta_tables.py            Delta table column definitions
├── pipelines/
│   └── full_qc_pipeline.py        Main entry point (CLI + importable)
├── notebooks/
│   ├── 01_setup_tables.py         Create Delta tables (run once)
│   ├── 02_run_qc_pipeline.py      Interactive pipeline run
│   └── 03_review_l2_flags.py      Human L2 review workflow
├── tests/
│   ├── conftest.py                Shared fixtures (sample DataFrame)
│   ├── unit/                      No Spark needed
│   └── integration/               Full pipeline, local CSV
└── databricks.yml                 Databricks Asset Bundle config
```

## Quick Start

### Local (with Databricks workspace)

```bash
# 1. Clone and install
git clone <repo>
cd databricks-agentic-tune
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env: set DATABRICKS_HOST and DATABRICKS_TOKEN

# 3. Run against a local CSV
python -m pipelines.full_qc_pipeline \
  --table-catalog local \
  --table-schema test \
  --table-name customers \
  --qc-checks null,format \
  --local-file tests/fixtures/sample_customers.csv \
  --dry-run

# 4. Run tests
pytest tests/unit/
pytest tests/integration/
```

### On Databricks (Asset Bundles)

```bash
# Install Databricks CLI v2
pip install databricks-cli

# Deploy
databricks bundle deploy --target dev

# Run job (one-shot)
databricks bundle run data_qc_pipeline \
  -p catalog=main \
  -p schema=sales \
  -p table_name=customers

# Or trigger via REST API / Databricks Jobs UI
```

## Configuration

### Thresholds (`configs/settings.py`)

```python
l1_auto_correct = 0.85    # >= 85% → auto-fix
l2_review_lower = 0.50    # 50-84% → flag for human
llm_borderline_lower = 0.75  # use LLM for 75-85% decisions
```

### LLM Model (`configs/settings.py`)

The default model is `databricks-meta-llama-3-3-70b-instruct`. Override via environment variable:

```bash
DATABRICKS_LLM_MODEL=databricks-meta-llama-3-1-405b-instruct
```

Any model available on your Databricks Model Serving endpoint can be used.

### QC Rules (`configs/qc_rules.yaml`)

Edit to add columns, change critical column list, or add format regex rules.

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `DATABRICKS_HOST` | Databricks workspace URL (auto-injected on clusters) |
| `DATABRICKS_TOKEN` | Workspace token (auto-injected on clusters) |
| `DATABRICKS_LLM_MODEL` | Model Serving endpoint name (default: `databricks-meta-llama-3-3-70b-instruct`) |
| `QC_ENV` | `local` or `databricks` |
| `QC_CATALOG` | Unity Catalog catalog name |
| `QC_SCHEMA` | Schema for QC tables |
| `GEOCODER` | `nominatim` (default) |
| `MLFLOW_TRACKING_URI` | MLflow tracking URI (local only) |

## MLflow Metrics Logged

Every pipeline run logs to MLflow:
- `total_records_scanned`, `total_issues_found`
- `l1_auto_corrected`, `l2_flagged`
- `avg_confidence_score`
- `issues_null`, `issues_format`, `issues_duplicate`, `issues_geo_typo`
- `llm_calls`, `llm_total_tokens`
- `pipeline_duration_seconds`

## CI/CD Deployment

This project uses GitHub Actions for automated deployment via Databricks Asset Bundles.

### Prerequisites

Set these GitHub repository secrets:

| Secret | Description |
|--------|-------------|
| `DATABRICKS_HOST` | Databricks workspace URL (e.g. `https://dbc-xxx.cloud.databricks.com`) |
| `DATABRICKS_TOKEN` | Databricks personal access token or service principal token |

### Deployment Workflow

| Trigger | Action |
|---------|--------|
| Push to `main` | Validates bundle and deploys to **dev** |
| Pull request to `main` | Validates bundle only (no deploy) |
| Manual dispatch (dev) | Validates and deploys to **dev** |
| Manual dispatch (prod) | Validates and deploys to **prod** |

```bash
# Manual deployment via CLI
databricks bundle deploy --target dev    # deploy to dev
databricks bundle deploy --target prod   # deploy to prod

# Run the pipeline after deployment
databricks bundle run data_qc_pipeline --target dev
```

### MCP Server (Docker)

```bash
# Build
docker build -t qc-mcp-server .

# Run (behind a proxy/load balancer)
docker run -d \
  -e MCP_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  -e PROXY_MODE=true \
  -p 8000:8000 \
  qc-mcp-server
```

## Dependencies

- **PySpark / Delta Lake** — data processing and storage
- **databricks-sdk** — Databricks Model Serving (LLM) + workspace auth
- **openai** — OpenAI-compatible client for Databricks Model Serving
- **rapidfuzz** — fuzzy string matching for address corrections
- **geopy** — geocoding (Nominatim by default, free)
- **usaddress** — US address structural parsing
- **mlflow** — experiment tracking
- **pydantic v2** — message and schema validation
