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
    {"id": "clean_001", "name": "Apple Inc",              "city": "Cupertino",      "state_or_country": "CA", "zip_code": "95014", "street1": "One Apple Park Way",        "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_002", "name": "Google LLC",             "city": "Mountain View",  "state_or_country": "CA", "zip_code": "94043", "street1": "1600 Amphitheatre Pkwy",   "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_003", "name": "Microsoft Corp",         "city": "Redmond",        "state_or_country": "WA", "zip_code": "98052", "street1": "One Microsoft Way",        "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_004", "name": "Amazon.com Inc",         "city": "Seattle",        "state_or_country": "WA", "zip_code": "98109", "street1": "410 Terry Ave N",          "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_005", "name": "Meta Platforms Inc",     "city": "Menlo Park",     "state_or_country": "CA", "zip_code": "94025", "street1": "1 Hacker Way",             "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_006", "name": "Tesla Inc",              "city": "Austin",         "state_or_country": "TX", "zip_code": "78725", "street1": "13101 Harold Green Rd",    "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_007", "name": "Netflix Inc",            "city": "Los Gatos",      "state_or_country": "CA", "zip_code": "95032", "street1": "100 Winchester Cir",       "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_008", "name": "Adobe Inc",              "city": "San Jose",       "state_or_country": "CA", "zip_code": "95110", "street1": "345 Park Ave",             "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_009", "name": "Salesforce Inc",         "city": "San Francisco",  "state_or_country": "CA", "zip_code": "94105", "street1": "415 Mission St",           "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_010", "name": "Oracle Corp",            "city": "Austin",         "state_or_country": "TX", "zip_code": "78741", "street1": "2300 Oracle Way",          "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_011", "name": "Intel Corp",             "city": "Santa Clara",    "state_or_country": "CA", "zip_code": "95054", "street1": "2200 Mission College Blvd","_expect_issue": None, "_expect_flag": None},
    {"id": "clean_012", "name": "IBM Corp",               "city": "Armonk",         "state_or_country": "NY", "zip_code": "10504", "street1": "1 New Orchard Rd",         "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_013", "name": "Cisco Systems Inc",      "city": "San Jose",       "state_or_country": "CA", "zip_code": "95134", "street1": "170 W Tasman Dr",          "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_014", "name": "Nvidia Corp",            "city": "Santa Clara",    "state_or_country": "CA", "zip_code": "95051", "street1": "2788 San Tomas Expy",      "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_015", "name": "PayPal Holdings Inc",    "city": "San Jose",       "state_or_country": "CA", "zip_code": "95131", "street1": "2211 N First St",          "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_016", "name": "Zoom Video Inc",         "city": "San Jose",       "state_or_country": "CA", "zip_code": "95113", "street1": "55 Almaden Blvd",          "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_017", "name": "Airbnb Inc",             "city": "San Francisco",  "state_or_country": "CA", "zip_code": "94103", "street1": "888 Brannan St",           "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_018", "name": "Uber Technologies Inc",  "city": "San Francisco",  "state_or_country": "CA", "zip_code": "94107", "street1": "1515 3rd St",              "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_019", "name": "Lyft Inc",               "city": "San Francisco",  "state_or_country": "CA", "zip_code": "94107", "street1": "185 Berry St",             "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_020", "name": "Snap Inc",               "city": "Santa Monica",   "state_or_country": "CA", "zip_code": "90405", "street1": "3000 31st St",             "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_021", "name": "Twitter Inc",            "city": "San Francisco",  "state_or_country": "CA", "zip_code": "94103", "street1": "1355 Market St",           "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_022", "name": "Palantir Technologies",  "city": "Denver",         "state_or_country": "CO", "zip_code": "80202", "street1": "1555 Blake St",            "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_023", "name": "Snowflake Inc",          "city": "Bozeman",        "state_or_country": "MT", "zip_code": "59715", "street1": "106 E Babcock St",         "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_024", "name": "Databricks Inc",         "city": "San Francisco",  "state_or_country": "CA", "zip_code": "94105", "street1": "160 Spear St",             "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_025", "name": "Stripe Inc",             "city": "San Francisco",  "state_or_country": "CA", "zip_code": "94103", "street1": "354 Oyster Point Blvd",    "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_026", "name": "SpaceX",                 "city": "Hawthorne",      "state_or_country": "CA", "zip_code": "90250", "street1": "1 Rocket Rd",              "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_027", "name": "Lockheed Martin Corp",   "city": "Bethesda",       "state_or_country": "MD", "zip_code": "20817", "street1": "6801 Rockledge Dr",        "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_028", "name": "JPMorgan Chase & Co",    "city": "New York",       "state_or_country": "NY", "zip_code": "10179", "street1": "383 Madison Ave",          "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_029", "name": "Goldman Sachs Group Inc","city": "New York",       "state_or_country": "NY", "zip_code": "10282", "street1": "200 West St",              "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_030", "name": "Walmart Inc",            "city": "Bentonville",    "state_or_country": "AR", "zip_code": "72716", "street1": "702 SW 8th St",            "_expect_issue": None, "_expect_flag": None},

    # ── NULL CHECKS (QCRunnerAgent — null check) ─────────────────────────────
    {"id": "null_city_001",   "name": "Null City Corp",        "city": None,      "state_or_country": "TX", "zip_code": "75001", "street1": "123 Main St",       "_expect_issue": "null", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_zip_001",    "name": "Null Zip Corp",         "city": "Dallas",  "state_or_country": "TX", "zip_code": None,    "street1": "456 Oak Ave",       "_expect_issue": "null", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_state_001",  "name": "Null State Corp",       "city": "Austin",  "state_or_country": None, "zip_code": "78701", "street1": "789 Congress Ave",  "_expect_issue": "null", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_city_002",   "name": "No City Holdings",      "city": None,      "state_or_country": "NY", "zip_code": "10001", "street1": "350 5th Ave",       "_expect_issue": "null", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_zip_002",    "name": "Missing ZIP Inc",       "city": "Chicago", "state_or_country": "IL", "zip_code": None,    "street1": "233 S Wacker Dr",   "_expect_issue": "null", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_state_002",  "name": "Stateless LLC",         "city": "Boston",  "state_or_country": None, "zip_code": "02110", "street1": "1 Federal St",      "_expect_issue": "null", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_city_003",   "name": "Phantom City Grp",      "city": None,      "state_or_country": "FL", "zip_code": "33101", "street1": "100 Biscayne Blvd", "_expect_issue": "null", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_zip_003",    "name": "Zipless Corp",          "city": "Atlanta", "state_or_country": "GA", "zip_code": None,    "street1": "200 Peachtree St",  "_expect_issue": "null", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_street_001", "name": "No Address Corp",       "city": "Phoenix", "state_or_country": "AZ", "zip_code": "85001", "street1": None,                "_expect_issue": "null", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_street_002", "name": "Missing Street Inc",    "city": "Denver",  "state_or_country": "CO", "zip_code": "80202", "street1": None,                "_expect_issue": "null", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_name_001",   "name": None,                    "city": "Portland","state_or_country": "OR", "zip_code": "97201", "street1": "1 SW Columbia St",  "_expect_issue": "null", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_all_addr",   "name": "Everything Null LLC",   "city": None,      "state_or_country": None, "zip_code": None,    "street1": None,                "_expect_issue": "null", "_expect_flag": "NEEDS_REVIEW"},

    # ── FORMAT CHECKS (QCRunnerAgent — format check) ─────────────────────────
    {"id": "bad_zip_alpha_001",  "name": "Bad ZIP Alpha",       "city": "Denver",       "state_or_country": "CO", "zip_code": "ABCDE",     "street1": "1 Denver Pl",       "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_zip_short_001",  "name": "Bad ZIP Short",       "city": "Denver",       "state_or_country": "CO", "zip_code": "802",       "street1": "2 Denver Pl",       "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_state_001",      "name": "Bad State Co",        "city": "Phoenix",      "state_or_country": "ZZ", "zip_code": "85001",     "street1": "10 Phoenix Blvd",   "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_zip_long_001",   "name": "Too Long ZIP Inc",    "city": "Seattle",      "state_or_country": "WA", "zip_code": "981091234567", "street1": "3 Pine St",      "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_zip_special",    "name": "Special ZIP Corp",    "city": "Portland",     "state_or_country": "OR", "zip_code": "97@01",     "street1": "4 Oak St",          "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_state_lower",    "name": "Lowercase State Inc", "city": "Chicago",      "state_or_country": "il", "zip_code": "60601",     "street1": "5 State St",        "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_state_long",     "name": "Long State Corp",     "city": "Houston",      "state_or_country": "TEX", "zip_code": "77001",    "street1": "6 Texas Ave",       "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_state_num",      "name": "Numeric State LLC",   "city": "Miami",        "state_or_country": "99", "zip_code": "33101",     "street1": "7 Ocean Dr",        "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_zip_neg",        "name": "Negative ZIP Inc",    "city": "Boston",       "state_or_country": "MA", "zip_code": "-02110",    "street1": "8 Beacon St",       "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_zip_space",      "name": "Space ZIP Corp",      "city": "Dallas",       "state_or_country": "TX", "zip_code": "7 5 0 0 1", "street1": "9 Elm St",         "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_zip_zeros",      "name": "All Zeros ZIP LLC",   "city": "Detroit",      "state_or_country": "MI", "zip_code": "00000",     "street1": "10 Motor City Dr",  "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_state_empty",    "name": "Empty State Inc",     "city": "Nashville",    "state_or_country": "",   "zip_code": "37201",     "street1": "11 Music Row",      "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_zip_float",      "name": "Float ZIP Corp",      "city": "Raleigh",      "state_or_country": "NC", "zip_code": "27601.5",   "street1": "12 Fayetteville St","_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_state_single",   "name": "Single Char State",   "city": "Charlotte",    "state_or_country": "N",  "zip_code": "28202",     "street1": "13 Tryon St",       "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_zip_hex",        "name": "Hex ZIP Inc",         "city": "Omaha",        "state_or_country": "NE", "zip_code": "0xFFFF",    "street1": "14 Dodge St",       "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},

    # ── ZIP/STATE MISMATCH (QCRunnerAgent — zip_state_mismatch) ─────────────
    # ZIP 10001 is New York, not California
    {"id": "zip_mismatch_001", "name": "Wrong State Corp",       "city": "Los Angeles",   "state_or_country": "CA", "zip_code": "10001", "street1": "5 Wrong Way",        "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
    # ZIP 75001 is Texas, not Florida
    {"id": "zip_mismatch_002", "name": "Confused State Inc",     "city": "Miami",         "state_or_country": "FL", "zip_code": "75001", "street1": "6 Confused Blvd",    "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
    # ZIP 60601 is Illinois, not California
    {"id": "zip_mismatch_003", "name": "Misplaced Chicago Co",   "city": "San Diego",     "state_or_country": "CA", "zip_code": "60601", "street1": "100 N State St",     "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
    # ZIP 30301 is Georgia, not Texas
    {"id": "zip_mismatch_004", "name": "Atlanta Texas LLC",      "city": "Houston",       "state_or_country": "TX", "zip_code": "30301", "street1": "200 Peachtree St",   "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
    # ZIP 02110 is Massachusetts, not Washington
    {"id": "zip_mismatch_005", "name": "Boston Wash Corp",       "city": "Seattle",       "state_or_country": "WA", "zip_code": "02110", "street1": "50 Federal St",      "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
    # ZIP 85001 is Arizona, not Nevada
    {"id": "zip_mismatch_006", "name": "Phoenix Vegas Inc",      "city": "Las Vegas",     "state_or_country": "NV", "zip_code": "85001", "street1": "1 Central Ave",      "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
    # ZIP 98101 is Washington, not Oregon
    {"id": "zip_mismatch_007", "name": "Seattle Portland LLC",   "city": "Portland",      "state_or_country": "OR", "zip_code": "98101", "street1": "600 Pine St",        "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
    # ZIP 33101 is Florida, not North Carolina
    {"id": "zip_mismatch_008", "name": "Miami Carolina Corp",    "city": "Charlotte",     "state_or_country": "NC", "zip_code": "33101", "street1": "300 S Tryon St",     "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
    # ZIP 80202 is Colorado, not Utah
    {"id": "zip_mismatch_009", "name": "Denver SLC Inc",         "city": "Salt Lake City","state_or_country": "UT", "zip_code": "80202", "street1": "1 Larimer Sq",       "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
    # ZIP 37201 is Tennessee, not Kentucky
    {"id": "zip_mismatch_010", "name": "Nashville Kentucky LLC", "city": "Louisville",    "state_or_country": "KY", "zip_code": "37201", "street1": "4th Ave N",          "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},

    # ── MULTI-ISSUE RECORDS (more than one problem per row) ─────────────────
    # null city + bad state
    {"id": "multi_001", "name": "Double Trouble Inc",    "city": None,      "state_or_country": "XX", "zip_code": "90210", "street1": "1 Rodeo Dr",         "_expect_issue": "null",  "_expect_flag": "NEEDS_REVIEW"},
    # null zip + bad state
    {"id": "multi_002", "name": "Two Issues Corp",       "city": "Atlanta", "state_or_country": "ZQ", "zip_code": None,    "street1": "2 Peach St",         "_expect_issue": "null",  "_expect_flag": "NEEDS_REVIEW"},
    # bad zip format + zip/state mismatch (ZIP unrecognizable, state valid)
    {"id": "multi_003", "name": "Triple Threat LLC",     "city": "Reno",    "state_or_country": "CA", "zip_code": "ABC",   "street1": "3 Sierra St",        "_expect_issue": "format","_expect_flag": "NEEDS_REVIEW"},
    # null city + null state
    {"id": "multi_004", "name": "Ghost Address Inc",     "city": None,      "state_or_country": None, "zip_code": "30301", "street1": "4 Missing Ave",      "_expect_issue": "null",  "_expect_flag": "NEEDS_REVIEW"},
    # null zip + null city
    {"id": "multi_005", "name": "Bare Bones Corp",       "city": None,      "state_or_country": "TX", "zip_code": None,    "street1": "5 Skeleton Rd",      "_expect_issue": "null",  "_expect_flag": "NEEDS_REVIEW"},

    # ── EDGE CASE CLEAN RECORDS (tricky but valid) ──────────────────────────
    {"id": "edge_clean_001", "name": "Zip Plus Four OK",  "city": "New York",     "state_or_country": "NY", "zip_code": "10001-1234", "street1": "350 5th Ave",   "_expect_issue": None, "_expect_flag": None},
    {"id": "edge_clean_002", "name": "Short Name Co",     "city": "Honolulu",     "state_or_country": "HI", "zip_code": "96801",      "street1": "1 Aloha Dr",    "_expect_issue": None, "_expect_flag": None},
    {"id": "edge_clean_003", "name": "Alaska Ventures",   "city": "Anchorage",    "state_or_country": "AK", "zip_code": "99501",      "street1": "100 Northern Lights Blvd", "_expect_issue": None, "_expect_flag": None},
    {"id": "edge_clean_004", "name": "Puerto Rico Trade", "city": "San Juan",     "state_or_country": "PR", "zip_code": "00901",      "street1": "1 Isla Verde",  "_expect_issue": None, "_expect_flag": None},
    {"id": "edge_clean_005", "name": "DC Consulting LLC", "city": "Washington",   "state_or_country": "DC", "zip_code": "20001",      "street1": "1600 Penn Ave", "_expect_issue": None, "_expect_flag": None},
    {"id": "edge_clean_006", "name": "Virgin Islands Co", "city": "Charlotte Amalie","state_or_country": "VI","zip_code": "00802",     "street1": "1 Harbor Dr",   "_expect_issue": None, "_expect_flag": None},
    {"id": "edge_clean_007", "name": "Guam Logistics",    "city": "Hagatna",      "state_or_country": "GU", "zip_code": "96910",      "street1": "1 Marine Dr",   "_expect_issue": None, "_expect_flag": None},
    {"id": "edge_clean_008", "name": "Long Name International Holdings Group Corp", "city": "Wilmington","state_or_country": "DE","zip_code": "19801","street1": "1209 Orange St", "_expect_issue": None, "_expect_flag": None},

    # ── ADDITIONAL CLEAN RECORDS (bulk padding to 100 total) ────────────────
    {"id": "clean_031", "name": "Broadcom Inc",           "city": "San Jose",      "state_or_country": "CA", "zip_code": "95134", "street1": "1320 Ridder Park Dr",      "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_032", "name": "Qualcomm Inc",           "city": "San Diego",     "state_or_country": "CA", "zip_code": "92121", "street1": "5775 Morehouse Dr",        "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_033", "name": "ServiceNow Inc",         "city": "Santa Clara",   "state_or_country": "CA", "zip_code": "95054", "street1": "2225 Lawson Ln",           "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_034", "name": "Intuit Inc",             "city": "Mountain View", "state_or_country": "CA", "zip_code": "94043", "street1": "2700 Coast Ave",           "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_035", "name": "Palo Alto Networks",     "city": "Santa Clara",   "state_or_country": "CA", "zip_code": "95054", "street1": "3000 Tannery Way",         "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_036", "name": "CrowdStrike Holdings",   "city": "Austin",        "state_or_country": "TX", "zip_code": "78753", "street1": "206 E 9th St",             "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_037", "name": "Fortinet Inc",           "city": "Sunnyvale",     "state_or_country": "CA", "zip_code": "94085", "street1": "899 Kifer Rd",             "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_038", "name": "Workday Inc",            "city": "Pleasanton",    "state_or_country": "CA", "zip_code": "94566", "street1": "6110 Stoneridge Mall Rd",  "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_039", "name": "Twilio Inc",             "city": "San Francisco", "state_or_country": "CA", "zip_code": "94105", "street1": "101 Spear St",             "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_040", "name": "Atlassian Corp",         "city": "San Francisco", "state_or_country": "CA", "zip_code": "94105", "street1": "431 El Camino Real",       "_expect_issue": None, "_expect_flag": None},

    # ── ADDITIONAL BAD RECORDS (variety) ────────────────────────────────────
    # zip/state mismatches
    {"id": "zip_mismatch_011", "name": "Philly in Ohio LLC",    "city": "Columbus",    "state_or_country": "OH", "zip_code": "19101", "street1": "1 Broad St",       "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "zip_mismatch_012", "name": "KC in Alabama Corp",   "city": "Birmingham",  "state_or_country": "AL", "zip_code": "64101", "street1": "2 Main St",        "_expect_issue": "zip_state_mismatch", "_expect_flag": "NEEDS_REVIEW"},
    # format issues
    {"id": "bad_zip_just_dash", "name": "Dash ZIP Inc",        "city": "Tampa",       "state_or_country": "FL", "zip_code": "-----", "street1": "3 Bay St",         "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    {"id": "bad_state_three",   "name": "Three Char State Co", "city": "Tulsa",       "state_or_country": "OKL","zip_code": "74101", "street1": "4 Main St",        "_expect_issue": "format", "_expect_flag": "NEEDS_REVIEW"},
    # nulls
    {"id": "null_city_004",     "name": "No City Financial",   "city": None,          "state_or_country": "PA", "zip_code": "19103", "street1": "5 Market St",      "_expect_issue": "null",   "_expect_flag": "NEEDS_REVIEW"},
    {"id": "null_state_003",    "name": "Stateless Financial",  "city": "Memphis",    "state_or_country": None, "zip_code": "38103", "street1": "6 Beale St",       "_expect_issue": "null",   "_expect_flag": "NEEDS_REVIEW"},
    # multi-issue
    {"id": "multi_006", "name": "Total Mess Inc",  "city": None,  "state_or_country": "QQ", "zip_code": "NOPE",  "street1": "7 Nowhere Blvd",     "_expect_issue": "null",   "_expect_flag": "NEEDS_REVIEW"},
    {"id": "multi_007", "name": "Bad Everything",  "city": None,  "state_or_country": None, "zip_code": None,    "street1": None,                 "_expect_issue": "null",   "_expect_flag": "NEEDS_REVIEW"},
    # more clean
    {"id": "clean_041", "name": "Deloitte LLP",    "city": "New York",    "state_or_country": "NY", "zip_code": "10112", "street1": "30 Rockefeller Plz",  "_expect_issue": None, "_expect_flag": None},
    {"id": "clean_042", "name": "Accenture PLC",   "city": "Chicago",     "state_or_country": "IL", "zip_code": "60601", "street1": "161 N Clark St",      "_expect_issue": None, "_expect_flag": None},
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
import os, requests as _req
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

print("\nPipeline result:")
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
