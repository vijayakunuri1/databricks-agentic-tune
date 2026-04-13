"""Delta table schema definitions."""
from __future__ import annotations

try:
    from pyspark.sql.types import (
        BooleanType,
        FloatType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

# Column definitions as plain dicts so they can be used without Spark too
QC_AUDIT_COLUMN_DEFS = [
    ("qc_status", "string"),           # CLEAN | AUTO_CORRECTED | NEEDS_REVIEW | FAILED
    ("qc_flag", "string"),             # NULL_VALUE | FORMAT_ERROR | DUPLICATE | GEO_TYPO | …
    ("qc_corrected_column", "string"),
    ("qc_original_value", "string"),
    ("qc_corrected_value", "string"),
    ("qc_confidence_score", "float"),
    ("qc_correction_method", "string"),
    ("qc_support_level", "string"),    # L1 | L2
    ("qc_run_id", "string"),
    ("qc_batch_id", "string"),
    ("qc_checked_at", "timestamp"),
    ("qc_corrected_at", "timestamp"),
    ("qc_agent_version", "string"),
    ("qc_all_issues", "string"),       # JSON blob of all issues on this record
    ("qc_human_reviewed", "boolean"),
    ("qc_human_decision", "string"),   # ACCEPT | REJECT | MANUAL_EDIT
    ("qc_human_reviewed_at", "timestamp"),
    ("qc_human_reviewer", "string"),
]

QC_AUDIT_LOG_SCHEMA_DEF = [
    ("audit_id", "string", False),
    ("run_id", "string", False),
    ("batch_id", "string", False),
    ("table_fqn", "string", False),
    ("record_id", "string", False),
    ("column_name", "string", True),
    ("check_type", "string", False),
    ("original_value", "string", True),
    ("corrected_value", "string", True),
    ("confidence_score", "float", True),
    ("support_level", "string", True),
    ("was_applied", "boolean", True),
    ("correction_method", "string", True),
    ("llm_reasoning", "string", True),
    ("created_at", "timestamp", False),
]


if SPARK_AVAILABLE:
    _TYPE_MAP = {
        "string": StringType(),
        "float": FloatType(),
        "boolean": BooleanType(),
        "timestamp": TimestampType(),
    }

    def _to_struct_field(name: str, type_str: str, nullable: bool = True) -> "StructField":
        return StructField(name, _TYPE_MAP[type_str], nullable)

    QC_AUDIT_COLUMNS: list["StructField"] = [
        _to_struct_field(name, type_str)
        for name, type_str in QC_AUDIT_COLUMN_DEFS
    ]

    QC_AUDIT_LOG_SCHEMA = StructType([
        _to_struct_field(name, type_str, nullable)
        for name, type_str, nullable in QC_AUDIT_LOG_SCHEMA_DEF
    ])
