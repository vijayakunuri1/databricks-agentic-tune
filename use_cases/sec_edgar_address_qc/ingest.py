"""
Ingest SEC EDGAR company data and GeoNames reference data.

Run this once before the QC pipeline:
    python -m use_cases.sec_edgar_address_qc.ingest
    python -m use_cases.sec_edgar_address_qc.ingest --databricks  # writes to Delta
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

DATA_DIR = Path("./data")


def ingest_geonames(cache_path: str, spark=None, delta_table: str | None = None) -> "GeonamesIndex":
    from data_ingestion.geonames_fetcher import GeonamesIndex

    logger.info("Ingesting GeoNames US postal code data...")
    index = GeonamesIndex.download(cache_path=cache_path)

    if spark and delta_table:
        logger.info("Writing GeoNames to Delta: %s", delta_table)
        sdf = spark.createDataFrame(index._df)
        sdf.write.format("delta").mode("overwrite").saveAsTable(delta_table)
        logger.info("GeoNames written to %s (%d rows)", delta_table, len(index._df))

    return index


def ingest_sec_edgar(
    limit: int,
    cache_path: str,
    spark=None,
    delta_table: str | None = None,
) -> pd.DataFrame:
    from data_ingestion.sec_edgar_fetcher import SECEdgarFetcher

    logger.info("Ingesting SEC EDGAR company data (limit=%d)...", limit)
    fetcher = SECEdgarFetcher()
    df = fetcher.fetch_companies(limit=limit, us_only=True, cache_path=cache_path)

    # Ensure 'id' column exists (used as primary key)
    if "id" not in df.columns:
        df["id"] = df["cik"].astype(str)

    logger.info("SEC EDGAR: %d companies loaded.", len(df))
    logger.info("Sample address issues (first 5 rows):")
    logger.info(df[["name", "city", "state_or_country", "zip_code"]].head(5).to_string())

    if spark and delta_table:
        # Create schema if needed
        catalog, schema, table = delta_table.split(".")
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
        logger.info("Writing SEC EDGAR data to Delta: %s", delta_table)
        sdf = spark.createDataFrame(df.astype(str).fillna(""))
        sdf.write.format("delta").mode("overwrite").saveAsTable(delta_table)
        logger.info("SEC EDGAR written to %s", delta_table)

    return df


def main():
    parser = argparse.ArgumentParser(description="Ingest SEC EDGAR + GeoNames data")
    parser.add_argument("--limit", type=int, default=500, help="Max companies to fetch")
    parser.add_argument("--databricks", action="store_true", help="Write to Databricks Delta")
    parser.add_argument("--catalog", default="main")
    parser.add_argument("--schema", default="sec_edgar")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    spark = None
    if args.databricks:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

    # 1. GeoNames
    geo_cache = str(DATA_DIR / "geonames_us_postal.parquet")
    geo_delta = f"{args.catalog}.reference.us_postal_codes" if args.databricks else None
    geonames_index = ingest_geonames(geo_cache, spark=spark, delta_table=geo_delta)

    # 2. SEC EDGAR
    edgar_cache = str(DATA_DIR / "sec_edgar_companies.parquet")
    edgar_delta = f"{args.catalog}.{args.schema}.companies" if args.databricks else None
    df = ingest_sec_edgar(
        limit=args.limit,
        cache_path=edgar_cache,
        spark=spark,
        delta_table=edgar_delta,
    )

    logger.info("")
    logger.info("━━━━━━━━━━━━ INGESTION COMPLETE ━━━━━━━━━━━━")
    logger.info("  SEC EDGAR companies: %d rows → %s", len(df), edgar_cache)
    logger.info("  GeoNames ZIPs:       %d entries → %s", len(geonames_index._zip_to_primary), geo_cache)
    logger.info("  Next step: run the QC pipeline")
    logger.info("    python -m use_cases.sec_edgar_address_qc.run_qc")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
