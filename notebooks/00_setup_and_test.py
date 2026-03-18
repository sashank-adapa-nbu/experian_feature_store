# Databricks notebook source
# notebooks/00_setup_and_test.py
# ─────────────────────────────────────────────────────────────────────────────
# Setup & Environment Check Notebook
#
# Run this first to:
#   1. Verify the repo is on sys.path
#   2. Check source table accessibility
#   3. Validate feature registry loads correctly
#   4. Run a quick smoke test on a small sample
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------
# Add project root to sys.path so all modules can be imported
# Adjust this path to where you cloned / uploaded the repo in Databricks

import sys
import os

REPO_ROOT = "/Workspace/Repos/<your-username>/experian_feature_store"  # ← Update this

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    print(f"✓ Added {REPO_ROOT} to sys.path")

# COMMAND ----------
# Test imports

from config import config
from core.logger import get_logger
from core.utils import get_all_scrub_dates
from features.registry import TRADELINE_FEATURE_REGISTRY, ENQUIRY_FEATURE_REGISTRY

logger = get_logger("setup_test")
logger.info("✓ All imports successful")

# COMMAND ----------
# Show registered feature categories

print(f"\n{'='*60}")
print(f"Tradeline Feature Categories ({len(TRADELINE_FEATURE_REGISTRY)}):")
for cat in TRADELINE_FEATURE_REGISTRY:
    print(f"  • {cat.CATEGORY}")

print(f"\nEnquiry Feature Categories ({len(ENQUIRY_FEATURE_REGISTRY)}):")
for cat in ENQUIRY_FEATURE_REGISTRY:
    print(f"  • {cat.CATEGORY}")
print(f"{'='*60}")

# COMMAND ----------
# Check source table accessibility

print("\nChecking source tables...")

for table_name in [config.TRADELINE_TABLE, config.ENQUIRY_TABLE, config.MASTER_TABLE]:
    try:
        count = spark.table(table_name).limit(1).count()
        print(f"  ✓ {table_name} — accessible")
    except Exception as e:
        print(f"  ✗ {table_name} — ERROR: {e}")

# COMMAND ----------
# List available scrub dates

print("\nAvailable scrub dates:")
try:
    dates = get_all_scrub_dates(spark, config.TRADELINE_TABLE)
    for d in dates[:10]:
        print(f"  {d}")
    if len(dates) > 10:
        print(f"  ... ({len(dates)} total)")
except Exception as e:
    print(f"  ERROR: {e}")

# COMMAND ----------
# Smoke test: run one category on a small sample

print("\n--- Smoke Test ---")
try:
    from features.tradeline.cat01_account_summary import AccountSummaryFeatures

    sample_df = (
        spark.table(config.TRADELINE_TABLE)
        .limit(500)
        .withColumn("scrub_output_date", F.col("scrub_output_date").cast("date"))
    )

    from pyspark.sql import functions as F
    cat = AccountSummaryFeatures()
    result = cat.compute(
        df=sample_df,
        pk_cols=config.SCRUB_PK_COLS,
        as_of_col=config.SCRUB_OUTPUT_DATE_COL,
    )
    print(f"  ✓ Smoke test passed | output rows={result.count()} | cols={len(result.columns)}")
    result.printSchema()
except Exception as e:
    print(f"  ✗ Smoke test FAILED: {e}")
    raise
