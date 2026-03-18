# Databricks notebook source
# notebooks/03_feature_dev_template.py
# ─────────────────────────────────────────────────────────────────────────────
# New Feature Category Development Template
#
# Use this notebook to:
#   1. Explore raw data interactively
#   2. Develop and test new feature logic on a sample
#   3. Copy the final compute() logic into a new cat0N_*.py file
#
# Steps:
#   1. Set CATEGORY_NAME and CATEGORY_NUMBER below
#   2. Write your feature logic in the "DEVELOP FEATURES HERE" cell
#   3. Validate output schema and PK uniqueness
#   4. Copy logic to features/tradeline/cat0N_your_category.py
#   5. Register in features/registry.py
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------
import sys
REPO_ROOT = "/Workspace/Repos/<your-username>/experian_feature_store"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------
# ── CONFIGURE YOUR NEW CATEGORY ──────────────────────────────────────────────

CATEGORY_NUMBER = "05"           # e.g. "05"
CATEGORY_NAME   = "my_category"  # e.g. "write_off_analysis"
SOURCE          = "tradeline"    # "tradeline" or "enquiry"

# Development scrub date (use a recent one for fast iteration)
DEV_SCRUB_DATE  = "2024-04-18"
SAMPLE_SIZE     = 5000           # rows to load for dev/test

print(f"Developing: cat{CATEGORY_NUMBER}_{CATEGORY_NAME} | source={SOURCE}")

# COMMAND ----------
# Load a sample of raw data

from config import config
import pyspark.sql.functions as F

table = config.TRADELINE_TABLE if SOURCE == "tradeline" else config.ENQUIRY_TABLE
pk_cols   = config.SCRUB_PK_COLS
as_of_col = config.SCRUB_OUTPUT_DATE_COL

sample_df = (
    spark.table(table)
    .filter(F.col("scrub_output_date") == DEV_SCRUB_DATE)
    .limit(SAMPLE_SIZE)
)

print(f"Sample loaded: {sample_df.count()} rows | {len(sample_df.columns)} cols")
sample_df.printSchema()

# COMMAND ----------
# Explore key columns

sample_df.select(
    "customer_scrub_key", "scrub_id", "scrub_output_date",
    # Add columns relevant to your category here
).show(10)

# COMMAND ----------
# ── DEVELOP FEATURES HERE ─────────────────────────────────────────────────────
# Write your feature derivation logic below.
# When done, copy this into compute() in your new cat0N_*.py file.

group_cols = pk_cols + [as_of_col]

# Example: simple aggregation
feature_df = sample_df.groupBy(group_cols).agg(
    F.count("*").alias("my_feature_row_count"),
    # Add your feature expressions here
)

feature_df.show(5)
print(f"Feature output: {feature_df.count()} rows | {len(feature_df.columns)} cols")

# COMMAND ----------
# ── VALIDATE OUTPUT ───────────────────────────────────────────────────────────

from core.utils import validate_pk_uniqueness

validate_pk_uniqueness(feature_df, group_cols, context=f"cat{CATEGORY_NUMBER}_{CATEGORY_NAME}")
feature_df.printSchema()

# COMMAND ----------
# ── COPY THIS INTO YOUR NEW FILE ─────────────────────────────────────────────
# Once satisfied with your logic, create the file:
#
# features/{SOURCE}/cat{CATEGORY_NUMBER}_{CATEGORY_NAME}.py
#
# Use this template:

template = f'''
# features/{SOURCE}/cat{CATEGORY_NUMBER}_{CATEGORY_NAME}.py

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.{"tradeline" if SOURCE == "tradeline" else "enquiry"}.base import {"TradelineFeatureBase" if SOURCE == "tradeline" else "EnquiryFeatureBase"}
from core.logger import get_logger

logger = get_logger(__name__)


class {"".join(w.capitalize() for w in CATEGORY_NAME.split("_"))}Features({"TradelineFeatureBase" if SOURCE == "tradeline" else "EnquiryFeatureBase"}):
    """
    Category {CATEGORY_NUMBER}: {CATEGORY_NAME.replace("_", " ").title()} Features
    TODO: add description
    """

    CATEGORY = "cat{CATEGORY_NUMBER}_{CATEGORY_NAME}"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        # ── YOUR FEATURE LOGIC HERE ───────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(
            F.count("*").alias("my_feature_row_count"),
            # Add your aggregations
        )

        self._log_done(feature_df)
        return feature_df
'''

print(template)

# COMMAND ----------
# ── REGISTER REMINDER ────────────────────────────────────────────────────────
print(f"""
Once your file is created, add this to features/registry.py:

  from features.{SOURCE}.cat{CATEGORY_NUMBER}_{CATEGORY_NAME} import {"".join(w.capitalize() for w in CATEGORY_NAME.split("_"))}Features

  {"TRADELINE" if SOURCE == "tradeline" else "ENQUIRY"}_FEATURE_REGISTRY.append(
      {"".join(w.capitalize() for w in CATEGORY_NAME.split("_"))}Features()
  )
""")
