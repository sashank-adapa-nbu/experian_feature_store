# Databricks notebook source
# notebooks/01_run_scrub_pipeline.py
# =============================================================================
# Scrub Pipeline — Databricks Job Entry Point
#
# Parameters (Job widgets):
#   scrub_output_date : YYYY-MM-DD  → process that single date
#                       ALL         → process all unprocessed dates (incremental)
#                       FULL        → reprocess ALL dates (full refresh)
#
# Schedule example (Databricks Jobs UI):
#   Trigger  : Every Sunday 02:00 AM
#   Parameter: scrub_output_date = ALL
# =============================================================================

# COMMAND ----------
import sys
REPO_ROOT = "/Workspace/Repos/<your-username>/experian_feature_store"  # ← update
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------
dbutils.widgets.text(
    "scrub_output_date",
    "ALL",
    "Date (YYYY-MM-DD), ALL (incremental), or FULL (reprocess all)"
)
param = dbutils.widgets.get("scrub_output_date").strip().upper()
print(f"scrub_output_date = '{param}'")

# COMMAND ----------
from pipeline.scrub_pipeline import ScrubPipeline
from core.logger import get_logger

logger  = get_logger("nb01_scrub")
pipeline = ScrubPipeline(spark)

if param == "ALL":
    logger.info("Mode: incremental — processing all unprocessed scrub dates")
    pipeline.run_all(skip_processed=True)

elif param == "FULL":
    logger.info("Mode: full refresh — reprocessing ALL scrub dates")
    pipeline.run_all(skip_processed=False)

else:
    # Specific date passed
    logger.info(f"Mode: single date — {param}")
    pipeline.run(scrub_date=param)

logger.info("Notebook 01 complete.")

# COMMAND ----------
# Validation — show row counts per scrub date in output
from config import config

tl_tbl = f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}.{config.TRADELINE_FEATURE_TABLE_PREFIX}_scrub"
enq_tbl = f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}.{config.ENQUIRY_FEATURE_TABLE_PREFIX}_scrub"

for tbl, label in [(tl_tbl, "Tradeline"), (enq_tbl, "Enquiry")]:
    try:
        df = spark.table(tbl)
        print(f"\n{label} → {tbl}")
        print(f"  Total rows : {df.count()}")
        print(f"  Columns    : {len(df.columns)}")
        print("  Rows per scrub date:")
        df.groupBy("scrub_output_date").count().orderBy("scrub_output_date").show(50, False)
    except Exception as e:
        print(f"  Could not read {tbl}: {e}")
