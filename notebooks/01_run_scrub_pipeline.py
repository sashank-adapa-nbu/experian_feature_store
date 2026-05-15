# Databricks notebook source
# notebooks/01_run_scrub_pipeline.py  [OPTIMISED]
# =============================================================================
# Scrub Pipeline — Databricks Job Entry Point
#
# Parameters (Job widgets):
#   scrub_output_date : YYYY-MM-DD  → process that single date
#                       ALL         → process all unprocessed dates (incremental)
#                       FULL        → reprocess ALL dates (full refresh)
#   start_date        : YYYY-MM-DD  → for date-range mode (used with end_date)
#   end_date          : YYYY-MM-DD  → for date-range mode (used with start_date)
#
# ARCHITECTURE NOTE — Sequential per-date processing:
#   With 100+ scrubs × 500M records each, loading all data at once (50B+ rows)
#   would crash any cluster.  This pipeline processes ONE scrub date at a time:
#     1. Load ~500M rows for date D
#     2. Compute all 365 features
#     3. Write to Delta (partition overwrite — idempotent, restartable)
#     4. Clear executor cache
#     5. Move to date D+1
#
#   This keeps peak memory at ~100GB (one scrub), not 10TB (all scrubs).
#   Each scrub partition is written atomically — failure at any date
#   leaves all previously processed dates intact and can resume from that date.
#
# Schedule example (Databricks Jobs UI):
#   Parameter: scrub_output_date = ALL
# =============================================================================

# COMMAND ----------

import sys
REPO_ROOT = "/Workspace/Users/sashanknagasai.adapa@angelone.in/experian_feature_store" 
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------

dbutils.widgets.text("scrub_output_date", "ALL",
    "Date (YYYY-MM-DD), ALL (incremental), FULL (reprocess all), or RANGE")
dbutils.widgets.text("start_date", "", "Start date for RANGE mode (YYYY-MM-DD)")
dbutils.widgets.text("end_date",   "", "End date for RANGE mode (YYYY-MM-DD)")

param      = dbutils.widgets.get("scrub_output_date").strip().upper()
start_date = dbutils.widgets.get("start_date").strip()
end_date   = dbutils.widgets.get("end_date").strip()
print(f"scrub_output_date = '{param}' | start={start_date} | end={end_date}")

# COMMAND ----------

from pipeline.scrub_pipeline import ScrubPipeline
from core.logger import get_logger

logger   = get_logger("nb01_scrub")
pipeline = ScrubPipeline(spark)  # applies Spark conf tuning at init

if param == "ALL":
    logger.info("Mode: incremental — skip already-processed dates")
    pipeline.run_all(skip_processed=True)

elif param == "FULL":
    logger.info("Mode: full refresh — reprocess ALL dates")
    pipeline.run_all(skip_processed=False)

elif param == "RANGE":
    assert start_date and end_date, "start_date and end_date required for RANGE mode"
    logger.info(f"Mode: date range {start_date} → {end_date}")
    pipeline.run_date_range(start_date, end_date, skip_processed=True)

else:
    # Specific single date
    logger.info(f"Mode: single date — {param}")
    pipeline.run(scrub_date=param)

logger.info("Notebook 01 complete.")

# COMMAND ----------

# Validation — show row counts per scrub date (metadata-only, no full scan)
# =============================================================================
# The previous version did df.groupBy("scrub_output_date").count() which
# triggers a FULL DATA SCAN of the output table at the end of a multi-hour
# job. If the driver was already memory-pressured from the run, this final
# scan could crash the notebook and mark the whole job as failed even though
# all data was successfully written.
#
# We now use Delta's transaction log via SHOW PARTITIONS — reads ~ms regardless
# of table size, no data scan, no shuffle, no driver memory pressure.
# Row counts per partition come from the Delta commit log via DESCRIBE DETAIL,
# which is also metadata-only.
# =============================================================================
from config import config

tl_tbl  = f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}.{config.TRADELINE_FEATURE_TABLE_PREFIX}_scrub"
enq_tbl = f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}.{config.ENQUIRY_FEATURE_TABLE_PREFIX}_scrub"

def _summarise_table(tbl: str, label: str):
    """Print partition list + total row count, all from Delta metadata only."""
    try:
        # Schema info — cheap, just reads catalog metadata
        df_schema = spark.table(tbl)
        n_cols = len(df_schema.columns)

        # Partition list — reads Delta transaction log, milliseconds
        parts = spark.sql(f"SHOW PARTITIONS {tbl}").collect()
        n_parts = len(parts)

        # Total table-level metrics — reads Delta DESCRIBE DETAIL (metadata-only)
        detail = spark.sql(f"DESCRIBE DETAIL {tbl}").collect()[0].asDict()
        # numFiles / sizeInBytes are always populated; numRecords is too for
        # delta tables written by recent Databricks runtimes
        size_gb   = round(detail.get("sizeInBytes", 0) / (1024 ** 3), 2)
        num_files = detail.get("numFiles", "n/a")

        print(f"\n{label} → {tbl}")
        print(f"  Columns          : {n_cols}")
        print(f"  Partitions       : {n_parts}")
        print(f"  Files            : {num_files}")
        print(f"  Size on disk (GB): {size_gb}")
        print(f"  Partitions list (first 20):")
        for p in parts[:20]:
            print(f"    {p[0]}")
        if n_parts > 20:
            print(f"    ... and {n_parts - 20} more")
    except Exception as e:
        print(f"  Could not summarise {tbl}: {e}")

for tbl, label in [(tl_tbl, "Tradeline"), (enq_tbl, "Enquiry")]:
    _summarise_table(tbl, label)


