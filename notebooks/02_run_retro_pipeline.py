# Databricks notebook source
# notebooks/02_run_retro_pipeline.py
# =============================================================================
# Retro Pipeline — Databricks Job Entry Point
#
# Derives bureau features as of a reference_dt for each party_code.
#
# Input table must have columns: party_code (string), reference_dt (date/string)
#
# Scrub selection window:
#   For each (party_code, reference_dt), picks the LATEST scrub_output_date
#   such that:
#       reference_dt <= scrub_output_date <= reference_dt + retro_window_months
#
# Parameters:
#   source_table           : Fully qualified input table [party_code, reference_dt]
#   tradeline_output_table : Where to write tradeline features
#   enquiry_output_table   : Where to write enquiry features
#   retro_window_months    : Max months after reference_dt to find a scrub (default 12)
# =============================================================================

# COMMAND ----------
import sys
REPO_ROOT = "/Workspace/Repos/<your-username>/experian_feature_store"  # ← update
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COMMAND ----------
dbutils.widgets.text(
    "source_table",
    "",
    "Input table: must have [party_code, reference_dt]"
)
dbutils.widgets.text(
    "tradeline_output_table",
    "uc_dataorg_prod.l3_nbu.<model_name>_tl_features",
    "Output table for tradeline features"
)
dbutils.widgets.text(
    "enquiry_output_table",
    "uc_dataorg_prod.l3_nbu.<model_name>_enq_features",
    "Output table for enquiry features"
)
dbutils.widgets.text(
    "retro_window_months",
    "12",
    "Max months after reference_dt to find a scrub (default 12)"
)

source_table           = dbutils.widgets.get("source_table").strip()
tradeline_output_table = dbutils.widgets.get("tradeline_output_table").strip()
enquiry_output_table   = dbutils.widgets.get("enquiry_output_table").strip()
retro_window_months    = int(dbutils.widgets.get("retro_window_months").strip())

# Validate
assert source_table, "source_table widget must be set to a fully qualified table name"
assert "<model_name>" not in tradeline_output_table, "Set tradeline_output_table widget"
assert "<model_name>" not in enquiry_output_table,   "Set enquiry_output_table widget"

print(f"source_table           : {source_table}")
print(f"tradeline_output_table : {tradeline_output_table}")
print(f"enquiry_output_table   : {enquiry_output_table}")
print(f"retro_window_months    : {retro_window_months}")

# COMMAND ----------
# Preview input
df_preview = spark.table(source_table).select("party_code", "reference_dt").limit(5)
print(f"\nInput table preview:")
df_preview.show()
total_input = spark.table(source_table).select("party_code", "reference_dt").dropDuplicates().count()
print(f"Distinct (party_code, reference_dt) rows: {total_input}")

# COMMAND ----------
from pipeline.retro_pipeline import RetroPipeline
from core.logger import get_logger

logger   = get_logger("nb02_retro")
pipeline = RetroPipeline(spark)

pipeline.run(
    source_table=source_table,
    tradeline_output_table=tradeline_output_table,
    enquiry_output_table=enquiry_output_table,
    retro_max_months=retro_window_months,
)

logger.info("Notebook 02 complete.")

# COMMAND ----------
# Validation
for tbl, label in [(tradeline_output_table, "Tradeline"), (enquiry_output_table, "Enquiry")]:
    try:
        df = spark.table(tbl)
        print(f"\n{label} → {tbl}")
        print(f"  Rows    : {df.count()}")
        print(f"  Columns : {len(df.columns)}")
        df.select("party_code", "reference_dt").show(5, False)
    except Exception as e:
        print(f"  Could not read {tbl}: {e}")
