# config/config.py
# =============================================================================
# Global configuration — Experian Feature Store  [OPTIMISED]
# =============================================================================

# ── Source Tables ─────────────────────────────────────────────────────────────
TRADELINE_TABLE = "uc_dataorg_prod.l1_experian.experian_tradeline_segment"
ENQUIRY_TABLE   = "uc_dataorg_prod.l1_experian.experian_enquiry_segment"
MASTER_TABLE    = "uc_dataorg_prod.l3_nbu.experian_master_table"

# ── Output Tables ─────────────────────────────────────────────────────────────
OUTPUT_CATALOG = "uc_dataorg_prod"
OUTPUT_SCHEMA  = "do_nbu_biz_sandbox"

TRADELINE_FEATURE_TABLE_PREFIX = "experian_tradeline_features"
ENQUIRY_FEATURE_TABLE_PREFIX   = "experian_enquiry_features"

# ── Primary Key Columns ───────────────────────────────────────────────────────
SCRUB_PK_COLS  = ["customer_scrub_key", "party_code", "scrub_output_date"]
RETRO_PK_COLS  = ["party_code", "reference_dt"]

# ── Key Column Names ──────────────────────────────────────────────────────────
SCRUB_OUTPUT_DATE_COL  = "scrub_output_date"
REFERENCE_DT_COL       = "reference_dt"
PARTY_CODE_COL         = "party_code"
CUSTOMER_SCRUB_KEY_COL = "customer_scrub_key"

# ── Retro Window ──────────────────────────────────────────────────────────────
RETRO_MAX_SCRUB_MONTHS = 12

# ── Write Mode ────────────────────────────────────────────────────────────────
SCRUB_WRITE_MODE = "append"
RETRO_WRITE_MODE = "overwrite"
PARTITION_COL    = "scrub_output_date"

# ── Max History Columns ───────────────────────────────────────────────────────
MAX_HISTORY_COLS = 36

# ── Spark Tuning — cluster: 2-10 workers × 4 cores (autoscaling) ────────────
# Max cores = 40  |  Target ~128 MB per shuffle partition
# 500M rows × ~300 bytes/row ≈ 150 GB per scrub date
# 150 GB / 128 MB ≈ 1200 partitions — use 1200 as floor, AQE coalesces upward.
#
# With autoscaling, AQE (ADAPTIVE_ENABLED=True) is critical — it re-optimises
# the plan as the cluster grows/shrinks mid-job.
#
# DISK_ONLY persist in base_pipeline means executors handle caching independently;
# driver does not track heap blocks so stays responsive even at 500M rows.
SHUFFLE_PARTITIONS     = 500       # 150GB / 128MB ≈ 1200 — AQE will coalesce smaller ones
ADAPTIVE_ENABLED       = True       # AQE on — essential for autoscaling clusters
SKEW_JOIN_ENABLED      = True       # AQE skew join — handles PL-heavy customers
BROADCAST_THRESHOLD_MB = 50         # master table ~5MB → auto-broadcast
CODEGEN_MAX_FIELDS     = 200        # grp07/08 produce 250+ col aggs — keep above 200
ARROW_ENABLED          = True       # faster Python<->JVM transfer for notebooks
SPECULATION_ENABLED    = False      # off — slow tasks on 500M rows are legitimate

# ── Per-scrub memory / GC settings ───────────────────────────────────────────
# Set in cluster Advanced Options → Spark Config (static, cannot be set at runtime):
#   spark.executor.memory              8g    (adjust to instance type)
#   spark.executor.memoryOverhead      2g    (>= 20% of executor.memory)
#   spark.memory.fraction              0.7
#   spark.memory.storageFraction       0.3   (lower storage — DISK_ONLY needs less)
#   spark.driver.memory                4g
#   spark.driver.maxResultSize         2g
EXECUTOR_MEMORY_OVERHEAD_FRACTION = 0.20   # raise to 20% for wide-schema joins
MEMORY_FRACTION                   = 0.70   # 70% heap for execution+storage

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"