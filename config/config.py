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


SHUFFLE_PARTITIONS     = 300       # spark.sql.shuffle.partitions
ADAPTIVE_ENABLED       = True       # AQE — auto-coalesces small partitions post-shuffle
SKEW_JOIN_ENABLED      = True       # AQE skew join handling
BROADCAST_THRESHOLD_MB = 50         # tables <= 50MB auto-broadcast (avoid shuffle joins)
CODEGEN_MAX_FIELDS     = 200        # default 100 — increase for wide schemas
ARROW_ENABLED          = True       # faster Python↔JVM columnar transfer
SPECULATION_ENABLED    = False      # disable — straggler re-launch wastes 500M-row resources

# ── Per-scrub memory / GC settings ───────────────────────────────────────────
# Passed as Spark SQL conf at runtime.  Override in cluster config for production.
EXECUTOR_MEMORY_OVERHEAD_FRACTION = 0.15   # extra JVM overhead beyond executor.memory
MEMORY_FRACTION                   = 0.70   # fraction of heap for execution+storage

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
