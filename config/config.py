# config/config.py

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

SHUFFLE_PARTITIONS     = 1200        # 150GB / 128MB ≈ 1200 — AQE will coalesce smaller ones
ADAPTIVE_ENABLED       = True       # AQE on — essential for autoscaling clusters
SKEW_JOIN_ENABLED      = True       # AQE skew join — handles PL-heavy customers
BROADCAST_THRESHOLD_MB = 50         # master table ~5MB → auto-broadcast
CODEGEN_MAX_FIELDS     = 200        # grp07/08 produce 250+ col aggs — keep above 200
ARROW_ENABLED          = True       # faster Python<->JVM transfer for notebooks
SPECULATION_ENABLED    = False      # off — slow tasks on 500M rows are legitimate


EXECUTOR_MEMORY_OVERHEAD_FRACTION = 0.20   # raise to 20% for wide-schema joins
MEMORY_FRACTION                   = 0.70   # 70% heap for execution+storage

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"