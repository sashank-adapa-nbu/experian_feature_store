# config/config.py
# =============================================================================
# Global configuration — Experian Feature Store
# Update table paths and output settings here — nowhere else.
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
# Scrub PK: customer_scrub_key + party_code + scrub_output_date
# party_code is resolved via master table join and included in scrub output
SCRUB_PK_COLS = ["customer_scrub_key", "party_code", "scrub_output_date"]

# Retro PK: party_code + reference_dt (the loan reference/application date)
RETRO_PK_COLS = ["party_code", "reference_dt"]

# ── Key Column Names ──────────────────────────────────────────────────────────
SCRUB_OUTPUT_DATE_COL  = "scrub_output_date"
REFERENCE_DT_COL       = "reference_dt"       # retro as-of date column
PARTY_CODE_COL         = "party_code"
CUSTOMER_SCRUB_KEY_COL = "customer_scrub_key"

# ── Retro Scrub Selection Window ──────────────────────────────────────────────
# For retro mode: find the latest scrub_output_date where:
#   reference_dt <= scrub_output_date <= reference_dt + RETRO_MAX_SCRUB_MONTHS
# Scrubs more than RETRO_MAX_SCRUB_MONTHS after reference_dt are excluded.
RETRO_MAX_SCRUB_MONTHS = 12

# ── Write Mode ────────────────────────────────────────────────────────────────
SCRUB_WRITE_MODE = "append"
RETRO_WRITE_MODE = "overwrite"
PARTITION_COL    = "scrub_output_date"

# ── Max History Columns ───────────────────────────────────────────────────────
MAX_HISTORY_COLS = 36

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
