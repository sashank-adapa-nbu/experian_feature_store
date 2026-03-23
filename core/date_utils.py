# core/date_utils.py
# =============================================================================
# Robust date parsing utility for Experian Bureau data
# =============================================================================
# Experian data contains dates in multiple string formats across columns.
# This module provides a single parse_date() function that handles all
# formats observed in experian_tradeline_segment and experian_enquiry_segment.
#
# Uses F.to_date(col, format_str) with spark.sql.legacy.timeParserPolicy=LEGACY
# (set in core/spark_conf.py). In LEGACY mode F.to_date silently returns NULL
# for values that don't match the format, instead of raising DateTimeException.
#
# IMPORTANT:
#   - F.to_date second argument must be a plain Python str, NOT F.lit().
#   - F.to_date on a non-StringType column (DateType, TimestampType) throws
#     CANNOT_PARSE_TIMESTAMP even in LEGACY mode. The input is cast to STRING
#     first so the coalesce chain always operates on text, never on a typed
#     date/timestamp column.
#
# Format priority (first match wins via F.coalesce):
#   1. dd/MM/yyyy              → Experian tradeline standard (most common)
#   2. yyyy-MM-dd              → ISO, enquiry table & partition dates
#   3. dd-MM-yyyy              → Hyphen Indian format
#   4. yyyy/MM/dd              → Slash ISO variant
#   5. yyyyMMdd                → Compact numeric
#   6. yyyy-MM-dd'T'HH:mm:ss.SSSXXX → Full ISO with tz offset (+00:00)
#   7. yyyy-MM-dd'T'HH:mm:ss.SSS'Z' → ISO with Z suffix
#   8. yyyy-MM-dd'T'HH:mm:ss.SSS    → ISO with milliseconds, no tz
#   9. yyyy-MM-dd'T'HH:mm:ss        → ISO datetime, no ms
#  10. yyyy-MM-dd HH:mm:ss          → Space-separated datetime
#  11. MM/dd/yyyy                   → US format (defensive fallback only)
#
# NOTE: dd/MM/yyyy is placed before MM/dd/yyyy because Experian is an Indian
# bureau — dates like 08/12/2021 = 8 Dec 2021, not 12 Aug 2021.
#
# REQUIRES: spark.sql.legacy.timeParserPolicy = LEGACY
# =============================================================================

from pyspark.sql import functions as F
from pyspark.sql import Column


def parse_date(col_name: str) -> Column:
    """
    Robustly parse a string date column to DateType.

    Handles all date formats seen in Experian Bureau data.
    Returns NULL for any value that does not match any known format.

    The column is cast to STRING before parsing so this function is safe
    to call on StringType, DateType, and TimestampType columns alike.
    Calling F.to_date directly on a DateType column throws
    CANNOT_PARSE_TIMESTAMP even in LEGACY mode.

    Requires spark.sql.legacy.timeParserPolicy = LEGACY so that
    F.to_date returns NULL (not exception) on format mismatch.

    Parameters
    ----------
    col_name : str
        Name of the DataFrame column containing date strings.

    Returns
    -------
    pyspark.sql.Column
        DateType column. NULL where value is unparseable or NULL.
    """
    # Cast to STRING first — F.to_date on DateType/TimestampType raises
    # CANNOT_PARSE_TIMESTAMP in LEGACY mode instead of returning NULL.
    c = F.col(col_name).cast("string")

    return F.coalesce(
        # ── 1. Experian standard (Indian format) ─────────────────────────────
        F.to_date(c, "dd/MM/yyyy"),

        # ── 2. ISO date (enquiry table, partition columns, scrub_output_date) ─
        F.to_date(c, "yyyy-MM-dd"),

        # ── 3. Indian hyphen variant ──────────────────────────────────────────
        F.to_date(c, "dd-MM-yyyy"),

        # ── 4. Slash ISO ──────────────────────────────────────────────────────
        F.to_date(c, "yyyy/MM/dd"),

        # ── 5. Compact numeric ────────────────────────────────────────────────
        F.to_date(c, "yyyyMMdd"),

        # ── 6. ISO datetime with timezone offset (+00:00, +05:30 etc.) ───────
        F.to_date(c, "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"),

        # ── 7. ISO datetime with Z suffix ────────────────────────────────────
        F.to_date(c, "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"),

        # ── 8. ISO datetime with milliseconds, no tz ─────────────────────────
        F.to_date(c, "yyyy-MM-dd'T'HH:mm:ss.SSS"),

        # ── 9. ISO datetime, no milliseconds ─────────────────────────────────
        F.to_date(c, "yyyy-MM-dd'T'HH:mm:ss"),

        # ── 10. Space-separated datetime ──────────────────────────────────────
        F.to_date(c, "yyyy-MM-dd HH:mm:ss"),

        # ── 11. US format — defensive fallback only ───────────────────────────
        # Placed last to avoid ambiguity with dd/MM/yyyy for day <= 12
        F.to_date(c, "MM/dd/yyyy"),
    )