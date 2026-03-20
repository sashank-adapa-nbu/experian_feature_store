# core/date_utils.py
# =============================================================================
# Robust date parsing utility for Experian Bureau data
# =============================================================================
# Experian data contains dates in multiple string formats across columns.
# This module provides a single parse_date() function that handles all
# formats observed in experian_tradeline_segment and experian_enquiry_segment.
#
# Uses F.try_to_date — silently returns NULL for unparseable values
# instead of raising DateTimeException (which F.to_date does in ANSI mode).
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
# =============================================================================

from pyspark.sql import functions as F
from pyspark.sql import Column


def parse_date(col_name: str) -> Column:
    """
    Robustly parse a string date column to DateType.

    Handles all date formats seen in Experian Bureau data.
    Returns NULL for any value that does not match any known format,
    rather than raising an exception.

    Parameters
    ----------
    col_name : str
        Name of the DataFrame column containing date strings.

    Returns
    -------
    pyspark.sql.Column
        DateType column. NULL where value is unparseable or NULL.
    """
    c = F.col(col_name)
    lit = F.lit

    return F.coalesce(
        # ── 1. Experian standard (Indian format) ─────────────────────────────
        F.try_to_date(c, lit("dd/MM/yyyy")),

        # ── 2. ISO date (enquiry table, partition columns) ────────────────────
        F.try_to_date(c, lit("yyyy-MM-dd")),

        # ── 3. Indian hyphen variant ──────────────────────────────────────────
        F.try_to_date(c, lit("dd-MM-yyyy")),

        # ── 4. Slash ISO ──────────────────────────────────────────────────────
        F.try_to_date(c, lit("yyyy/MM/dd")),

        # ── 5. Compact numeric ────────────────────────────────────────────────
        F.try_to_date(c, lit("yyyyMMdd")),

        # ── 6. ISO datetime with timezone offset (+00:00, +05:30 etc.) ───────
        F.try_to_date(c, lit("yyyy-MM-dd'T'HH:mm:ss.SSSXXX")),

        # ── 7. ISO datetime with Z suffix ────────────────────────────────────
        F.try_to_date(c, lit("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")),

        # ── 8. ISO datetime with milliseconds, no tz ─────────────────────────
        F.try_to_date(c, lit("yyyy-MM-dd'T'HH:mm:ss.SSS")),

        # ── 9. ISO datetime, no milliseconds ─────────────────────────────────
        F.try_to_date(c, lit("yyyy-MM-dd'T'HH:mm:ss")),

        # ── 10. Space-separated datetime ──────────────────────────────────────
        F.try_to_date(c, lit("yyyy-MM-dd HH:mm:ss")),

        # ── 11. US format — defensive fallback only ───────────────────────────
        # Placed last to avoid ambiguity with dd/MM/yyyy for day ≤ 12
        F.try_to_date(c, lit("MM/dd/yyyy")),
    )
