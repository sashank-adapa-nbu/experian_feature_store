# core/utils.py  [OPTIMISED]
# =============================================================================
# Shared helper functions — scale-safe, 500M record compatible.
# =============================================================================

from typing import List, Optional
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from core.logger import get_logger

logger = get_logger(__name__)


# ── DataFrame Utilities ───────────────────────────────────────────────────────

def safe_join(
    left: DataFrame,
    right: DataFrame,
    on: List[str],
    how: str = "left",
    right_prefix: str = "",
) -> DataFrame:
    """Join two DataFrames avoiding duplicate column names."""
    if right_prefix:
        for col in right.columns:
            if col not in on:
                right = right.withColumnRenamed(col, f"{right_prefix}{col}")
    return left.join(right, on=on, how=how)


def coalesce_nulls(df: DataFrame, columns: List[str], fill_value=0) -> DataFrame:
    for col in columns:
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(fill_value)))
    return df


def add_feature_prefix(df: DataFrame, prefix: str, exclude_cols: List[str]) -> DataFrame:
    for col in df.columns:
        if col not in exclude_cols:
            df = df.withColumnRenamed(col, f"{prefix}_{col}")
    return df


# ── Scrub Date Utilities ──────────────────────────────────────────────────────

def get_all_scrub_dates(spark: SparkSession, tradeline_table: str) -> List[str]:
    """Return sorted list of all distinct scrub_output_date values.
    Uses partition pruning — no full scan."""
    rows = (
        spark.table(tradeline_table)
        .select("scrub_output_date")
        .distinct()
        .orderBy("scrub_output_date")
        .collect()
    )
    dates = [str(r["scrub_output_date"]) for r in rows]
    logger.info(f"Found {len(dates)} scrub dates in {tradeline_table}")
    return dates


def get_already_processed_dates(spark: SparkSession, output_table: str) -> List[str]:
    """Return scrub dates already written to the output feature table."""
    try:
        rows = (
            spark.table(output_table)
            .select("scrub_output_date")
            .distinct()
            .collect()
        )
        return [str(r["scrub_output_date"]) for r in rows]
    except Exception:
        logger.warning(f"Output table {output_table} not found — treating as empty.")
        return []


def get_pending_scrub_dates(
    spark: SparkSession,
    tradeline_table: str,
    output_table: str,
) -> List[str]:
    all_dates  = set(get_all_scrub_dates(spark, tradeline_table))
    done_dates = set(get_already_processed_dates(spark, output_table))
    pending    = sorted(all_dates - done_dates)
    logger.info(f"Pending: {len(pending)} | Done: {len(done_dates)}")
    return pending


# ── Numeric Column Helpers ────────────────────────────────────────────────────

def month_cols(prefix: str, n: int = 36) -> List[str]:
    return [f"{prefix}_{str(i).zfill(2)}" for i in range(1, n + 1)]


def safe_divide(numerator_col: str, denominator_col: str, alias: str) -> F.Column:
    return F.when(
        F.col(denominator_col).isNotNull() & (F.col(denominator_col) != 0),
        F.col(numerator_col) / F.col(denominator_col),
    ).otherwise(F.lit(None)).alias(alias)


# ── Array-based slot helpers (shared across grp07/08/09/03b) ─────────────────
# These replace the 73-WHEN nested chain with O(1) arithmetic + element_at.

def build_history_array(df: DataFrame, base_col: str, history_cols: List[str],
                        out_col: str, clean_negative: bool = True) -> DataFrame:
    """
    Build a single array column from base_col + history_cols.
    Array is 1-indexed in Spark: arr[1] = base_col (slot 0), arr[2] = history_cols[0], ...

    Parameters
    ----------
    df             : Input DataFrame
    base_col       : Slot-0 column name (e.g. "days_past_due", "balance_am")
    history_cols   : List of N_HISTORY history column names (e.g. ["days_past_due_01"...])
    out_col        : Name for the output array column
    clean_negative : If True, treat negative values as NULL (Experian -1 sentinel)
    """
    all_cols = [base_col] + history_cols

    def clean(c):
        col = F.col(c).cast("double")
        if clean_negative:
            return F.when(col >= 0, col).otherwise(F.lit(None).cast("double"))
        return col

    return df.withColumn(out_col, F.array(*[clean(c) for c in all_cols]))


def resolve_slot(arr_col: str, k: int, md_col: str = "_md",
                 product_filter=None) -> F.Column:
    """
    Resolve value at window offset k from as_of_dt using array indexing.

    slot_idx = k - _md   (where _md = ceil(months_between(as_of_dt, report_dt)))
    Spark arrays are 1-based → index = slot_idx + 1.
    slot < 0 (gap) → NULL.
    slot > N (beyond history) → NULL (element_at returns NULL automatically).

    Parameters
    ----------
    arr_col        : Name of the pre-built array column
    k              : Window offset (0 = as_of, 1 = 1m before, ...)
    md_col         : Column name holding month_diff
    product_filter : Optional boolean Column to mask rows (returns NULL when False)
    """
    slot_idx  = F.lit(k) - F.col(md_col)
    idx1based = (slot_idx + F.lit(1)).cast("int")
    val = F.when(
        slot_idx >= 0,
        F.element_at(F.col(arr_col), idx1based)
    ).otherwise(F.lit(None).cast("double"))

    if product_filter is not None:
        val = F.when(product_filter, val).otherwise(F.lit(None).cast("double"))
    return val


def resolve_slot_at_asof(arr_col: str, md_col: str = "_month_diff") -> F.Column:
    """
    Resolve balance/value at as_of_dt for grp03a-style slot resolution
    where month_diff = ceil(months_between(balance_dt, as_of_dt)) — note
    the REVERSED argument order (balance_dt first → negative when balance < as_of).

    _md <= 0 → use arr[1] (slot 0 = base_col, i.e. balance_am)
    _md 1..N → use arr[_md+1] (the _md-th history column)
    _md > N  → NULL
    """
    return F.when(
        F.col(md_col) <= 0,
        F.element_at(F.col(arr_col), F.lit(1))
    ).when(
        (F.col(md_col) >= 1) & (F.col(md_col) <= 36),
        F.element_at(F.col(arr_col), (F.col(md_col) + F.lit(1)).cast("int"))
    ).otherwise(F.lit(None).cast("double"))


def build_window_cols(arr_col: str, w: int, md_col: str = "_md",
                      product_filter=None) -> List[F.Column]:
    """Return list of w slot-Column objects for aggregation."""
    return [resolve_slot(arr_col, k, md_col, product_filter) for k in range(w)]


# ── Validation ────────────────────────────────────────────────────────────────

def validate_pk_uniqueness(df: DataFrame, pk_cols: List[str], context: str = "") -> bool:
    """Assert PK uniqueness and log a warning if violated.
    NOTE: calls .count() twice — skip in production for 500M+ rows.
    """
    total    = df.count()
    distinct = df.select(pk_cols).distinct().count()
    if total != distinct:
        logger.warning(
            f"[{context}] PK not unique! total={total}, distinct={distinct}, "
            f"duplicates={total - distinct}"
        )
        return False
    logger.info(f"[{context}] PK uniqueness OK. rows={total}")
    return True
