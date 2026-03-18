# core/utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared helper functions used across feature modules and pipelines.
# ─────────────────────────────────────────────────────────────────────────────

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
    """
    Join two DataFrames, avoiding duplicate column names by optionally
    prefixing right-side columns.
    """
    if right_prefix:
        for col in right.columns:
            if col not in on:
                right = right.withColumnRenamed(col, f"{right_prefix}{col}")
    return left.join(right, on=on, how=how)


def coalesce_nulls(df: DataFrame, columns: List[str], fill_value=0) -> DataFrame:
    """Replace nulls in specified columns with fill_value."""
    for col in columns:
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(fill_value)))
    return df


def add_feature_prefix(df: DataFrame, prefix: str, exclude_cols: List[str]) -> DataFrame:
    """
    Prefix all feature columns (not in exclude_cols) with a category prefix.
    This prevents column name collisions when joining multiple categories.
    """
    for col in df.columns:
        if col not in exclude_cols:
            df = df.withColumnRenamed(col, f"{prefix}_{col}")
    return df


# ── Scrub Date Utilities ──────────────────────────────────────────────────────

def get_all_scrub_dates(spark: SparkSession, tradeline_table: str) -> List[str]:
    """Return sorted list of all distinct scrub_output_date values."""
    dates = (
        spark.table(tradeline_table)
        .select("scrub_output_date")
        .distinct()
        .orderBy("scrub_output_date")
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    logger.info(f"Found {len(dates)} scrub dates in {tradeline_table}")
    return [str(d) for d in dates]


def get_already_processed_dates(spark: SparkSession, output_table: str) -> List[str]:
    """Return scrub dates already written to the output feature table."""
    try:
        dates = (
            spark.table(output_table)
            .select("scrub_output_date")
            .distinct()
            .rdd.flatMap(lambda x: x)
            .collect()
        )
        return [str(d) for d in dates]
    except Exception:
        logger.warning(f"Output table {output_table} not found — treating as empty.")
        return []


def get_pending_scrub_dates(
    spark: SparkSession,
    tradeline_table: str,
    output_table: str,
) -> List[str]:
    """Return scrub dates not yet processed."""
    all_dates = set(get_all_scrub_dates(spark, tradeline_table))
    done_dates = set(get_already_processed_dates(spark, output_table))
    pending = sorted(all_dates - done_dates)
    logger.info(f"Pending scrub dates: {len(pending)} | Already done: {len(done_dates)}")
    return pending


# ── Numeric Column Helpers ────────────────────────────────────────────────────

def month_cols(prefix: str, n: int = 36) -> List[str]:
    """Generate list like ['balance_am_01', ..., 'balance_am_36']"""
    return [f"{prefix}_{str(i).zfill(2)}" for i in range(1, n + 1)]


def safe_divide(numerator_col: str, denominator_col: str, alias: str) -> F.Column:
    """Null-safe division returning None when denominator is 0."""
    return F.when(
        F.col(denominator_col).isNotNull() & (F.col(denominator_col) != 0),
        F.col(numerator_col) / F.col(denominator_col),
    ).otherwise(F.lit(None)).alias(alias)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_pk_uniqueness(df: DataFrame, pk_cols: List[str], context: str = "") -> bool:
    """Assert PK uniqueness and log a warning if violated."""
    total = df.count()
    distinct = df.select(pk_cols).distinct().count()
    if total != distinct:
        logger.warning(
            f"[{context}] PK not unique! total={total}, distinct={distinct}, "
            f"duplicates={total - distinct}"
        )
        return False
    logger.info(f"[{context}] PK uniqueness OK. rows={total}")
    return True
