# output/writer.py  [OPTIMISED]
# =============================================================================
# Delta table writer for feature outputs.
# =============================================================================
# OPTIMISATION: removed df.count() before write (was triggering a full
# extra scan of 500M rows purely for log output).
# Uses Delta commit metrics for row counts instead.
# =============================================================================

from typing import Optional, List
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from delta.tables import DeltaTable

from core.logger import get_logger

logger = get_logger(__name__) 


class FeatureWriter:
    """
    Handles writing feature DataFrames to Unity Catalog Delta tables.

    write_mode options:
      "append"    → partition overwrite (dynamic) on partition_col
      "overwrite" → full table overwrite
      "merge"     → upsert on merge_keys
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def write(
        self,
        df: DataFrame,
        table_name: str,
        write_mode: str = "append",
        partition_col: Optional[str] = None,
        merge_keys: Optional[List[str]] = None,
    ):
        """
        Write a feature DataFrame to a Delta table.
        Does NOT call df.count() — avoids extra full scan on 500M-row data.
        Delta commit log provides row count metrics after write.
        """
        logger.info(f"[Writer] → {table_name} | mode={write_mode}")

        if write_mode == "append":
            self._append(df, table_name, partition_col)
        elif write_mode == "overwrite":
            self._overwrite(df, table_name, partition_col)
        elif write_mode == "merge":
            if not merge_keys:
                raise ValueError("merge_keys must be provided for write_mode='merge'")
            self._merge(df, table_name, merge_keys)
        else:
            raise ValueError(f"Unsupported write_mode: {write_mode}")

    @staticmethod
    def _dedup_columns(df: DataFrame) -> DataFrame:
        """
        Remove duplicate column names from a DataFrame.
        Spark raises AMBIGUOUS_COLUMN_OR_FIELD if a column name appears more
        than once — this can happen when multiple feature groups share join keys
        that survive through the join chain. Keep the first occurrence only.
        """
        seen: set = set()
        cols = []
        for c in df.columns:
            if c not in seen:
                cols.append(c)
                seen.add(c)
        return df.select(cols) if len(cols) < len(df.columns) else df

    def _append(self, df: DataFrame, table_name: str, partition_col: Optional[str]):
        """
        Append with dynamic partition overwrite — idempotent per partition.
        Re-running for the same scrub_output_date overwrites that partition only,
        making the pipeline safe to retry on failure.
        """
        df = self._dedup_columns(df)
        # ── PARTITION TYPE FIX ────────────────────────────────────────────────
        # After 16 groupBy→join passes the partition column (scrub_output_date)
        # can end up as StringType. Delta's partition writer then tries to parse
        # the string as a timestamp and throws CANNOT_PARSE_TIMESTAMP.
        # Explicitly cast to DateType before writing so Delta sees the correct
        # type and writes clean date partition paths (2023-12-05/ not a ts).
        if partition_col and partition_col in df.columns:
            from pyspark.sql import functions as F
            df = df.withColumn(partition_col, F.col(partition_col).cast("date"))
            writer = (
                df.write
                .format("delta")
                .mode("overwrite")
                .option("partitionOverwriteMode", "dynamic")
                .partitionBy(partition_col)
            )
            logger.info(f"[Writer] Dynamic partition overwrite on '{partition_col}'")
        else:
            writer = df.write.format("delta").mode("append")

        try:
            writer.saveAsTable(table_name)
            logger.info(f"[Writer] ✓ Done → {table_name}")
        except Exception as e:
            logger.error(f"[Writer] ✗ Failed → {table_name} | {e}")
            raise

    def _overwrite(self, df: DataFrame, table_name: str, partition_col: Optional[str]):
        df = self._dedup_columns(df)
        if partition_col and partition_col in df.columns:
            from pyspark.sql import functions as F
            df = df.withColumn(partition_col, F.col(partition_col).cast("date"))
        writer = df.write.format("delta").mode("overwrite")
        if partition_col and partition_col in df.columns:
            writer = writer.partitionBy(partition_col)
        try:
            writer.saveAsTable(table_name)
            logger.info(f"[Writer] ✓ Overwrite → {table_name}")
        except Exception as e:
            logger.error(f"[Writer] ✗ Failed → {table_name} | {e}")
            raise

    def _merge(self, df: DataFrame, table_name: str, merge_keys: List[str]):
        try:
            delta_table = DeltaTable.forName(self.spark, table_name)
        except Exception:
            logger.info(f"[Writer] Table {table_name} not found — creating via overwrite.")
            df.write.format("delta").mode("overwrite").saveAsTable(table_name)
            logger.info(f"[Writer] ✓ Created → {table_name}")
            return

        merge_condition = " AND ".join([f"target.{k} = source.{k}" for k in merge_keys])
        update_map = {col: f"source.{col}" for col in df.columns if col not in merge_keys}

        try:
            (
                delta_table.alias("target")
                .merge(df.alias("source"), merge_condition)
                .whenMatchedUpdate(set=update_map)
                .whenNotMatchedInsertAll()
                .execute()
            )
            logger.info(f"[Writer] ✓ Merge → {table_name}")
        except Exception as e:
            logger.error(f"[Writer] ✗ Merge failed → {table_name} | {e}")
            raise