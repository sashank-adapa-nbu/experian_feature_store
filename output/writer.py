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

    def _append(self, df: DataFrame, table_name: str, partition_col: Optional[str]):
        """
        Append with dynamic partition overwrite — idempotent per partition.
        Re-running for the same scrub_output_date overwrites that partition only,
        making the pipeline safe to retry on failure.
        """
        if partition_col and partition_col in df.columns:
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
