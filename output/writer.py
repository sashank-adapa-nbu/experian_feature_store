# output/writer.py
# ─────────────────────────────────────────────────────────────────────────────
# Delta table writer for feature outputs.
# Supports: append (with partition), overwrite, and merge (upsert).
# ─────────────────────────────────────────────────────────────────────────────

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
      "append"    → appends rows, partitioned by partition_col if provided
      "overwrite" → full overwrite of the table
      "merge"     → upsert on merge_keys (requires merge_keys param)
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

        Parameters
        ----------
        df            : Feature DataFrame to write
        table_name    : Fully qualified table name (catalog.schema.table)
        write_mode    : "append" | "overwrite" | "merge"
        partition_col : Column to partition by (used for append mode)
        merge_keys    : PK columns for upsert (used for merge mode)
        """
        logger.info(f"[Writer] Writing to {table_name} | mode={write_mode} | rows={df.count()}")

        if write_mode == "append":
            self._append(df, table_name, partition_col)

        elif write_mode == "overwrite":
            self._overwrite(df, table_name, partition_col)

        elif write_mode == "merge":
            if not merge_keys:
                raise ValueError("merge_keys must be provided for write_mode='merge'")
            self._merge(df, table_name, merge_keys)

        else:
            raise ValueError(f"Unsupported write_mode: {write_mode}. Use 'append', 'overwrite', or 'merge'.")

    def _append(self, df: DataFrame, table_name: str, partition_col: Optional[str]):
        """Append rows. If partition_col provided, overwrites only that partition."""
        writer = df.write.format("delta").mode("append")

        if partition_col and partition_col in df.columns:
            writer = (
                df.write
                .format("delta")
                .mode("overwrite")                          # overwrite the partition only
                .option("partitionOverwriteMode", "dynamic")
                .partitionBy(partition_col)
            )
            logger.info(f"[Writer] Using dynamic partition overwrite on '{partition_col}'")

        try:
            writer.saveAsTable(table_name)
            logger.info(f"[Writer] ✓ Append complete → {table_name}")
        except Exception as e:
            logger.error(f"[Writer] ✗ Append failed → {table_name} | {e}")
            raise

    def _overwrite(self, df: DataFrame, table_name: str, partition_col: Optional[str]):
        """Full overwrite of the table."""
        writer = df.write.format("delta").mode("overwrite")
        if partition_col and partition_col in df.columns:
            writer = writer.partitionBy(partition_col)
        try:
            writer.saveAsTable(table_name)
            logger.info(f"[Writer] ✓ Overwrite complete → {table_name}")
        except Exception as e:
            logger.error(f"[Writer] ✗ Overwrite failed → {table_name} | {e}")
            raise

    def _merge(self, df: DataFrame, table_name: str, merge_keys: List[str]):
        """
        MERGE (upsert) into existing Delta table.
        Inserts new rows, updates existing rows matched on merge_keys.
        """
        try:
            delta_table = DeltaTable.forName(self.spark, table_name)
        except Exception:
            # Table doesn't exist yet → create it
            logger.info(f"[Writer] Table {table_name} does not exist — creating via overwrite.")
            df.write.format("delta").mode("overwrite").saveAsTable(table_name)
            logger.info(f"[Writer] ✓ Created → {table_name}")
            return

        # Build merge condition string
        merge_condition = " AND ".join([f"target.{k} = source.{k}" for k in merge_keys])

        # Build update map for all non-key columns
        update_map = {
            col: f"source.{col}"
            for col in df.columns
            if col not in merge_keys
        }

        try:
            (
                delta_table.alias("target")
                .merge(df.alias("source"), merge_condition)
                .whenMatchedUpdate(set=update_map)
                .whenNotMatchedInsertAll()
                .execute()
            )
            logger.info(f"[Writer] ✓ Merge complete → {table_name}")
        except Exception as e:
            logger.error(f"[Writer] ✗ Merge failed → {table_name} | {e}")
            raise
