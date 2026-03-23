# pipeline/base_pipeline.py  [OPTIMISED]
# =============================================================================
# Abstract base pipeline — shared logic for scrub and retro modes.
# =============================================================================
# OPTIMISATION:
#   - Removed df.count() from write_features (double-materialise on 500M rows)
#   - _join() uses column selection to avoid bringing all columns to driver
#   - Groups compute in sequence, each groupBy triggers only its own shuffle
# =============================================================================

from abc import ABC, abstractmethod
from typing import List, Optional
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

from features.registry import TRADELINE_FEATURE_CLASSES, ENQUIRY_FEATURE_CLASSES
from output.writer import FeatureWriter
from core.logger import get_logger
from config import config

logger = get_logger(__name__)


class BasePipeline(ABC):

    def __init__(self, spark: SparkSession):
        self.spark  = spark
        self.writer = FeatureWriter(spark)

    @abstractmethod
    def get_pk_cols(self) -> List[str]: ...
    @abstractmethod
    def get_as_of_col(self) -> str: ...
    @abstractmethod
    def get_mode_suffix(self) -> str: ...

    def run_tradeline_categories(self, df: DataFrame, batch_key: str) -> Optional[DataFrame]:
        pk_cols   = self.get_pk_cols()
        as_of_col = self.get_as_of_col()
        join_cols = pk_cols + [as_of_col]
        logger.info(f"[{self.get_mode_suffix()}] {len(TRADELINE_FEATURE_CLASSES)} tradeline groups | batch={batch_key}")

        # DISK_ONLY avoids executor heap pressure on 500M rows — spills to local
        # disk instead of filling executor memory. Driver stays responsive because
        # it is not managing thousands of in-memory RDD blocks.
        # Do NOT use MEMORY_AND_DISK or MEMORY_ONLY on 500M rows — driver OOM.
        from pyspark import StorageLevel
        df = df.persist(StorageLevel.DISK_ONLY)

        dfs = []
        for cls in TRADELINE_FEATURE_CLASSES:
            inst = cls()
            try:
                logger.info(f"  → {inst.CATEGORY}")
                result = inst.compute(df, pk_cols, as_of_col)
                dfs.append((inst.CATEGORY, result))
            except Exception as e:
                logger.error(f"  ✗ {inst.CATEGORY}: {e}")
                df.unpersist()
                raise

        df.unpersist()
        if not dfs:
            return None
        result = self._join(dfs, join_cols)
        # Persist the joined result (DISK_ONLY) so write_features does not
        # re-execute the full join chain during saveAsTable.
        # Do NOT use localCheckpoint — it requires an extra Spark action to
        # materialise which adds a full scan, and eager=False couples it with
        # saveAsTable making the write job unpredictably long.
        result = result.persist(StorageLevel.DISK_ONLY)
        return result

    def run_enquiry_categories(self, df: DataFrame, batch_key: str) -> Optional[DataFrame]:
        pk_cols   = self.get_pk_cols()
        as_of_col = self.get_as_of_col()
        join_cols = pk_cols + [as_of_col]
        logger.info(f"[{self.get_mode_suffix()}] {len(ENQUIRY_FEATURE_CLASSES)} enquiry groups | batch={batch_key}")

        from pyspark import StorageLevel
        df = df.persist(StorageLevel.DISK_ONLY)
        dfs = []
        for cls in ENQUIRY_FEATURE_CLASSES:
            inst = cls()
            try:
                logger.info(f"  → {inst.CATEGORY}")
                result = inst.compute(df, pk_cols, as_of_col)
                dfs.append((inst.CATEGORY, result))
            except Exception as e:
                logger.error(f"  ✗ {inst.CATEGORY}: {e}")
                df.unpersist()
                raise

        df.unpersist()
        if not dfs:
            return None
        result = self._join(dfs, join_cols)
        result = result.persist(StorageLevel.DISK_ONLY)
        return result

    # Date/timestamp columns that must stay as DateType through the join chain.
    # Spark Catalyst can silently widen these to StringType after many chained
    # joins because each group DataFrame is an independent plan — the type is
    # re-resolved from string column names, not from the original schema object.
    _DATE_COLS = {"scrub_output_date", "reference_dt"}

    def _join(self, dfs: list, join_cols: List[str]) -> DataFrame:
        """
        Left-join all feature DataFrames on pk_cols + as_of_col.

        Each group DataFrame is produced independently by groupBy — they share
        no column lineage with each other so the join is clean.

        We select only feature columns from the right side (never join_cols)
        so scrub_output_date / party_code etc. appear exactly once in the
        result — from the base (left-most) DataFrame only.
        This prevents AMBIGUOUS_COLUMN_OR_FIELD on write.

        After all joins, known date columns are re-cast to DateType to prevent
        Catalyst type widening (StringType drift) across the deep join chain.
        """
        base_name, base_df = dfs[0]
        for cat_name, cat_df in dfs[1:]:
            # Take only non-key columns from right side — keys come from base_df
            feat_cols = [c for c in cat_df.columns if c not in join_cols]
            base_df = base_df.join(
                cat_df.select(join_cols + feat_cols),
                on=join_cols,
                how="left",
            )
            logger.info(f"  Joined {cat_name} | total_cols={len(base_df.columns)}")

        # ── Dedup guard ───────────────────────────────────────────────────────
        seen = set()
        final_cols = []
        for c in base_df.columns:
            if c not in seen:
                final_cols.append(c)
                seen.add(c)
        if len(final_cols) < len(base_df.columns):
            logger.warning(f"  Dropped {len(base_df.columns)-len(final_cols)} duplicate column(s) after join")
            base_df = base_df.select(final_cols)

        # ── Type enforcement: re-cast date columns after join chain ───────────
        # Catalyst can widen DateType → StringType when resolving column names
        # across 15+ chained independent groupBy plans. Enforce DateType here
        # so Delta partition writes never see a string where a date is expected.
        for col_name in self._DATE_COLS:
            if col_name in base_df.columns:
                base_df = base_df.withColumn(col_name, F.col(col_name).cast("date"))

        return base_df

    def write_features(self, df: DataFrame, source: str, batch_key: str,
                       table_name: Optional[str] = None):
        if table_name is None:
            prefix = (config.TRADELINE_FEATURE_TABLE_PREFIX if source == "tradeline"
                      else config.ENQUIRY_FEATURE_TABLE_PREFIX)
            table_name = f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}.{prefix}_scrub"

        mode     = config.SCRUB_WRITE_MODE if self.get_mode_suffix() == "scrub" else config.RETRO_WRITE_MODE
        part_col = config.PARTITION_COL    if self.get_mode_suffix() == "scrub" else None

        # Do NOT call df.count() here — costs a full extra scan on 500M rows.
        # The writer logs via Delta commit metrics instead.
        try:
            self.writer.write(df=df, table_name=table_name, write_mode=mode,
                              partition_col=part_col)
            logger.info(f"  ✓ Written → {table_name} | mode={mode}")
        finally:
            # ── LEAK FIX: always unpersist after write ────────────────────────
            # df was persisted(DISK_ONLY) in run_*_categories() before being
            # returned here. Without an explicit unpersist the DISK_ONLY blocks
            # stay pinned on executors for the rest of the driver lifetime.
            # Over 108 scrub dates this accumulates 108× the per-date footprint
            # (tradeline feats ≈ 150GB, enquiry feats ≈ smaller) until executor
            # local disk fills up and tasks start failing.
            # finally: ensures we unpersist even when write raises an exception.
            df.unpersist()
            logger.debug(f"  [GC] Unpersisted {source} features DF | date={batch_key}")