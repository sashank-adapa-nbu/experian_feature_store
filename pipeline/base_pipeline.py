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

        # Cache the input once — each group reads it independently via groupBy.
        # For 500M rows this avoids repeated table scans.
        df.cache()

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
        return self._join(dfs, join_cols) if dfs else None

    def run_enquiry_categories(self, df: DataFrame, batch_key: str) -> Optional[DataFrame]:
        pk_cols   = self.get_pk_cols()
        as_of_col = self.get_as_of_col()
        join_cols = pk_cols + [as_of_col]
        logger.info(f"[{self.get_mode_suffix()}] {len(ENQUIRY_FEATURE_CLASSES)} enquiry groups | batch={batch_key}")

        df.cache()
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
        return self._join(dfs, join_cols) if dfs else None

    def _join(self, dfs: list, join_cols: List[str]) -> DataFrame:
        """
        Left-join all feature DataFrames on pk_cols + as_of_col.
        Select only feature columns (not join_cols) from right side to avoid
        duplicate column warnings on 365-column wide output.
        """
        base_name, base_df = dfs[0]
        for cat_name, cat_df in dfs[1:]:
            feat_cols = [c for c in cat_df.columns if c not in join_cols]
            base_df   = base_df.join(cat_df.select(join_cols + feat_cols), on=join_cols, how="left")
            logger.info(f"  Joined {cat_name} | total_cols={len(base_df.columns)}")
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
        self.writer.write(df=df, table_name=table_name, write_mode=mode,
                          partition_col=part_col)
        logger.info(f"  ✓ Written → {table_name} | mode={mode}")
