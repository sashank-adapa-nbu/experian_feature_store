# pipeline/base_pipeline.py
# =============================================================================
# Abstract base pipeline — shared logic for scrub and retro modes.
# =============================================================================

from abc import ABC, abstractmethod
from typing import List, Optional
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

from features.tradeline.registry import TRADELINE_FEATURE_CLASSES, ENQUIRY_FEATURE_CLASSES
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
        dfs = []
        for cls in TRADELINE_FEATURE_CLASSES:
            inst = cls()
            try:
                logger.info(f"  → {inst.CATEGORY}")
                dfs.append((inst.CATEGORY, inst.compute(df, pk_cols, as_of_col)))
            except Exception as e:
                logger.error(f"  ✗ {inst.CATEGORY}: {e}")
                raise
        return self._join(dfs, join_cols) if dfs else None

    def run_enquiry_categories(self, df: DataFrame, batch_key: str) -> Optional[DataFrame]:
        pk_cols   = self.get_pk_cols()
        as_of_col = self.get_as_of_col()
        join_cols = pk_cols + [as_of_col]
        logger.info(f"[{self.get_mode_suffix()}] {len(ENQUIRY_FEATURE_CLASSES)} enquiry groups | batch={batch_key}")
        dfs = []
        for cls in ENQUIRY_FEATURE_CLASSES:
            inst = cls()
            try:
                logger.info(f"  → {inst.CATEGORY}")
                dfs.append((inst.CATEGORY, inst.compute(df, pk_cols, as_of_col)))
            except Exception as e:
                logger.error(f"  ✗ {inst.CATEGORY}: {e}")
                raise
        return self._join(dfs, join_cols) if dfs else None

    def _join(self, dfs: list, join_cols: List[str]) -> DataFrame:
        base_name, base_df = dfs[0]
        for cat_name, cat_df in dfs[1:]:
            feat_cols = [c for c in cat_df.columns if c not in join_cols]
            base_df   = base_df.join(cat_df.select(join_cols + feat_cols), on=join_cols, how="left")
            logger.info(f"  Joined {cat_name} | cols={len(base_df.columns)}")
        return base_df

    def write_features(self, df: DataFrame, source: str, batch_key: str,
                       table_name: Optional[str] = None):
        if table_name is None:
            prefix = (config.TRADELINE_FEATURE_TABLE_PREFIX if source == "tradeline"
                      else config.ENQUIRY_FEATURE_TABLE_PREFIX)
            table_name = f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}.{prefix}_scrub"
        mode      = config.SCRUB_WRITE_MODE if self.get_mode_suffix() == "scrub" else config.RETRO_WRITE_MODE
        part_col  = config.PARTITION_COL    if self.get_mode_suffix() == "scrub" else None
        self.writer.write(df=df, table_name=table_name, write_mode=mode, partition_col=part_col)
        logger.info(f"  ✓ Written → {table_name} | mode={mode}")
