# features/tradeline/base.py
# =============================================================================
# Abstract base class for all tradeline feature group modules.
# Every grp*.py class must extend TradelineFeatureBase and implement compute().
# =============================================================================

from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from typing import List
from core.logger import get_logger

logger = get_logger(__name__)


class TradelineFeatureBase(ABC):
    """
    Base class for tradeline feature groups.

    Usage:
        class MyFeatures(TradelineFeatureBase):
            CATEGORY = "grp_my_category"

            def compute(self, df, pk_cols, as_of_col):
                ...
                return feature_df
    """

    CATEGORY: str = "base"

    def __repr__(self):
        return f"<TradelineFeatureModule: {self.CATEGORY}>"

    @abstractmethod
    def compute(
        self,
        df: DataFrame,
        pk_cols: List[str],
        as_of_col: str,
    ) -> DataFrame:
        """
        Derive features from the tradeline DataFrame.

        Parameters
        ----------
        df : DataFrame
            Raw tradeline data (all rows for the current batch).
        pk_cols : List[str]
            Primary key columns — must be present in returned DataFrame.
            Scrub : ["customer_scrub_key", "party_code", "scrub_output_date"]
            Retro : ["party_code", "reference_dt"]
        as_of_col : str
            Column name used as the as-of reference date for all features.
            Scrub : "scrub_output_date"
            Retro : "reference_dt"

        Returns
        -------
        DataFrame
            One row per unique (pk_cols + as_of_col).
            Columns = pk_cols + [as_of_col] + feature columns.
        """
        ...

    def _log_start(self, mode: str = "batch", date: str = ""):
        logger.info(f"[{self.CATEGORY}] Starting | mode={mode} | date={date}")

    def _log_done(self, df: DataFrame):
        logger.info(f"[{self.CATEGORY}] Done | cols={len(df.columns)}")
