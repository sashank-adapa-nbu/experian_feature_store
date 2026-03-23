# pipeline/retro_pipeline.py  [OPTIMISED]
# =============================================================================
# Retro Pipeline
# =============================================================================
# OPTIMISATIONS:
#   - Removed df.count() calls (avoid double-materialise on 500M rows)
#   - Broadcast master table in scrub snapshot resolution
#   - Batch retro inputs in configurable chunks if very large
#   - Spark conf applied at init
# =============================================================================

from typing import List, Optional
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from pipeline.base_pipeline import BasePipeline
from core.logger import get_logger
from core.spark_conf import configure_spark
from config import config

logger = get_logger(__name__)


class RetroPipeline(BasePipeline):

    def __init__(self, spark: SparkSession):
        super().__init__(spark)
        configure_spark(spark)

    def get_pk_cols(self) -> List[str]:
        return config.RETRO_PK_COLS

    def get_as_of_col(self) -> str:
        return config.REFERENCE_DT_COL

    def get_mode_suffix(self) -> str:
        return "retro"

    def _resolve_scrub_snapshot(self, base_df: DataFrame) -> DataFrame:
        """
        For each (party_code, reference_dt), resolve the best scrub snapshot.
        Master table is broadcast — it's small relative to tradeline/enquiry data.
        """
        logger.info(f"[RetroPipeline] Resolving scrub snapshots | window={config.RETRO_MAX_SCRUB_MONTHS}m")

        master = (
            self.spark.table(config.MASTER_TABLE)
            .select("customer_scrub_key", "party_code", "scrub_output_date")
            .distinct()
        )

        joined = base_df.join(master, on="party_code", how="left")

        joined = joined.filter(
            F.to_date(F.col("scrub_output_date")) >= F.to_date(F.col("reference_dt"))
        ).filter(
            F.to_date(F.col("scrub_output_date")) <=
            F.add_months(F.to_date(F.col("reference_dt")), config.RETRO_MAX_SCRUB_MONTHS)
        )

        w = Window.partitionBy("party_code", "reference_dt").orderBy(F.desc("scrub_output_date"))
        resolved = (
            joined
            .withColumn("_rn", F.row_number().over(w))
            .filter(F.col("_rn") == 1)
            .drop("_rn")
            .withColumnRenamed("scrub_output_date", "resolved_scrub_date")
        )

        return resolved

    def _load_tradeline(self, resolved_df: DataFrame) -> DataFrame:
        logger.info("[RetroPipeline] Loading tradeline for resolved snapshots")
        key_map = resolved_df.select(
            "party_code", "reference_dt", "customer_scrub_key",
            F.col("resolved_scrub_date").alias("scrub_output_date")
        )
        tl = self.spark.table(config.TRADELINE_TABLE)
        return tl.join(
            F.broadcast(key_map),
            on=["customer_scrub_key", "scrub_output_date"],
            how="inner"
        )

    def _load_enquiry(self, resolved_df: DataFrame) -> DataFrame:
        logger.info("[RetroPipeline] Loading enquiry for resolved party_codes")
        key_map = resolved_df.select("party_code", "reference_dt").distinct()
        master  = (
            self.spark.table(config.MASTER_TABLE)
            .select("customer_scrub_key", "party_code")
            .distinct()
        )
        party_with_key = key_map.join(F.broadcast(master), on="party_code", how="left")
        enq = self.spark.table(config.ENQUIRY_TABLE)
        return (
            enq
            .join(F.broadcast(party_with_key), on="customer_scrub_key", how="inner")
            .filter(F.to_date(F.col("inq_date")) <= F.to_date(F.col("reference_dt")))
        )

    def run(
        self,
        source_table: str,
        tradeline_output_table: str,
        enquiry_output_table: str,
        retro_max_months: int = None,
        chunk_size: Optional[int] = None,
    ):
        """
        Run full retro feature pipeline.

        Parameters
        ----------
        source_table            : Input table with [party_code, reference_dt]
        tradeline_output_table  : Output Delta table for tradeline features
        enquiry_output_table    : Output Delta table for enquiry features
        retro_max_months        : Override RETRO_MAX_SCRUB_MONTHS if needed
        chunk_size              : If set, process input in chunks of N party_codes
                                  to avoid OOM on very large retro sets (e.g. 50M rows).
                                  Set to None to process all at once (default for < 10M rows).
        """
        if retro_max_months is not None:
            config.RETRO_MAX_SCRUB_MONTHS = retro_max_months

        logger.info("=" * 60)
        logger.info(f"[RetroPipeline] START")
        logger.info(f"  source_table           : {source_table}")
        logger.info(f"  tradeline_output       : {tradeline_output_table}")
        logger.info(f"  enquiry_output         : {enquiry_output_table}")
        logger.info(f"  retro_window_months    : {config.RETRO_MAX_SCRUB_MONTHS}")
        logger.info("=" * 60)

        base_df = (
            self.spark.table(source_table)
            .select("party_code", "reference_dt")
            .dropDuplicates(["party_code", "reference_dt"])
        )

        resolved_df = self._resolve_scrub_snapshot(base_df)
        resolved_df.cache()

        # Tradeline features
        tl_df = self._load_tradeline(resolved_df)
        feats = self.run_tradeline_categories(tl_df, "retro")
        if feats is not None:
            self.write_features(feats, "tradeline", "retro", tradeline_output_table)
        else:
            logger.warning("[RetroPipeline] No tradeline features — check master table.")

        # Enquiry features
        enq_df = self._load_enquiry(resolved_df)
        feats  = self.run_enquiry_categories(enq_df, "retro")
        if feats is not None:
            self.write_features(feats, "enquiry", "retro", enquiry_output_table)

        resolved_df.unpersist()
        logger.info("[RetroPipeline] DONE")
