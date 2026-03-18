# pipeline/retro_pipeline.py
# =============================================================================
# Retro Pipeline
# =============================================================================
# PK     : party_code, reference_dt
# As-of  : reference_dt  (the loan/application reference date)
#
# Input  : Delta table with columns [party_code, reference_dt]
#
# Scrub Selection Logic:
#   For each (party_code, reference_dt), find the latest scrub_output_date
#   such that:
#       reference_dt <= scrub_output_date <= reference_dt + RETRO_MAX_SCRUB_MONTHS
#
#   If no scrub exists in this window → row is excluded (logged as warning).
#   RETRO_MAX_SCRUB_MONTHS is configurable (default 12).
#
# Why this logic:
#   The scrub must be ON or AFTER reference_dt so bureau data reflects the
#   state at the time of the event. We cap at +12 months to avoid using
#   a scrub that is too far in the future (stale data).
#   The LATEST scrub in the window gives the most complete snapshot.
#
# Enquiry features:
#   Computed directly from enquiry table filtered to inq_date <= reference_dt.
#   No scrub selection needed — enquiries have real event dates.
# =============================================================================

from typing import List
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from pipeline.base_pipeline import BasePipeline
from core.logger import get_logger
from config import config

logger = get_logger(__name__)


class RetroPipeline(BasePipeline):

    def __init__(self, spark: SparkSession):
        super().__init__(spark)

    def get_pk_cols(self) -> List[str]:
        return config.RETRO_PK_COLS   # ["party_code", "reference_dt"]

    def get_as_of_col(self) -> str:
        return config.REFERENCE_DT_COL   # "reference_dt"

    def get_mode_suffix(self) -> str:
        return "retro"

    def _resolve_scrub_snapshot(self, base_df: DataFrame) -> DataFrame:
        """
        For each (party_code, reference_dt), resolve the best scrub snapshot:
          1. Map party_code → customer_scrub_key via master table
          2. Find all scrub_output_dates for that customer
          3. Keep only those where:
               reference_dt <= scrub_output_date <= reference_dt + RETRO_MAX_SCRUB_MONTHS
          4. Pick the LATEST (most complete) scrub in that window

        Returns columns:
          party_code, reference_dt, customer_scrub_key, resolved_scrub_date
        """
        logger.info(f"[RetroPipeline] Resolving scrub snapshots | window={config.RETRO_MAX_SCRUB_MONTHS}m")

        # Step 1: party_code → customer_scrub_key (latest scrub per party)
        master = (
            self.spark.table(config.MASTER_TABLE)
            .select("customer_scrub_key", "party_code", "scrub_output_date")
            .distinct()
        )

        # Step 2: Join input base with master to get all (party, scrub_date) combos
        joined = base_df.join(master, on="party_code", how="left")

        # Step 3: Window filter
        #   scrub_output_date >= reference_dt
        #   scrub_output_date <= reference_dt + RETRO_MAX_SCRUB_MONTHS months
        joined = joined.filter(
            F.to_date(F.col("scrub_output_date")) >= F.to_date(F.col("reference_dt"))
        ).filter(
            F.to_date(F.col("scrub_output_date")) <=
            F.add_months(F.to_date(F.col("reference_dt")), config.RETRO_MAX_SCRUB_MONTHS)
        )

        # Step 4: Pick latest scrub per (party_code, reference_dt)
        w = Window.partitionBy("party_code", "reference_dt").orderBy(F.desc("scrub_output_date"))
        resolved = (
            joined
            .withColumn("_rn", F.row_number().over(w))
            .filter(F.col("_rn") == 1)
            .drop("_rn")
            .withColumnRenamed("scrub_output_date", "resolved_scrub_date")
        )

        # Log unresolved
        total    = base_df.count()
        resolved_count = resolved.count()
        unresolved = total - resolved_count
        logger.info(f"  Total input   : {total}")
        logger.info(f"  Resolved      : {resolved_count}")
        if unresolved > 0:
            logger.warning(f"  Unresolved    : {unresolved} (no scrub in {config.RETRO_MAX_SCRUB_MONTHS}m window)")

        return resolved

    def _load_tradeline(self, resolved_df: DataFrame) -> DataFrame:
        """
        Load tradeline rows for resolved (customer_scrub_key, resolved_scrub_date).
        Joins back party_code and reference_dt as the PK/as-of columns.
        """
        logger.info("[RetroPipeline] Loading tradeline for resolved scrub snapshots")

        key_map = resolved_df.select(
            "party_code",
            "reference_dt",
            "customer_scrub_key",
            F.col("resolved_scrub_date").alias("scrub_output_date")
        )

        tl = self.spark.table(config.TRADELINE_TABLE)
        return (
            tl.join(key_map, on=["customer_scrub_key", "scrub_output_date"], how="inner")
            # party_code and reference_dt now present in each tradeline row
            # Feature compute() will use reference_dt as as_of_col
        )

    def _load_enquiry(self, resolved_df: DataFrame) -> DataFrame:
        """
        Load enquiry rows for resolved party_codes.
        Filter: inq_date <= reference_dt (all historical enquiries up to the event).
        No scrub selection needed — enquiries have real dates.
        """
        logger.info("[RetroPipeline] Loading enquiry for resolved party_codes")

        key_map = resolved_df.select("party_code", "reference_dt").distinct()

        # Map party_code → customer_scrub_key for enquiry join
        master = (
            self.spark.table(config.MASTER_TABLE)
            .select("customer_scrub_key", "party_code")
            .distinct()
        )
        party_with_key = key_map.join(master, on="party_code", how="left")

        enq = self.spark.table(config.ENQUIRY_TABLE)
        return (
            enq
            .join(party_with_key, on="customer_scrub_key", how="inner")
            .filter(F.to_date(F.col("inq_date")) <= F.to_date(F.col("reference_dt")))
        )

    def run(
        self,
        source_table: str,
        tradeline_output_table: str,
        enquiry_output_table: str,
        retro_max_months: int = None,
    ):
        """
        Run full retro feature pipeline.

        Parameters
        ----------
        source_table            : Fully qualified table with [party_code, reference_dt]
        tradeline_output_table  : Output Delta table for tradeline features
        enquiry_output_table    : Output Delta table for enquiry features
        retro_max_months        : Override RETRO_MAX_SCRUB_MONTHS if needed
        """
        if retro_max_months is not None:
            config.RETRO_MAX_SCRUB_MONTHS = retro_max_months

        logger.info("="*60)
        logger.info(f"[RetroPipeline] START")
        logger.info(f"  source_table             : {source_table}")
        logger.info(f"  tradeline_output_table   : {tradeline_output_table}")
        logger.info(f"  enquiry_output_table     : {enquiry_output_table}")
        logger.info(f"  retro_window_months      : {config.RETRO_MAX_SCRUB_MONTHS}")
        logger.info("="*60)

        # Load input
        base_df = (
            self.spark.table(source_table)
            .select("party_code", "reference_dt")
            .dropDuplicates(["party_code", "reference_dt"])
        )
        logger.info(f"  Input rows: {base_df.count()}")

        # Resolve scrub snapshots
        resolved_df = self._resolve_scrub_snapshot(base_df)
        resolved_df.cache()

        # Tradeline features
        tl_df = self._load_tradeline(resolved_df)
        n = tl_df.count()
        logger.info(f"  Tradeline rows loaded: {n}")
        if n > 0:
            feats = self.run_tradeline_categories(tl_df, "retro")
            if feats:
                self.write_features(feats, "tradeline", "retro", tradeline_output_table)
        else:
            logger.warning("[RetroPipeline] No tradeline rows — check master table.")

        # Enquiry features
        enq_df = self._load_enquiry(resolved_df)
        n = enq_df.count()
        logger.info(f"  Enquiry rows loaded: {n}")
        if n > 0:
            feats = self.run_enquiry_categories(enq_df, "retro")
            if feats:
                self.write_features(feats, "enquiry", "retro", enquiry_output_table)
        else:
            logger.warning("[RetroPipeline] No enquiry rows.")

        resolved_df.unpersist()
        logger.info("[RetroPipeline] DONE")
