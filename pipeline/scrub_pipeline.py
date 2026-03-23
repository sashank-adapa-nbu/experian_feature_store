# pipeline/scrub_pipeline.py  [OPTIMISED]
# =============================================================================
# Scrub Pipeline
# =============================================================================
# CRITICAL REQUIREMENT: Process one scrub_output_date at a time.
# With 100+ scrubs × 500M records each = 50B+ total rows — loading all at
# once crashes the cluster.  run_all() iterates scrub dates sequentially,
# processing and writing each before loading the next.
#
# OPTIMISATIONS:
#   - No df.count() on 500M rows (removed double-materialise)
#   - master table broadcast-joined (< 10MB) instead of shuffle-joined
#   - Partition pruning on TRADELINE_TABLE via scrub_output_date filter
#   - Incremental skip of already-processed dates
#   - Spark conf applied once at pipeline init
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


class ScrubPipeline(BasePipeline):

    def __init__(self, spark: SparkSession):
        super().__init__(spark)
        # Apply scale-tuned Spark conf once at startup
        configure_spark(spark)

    def get_pk_cols(self) -> List[str]:
        return config.SCRUB_PK_COLS

    def get_as_of_col(self) -> str:
        return config.SCRUB_OUTPUT_DATE_COL

    def get_mode_suffix(self) -> str:
        return "scrub"

    def _load_master(self) -> DataFrame:
        """
        Load master table deduplicated to one party_code per customer_scrub_key.
        Cached and broadcast — master is small (< cluster broadcast threshold).
        """
        master = (
            self.spark.table(config.MASTER_TABLE)
            .select("customer_scrub_key", "party_code")
            .distinct()
        )
        w = Window.partitionBy("customer_scrub_key").orderBy("party_code")
        master = (
            master
            .withColumn("_rn", F.row_number().over(w))
            .filter(F.col("_rn") == 1)
            .drop("_rn")
        )
        # Hint Spark to broadcast — master is typically < 5MB
        return F.broadcast(master)

    def _add_party_code(self, df: DataFrame, master: DataFrame) -> DataFrame:
        """Join pre-loaded (broadcast) master to add party_code."""
        return df.join(master, on="customer_scrub_key", how="left")

    def _load_tradeline(self, scrub_date: str, master: DataFrame) -> DataFrame:
        """
        Load tradeline for one scrub_date only.
        Partition pruning: Delta skips all other date partitions.
        """
        logger.info(f"[ScrubPipeline] Loading tradeline | date={scrub_date}")
        df = (
            self.spark.table(config.TRADELINE_TABLE)
            .filter(F.col("scrub_output_date") == scrub_date)
        )
        return self._add_party_code(df, master)

    def _load_enquiry(self, scrub_date: str, master: DataFrame) -> DataFrame:
        """
        Load enquiry for customers in this scrub_date.
        Filters inq_date <= scrub_output_date (no leakage).
        """
        logger.info(f"[ScrubPipeline] Loading enquiry | date={scrub_date}")
        keys = (
            self.spark.table(config.TRADELINE_TABLE)
            .filter(F.col("scrub_output_date") == scrub_date)
            .select("customer_scrub_key", "scrub_output_date")
            .distinct()
        )
        enq = (
            self.spark.table(config.ENQUIRY_TABLE)
            .join(F.broadcast(keys), on="customer_scrub_key", how="inner")
            .filter(F.to_date(F.col("inq_date")) <= F.to_date(F.col("scrub_output_date")))
        )
        return self._add_party_code(enq, master)

    def _get_all_scrub_dates(self) -> List[str]:
        rows = (
            self.spark.table(config.TRADELINE_TABLE)
            .select("scrub_output_date")
            .distinct()
            .orderBy("scrub_output_date")
            .collect()
        )
        return [str(r["scrub_output_date"]) for r in rows]

    def _get_processed_dates(self) -> List[str]:
        out = (f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}."
               f"{config.TRADELINE_FEATURE_TABLE_PREFIX}_scrub")
        try:
            rows = (
                self.spark.table(out)
                .select("scrub_output_date")
                .distinct()
                .collect()
            )
            return [str(r["scrub_output_date"]) for r in rows]
        except Exception:
            return []

    def run(self, scrub_date: str):
        """
        Run full feature pipeline for ONE scrub_output_date.

        Loading, computing and writing one date at a time ensures that
        only ~500M rows are in memory at any point — not 50B+.
        """
        logger.info("=" * 60)
        logger.info(f"[ScrubPipeline] START | date={scrub_date}")

        # Load master once per scrub date (broadcast — stays in executor memory)
        master = self._load_master()

        # ── Tradeline features ────────────────────────────────────────────────
        tl_df = self._load_tradeline(scrub_date, master)
        # DO NOT call .count() here — it triggers a full scan of 500M rows just for logging
        feats = self.run_tradeline_categories(tl_df, scrub_date)
        if feats is not None:
            self.write_features(feats, "tradeline", scrub_date)
        else:
            logger.warning(f"  No tradeline features for {scrub_date}")

        # ── Enquiry features ──────────────────────────────────────────────────
        enq_df = self._load_enquiry(scrub_date, master)
        feats  = self.run_enquiry_categories(enq_df, scrub_date)
        if feats is not None:
            self.write_features(feats, "enquiry", scrub_date)

        logger.info(f"[ScrubPipeline] DONE | date={scrub_date}")

    def run_all(self, skip_processed: bool = True):
        """
        Process all scrub dates ONE AT A TIME.

        WHY SEQUENTIAL (not parallel):
          - Each scrub = ~500M rows × ~200 bytes = ~100 GB of data
          - Running 2 scrubs simultaneously = 200 GB in shuffle memory → OOM
          - Sequential processing with partition overwrite is safe and restartable
          - Databricks Jobs can be scheduled daily; each job processes one new date

        skip_processed=True  → incremental mode (skip already-written dates)
        skip_processed=False → full reprocess (for schema changes / bug fixes)
        """
        all_dates  = self._get_all_scrub_dates()
        done_dates = set(self._get_processed_dates()) if skip_processed else set()
        pending    = [d for d in all_dates if d not in done_dates]

        logger.info(
            f"[ScrubPipeline] Total={len(all_dates)} | Done={len(done_dates)} | "
            f"Pending={len(pending)}"
        )

        for i, d in enumerate(pending):
            logger.info(f"\n{'='*60}")
            logger.info(f"[ScrubPipeline] Processing {i+1}/{len(pending)} | date={d}")
            try:
                self.run(d)
                # Explicitly clear executor caches between dates
                self.spark.catalog.clearCache()
            except Exception as e:
                logger.error(f"[ScrubPipeline] FAILED | date={d} | error={e}")
                raise   # fail fast — do not silently skip a broken date

        logger.info("\n[ScrubPipeline] All pending dates complete.")

    def run_date_range(self, start_date: str, end_date: str,
                       skip_processed: bool = True):
        """
        Process scrub dates within [start_date, end_date] inclusive.
        Useful for backfill of a specific date range without reprocessing all history.
        """
        all_dates  = self._get_all_scrub_dates()
        done_dates = set(self._get_processed_dates()) if skip_processed else set()
        pending    = [
            d for d in all_dates
            if start_date <= d <= end_date and d not in done_dates
        ]
        logger.info(f"[ScrubPipeline] Date range {start_date}..{end_date} | Pending={len(pending)}")
        for i, d in enumerate(pending):
            logger.info(f"\n[ScrubPipeline] {i+1}/{len(pending)} | date={d}")
            try:
                self.run(d)
                self.spark.catalog.clearCache()
            except Exception as e:
                logger.error(f"[ScrubPipeline] FAILED | date={d} | error={e}")
                raise
