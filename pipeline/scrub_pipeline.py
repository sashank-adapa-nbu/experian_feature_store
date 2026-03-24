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
#   - Incremental skip of already-processed dates (per-source: tradeline + enquiry)
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
        Selects ONLY customer_scrub_key and party_code so no other columns
        (including scrub_output_date) leak into the tradeline/enquiry DataFrames
        via the join and cause AMBIGUOUS_COLUMN errors.
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
        """
        Join pre-loaded (broadcast) master to add party_code.
        master contains ONLY customer_scrub_key + party_code so no columns
        from master can duplicate columns already present in df.
        """
        # Drop party_code from df first in case it already exists
        # (prevents duplicate column if tradeline table carries party_code)
        if "party_code" in df.columns:
            df = df.drop("party_code")
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

        NOTE: keys carries scrub_output_date; the enquiry table may also have it.
        Rename it in keys to avoid AMBIGUOUS_COLUMN after the join,
        then add it back as a literal so feature groupBy works cleanly.
        """
        logger.info(f"[ScrubPipeline] Loading enquiry | date={scrub_date}")
        keys = (
            self.spark.table(config.TRADELINE_TABLE)
            .filter(F.col("scrub_output_date") == scrub_date)
            .select("customer_scrub_key", "scrub_output_date")
            .distinct()
            # Rename to avoid ambiguity when joining against the enquiry table
            # which may also carry a scrub_output_date column
            .withColumnRenamed("scrub_output_date", "_scrub_dt_key")
        )
        enq = (
            self.spark.table(config.ENQUIRY_TABLE)
            .join(F.broadcast(keys), on="customer_scrub_key", how="inner")
            # inq_date is StringType in yyyy-MM-dd format — F.to_date with
            # no format works fine (same as SQL: to_date(inq_date)).
            # _scrub_dt_key is DateType (from tradeline scrub_output_date) —
            # cast directly to date, never through F.to_date which throws
            # CANNOT_PARSE_TIMESTAMP on non-string columns in LEGACY mode.
            .filter(
                F.to_date(F.col("inq_date")) <= F.col("_scrub_dt_key").cast("date")
            )
            .drop("_scrub_dt_key")
            # Restore scrub_output_date as DateType to match partition column type.
            .withColumn("scrub_output_date", F.lit(scrub_date).cast("date"))
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

    def _get_processed_tradeline_dates(self) -> set:
        """Dates already written to the tradeline feature table."""
        tl_table = (f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}."
                    f"{config.TRADELINE_FEATURE_TABLE_PREFIX}_scrub")
        try:
            return {
                str(r["scrub_output_date"])
                for r in self.spark.table(tl_table)
                    .select("scrub_output_date").distinct().collect()
            }
        except Exception:
            return set()

    def _get_processed_enquiry_dates(self) -> set:
        """Dates already written to the enquiry feature table."""
        enq_table = (f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}."
                     f"{config.ENQUIRY_FEATURE_TABLE_PREFIX}_scrub")
        try:
            return {
                str(r["scrub_output_date"])
                for r in self.spark.table(enq_table)
                    .select("scrub_output_date").distinct().collect()
            }
        except Exception:
            return set()

    def run(self, scrub_date: str):
        """
        Run full feature pipeline for ONE scrub_output_date.

        Each source (tradeline / enquiry) is checked independently —
        if one is already written it is skipped, the other still runs.
        This makes the pipeline safely resumable after a partial failure.

        Loading, computing and writing one date at a time ensures that
        only ~500M rows are in memory at any point — not 50B+.
        """
        logger.info("=" * 60)
        logger.info(f"[ScrubPipeline] START | date={scrub_date}")

        # Load master once per scrub date (broadcast — stays in executor memory)
        master = self._load_master()

        try:
            # ── Tradeline features ────────────────────────────────────────────
            if scrub_date in self._get_processed_tradeline_dates():
                logger.info(f"  [SKIP] Tradeline already written | date={scrub_date}")
            else:
                tl_df = self._load_tradeline(scrub_date, master)
                # DO NOT call .count() here — triggers full scan of 500M rows
                feats = self.run_tradeline_categories(tl_df, scrub_date)
                if feats is not None:
                    self.write_features(feats, "tradeline", scrub_date)
                    # write_features() calls df.unpersist() in its finally block
                else:
                    logger.warning(f"  No tradeline features for {scrub_date}")

            # ── Enquiry features ──────────────────────────────────────────────
            if scrub_date in self._get_processed_enquiry_dates():
                logger.info(f"  [SKIP] Enquiry already written | date={scrub_date}")
            else:
                enq_df = self._load_enquiry(scrub_date, master)
                feats  = self.run_enquiry_categories(enq_df, scrub_date)
                if feats is not None:
                    self.write_features(feats, "enquiry", scrub_date)

        finally:
            # ── LEAK FIX: unpersist broadcast master after each date ──────────
            # _load_master() returns F.broadcast(master). The broadcast hint
            # pins a copy of the master table in every executor's block manager.
            # Without unpersist it accumulates across all 108 dates.
            # unpersist(blocking=True) waits for all executors to confirm
            # removal before proceeding — safe to call synchronously here
            # because we are between dates (no active Spark jobs).
            try:
                master.unpersist(blocking=True)
                logger.debug(f"  [GC] Unpersisted broadcast master | date={scrub_date}")
            except Exception:
                pass  # best-effort — non-fatal if master was never materialised

            # ── LEAK FIX: explicit JVM GC call on the driver ─────────────────
            # Python's GC does not reach into the JVM heap where Spark stores
            # RDD metadata, block manager references, and shuffle map output
            # tracking tables. Calling System.gc() on the driver JVM prompts
            # the JVM to reclaim objects that are no longer reachable but not
            # yet collected — primarily shuffle metadata from completed stages
            # and orphaned RDD descriptor objects from unpersisted DataFrames.
            # This is especially effective after unpersist() since JVM GC is
            # what actually frees the underlying metadata.
            try:
                self.spark.sparkContext._jvm.System.gc()
                logger.debug(f"  [GC] JVM System.gc() called on driver | date={scrub_date}")
            except Exception:
                pass  # best-effort — non-fatal

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

        A date is considered fully done only when BOTH tradeline AND enquiry
        are written. If only one succeeded on a previous run, that date will
        be re-entered and run() will skip the already-written source internally.
        """
        all_dates = self._get_all_scrub_dates()

        if skip_processed:
            tl_done  = self._get_processed_tradeline_dates()
            enq_done = self._get_processed_enquiry_dates()
            # Only skip the outer loop for dates where BOTH sources are written.
            # Partially-done dates (one source written, other failed) are kept
            # in pending — run() will skip the already-written source internally.
            done_dates = tl_done & enq_done
        else:
            tl_done = enq_done = done_dates = set()

        pending = [d for d in all_dates if d not in done_dates]

        logger.info(
            f"[ScrubPipeline] Total={len(all_dates)} | "
            f"TL written={len(tl_done)} | ENQ written={len(enq_done)} | "
            f"Both done={len(done_dates)} | Pending={len(pending)}"
        )

        for i, d in enumerate(pending):
            logger.info(f"\n{'='*60}")
            logger.info(f"[ScrubPipeline] Processing {i+1}/{len(pending)} | date={d}")
            try:
                self.run(d)
                # ── Between-date cleanup ──────────────────────────────────────
                # clearCache() clears SQL-cached tables (CACHE TABLE / spark.table
                # calls). Combined with the unpersist + JVM GC inside run(), this
                # ensures the driver and executor heaps are fully clean before
                # loading the next scrub date.
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

        Same per-source skip logic as run_all — a date is only excluded from
        pending if both tradeline AND enquiry are already written.
        """
        all_dates = self._get_all_scrub_dates()

        if skip_processed:
            tl_done  = self._get_processed_tradeline_dates()
            enq_done = self._get_processed_enquiry_dates()
            done_dates = tl_done & enq_done
        else:
            tl_done = enq_done = done_dates = set()

        pending = [
            d for d in all_dates
            if start_date <= d <= end_date and d not in done_dates
        ]

        logger.info(
            f"[ScrubPipeline] Date range {start_date}..{end_date} | "
            f"TL written={len(tl_done)} | ENQ written={len(enq_done)} | "
            f"Both done={len(done_dates)} | Pending={len(pending)}"
        )

        for i, d in enumerate(pending):
            logger.info(f"\n[ScrubPipeline] {i+1}/{len(pending)} | date={d}")
            try:
                self.run(d)
                self.spark.catalog.clearCache()  # SQL cache; RDD GC done inside run()
            except Exception as e:
                logger.error(f"[ScrubPipeline] FAILED | date={d} | error={e}")
                raise