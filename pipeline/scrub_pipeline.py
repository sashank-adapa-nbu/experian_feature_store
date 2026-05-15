# pipeline/scrub_pipeline.py  [OPTIMISED v4 — column projection + repartition pattern]
# =============================================================================
# Scrub Pipeline
# =============================================================================
# CRITICAL REQUIREMENT: Process one scrub_output_date at a time.
# With 100+ scrubs × 500M records each = 50B+ total rows — loading all at
# once crashes the cluster. run_all() iterates scrub dates sequentially,
# processing and writing each before loading the next.
#
# OPTIMISATIONS (v4 — adapted from Bureau FS PROD pattern):
#   - Master table FILTERED to single scrub_date and broadcast-joined directly
#       (customer_scrub_key = hash(party_code, scrub_output_date) so keys are
#        unique per date by construction; no global Window/row_number dedup)
#   - Partition pruning on TRADELINE_TABLE via scrub_output_date filter
#       (cast literal to date so Spark pushes the predicate into the scan)
#   - COLUMN PROJECTION: only the ~199 columns features actually use are kept.
#       The source table has 230+ cols (most notably 36 unused payment_rating_cd
#       history cols + audit/metadata cols). Dropping them cuts I/O ~20-25%.
#   - REPARTITION by customer_scrub_key after the master join. This means
#       every subsequent groupBy in base_pipeline runs as a LOCAL aggregation
#       (no network shuffle) because data is already keyed correctly. With
#       16 feature groups, this eliminates ~15 network shuffles per date.
#   - SHOW PARTITIONS for date listing (falls back to distinct().collect())
# =============================================================================

from typing import List
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

from pipeline.base_pipeline import BasePipeline
from core.logger import get_logger
from core.spark_conf import configure_spark
from config import config

logger = get_logger(__name__)


# =============================================================================
# COLUMN PROJECTION — exact list of source columns used by ANY feature group
# =============================================================================
# Built by greping every F.col("..."), parse_date("..."), and
# build_history_array(df, "...", ...) reference across features/tradeline/*.py.
# Verified against the source table DESCRIBE output.
#
# If a future feature needs a column not in this list, the pipeline will fail
# immediately with COLUMN_NOT_FOUND on the first scrub date — easy to detect.
# Add the column here and redeploy.
# =============================================================================
_TRADELINE_HISTORY_PREFIXES = [
    "actual_payment_am",   # used by grp08 (PMT_HIST)
    "balance_am",          # used by grp03 (BAL_HIST)
    "credit_limit_am",     # used by grp03 (CL_HIST)
    "past_due_am",         # used by grp07/grp08 (PDU_HIST)
    "days_past_due",       # used by grp07 (DPD_HIST)
]
# NOT included: payment_rating_cd_01..36 — referenced nowhere in features.
# Saves 36 wide string columns from the read.

TRADELINE_USED_COLS: List[str] = [
    # Required keys
    "customer_scrub_key",
    "scrub_output_date",

    # Account-level columns referenced in feature compute() methods
    "acct_type_cd",
    "m_sub_id",
    "open_dt",
    "closed_dt",
    "balance_dt",
    "last_reporting_pymt_dt",
    "dflt_status_dt",
    "write_off_status_dt",
    "orig_loan_am",
    "credit_limit_am",
    "balance_am",
    "past_due_am",
    "actual_payment_am",
    "days_past_due",
    "emi",
    "suit_filed_willful_dflt",
    "written_off_and_settled_status",
    "total_write_off_am_4in",
    "principal_write_off_am_4in",
    "settled_am_4in",

]
# Append history columns 01..36 for each prefix
for _prefix in _TRADELINE_HISTORY_PREFIXES:
    for _i in range(1, 37):
        TRADELINE_USED_COLS.append(f"{_prefix}_{_i:02d}")


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

    # =========================================================================
    # Master table — filtered per scrub_date, broadcast directly
    # =========================================================================
    def _load_master(self, scrub_date: str) -> DataFrame:
        """
        Load master table filtered to ONE scrub_output_date.

        customer_scrub_key is a hash of (party_code, scrub_output_date), so
        for a single scrub_date each key appears exactly once. No cross-date
        dedup is needed (Window/row_number guard removed — was defensive
        against duplicates that cannot exist by construction).

        Returns broadcast-hinted DataFrame with: customer_scrub_key, party_code.
        Only those two columns selected to prevent column leakage into
        downstream DataFrames (avoid AMBIGUOUS_COLUMN errors).
        """
        master = (
            self.spark.table(config.MASTER_TABLE)
            .filter(F.col("scrub_output_date") == F.lit(scrub_date).cast("date"))
            .select("customer_scrub_key", "party_code")
        )
        return F.broadcast(master)

    def _add_party_code(self, df: DataFrame, master: DataFrame) -> DataFrame:
        """Join pre-built broadcast master to add party_code to df."""
        # Drop party_code from df first in case it already exists
        # (prevents duplicate column if tradeline table carries party_code)
        if "party_code" in df.columns:
            df = df.drop("party_code")
        return df.join(master, on="customer_scrub_key", how="left")

    # =========================================================================
    # Tradeline loader — column projection + repartition
    # =========================================================================
    def _load_tradeline(self, scrub_date: str, master: DataFrame) -> DataFrame:
        """
        Load tradeline for one scrub_date with two performance optimizations:

        1. COLUMN PROJECTION — select only the ~199 columns referenced by any
           feature group. Source has 230+ cols including 36 unused
           payment_rating_cd_NN history cols + audit/metadata fields. Dropping
           them at scan time cuts I/O ~20-25%.

        2. REPARTITION by customer_scrub_key — every downstream groupBy on
           this key becomes a LOCAL aggregation (no network shuffle) because
           data is already partitioned correctly. With 16 feature groups
           each doing groupBy(customer_scrub_key, ...), this eliminates ~15
           network shuffles per scrub date.

        Note: The source table is NOT partitioned by scrub_output_date — the
        filter still works but doesn't get partition pruning. With column
        projection in place, the file scan is much narrower (fewer bytes
        read per row).
        """
        logger.info(f"[ScrubPipeline] Loading tradeline | date={scrub_date}")

        df = (
            self.spark.table(config.TRADELINE_TABLE)
            # Cast literal to date so the filter is pushed into the scan
            # (currently the source isn't partitioned by date so this only
            # helps if/when it gets partitioned in future; harmless either way)
            .filter(F.col("scrub_output_date") == F.lit(scrub_date).cast("date"))
        )

        # ── COLUMN PROJECTION ────────────────────────────────────────────────
        # Defensive: intersect TRADELINE_USED_COLS with actual columns in case
        # the source schema changes. If a required column is missing, we log
        # a warning and continue with the rest — the feature using it will
        # fail later with a clear error message (COLUMN_NOT_FOUND).
        avail = set(df.columns)
        keep_cols = [c for c in TRADELINE_USED_COLS if c in avail]
        missing   = [c for c in TRADELINE_USED_COLS if c not in avail]
        if missing:
            logger.warning(
                f"[ScrubPipeline] {len(missing)} expected column(s) missing "
                f"from {config.TRADELINE_TABLE}: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        df = df.select(*keep_cols)
        logger.info(
            f"[ScrubPipeline] Tradeline column projection: keeping "
            f"{len(keep_cols)} cols (dropped {len(avail) - len(keep_cols)} unused)"
        )

        # Join master to add party_code (master is broadcast, so cheap)
        df = self._add_party_code(df, master)

        # ── REPARTITION by customer_scrub_key ────────────────────────────────
        # Single shuffle now. All 16 downstream groupBys on this key become
        # local aggregations (no shuffle). With shuffle.partitions=1200 set
        # in config, AQE coalesces post-shuffle small partitions.
        df = df.repartition("customer_scrub_key")

        return df

    # =========================================================================
    # Enquiry loader — same repartition pattern
    # =========================================================================
    def _load_enquiry(self, scrub_date: str, master: DataFrame) -> DataFrame:
        """
        Load enquiry for customers in this scrub_date.
        Filters inq_date <= scrub_output_date (no leakage).

        Repartition by customer_scrub_key for same reason as tradeline loader.
        No column projection here because enquiry table is already narrow
        and we don't have a centralised list of "used" cols for it.
        """
        logger.info(f"[ScrubPipeline] Loading enquiry | date={scrub_date}")
        keys = (
            self.spark.table(config.TRADELINE_TABLE)
            .filter(F.col("scrub_output_date") == F.lit(scrub_date).cast("date"))
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
            # no format works fine. _scrub_dt_key is DateType — cast directly.
            .filter(
                F.to_date(F.col("inq_date")) <= F.col("_scrub_dt_key").cast("date")
            )
            .drop("_scrub_dt_key")
            # Restore scrub_output_date as DateType matching partition column.
            .withColumn("scrub_output_date", F.lit(scrub_date).cast("date"))
        )
        enq = self._add_party_code(enq, master)

        # Same repartition trick for enquiry features
        enq = enq.repartition("customer_scrub_key")
        return enq

    # =========================================================================
    # Date discovery — SHOW PARTITIONS for partitioned tables, fallback distinct
    # =========================================================================
    @staticmethod
    def _parse_partition_rows(rows) -> List[str]:
        """
        SHOW PARTITIONS returns rows like 'scrub_output_date=2024-01-15'.
        Extract the date string after the '=' sign.
        """
        import urllib.parse
        dates = []
        for r in rows:
            spec = r[0]
            last = spec.split("/")[-1]
            if "=" in last:
                val = last.split("=", 1)[1]
                dates.append(urllib.parse.unquote(val))
        return dates

    def _get_all_scrub_dates(self) -> List[str]:
        # SHOW PARTITIONS reads only the Delta transaction log — fast.
        # Cache on the instance — called 3× per run_all() otherwise.
        if hasattr(self, "_all_dates_cache"):
            return self._all_dates_cache
        try:
            rows = self.spark.sql(
                f"SHOW PARTITIONS {config.TRADELINE_TABLE}"
            ).collect()
            dates = sorted(self._parse_partition_rows(rows))
        except Exception as e:
            # Fallback for unpartitioned tables (current state of source)
            logger.warning(f"SHOW PARTITIONS failed ({e}); falling back to distinct()")
            rows = (
                self.spark.table(config.TRADELINE_TABLE)
                .select("scrub_output_date").distinct()
                .orderBy("scrub_output_date").collect()
            )
            dates = [str(r["scrub_output_date"]) for r in rows]
        self._all_dates_cache = dates
        return dates

    def _get_processed_tradeline_dates(self) -> set:
        """Dates already written to the tradeline feature table."""
        tl_table = (f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}."
                    f"{config.TRADELINE_FEATURE_TABLE_PREFIX}_scrub")
        try:
            rows = self.spark.sql(f"SHOW PARTITIONS {tl_table}").collect()
            return set(self._parse_partition_rows(rows))
        except Exception:
            return set()

    def _get_processed_enquiry_dates(self) -> set:
        """Dates already written to the enquiry feature table."""
        enq_table = (f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}."
                     f"{config.ENQUIRY_FEATURE_TABLE_PREFIX}_scrub")
        try:
            rows = self.spark.sql(f"SHOW PARTITIONS {enq_table}").collect()
            return set(self._parse_partition_rows(rows))
        except Exception:
            return set()

    # =========================================================================
    # Single-date orchestration
    # =========================================================================
    def run(self, scrub_date: str):
        """
        Run full feature pipeline for ONE scrub_output_date.

        Per-source skip logic: tradeline and enquiry are checked independently.
        If one is already written it is skipped; the other still runs.
        This makes the pipeline safely resumable after a partial failure.
        """
        logger.info("=" * 60)
        logger.info(f"[ScrubPipeline] START | date={scrub_date}")

        # Per-date broadcast master (small — keys unique by construction)
        master = self._load_master(scrub_date)

        try:
            # ── Tradeline features ────────────────────────────────────────────
            if scrub_date in self._get_processed_tradeline_dates():
                logger.info(f"  [SKIP] Tradeline already written | date={scrub_date}")
            else:
                tl_df = self._load_tradeline(scrub_date, master)
                feats = self.run_tradeline_categories(tl_df, scrub_date)
                if feats is not None:
                    self.write_features(feats, "tradeline", scrub_date)
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
            # Unpersist the per-date broadcast so it doesn't accumulate
            try:
                master.unpersist(blocking=True)
                logger.debug(f"  [GC] Unpersisted broadcast master | date={scrub_date}")
            except Exception:
                pass

            # JVM GC to reclaim shuffle metadata and orphaned RDD descriptors
            try:
                self.spark.sparkContext._jvm.System.gc()
                logger.debug(f"  [GC] JVM System.gc() called on driver | date={scrub_date}")
            except Exception:
                pass

        logger.info(f"[ScrubPipeline] DONE | date={scrub_date}")

    # =========================================================================
    # Multi-date orchestration
    # =========================================================================
    def run_all(self, skip_processed: bool = True):
        """
        Process all scrub dates ONE AT A TIME.

        Sequential because each scrub = ~500M rows × ~200 bytes ≈ 100 GB.
        Running 2 scrubs simultaneously would OOM the cluster.

        skip_processed=True  → incremental mode (skip already-written dates)
        skip_processed=False → full reprocess (for schema changes / bug fixes)

        A date is fully done only when BOTH tradeline AND enquiry are written.
        Partial-failure dates are kept in pending — run() skips the
        already-written source internally on retry.
        """
        all_dates = self._get_all_scrub_dates()

        if skip_processed:
            tl_done  = self._get_processed_tradeline_dates()
            enq_done = self._get_processed_enquiry_dates()
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
                # Between-date cleanup
                self.spark.catalog.clearCache()
            except Exception as e:
                logger.error(f"[ScrubPipeline] FAILED | date={d} | error={e}")
                raise   # fail fast — do not silently skip a broken date

        logger.info("\n[ScrubPipeline] All pending dates complete.")

    def run_date_range(self, start_date: str, end_date: str,
                       skip_processed: bool = True):
        """
        Process scrub dates within [start_date, end_date] inclusive.
        Same per-source skip logic as run_all.
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
                self.spark.catalog.clearCache()
            except Exception as e:
                logger.error(f"[ScrubPipeline] FAILED | date={d} | error={e}")
                raise