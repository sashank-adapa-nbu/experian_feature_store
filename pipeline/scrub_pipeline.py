# pipeline/scrub_pipeline.py
# =============================================================================
# Scrub Pipeline
# =============================================================================
# PK     : customer_scrub_key, party_code, scrub_output_date
# As-of  : scrub_output_date
#
# party_code is resolved by joining master table on customer_scrub_key.
# This ensures party_code is always present in scrub output for downstream joins.
# =============================================================================

from typing import List
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from pipeline.base_pipeline import BasePipeline
from core.logger import get_logger
from config import config

logger = get_logger(__name__)


class ScrubPipeline(BasePipeline):

    def __init__(self, spark: SparkSession):
        super().__init__(spark)

    def get_pk_cols(self) -> List[str]:
        return config.SCRUB_PK_COLS   # ["customer_scrub_key", "party_code", "scrub_output_date"]

    def get_as_of_col(self) -> str:
        return config.SCRUB_OUTPUT_DATE_COL   # "scrub_output_date"

    def get_mode_suffix(self) -> str:
        return "scrub"

    def _add_party_code(self, df: DataFrame) -> DataFrame:
        """
        Join master table to add party_code to tradeline/enquiry data.
        Master table: customer_scrub_key → party_code (many-to-one).
        We use the latest party_code per customer_scrub_key.
        """
        master = (
            self.spark.table(config.MASTER_TABLE)
            .select("customer_scrub_key", "party_code")
            .distinct()
        )
        # In case of multiple party_codes per key (shouldn't happen), take first
        w = Window.partitionBy("customer_scrub_key").orderBy("party_code")
        master = (master
                  .withColumn("_rn", F.row_number().over(w))
                  .filter(F.col("_rn") == 1)
                  .drop("_rn"))
        return df.join(master, on="customer_scrub_key", how="left")

    def _load_tradeline(self, scrub_date: str) -> DataFrame:
        logger.info(f"[ScrubPipeline] Loading tradeline | date={scrub_date}")
        df = (
            self.spark.table(config.TRADELINE_TABLE)
            .filter(F.col("scrub_output_date") == scrub_date)
        )
        return self._add_party_code(df)

    def _load_enquiry(self, scrub_date: str) -> DataFrame:
        """
        Enquiry rows for customers active in this scrub date.
        Filter: inq_date <= scrub_output_date (no future enquiries).
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
            .join(keys, on="customer_scrub_key", how="inner")
            .filter(F.to_date(F.col("inq_date")) <= F.to_date(F.col("scrub_output_date")))
        )
        return self._add_party_code(enq)

    def _get_all_scrub_dates(self) -> List[str]:
        """All distinct scrub_output_dates in the tradeline table, ordered ascending."""
        rows = (
            self.spark.table(config.TRADELINE_TABLE)
            .select("scrub_output_date")
            .distinct()
            .orderBy("scrub_output_date")
            .collect()
        )
        return [r["scrub_output_date"].strftime("%Y-%m-%d") for r in rows]

    def _get_processed_dates(self) -> List[str]:
        """Scrub dates already written to the output table."""
        out = (f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}."
               f"{config.TRADELINE_FEATURE_TABLE_PREFIX}_scrub")
        try:
            rows = (
                self.spark.table(out)
                .select("scrub_output_date")
                .distinct()
                .collect()
            )
            return [r["scrub_output_date"].strftime("%Y-%m-%d") for r in rows]
        except Exception:
            return []

    def run(self, scrub_date: str):
        """Run full feature pipeline for one scrub_output_date."""
        logger.info(f"{'='*60}")
        logger.info(f"[ScrubPipeline] START | date={scrub_date}")

        # Tradeline features
        tl_df = self._load_tradeline(scrub_date)
        n = tl_df.count()
        logger.info(f"  Tradeline rows: {n}")
        if n > 0:
            feats = self.run_tradeline_categories(tl_df, scrub_date)
            if feats:
                self.write_features(feats, "tradeline", scrub_date)

        # Enquiry features
        enq_df = self._load_enquiry(scrub_date)
        n = enq_df.count()
        logger.info(f"  Enquiry rows: {n}")
        if n > 0:
            feats = self.run_enquiry_categories(enq_df, scrub_date)
            if feats:
                self.write_features(feats, "enquiry", scrub_date)

        logger.info(f"[ScrubPipeline] DONE | date={scrub_date}")

    def run_all(self, skip_processed: bool = True):
        """
        Process all scrub dates.
        skip_processed=True → skip dates already in output table (incremental).
        skip_processed=False → reprocess everything (full refresh).
        """
        all_dates  = self._get_all_scrub_dates()
        done_dates = set(self._get_processed_dates()) if skip_processed else set()
        pending    = [d for d in all_dates if d not in done_dates]

        logger.info(f"[ScrubPipeline] Total={len(all_dates)} | Done={len(done_dates)} | Pending={len(pending)}")

        for i, d in enumerate(pending):
            logger.info(f"\n[ScrubPipeline] {i+1}/{len(pending)} | date={d}")
            try:
                self.run(d)
            except Exception as e:
                logger.error(f"[ScrubPipeline] FAILED | date={d} | error={e}")
                raise

        logger.info("[ScrubPipeline] All dates complete.")
