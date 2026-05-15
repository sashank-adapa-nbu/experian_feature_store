# pipeline/base_pipeline.py  [OPTIMISED v3 — broadcast-join pattern, no staging]
# =============================================================================
# Abstract base pipeline — shared logic for scrub and retro modes.
# =============================================================================
# CHANGE LOG (v3) — adapted from Bureau FS PROD pattern:
#   - REMOVED: Delta staging tables. Each group's output is no longer written
#     to a temporary Delta table between compute and join.
#   - REPLACED with: in-memory broadcast-join pattern (same as Bureau FS).
#     Source DataFrame is persisted once (MEMORY_AND_DISK), all 16 feature
#     groups compute their aggregations against this single cached source,
#     and the small per-group outputs are broadcast-joined together.
#   - WHY this is faster:
#       * No Delta write per group (each used to take seconds-to-minutes
#         of metadata + file writes for trivially small outputs).
#       * No Delta read per group during the join phase.
#       * Source DataFrame is repartitioned by customer_scrub_key in
#         _load_tradeline (see scrub_pipeline.py v4), so each compute()'s
#         groupBy runs as a LOCAL aggregation — no shuffle.
#       * Each per-group output is small (one row per customer, max ~10M
#         rows × ~30 cols ≈ few hundred MB). Easily broadcast-joinable.
#   - KEPT: key-list dedup (pk_cols already contains as_of_col in scrub mode,
#     so we strip the duplicate before passing to each compute()).
#   - KEPT: defensive column dedup before final write.
#   - KEPT: blocking=True on unpersist so executors confirm release before
#     the next date starts.
# =============================================================================
# FEATURE LOGIC: unchanged. Each cls().compute(df, pk_cols, as_of_col) is
# called exactly as before (with deduped pk_cols). Left-join sequence on
# join_cols is preserved. Output schema and values are identical.
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

    # =========================================================================
    # Tradeline groups — compute → broadcast-join
    # =========================================================================
    def run_tradeline_categories(self, df: DataFrame, batch_key: str) -> Optional[DataFrame]:
        return self._run_categories(
            df=df,
            batch_key=batch_key,
            feature_classes=TRADELINE_FEATURE_CLASSES,
            source="tradeline",
        )

    # =========================================================================
    # Enquiry groups — compute → broadcast-join
    # =========================================================================
    def run_enquiry_categories(self, df: DataFrame, batch_key: str) -> Optional[DataFrame]:
        return self._run_categories(
            df=df,
            batch_key=batch_key,
            feature_classes=ENQUIRY_FEATURE_CLASSES,
            source="enquiry",
        )

    # =========================================================================
    # Shared compute+broadcast-join driver
    # =========================================================================
    def _run_categories(self, df: DataFrame, batch_key: str,
                        feature_classes: list, source: str) -> Optional[DataFrame]:
        """
        Pattern (adapted from Bureau FS PROD):

          1. Persist the prepared source DataFrame (MEMORY_AND_DISK).
             Source is already repartitioned by customer_scrub_key in
             _load_tradeline/_load_enquiry, so every groupBy below runs
             as a LOCAL aggregation — no network shuffle.

          2. For each feature class:
               result = cls().compute(df, pk_cols_clean, as_of_col)
             Each result is small: one row per (customer, scrub_date)
             with N feature columns.

          3. Reduce all results with broadcast left-joins on join_cols.
             Each feature DataFrame is small enough to broadcast (max ~10M
             rows × few hundred MB), so joins skip shuffles entirely.

          4. Unpersist source, return final joined DataFrame.

        No Delta staging tables are created or written.
        """
        pk_cols   = self.get_pk_cols()
        as_of_col = self.get_as_of_col()

        # ── KEY-LIST DEDUP ────────────────────────────────────────────────────
        # In scrub mode, config.SCRUB_PK_COLS contains "scrub_output_date" AND
        # config.SCRUB_OUTPUT_DATE_COL is also "scrub_output_date". The naive
        # pk_cols + [as_of_col] produces a list with the same column twice.
        # Spark accepts duplicate column names in groupBy() but the resulting
        # DataFrame then has two columns with the same name — Delta refuses
        # to write that schema (COLUMN_ALREADY_EXISTS). Dedup here, preserving
        # the order of first occurrence.
        seen_keys = set()
        join_cols: List[str] = []
        for c in pk_cols + [as_of_col]:
            if c not in seen_keys:
                join_cols.append(c)
                seen_keys.add(c)

        # Build a deduplicated pk_cols WITHOUT as_of_col to pass to each
        # compute(). Each grp's compute() does `group_cols = pk_cols + [as_of_col]`
        # then groupBy(group_cols). If pk_cols already contains as_of_col,
        # groupBy gets a duplicate name and the result DataFrame has two
        # columns with the same name — Delta refuses to write that.
        # Grouping keys are mathematically identical:
        #   groupBy(["a","b","x"] + ["x"]) == groupBy(["a","b","x"])
        pk_cols_clean = [c for c in pk_cols if c != as_of_col]

        logger.info(f"[{self.get_mode_suffix()}] {len(feature_classes)} {source} groups | batch={batch_key}")

        # ── STEP 1: Persist the source ────────────────────────────────────────
        # MEMORY_AND_DISK lets Spark keep hot data in RAM and spill cold
        # partitions to local SSD. With column projection in scrub_pipeline.py
        # (drops ~31 unused cols), the prepared DataFrame is small enough that
        # most of it fits in executor memory at 10-worker scale.
        # DISK_ONLY was used before but every group then re-read from disk —
        # MEMORY_AND_DISK is dramatically faster for repeated reads.
        from pyspark import StorageLevel
        df = df.persist(StorageLevel.MEMORY_AND_DISK)

        # Trigger materialization with a cheap action so the persist actually
        # happens before the first groupBy. Without this, the first group
        # pays the persist cost AND its own compute cost, making timings
        # confusing. df.count() is fine here because we've already repartitioned
        # and projected — counting is fast on a partitioned cached DataFrame.
        try:
            row_count = df.count()
            logger.info(f"  [persist] source materialized | rows={row_count:,}")
        except Exception as e:
            logger.warning(f"  [persist] count() failed (non-fatal): {e}")

        feature_dfs: List[tuple] = []   # list of (category, DataFrame)

        try:
            # ── STEP 2: Compute each group ────────────────────────────────────
            for cls in feature_classes:
                inst = cls()
                category = inst.CATEGORY
                try:
                    logger.info(f"  → {category}")
                    # Pass deduped pk_cols. compute() adds as_of_col itself.
                    result = inst.compute(df, pk_cols_clean, as_of_col)

                    # Defensive column dedup on the per-group result.
                    # If any compute() accidentally returns duplicate column
                    # names, drop them here so the final write doesn't fail.
                    result_cols_seen = set()
                    result_cols_keep = []
                    for c in result.columns:
                        if c not in result_cols_seen:
                            result_cols_keep.append(c)
                            result_cols_seen.add(c)
                    if len(result_cols_keep) < len(result.columns):
                        logger.warning(
                            f"    [{category}] dropped "
                            f"{len(result.columns) - len(result_cols_keep)} "
                            f"duplicate column(s) before join"
                        )
                        result = result.select(result_cols_keep)

                    feature_dfs.append((category, result))
                except Exception as e:
                    logger.error(f"  ✗ {category}: {e}")
                    # Unpersist source before re-raising so we don't leak it
                    df.unpersist(blocking=True)
                    raise
        except Exception:
            raise

        if not feature_dfs:
            try:
                df.unpersist(blocking=True)
            except Exception:
                pass
            return None

        # ── STEP 3: Broadcast-join all per-group results ──────────────────────
        # Each feature DataFrame is small (one row per customer) after the
        # groupBy. Broadcasting them turns each join into a shuffle-free map-
        # side join. The reduce() pattern matches Bureau FS exactly.
        # base = first group's output; subsequent groups are broadcast-joined.
        base_name, base_df = feature_dfs[0]
        logger.info(f"  [join] base = {base_name} | cols={len(base_df.columns)}")

        for cat_name, cat_df in feature_dfs[1:]:
            # Take only non-key columns from the right side so join keys
            # appear exactly once in the output (from base_df).
            feat_cols = [c for c in cat_df.columns if c not in join_cols]
            base_df = base_df.join(
                F.broadcast(cat_df.select(join_cols + feat_cols)),
                on=join_cols,
                how="left",
            )
            logger.info(f"  [join] +{cat_name} | total_cols={len(base_df.columns)}")

        # ── STEP 4: Final-result dedup + date-type enforcement ────────────────
        # Same guards as before — protect against duplicate column names that
        # somehow survived (e.g. two groups returning the same feature) and
        # against Catalyst widening DateType → StringType across the join chain.
        seen = set()
        final_cols = []
        for c in base_df.columns:
            if c not in seen:
                final_cols.append(c)
                seen.add(c)
        if len(final_cols) < len(base_df.columns):
            logger.warning(f"  Dropped {len(base_df.columns)-len(final_cols)} duplicate column(s) after join")
            base_df = base_df.select(final_cols)

        # Re-cast date columns to DateType after long join chain.
        # Catalyst can widen DateType → StringType when resolving column names
        # across many chained joins; this enforcement keeps Delta happy.
        for col_name in self._DATE_COLS:
            if col_name in base_df.columns:
                base_df = base_df.withColumn(col_name, F.col(col_name).cast("date"))

        # ── STEP 5: Attach source-handle for cleanup after write ──────────────
        # We can't unpersist df here because base_df still references it
        # transitively through the broadcast-joined plan. The actual unpersist
        # happens in write_features() after the final write triggers execution.
        base_df._source_df_to_unpersist = df

        return base_df

    # =========================================================================
    # Constants used across pipeline modes
    # =========================================================================
    # Date/timestamp columns that must stay as DateType through the join chain.
    # Spark Catalyst can silently widen these to StringType because each
    # per-group DataFrame is an independent plan — the type is re-resolved
    # from string column names, not from the original schema object.
    _DATE_COLS = {"scrub_output_date", "reference_dt"}

    # =========================================================================
    # Legacy _join — kept for any external callers that pass a list of dfs.
    # Not used by _run_categories anymore (we inline the join logic with
    # F.broadcast() now), but retro_pipeline or ad-hoc notebooks might call it.
    # =========================================================================
    def _join(self, dfs: list, join_cols: List[str]) -> DataFrame:
        """
        Left-join all feature DataFrames on join_cols (legacy entry point).

        Per-group dataframes are small — wrap each right side in F.broadcast()
        for shuffle-free joins. Same semantics as the inline join in
        _run_categories above.
        """
        base_name, base_df = dfs[0]
        for cat_name, cat_df in dfs[1:]:
            feat_cols = [c for c in cat_df.columns if c not in join_cols]
            base_df = base_df.join(
                F.broadcast(cat_df.select(join_cols + feat_cols)),
                on=join_cols,
                how="left",
            )
            logger.info(f"  Joined {cat_name} | total_cols={len(base_df.columns)}")

        seen = set()
        final_cols = []
        for c in base_df.columns:
            if c not in seen:
                final_cols.append(c)
                seen.add(c)
        if len(final_cols) < len(base_df.columns):
            logger.warning(f"  Dropped {len(base_df.columns)-len(final_cols)} duplicate column(s) after join")
            base_df = base_df.select(final_cols)

        for col_name in self._DATE_COLS:
            if col_name in base_df.columns:
                base_df = base_df.withColumn(col_name, F.col(col_name).cast("date"))

        return base_df

    # =========================================================================
    # Write — same as before; additionally unpersists source after success
    # =========================================================================
    def write_features(self, df: DataFrame, source: str, batch_key: str,
                       table_name: Optional[str] = None):
        if table_name is None:
            prefix = (config.TRADELINE_FEATURE_TABLE_PREFIX if source == "tradeline"
                      else config.ENQUIRY_FEATURE_TABLE_PREFIX)
            table_name = f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}.{prefix}_scrub"

        mode     = config.SCRUB_WRITE_MODE if self.get_mode_suffix() == "scrub" else config.RETRO_WRITE_MODE
        part_col = config.PARTITION_COL    if self.get_mode_suffix() == "scrub" else None

        # Pick up the source-DataFrame handle set by _run_categories so we
        # can unpersist it after the final write triggers execution.
        # getattr-with-default keeps this safe when write_features is called
        # from outside the pipeline (e.g. ad-hoc notebooks).
        source_df_to_unpersist = getattr(df, "_source_df_to_unpersist", None)

        try:
            self.writer.write(df=df, table_name=table_name, write_mode=mode,
                              partition_col=part_col)
            logger.info(f"  ✓ Written → {table_name} | mode={mode}")
        finally:
            # Unpersist the cached source AFTER the write finishes — at this
            # point all groups have been materialized through the write, so
            # nothing in the executor block managers references it anymore.
            # blocking=True so executors confirm release before the next date.
            if source_df_to_unpersist is not None:
                try:
                    source_df_to_unpersist.unpersist(blocking=True)
                    logger.debug(f"  [GC] Unpersisted source DF | {source} | date={batch_key}")
                except Exception:
                    pass