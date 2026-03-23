# features/tradeline/grp07_delinquency.py  [OPTIMISED]
# =============================================================================
# Group 07 — Delinquency / DPD Behaviour
# =============================================================================
# CRITICAL FIX: replaced 73-WHEN nested _slot() chain with F.array + element_at.
# Old plan nodes: ~40,000.  New plan nodes: ~300.  Eliminates executor OOM crash.
#
# Slot semantics (unchanged):
#   _md  = ceil(months_between(as_of_dt, last_reporting_pymt_dt))
#   slot = k - _md  (k=0=as_of, k=1=1m before, ...)
#   slot<0 → gap→NULL  | slot 0 → days_past_due | slot 1-36 → _NN | >36 → NULL
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List, Optional

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger
from core.date_utils import parse_date
from core.utils import build_history_array, resolve_slot, build_window_cols

logger = get_logger(__name__)

N_HISTORY = 36
DPD_HIST  = [f"days_past_due_{str(i).zfill(2)}" for i in range(1, N_HISTORY + 1)]

SECURED_CODES = {
    "47","58","168","172","173","175","181","184","185","191","195",
    "197","198","199","200","219","221","222","223","240","241","243","246","248",
}
CC_CODES = {"5","213","214","220","224","225"}
HL_CODES = {"58","195","168","240"}
GL_CODES = {"191","243"}
AL_CODES = {"47","173","172","221","222","223","246"}
PL_CODE  = "123"


def _all_history(product_filter: Optional[F.Column] = None) -> F.Column:
    """MAX DPD across all 37 raw slots via array_max — no as_of alignment, plan-cheap."""
    val = F.array_max(F.col("_dpd_arr"))
    if product_filter is not None:
        val = F.when(product_filter, val).otherwise(F.lit(None).cast("double"))
    return val


class DelinquencyDPDFeatures(TradelineFeatureBase):
    """Group 07: Delinquency / DPD Behaviour (array-optimised)."""

    CATEGORY = "grp07_delinquency"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        df = (
            df
            .withColumn("_rpt_dt",    parse_date("last_reporting_pymt_dt"))
            .withColumn("_as_of_dt",  parse_date(as_of_col))
            .withColumn("_open_dt",   parse_date("open_dt"))
            .withColumn("_closed_dt", parse_date("closed_dt"))
        )

        df = df.withColumn(
            "_is_active",
            F.when(
                (F.col("_open_dt") <= F.col("_as_of_dt")) &
                (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt"))),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        df = df.withColumn(
            "_md",
            F.ceil(F.months_between(F.col("_as_of_dt"), F.col("_rpt_dt"))).cast("int")
        )

        df = (
            df
            .withColumn("_loan_am",
                F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
                 .otherwise(F.lit(None).cast("double")))
            .withColumn("_past_due_am",
                F.when(F.col("past_due_am") > 0, F.col("past_due_am").cast("double"))
                 .otherwise(F.lit(0.0)))
        )

        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))
        df = (
            df
            .withColumn("_is_usl",       ~F.col("_acct").isin(SECURED_CODES))
            .withColumn("_is_usl_gt50k",
                ~F.col("_acct").isin(SECURED_CODES) &
                F.col("_loan_am").isNotNull() & (F.col("_loan_am") > 50000))
            .withColumn("_is_cc",  F.col("_acct").isin(CC_CODES))
            .withColumn("_is_hl",  F.col("_acct").isin(HL_CODES))
            .withColumn("_is_gl",  F.col("_acct").isin(GL_CODES))
            .withColumn("_is_al",  F.col("_acct").isin(AL_CODES))
            .withColumn("_is_pl",  F.col("_acct") == PL_CODE)
        )

        # ── KEY OPTIMISATION: build DPD array ONCE (replaces 73-WHEN loop) ───
        df = build_history_array(df, "days_past_due", DPD_HIST, "_dpd_arr",
                                 clean_negative=True)

        # ── Per-row delinquency recency (array_min of distance-to-DPD) ────────
        # months_from_as_of when DPD at slot s ≥ threshold = _md + s
        df = (
            df
            .withColumn("_dpd30_dist",
                F.array_min(F.transform(
                    F.sequence(F.lit(0), F.lit(N_HISTORY)),
                    lambda s: F.when(
                        F.element_at(F.col("_dpd_arr"), (s + F.lit(1)).cast("int")) >= F.lit(30.0) & ((F.col("_md") + s) >= F.lit(0)),
                        (F.col("_md") + s).cast("double")
                    ).otherwise(F.lit(None).cast("double"))
                ))
            )
            .withColumn("_dpd15_dist",
                F.array_min(F.transform(
                    F.sequence(F.lit(0), F.lit(N_HISTORY)),
                    lambda s: F.when(
                        F.element_at(F.col("_dpd_arr"), (s + F.lit(1)).cast("int")) > F.lit(15.0) & ((F.col("_md") + s) >= F.lit(0)),
                        (F.col("_md") + s).cast("double")
                    ).otherwise(F.lit(None).cast("double"))
                ))
            )
            .withColumn("_dpd0_dist",
                F.array_min(F.transform(
                    F.sequence(F.lit(0), F.lit(N_HISTORY)),
                    lambda s: F.when(
                        F.element_at(F.col("_dpd_arr"), (s + F.lit(1)).cast("int")) > F.lit(0.0) & ((F.col("_md") + s) >= F.lit(0)),
                        (F.col("_md") + s).cast("double")
                    ).otherwise(F.lit(None).cast("double"))
                ))
            )
        )

        # ── Build window slot-Column lists (array-indexed, plan-cheap) ─────────
        pl_f   = F.col("_is_pl")
        w3     = build_window_cols("_dpd_arr", 3)
        w6     = build_window_cols("_dpd_arr", 6)
        w12    = build_window_cols("_dpd_arr", 12)
        w36    = build_window_cols("_dpd_arr", 36)
        w15    = build_window_cols("_dpd_arr", 15)
        w12_pl = build_window_cols("_dpd_arr", 12, product_filter=pl_f)
        w36_pl = build_window_cols("_dpd_arr", 36, product_filter=pl_f)

        feature_df = df.groupBy(group_cols).agg(

            # Max DPD — all-history (array_max = plan-cheap vs F.greatest(*37cols))
            F.max(_all_history()).alias("max_dpd"),
            F.max(_all_history(F.col("_is_usl"))).alias("max_dpd_usl"),
            F.max(_all_history(F.col("_is_usl_gt50k"))).alias("max_dpd_usl_gt50k"),
            F.max(_all_history(F.col("_is_cc"))).alias("max_dpd_CC"),
            F.max(_all_history(F.col("_is_hl"))).alias("max_dpd_HL"),
            F.max(_all_history(F.col("_is_gl"))).alias("max_dpd_GL"),
            F.max(_all_history(F.col("_is_al"))).alias("max_dpd_AL"),
            F.max(_all_history(F.col("_is_pl"))).alias("max_dpd_pl"),

            # Windowed max DPD (greatest over slot-columns)
            F.max(F.greatest(*w36)).alias("max_dpd_in_3_years"),
            F.max(F.greatest(*w12)).alias("max_dpd_in_12_months"),
            F.max(F.greatest(*w15[3:15])).alias("max_dpd_in_last_3_15_mon"),
            F.max(F.greatest(*w12_pl)).alias("max_dpd_pl_12m"),
            F.max(F.greatest(*w36_pl)).alias("max_dpd_pl_3y"),

            # DPD counts
            F.sum(F.when(F.greatest(*w12).isNotNull(), F.when(F.greatest(*w12) > 0,  F.lit(1)).otherwise(F.lit(0)))).alias("no_of_dpd_in_12_months"),
            F.sum(F.when(F.greatest(*w12).isNotNull(), F.when(F.greatest(*w12) >= 30, F.lit(1)).otherwise(F.lit(0)))).alias("no_of_dpd30_in_12_months"),
            F.sum(F.when(F.greatest(*w12).isNotNull(), F.when(F.greatest(*w12) >= 60, F.lit(1)).otherwise(F.lit(0)))).alias("no_of_dpd60_in_12_months"),
            F.sum(F.when(F.greatest(*w12).isNotNull(), F.when(F.greatest(*w12) >= 90, F.lit(1)).otherwise(F.lit(0)))).alias("no_of_dpd90_in_12_months"),
            F.sum(F.when(F.greatest(*w36).isNotNull(), F.when(F.greatest(*w36) > 0,  F.lit(1)).otherwise(F.lit(0)))).alias("no_of_dpd_in_36_months"),
            F.sum(F.when(F.greatest(*w12_pl).isNotNull(), F.when(F.greatest(*w12_pl) >= 30, F.lit(1)).otherwise(F.lit(0)))).alias("no_of_dpd30_pl_12m"),
            F.sum(F.when(F.greatest(*w36_pl).isNotNull(), F.when(F.greatest(*w36_pl) > 0, F.lit(1)).otherwise(F.lit(0)))).alias("no_of_dpd_pl_36m"),

            # DPD bucket flags
            F.max(F.when((F.greatest(*w3)  >= 10) & (F.greatest(*w3)  < 30), F.lit(1)).otherwise(F.lit(0))).alias("is_10to30_last3_months"),
            F.max(F.when((F.greatest(*w3)  >= 30) & (F.greatest(*w3)  < 60), F.lit(1)).otherwise(F.lit(0))).alias("is_30to60_last3_months"),
            F.max(F.when((F.greatest(*w3)  >= 60) & (F.greatest(*w3)  < 90), F.lit(1)).otherwise(F.lit(0))).alias("is_60to90_last3_months"),
            F.max(F.when((F.greatest(*w6)  >= 10) & (F.greatest(*w6)  < 30), F.lit(1)).otherwise(F.lit(0))).alias("is_10to30_last6_months"),
            F.max(F.when((F.greatest(*w6)  >= 30) & (F.greatest(*w6)  < 60), F.lit(1)).otherwise(F.lit(0))).alias("is_30to60_last6_months"),
            F.max(F.when((F.greatest(*w6)  >= 60) & (F.greatest(*w6)  < 90), F.lit(1)).otherwise(F.lit(0))).alias("is_60to90_last6_months"),
            F.max(F.when((F.greatest(*w12) >= 10) & (F.greatest(*w12) < 30), F.lit(1)).otherwise(F.lit(0))).alias("is_10to30_last12_months"),
            F.max(F.when((F.greatest(*w12) >= 30) & (F.greatest(*w12) < 60), F.lit(1)).otherwise(F.lit(0))).alias("is_30to60_last12_months"),
            F.max(F.when((F.greatest(*w12) >= 60) & (F.greatest(*w12) < 90), F.lit(1)).otherwise(F.lit(0))).alias("is_60to90_last12_months"),

            # Severity flags — all-history (array_max = O(1) per row)
            F.max(F.when(_all_history() >= 180, F.lit(1)).otherwise(F.lit(0))).alias("is_dpd_180"),
            F.max(F.when(_all_history() >=  90, F.lit(1)).otherwise(F.lit(0))).alias("is_dpd_90"),
            F.max(F.when(_all_history() >=  60, F.lit(1)).otherwise(F.lit(0))).alias("is_dpd_60"),
            F.max(F.when(_all_history() >=  30, F.lit(1)).otherwise(F.lit(0))).alias("is_dpd_30"),

            # Severity flags — windowed
            F.max(F.when(F.greatest(*w12) >= 180, F.lit(1)).otherwise(F.lit(0))).alias("is_dpd_180_last12m"),
            F.max(F.when(F.greatest(*w12) >=  90, F.lit(1)).otherwise(F.lit(0))).alias("is_dpd_90_last12m"),
            F.max(F.when(F.greatest(*w12) >=  60, F.lit(1)).otherwise(F.lit(0))).alias("is_dpd_60_last12m"),
            F.max(F.when(F.greatest(*w12) >=  30, F.lit(1)).otherwise(F.lit(0))).alias("is_dpd_30_last12m"),
            F.max(F.when(F.greatest(*w36) >= 180, F.lit(1)).otherwise(F.lit(0))).alias("is_dpd_180_last36m"),
            F.max(F.when(F.greatest(*w36) >=  90, F.lit(1)).otherwise(F.lit(0))).alias("is_dpd_90_last36m"),

            # Recency (pre-computed per-row above — just aggregate with min)
            F.min("_dpd30_dist").alias("months_since_last_dpd30"),
            F.min("_dpd15_dist").alias("MostRecentDPDGT15Month"),
            F.min("_dpd0_dist").alias("MostRecentDPDGT0Month"),

            # Current DPD — slot 0 of array = days_past_due at rpt_dt
            F.max(
                F.when(F.col("_is_active") == 1, F.element_at(F.col("_dpd_arr"), F.lit(1)))
            ).alias("current_dpd"),

            F.sum(F.when(F.col("_is_active") == 1, F.col("_past_due_am"))).alias("current_dpd_due"),
        )

        self._log_done(feature_df)
        return feature_df
