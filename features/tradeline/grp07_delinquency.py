# features/tradeline/cat12_delinquency_dpd.py
# =============================================================================
# Category 12 — Delinquency / DPD Behaviour
# =============================================================================
# Source table  : experian_tradeline_segment
#
# ── COLUMN STRUCTURE ─────────────────────────────────────────────────────────
#   last_reporting_pymt_dt  → Payment report anchor date
#   DAYS_PAST_DUE           → Numeric DPD at last_reporting_pymt_dt  (slot 0)
#   DAYS_PAST_DUE_01..36    → Numeric DPD 1–36 months before         (slots 1–36)
#
#   DPD = -1 → Experian null placeholder → treated as NULL
#   DPD =  0 → current (no overdue)
#   DPD >  0 → overdue by that many days
#
# ── SLOT RESOLUTION ──────────────────────────────────────────────────────────
#   month_diff = round(months_between(as_of_dt, last_reporting_pymt_dt))
#
#   +ve → as_of AHEAD of reporting → gap months exist → NULL for those offsets
#   -ve → as_of BEHIND reporting   → skip early slots, start from |month_diff|
#
#   For window offset k (k=0 = as_of month, k=1 = 1m before, ...):
#     actual_slot = k - month_diff
#     slot < 0   → gap (not yet reported) → NULL
#     slot == 0  → DAYS_PAST_DUE
#     slot 1–36  → DAYS_PAST_DUE_NN
#     slot > 36  → beyond history → NULL
#
# ── EXAMPLES ─────────────────────────────────────────────────────────────────
#   Case A: as_of=Apr2025, last_rpt=Jan2025  → month_diff=+3
#     k=0(Apr): slot=-3 → NULL (gap)
#     k=3(Jan): slot= 0 → DAYS_PAST_DUE
#     k=4(Dec): slot= 1 → DAYS_PAST_DUE_01
#
#   Case B: as_of=Oct2024, last_rpt=Jan2025  → month_diff=-3
#     k=0(Oct): slot= 3 → DAYS_PAST_DUE_03
#     k=1(Sep): slot= 4 → DAYS_PAST_DUE_04
#     k=5(May): slot= 8 → DAYS_PAST_DUE_08
#
# ── AGGREGATION ──────────────────────────────────────────────────────────────
#   Each row = one tradeline. groupBy(pk_cols + as_of_col) collapses all
#   tradelines per customer → MAX/SUM/MIN across tradelines gives customer-level.
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List, Optional

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

USL_CODES = {
    "123", "189", "187", "130", "242", "244", "245", "247",
    "167", "169", "170", "176", "177", "178", "179",
    "228", "227", "226", "249",
}
CC_CODES = {"5", "213", "214", "220", "224", "225"}
HL_CODES = {"58", "195", "168", "240"}
GL_CODES = {"191", "243"}
AL_CODES = {"47", "173", "172", "221", "222", "223", "246"}
PL_CODE  = "123"

N_HISTORY = 36
DPD_COLS  = [f"days_past_due_{str(i).zfill(2)}" for i in range(1, N_HISTORY + 1)]


# =============================================================================
# DPD SIGNAL HELPER
# =============================================================================

def _clean(dpd: F.Column) -> F.Column:
    """Clean numeric DPD: treat -1 as NULL."""
    return F.when(dpd >= 0, dpd.cast("double")).otherwise(F.lit(None).cast("double"))


def _all_history(product_filter: Optional[F.Column] = None) -> F.Column:
    """MAX DPD across all 37 raw slots (no as_of alignment). For all-time max features."""
    val = F.greatest(
        _clean(F.col("days_past_due")),
        *[_clean(F.col(c)) for c in DPD_COLS]
    )
    return F.when(product_filter, val) if product_filter is not None else val


class DelinquencyDPDFeatures(TradelineFeatureBase):
    """
    Category 12: Delinquency / DPD Behaviour
    All features use DAYS_PAST_DUE numeric columns only.
    Slot resolution: actual_slot = k - month_diff
    """

    CATEGORY = "grp07_delinquency"

    @staticmethod
    def _slot(k: int, product_filter: Optional[F.Column] = None) -> F.Column:
        """
        Resolve DPD at window offset k from as_of_dt.
        actual_slot = k - month_diff (_md).
        """
        expr = F.lit(None).cast("double")
        for m in range(-N_HISTORY, N_HISTORY + 1):
            slot = k - m
            if slot < 0:
                raw = F.lit(None).cast("double")          # gap
            elif slot == 0:
                raw = _clean(F.col("days_past_due"))
            elif slot <= N_HISTORY:
                raw = _clean(F.col(DPD_COLS[slot - 1]))
            else:
                raw = F.lit(None).cast("double")           # beyond history
            expr = F.when(F.col("_md") == m, raw).otherwise(expr)
        if product_filter is not None:
            expr = F.when(product_filter, expr).otherwise(F.lit(None).cast("double"))
        return expr

    @classmethod
    def _window(cls, w: int,
                product_filter: Optional[F.Column] = None) -> List[F.Column]:
        """Build list of w slot-columns for a window of w months."""
        return [cls._slot(k, product_filter) for k in range(w)]

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        # ── STEP 1: Parse dates ───────────────────────────────────────────────
        def parse_date(c):
            return F.coalesce(
                F.to_date(F.col(c), "dd/MM/yyyy"),
                F.to_date(F.col(c), "yyyy-MM-dd"),
                F.to_date(F.col(c), "MM/dd/yyyy"),
            )

        df = (
            df
            .withColumn("_rpt_dt",   parse_date("last_reporting_pymt_dt"))
            .withColumn("_as_of_dt", parse_date(as_of_col))
        )

        # ── STEP 2: month_diff ────────────────────────────────────────────────
        df = df.withColumn(
            "_md",
            F.round(F.months_between(
                F.col("_as_of_dt"), F.col("_rpt_dt")
            )).cast("int")
        )

        # ── STEP 3: Clean amounts ─────────────────────────────────────────────
        df = (
            df
            .withColumn("_loan_am",
                F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
                 .otherwise(F.lit(None).cast("double")))
            .withColumn("_past_due_am",
                F.when(F.col("past_due_am") > 0, F.col("past_due_am").cast("double"))
                 .otherwise(F.lit(0.0)))
        )

        # ── STEP 4: Product flags ─────────────────────────────────────────────
        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))
        df = (
            df
            .withColumn("_is_usl",
                F.col("_acct").isin(USL_CODES))
            .withColumn("_is_usl_gt50k",
                F.col("_acct").isin(USL_CODES) &
                F.col("_loan_am").isNotNull() & (F.col("_loan_am") > 50000))
            .withColumn("_is_cc",  F.col("_acct").isin(CC_CODES))
            .withColumn("_is_hl",  F.col("_acct").isin(HL_CODES))
            .withColumn("_is_gl",  F.col("_acct").isin(GL_CODES))
            .withColumn("_is_al",  F.col("_acct").isin(AL_CODES))
            .withColumn("_is_pl",  F.col("_acct") == PL_CODE)
        )

        # ── STEP 5: Build windowed slot arrays ────────────────────────────────
        pl_f    = F.col("_is_pl")

        w3      = self._window(3)
        w6      = self._window(6)
        w12     = self._window(12)
        w36     = self._window(36)
        w15     = self._window(15)          # use [3:15] for 3–15m window
        w12_pl  = self._window(12, pl_f)
        w36_pl  = self._window(36, pl_f)

        # ── STEP 6: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # ── Max DPD — all history (no as_of alignment) ───────────────────
            F.max(_all_history()).alias("max_dpd"),
            F.max(_all_history(F.col("_is_usl"))).alias("max_dpd_usl"),
            F.max(_all_history(F.col("_is_usl_gt50k"))).alias("max_dpd_usl_gt50k"),

            # Product max DPD at as_of_dt (k=0)
            F.max(self._slot(0, F.col("_is_cc"))).alias("max_dpd_CC"),
            F.max(self._slot(0, F.col("_is_hl"))).alias("max_dpd_HL"),
            F.max(self._slot(0, F.col("_is_gl"))).alias("max_dpd_GL"),
            F.max(self._slot(0, F.col("_is_al"))).alias("max_dpd_AL"),
            F.max(self._slot(0, F.col("_is_pl"))).alias("max_dpd_pl"),

            # ── Windowed max DPD ──────────────────────────────────────────────
            F.max(F.greatest(*w36)).alias("max_dpd_in_3_years"),
            F.max(F.greatest(*w12)).alias("max_dpd_in_12_months"),
            F.max(F.greatest(*w15[3:15])).alias("max_dpd_in_last_3_15_mon"),
            F.max(F.greatest(*w12_pl)).alias("max_dpd_pl_12m"),
            F.max(F.greatest(*w36_pl)).alias("max_dpd_pl_3y"),

            # ── DPD counts ────────────────────────────────────────────────────
            F.sum(F.when(F.greatest(*w12)    >  0, F.lit(1)).otherwise(F.lit(0))
            ).alias("no_of_dpd_in_12_months"),
            F.sum(F.when(F.greatest(*w12)    >= 30, F.lit(1)).otherwise(F.lit(0))
            ).alias("no_of_dpd30_in_12_months"),
            F.sum(F.when(F.greatest(*w12)    >= 60, F.lit(1)).otherwise(F.lit(0))
            ).alias("no_of_dpd60_in_12_months"),
            F.sum(F.when(F.greatest(*w12)    >= 90, F.lit(1)).otherwise(F.lit(0))
            ).alias("no_of_dpd90_in_12_months"),
            F.sum(F.when(F.greatest(*w36)    >  0, F.lit(1)).otherwise(F.lit(0))
            ).alias("no_of_dpd_in_36_months"),
            F.sum(F.when(F.greatest(*w12_pl) >= 30, F.lit(1)).otherwise(F.lit(0))
            ).alias("no_of_dpd30_pl_12m"),
            F.sum(F.when(F.greatest(*w36_pl) >  0, F.lit(1)).otherwise(F.lit(0))
            ).alias("no_of_dpd_pl_36m"),

            # ── DPD bucket flags ──────────────────────────────────────────────
            F.max(F.when((F.greatest(*w3)  >= 10) & (F.greatest(*w3)  < 30), F.lit(1)).otherwise(F.lit(0))).alias("is_10to30_last3_months"),
            F.max(F.when((F.greatest(*w3)  >= 30) & (F.greatest(*w3)  < 60), F.lit(1)).otherwise(F.lit(0))).alias("is_30to60_last3_months"),
            F.max(F.when((F.greatest(*w3)  >= 60) & (F.greatest(*w3)  < 90), F.lit(1)).otherwise(F.lit(0))).alias("is_60to90_last3_months"),
            F.max(F.when((F.greatest(*w6)  >= 10) & (F.greatest(*w6)  < 30), F.lit(1)).otherwise(F.lit(0))).alias("is_10to30_last6_months"),
            F.max(F.when((F.greatest(*w6)  >= 30) & (F.greatest(*w6)  < 60), F.lit(1)).otherwise(F.lit(0))).alias("is_30to60_last6_months"),
            F.max(F.when((F.greatest(*w6)  >= 60) & (F.greatest(*w6)  < 90), F.lit(1)).otherwise(F.lit(0))).alias("is_60to90_last6_months"),
            F.max(F.when((F.greatest(*w12) >= 10) & (F.greatest(*w12) < 30), F.lit(1)).otherwise(F.lit(0))).alias("is_10to30_last12_months"),
            F.max(F.when((F.greatest(*w12) >= 30) & (F.greatest(*w12) < 60), F.lit(1)).otherwise(F.lit(0))).alias("is_30to60_last12_months"),
            F.max(F.when((F.greatest(*w12) >= 60) & (F.greatest(*w12) < 90), F.lit(1)).otherwise(F.lit(0))).alias("is_60to90_last12_months"),

            # ── Severity flags — all-history ──────────────────────────────────
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

            # ── Delinquency recency ───────────────────────────────────────────
            # MIN(k) where DPD[k] >= threshold = months ago of most recent event
            F.min(F.coalesce(*[
                F.when(self._slot(k) >= 30, F.lit(float(k)))
                for k in range(N_HISTORY + 1)
            ])).alias("months_since_last_dpd30"),

            F.min(F.coalesce(*[
                F.when(self._slot(k) > 15, F.lit(float(k)))
                for k in range(N_HISTORY + 1)
            ])).alias("MostRecentDPDGT15Month"),

            F.min(F.coalesce(*[
                F.when(self._slot(k) > 0, F.lit(float(k)))
                for k in range(N_HISTORY + 1)
            ])).alias("MostRecentDPDGT0Month"),

            # ── Current DPD — slot k=0 ────────────────────────────────────────
            # as_of >= reporting → DAYS_PAST_DUE
            # as_of <  reporting → DAYS_PAST_DUE_NN (correct retro slot)
            F.max(self._slot(0)).alias("current_dpd"),

            # Total past-due amount
            F.sum(F.col("_past_due_am")).alias("current_dpd_due"),
        )

        self._log_done(feature_df)
        return feature_df
