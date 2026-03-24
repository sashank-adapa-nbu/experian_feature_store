# features/tradeline/grp08_payment_repayment.py  [OPTIMISED]
# =============================================================================
# Group 08a: Payment Behaviour + Group 08b: Repayment Ratio
# =============================================================================
# CRITICAL FIX: replaced 73-WHEN nested _pmt_slot() / _balance_at_asof() chains
# with F.array() + F.element_at() indexing.
# Plan nodes: ~52,000 → ~400. Eliminates executor OOM crash.
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List, Optional

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger
from core.date_utils import parse_date
from core.utils import build_history_array, resolve_slot, build_window_cols, resolve_slot_at_asof

logger = get_logger(__name__)

CC_CODES  = {"5","213","214","220","224","225"}
PL_CODE   = "123"
STPL_CODE = "242"
SECURED_CODES = {
    "47","58","168","172","173","175","181","184","185","191","195",
    "197","198","199","200","219","221","222","223","240","241","243","246","248",
}

N_HISTORY = 36
PMT_HIST  = [f"actual_payment_am_{str(i).zfill(2)}" for i in range(1, N_HISTORY + 1)]
BAL_HIST  = [f"balance_am_{str(i).zfill(2)}"        for i in range(1, N_HISTORY + 1)]


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _win_sum(slots: List[F.Column]) -> F.Column:
    """Sum of non-null payment slots per row. NULL if no data at all."""
    slot_sum = sum(F.coalesce(s, F.lit(0.0)) for s in slots)
    return F.when(F.greatest(*slots).isNotNull(), slot_sum)


def _win_freq(slots: List[F.Column]) -> F.Column:
    """Count of months with a reported (non-null) payment per row."""
    cnt = sum(F.when(s.isNotNull(), F.lit(1)).otherwise(F.lit(0)) for s in slots)
    return F.when(F.greatest(*slots).isNotNull(), cnt)


def _missed(slots: List[F.Column]) -> F.Column:
    return F.sum(
        F.when(F.coalesce(*slots).isNotNull() & (F.coalesce(*slots) <= 0), F.lit(1))
         .otherwise(F.lit(0))
    )


def _flag_missed(slots: List[F.Column]) -> F.Column:
    return F.max(
        F.when(F.coalesce(*slots).isNotNull() & (F.coalesce(*slots) <= 0), F.lit(1))
         .otherwise(F.lit(0))
    )


# =============================================================================
# GROUP 08a — Payment Behaviour
# =============================================================================

class PaymentBehaviourFeatures(TradelineFeatureBase):
    """Group 08a: Payment Behaviour (array-optimised)."""

    CATEGORY = "grp08a_payment_behaviour"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        df = (
            df
            .withColumn("_bal_dt",    parse_date("balance_dt"))
            .withColumn("_as_of_dt", F.col(as_of_col).cast("date"))
            .withColumn("_open_dt",   parse_date("open_dt"))
            .withColumn("_closed_dt", parse_date("closed_dt"))
        )

        df = df.withColumn(
            "_md",
            F.ceil(F.months_between(F.col("_as_of_dt"), F.col("_bal_dt"))).cast("int")
        )

        df = df.withColumn(
            "_is_active",
            F.when(
                (F.col("_open_dt") <= F.col("_as_of_dt")) &
                (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt"))),
                F.lit(True)
            ).otherwise(F.lit(False))
        )

        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))
        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )
        df = (
            df
            .withColumn("_is_cc",        F.col("_acct").isin(CC_CODES))
            .withColumn("_is_pl",        F.col("_acct") == PL_CODE)
            .withColumn("_is_stpl",      F.col("_acct") == STPL_CODE)
            .withColumn("_is_pl_gt30k",
                (F.col("_acct") == PL_CODE) &
                F.col("_loan_am").isNotNull() & (F.col("_loan_am") > 30000))
        )

        # ── Build payment array ONCE (replaces 73-WHEN nested loop) ───────────
        df = build_history_array(df, "actual_payment_am", PMT_HIST, "_pmt_arr",
                                 clean_negative=True)

        # ── Composite filters ─────────────────────────────────────────────────
        active_cc_f     = F.col("_is_active") & F.col("_is_cc")
        active_pl_f     = F.col("_is_active") & F.col("_is_pl")
        active_stpl_f   = F.col("_is_active") & F.col("_is_stpl")
        active_pl_gt30k = F.col("_is_active") & F.col("_is_pl_gt30k")

        # ── Window slot-Column lists (array-indexed) ───────────────────────────
        w3           = build_window_cols("_pmt_arr", 3)
        w6           = build_window_cols("_pmt_arr", 6)
        w12          = build_window_cols("_pmt_arr", 12)
        w3_cc        = build_window_cols("_pmt_arr", 3,  product_filter=active_cc_f)
        w6_cc        = build_window_cols("_pmt_arr", 6,  product_filter=active_cc_f)
        w12_cc       = build_window_cols("_pmt_arr", 12, product_filter=active_cc_f)
        w36_cc       = build_window_cols("_pmt_arr", 36, product_filter=active_cc_f)
        w3_pl        = build_window_cols("_pmt_arr", 3,  product_filter=active_pl_f)
        w6_pl        = build_window_cols("_pmt_arr", 6,  product_filter=active_pl_f)
        w12_pl       = build_window_cols("_pmt_arr", 12, product_filter=active_pl_f)
        w36_pl       = build_window_cols("_pmt_arr", 36, product_filter=active_pl_f)
        w3_stpl      = build_window_cols("_pmt_arr", 3,  product_filter=active_stpl_f)
        w6_stpl      = build_window_cols("_pmt_arr", 6,  product_filter=active_stpl_f)
        w12_stpl     = build_window_cols("_pmt_arr", 12, product_filter=active_stpl_f)
        w3_pl_gt30k  = build_window_cols("_pmt_arr", 3,  product_filter=active_pl_gt30k)
        w6_pl_gt30k  = build_window_cols("_pmt_arr", 6,  product_filter=active_pl_gt30k)
        w12_pl_gt30k = build_window_cols("_pmt_arr", 12, product_filter=active_pl_gt30k)

        feature_df = df.groupBy(group_cols).agg(

            # Windowed payment sums — all accounts
            F.sum(_win_sum(w3)).alias("payments_3_month"),
            F.sum(_win_sum(w6)).alias("payments_6_month"),
            F.sum(_win_sum(w12)).alias("payments_12_month"),

            # Frequency
            F.sum(_win_freq(w3)).alias("frequency_of_payments_3m"),
            F.sum(_win_freq(w6)).alias("frequency_of_payments_6m"),
            F.sum(_win_freq(w12)).alias("frequency_of_payments_12m"),

            # Active CC repayment max/min
            F.max(F.greatest(*w36_cc)).alias("MaxRepaymentOnActiveCreditCardAccounts"),
            F.min(F.least(*[F.coalesce(s, F.lit(None).cast("double")) for s in w36_cc])).alias("MinRepaymentOnActiveCreditCardAccounts"),
            F.max(F.greatest(*w12_cc)).alias("MaxRepaymentOnActiveCreditCardAccounts_12m"),
            F.min(F.least(*[F.coalesce(s, F.lit(None).cast("double")) for s in w12_cc])).alias("MinRepaymentOnActiveCreditCardAccounts_12m"),
            F.max(F.greatest(*w6_cc)).alias("MaxRepaymentOnActiveCreditCardAccounts_6m"),
            F.min(F.least(*[F.coalesce(s, F.lit(None).cast("double")) for s in w6_cc])).alias("MinRepaymentOnActiveCreditCardAccounts_6m"),

            # Active PL repayment max/min
            F.max(F.greatest(*w36_pl)).alias("MaxRepaymentOnActivePersonalLoanAccounts"),
            F.min(F.least(*[F.coalesce(s, F.lit(None).cast("double")) for s in w36_pl])).alias("MinRepaymentOnActivePersonalLoanAccounts"),
            F.max(F.greatest(*w12_pl)).alias("MaxRepaymentOnActivePersonalLoanAccounts_12m"),
            F.min(F.least(*[F.coalesce(s, F.lit(None).cast("double")) for s in w12_pl])).alias("MinRepaymentOnActivePersonalLoanAccounts_12m"),
            F.max(F.greatest(*w6_pl)).alias("MaxRepaymentOnActivePersonalLoanAccounts_6m"),
            F.min(F.least(*[F.coalesce(s, F.lit(None).cast("double")) for s in w6_pl])).alias("MinRepaymentOnActivePersonalLoanAccounts_6m"),

            # Missed payments
            _missed(w3).alias("no_of_missed_payments_last_3m"),
            _missed(w6).alias("no_of_missed_payments_last_6m"),
            _missed(w12).alias("no_of_missed_payments_last_12m"),
            _missed(w3_cc).alias("no_of_missed_payments_cc_last_3m"),
            _missed(w6_cc).alias("no_of_missed_payments_cc_last_6m"),
            _missed(w12_cc).alias("no_of_missed_payments_cc_last_12m"),
            _missed(w3_pl).alias("no_of_missed_payments_pl_last_3m"),
            _missed(w6_pl).alias("no_of_missed_payments_pl_last_6m"),
            _missed(w12_pl).alias("no_of_missed_payments_pl_last_12m"),
            _missed(w3_stpl).alias("no_of_missed_payments_stpl_last_3m"),
            _missed(w6_stpl).alias("no_of_missed_payments_stpl_last_6m"),
            _missed(w12_stpl).alias("no_of_missed_payments_stpl_last_12m"),
            _missed(w3_pl_gt30k).alias("no_of_missed_payments_pl_gt30k_last_3m"),
            _missed(w6_pl_gt30k).alias("no_of_missed_payments_pl_gt30k_last_6m"),
            _missed(w12_pl_gt30k).alias("no_of_missed_payments_pl_gt30k_last_12m"),

            # Binary missed flags
            _flag_missed(w3).alias("flag_missed_payment_last_3m"),
            _flag_missed(w12).alias("flag_missed_payment_last_12m"),
            _flag_missed(w3_stpl).alias("flag_missed_stpl_last_3m"),
            _flag_missed(w12_stpl).alias("flag_missed_stpl_last_12m"),
            _flag_missed(w3_pl_gt30k).alias("flag_missed_pl_gt30k_last_3m"),
            _flag_missed(w12_pl_gt30k).alias("flag_missed_pl_gt30k_last_12m"),

            # Payment quality
            F.max(F.greatest(*w12)).alias("max_payment_last_12m"),

            # Intermediates for derived features
            F.sum(_win_sum(w12)).alias("_pmt_sum_12m_all"),
            F.sum(_win_freq(w12)).alias("_pmt_freq_12m_all"),
            F.sum(_win_sum(w3)).alias("_pmt_sum_3m"),
            F.sum(_win_sum(w12)).alias("_pmt_sum_12m"),
        )

        feature_df = feature_df.withColumn(
            "mean_payment_last_12m",
            F.when(
                F.col("_pmt_freq_12m_all").isNotNull() & (F.col("_pmt_freq_12m_all") > 0),
                F.col("_pmt_sum_12m_all") / F.col("_pmt_freq_12m_all")
            ).otherwise(F.lit(None).cast("double"))
        ).withColumn(
            "ratio_payment_3m_to_12m",
            F.when(
                F.col("_pmt_sum_12m").isNotNull() & (F.col("_pmt_sum_12m") > 0),
                F.col("_pmt_sum_3m") / F.col("_pmt_sum_12m")
            ).otherwise(F.lit(None).cast("double"))
        ).drop("_pmt_sum_3m", "_pmt_sum_12m", "_pmt_sum_12m_all", "_pmt_freq_12m_all")

        self._log_done(feature_df)
        return feature_df


# =============================================================================
# GROUP 08b — Repayment Ratio
# =============================================================================

class RepaymentRatioFeatures(TradelineFeatureBase):
    """Group 08b: Repayment Ratio / Balance Reduction (array-optimised)."""

    CATEGORY = "grp08b_repayment_ratio"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        df = (
            df
            .withColumn("_bal_dt",    parse_date("balance_dt"))
            .withColumn("_as_of_dt", F.col(as_of_col).cast("date"))
            .withColumn("_open_dt",   parse_date("open_dt"))
            .withColumn("_closed_dt", parse_date("closed_dt"))
        )

        # grp08b uses months_between(balance_dt, as_of_dt) — balance_dt FIRST
        # This gives negative _md when as_of > balance_dt, so _md <= 0 → use slot 0
        df = df.withColumn(
            "_month_diff",
            F.ceil(F.months_between(F.col("_bal_dt"), F.col("_as_of_dt"))).cast("int")
        )

        df = df.withColumn(
            "_is_active",
            F.when(
                (F.col("_open_dt") <= F.col("_as_of_dt")) &
                (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt"))),
                F.lit(True)
            ).otherwise(F.lit(False))
        )

        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))
        df = (
            df
            .withColumn("_is_pl",   F.col("_acct") == PL_CODE)
            .withColumn("_is_stpl", F.col("_acct") == STPL_CODE)
            .withColumn("_is_usl",  ~F.col("_acct").isin(SECURED_CODES))
        )

        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )

        # ── Build balance array ONCE ──────────────────────────────────────────
        df = build_history_array(df, "balance_am", BAL_HIST, "_bal_arr",
                                 clean_negative=False)  # allow 0 balance

        # ── Resolve balance at as_of_dt (grp03a-style: _md = balance_dt - as_of) ─
        df = df.withColumn("_bal_asof", resolve_slot_at_asof("_bal_arr", "_month_diff"))

        df = df.withColumn(
            "_repay_pct",
            F.when(
                F.col("_loan_am").isNotNull() & F.col("_bal_asof").isNotNull() &
                (F.col("_loan_am") > 0),
                F.lit(1.0) - (F.col("_bal_asof") / F.col("_loan_am"))
            ).otherwise(F.lit(None).cast("double"))
        )

        feature_df = df.groupBy(group_cols).agg(
            F.avg(F.when(F.col("_is_stpl"),                           F.col("_repay_pct"))).alias("RepaymentPercentageTotalSTPL"),
            F.avg(F.when(F.col("_is_stpl") & F.col("_is_active"),     F.col("_repay_pct"))).alias("RepaymentPercentageActiveSTPL"),
            F.avg(F.when(F.col("_is_usl"),                            F.col("_repay_pct"))).alias("Paid_ratio_unsecured"),
            F.avg(F.when(F.col("_is_pl"),                             F.col("_repay_pct"))).alias("RepaymentPercentageTotalPL"),
            F.avg(F.when(F.col("_is_pl") & F.col("_is_active"),       F.col("_repay_pct"))).alias("RepaymentPercentageActivePL"),
            F.avg("_repay_pct").alias("repayment_pct_all"),
            F.avg(F.when(F.col("_is_active"),                         F.col("_repay_pct"))).alias("repayment_pct_active"),
            F.min(F.when(F.col("_is_active"),                         F.col("_repay_pct"))).alias("min_repayment_pct_active"),
            F.max(F.when(F.col("_is_active"),                         F.col("_repay_pct"))).alias("max_repayment_pct_active"),
            F.max(F.when(F.col("_is_pl") & (F.col("_repay_pct") >= 1.0), F.lit(1)).otherwise(F.lit(0))).alias("flag_fully_repaid_pl"),
            F.max(F.when(F.col("_is_stpl") & (F.col("_repay_pct") < 0), F.lit(1)).otherwise(F.lit(0))).alias("flag_balance_grown_stpl"),
            F.max(F.when(F.col("_is_pl") & (F.col("_repay_pct") < 0), F.lit(1)).otherwise(F.lit(0))).alias("flag_balance_grown_pl"),
            F.sum(F.when(F.col("_is_stpl") & (F.col("_repay_pct") >= 1.0), F.lit(1)).otherwise(F.lit(0))).alias("count_stpl_fully_repaid"),
            F.sum(F.when(F.col("_is_pl")   & (F.col("_repay_pct") >= 1.0), F.lit(1)).otherwise(F.lit(0))).alias("count_pl_fully_repaid"),
            F.avg(F.when(F.col("_is_pl") & F.col("_is_active"),       F.col("_repay_pct"))).alias("_rp_pl"),
            F.avg(F.when(F.col("_is_stpl") & F.col("_is_active"),     F.col("_repay_pct"))).alias("_rp_stpl"),
        )

        feature_df = feature_df.withColumn(
            "ratio_repayment_pl_vs_stpl",
            F.when(
                F.col("_rp_stpl").isNotNull() & (F.col("_rp_stpl") > 0),
                F.col("_rp_pl") / F.col("_rp_stpl")
            ).otherwise(F.lit(None).cast("double"))
        ).drop("_rp_pl", "_rp_stpl")

        self._log_done(feature_df)
        return feature_df
