# features/tradeline/cat13_payment_behaviour.py
# =============================================================================
# Category 13 — Payment Behaviour
# =============================================================================
# Source table  : experian_tradeline_segment
#
# ── COLUMN STRUCTURE ─────────────────────────────────────────────────────────
#   balance_dt                → Last reported date (anchor for payment history)
#   ACTUAL_PAYMENT_AM         → Payment amount at balance_dt              (slot 0)
#   ACTUAL_PAYMENT_AM_01..36  → Payment amounts 1–36 months before        (slots 1–36)
#   EMI                       → Monthly EMI payable
#
# ── SLOT RESOLUTION (same logic as cat12 DPD) ────────────────────────────────
#   month_diff = round(months_between(as_of_dt, balance_dt))
#
#   +ve → as_of AHEAD of balance_dt → gap months → NULL for those offsets
#   -ve → as_of BEHIND balance_dt   → skip early slots, start from |month_diff|
#
#   For window offset k (k=0 = as_of month, k=1 = 1m before, ...):
#     actual_slot = k - month_diff
#     slot < 0   → gap (not yet reported) → NULL
#     slot == 0  → ACTUAL_PAYMENT_AM
#     slot 1–36  → ACTUAL_PAYMENT_AM_NN
#     slot > 36  → beyond history → NULL
#
# ── STALENESS NULL RULE ───────────────────────────────────────────────────────
#   If month_diff > window_size, the ENTIRE window falls in the gap → NULL.
#   Example: month_diff=4, window=3 → all 3 slots are gap → NULL.
#   Example: month_diff=2, window=6 → slots 0,1 are gap, slots 2-5 available.
#
# ── MISSED PAYMENT DEFINITION ────────────────────────────────────────────────
#   A payment is "missed" if ACTUAL_PAYMENT_AM <= 0 for that slot
#   (zero or negative = no payment made).
#   NULL slots (gap or beyond history) are NOT counted as missed.
#
# ── ACTIVE DEFINITION ────────────────────────────────────────────────────────
#   open_dt <= as_of_dt AND (closed_dt IS NULL OR closed_dt > as_of_dt)
#
# Product filters:
#   CC  : '5','213','214','220','224','225'
#   PL  : '123'
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

CC_CODES  = {"5", "213", "214", "220", "224", "225"}
PL_CODE   = "123"
STPL_CODE = "242"

N_HISTORY = 36
PMT_COLS  = [f"actual_payment_am_{str(i).zfill(2)}" for i in range(1, N_HISTORY + 1)]


# =============================================================================
# HELPERS
# =============================================================================

def _clean_pmt(col: F.Column) -> F.Column:
    """Payment amount: treat negative values as NULL (not available)."""
    return F.when(col >= 0, col.cast("double")).otherwise(F.lit(None).cast("double"))




class PaymentBehaviourFeatures(TradelineFeatureBase):
    """
    Category 13: Payment Behaviour

    Uses balance_dt as the anchor for payment history lookup.
    Slot resolution: actual_slot = k - month_diff (same as cat12).

    _pmt_slot(k)  → resolved payment amount at window offset k from as_of_dt
    _window(w)    → list of w slot-columns for a window

    Staleness NULL rule: if month_diff > window_size → entire window = NULL.
    Missed payment: payment amount <= 0 for a valid (non-NULL) slot.
    """

    CATEGORY = "cat13_payment_behaviour"

    @staticmethod
    def _pmt_slot(k: int,
                  month_diff_col: str = "_md",
                  product_filter: Optional[F.Column] = None,
                  active_filter: Optional[F.Column] = None) -> F.Column:
        """
        Resolved payment amount at window offset k from as_of_dt.
        actual_slot = k - month_diff.
        """
        expr = F.lit(None).cast("double")
        for m in range(-N_HISTORY, N_HISTORY + 1):
            slot = k - m
            if slot < 0:
                raw = F.lit(None).cast("double")            # gap
            elif slot == 0:
                raw = _clean_pmt(F.col("actual_payment_am"))
            elif slot <= N_HISTORY:
                raw = _clean_pmt(F.col(PMT_COLS[slot - 1]))
            else:
                raw = F.lit(None).cast("double")            # beyond history
            expr = F.when(F.col(month_diff_col) == m, raw).otherwise(expr)

        if product_filter is not None:
            expr = F.when(product_filter, expr).otherwise(F.lit(None).cast("double"))
        if active_filter is not None:
            expr = F.when(active_filter, expr).otherwise(F.lit(None).cast("double"))
        return expr

    @classmethod
    def _window(cls, w: int,
                product_filter: Optional[F.Column] = None,
                active_filter: Optional[F.Column] = None) -> List[F.Column]:
        """Build list of w slot-columns for a window of w months."""
        return [cls._pmt_slot(k, product_filter=product_filter,
                              active_filter=active_filter)
                for k in range(w)]

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
            .withColumn("_bal_dt",   parse_date("balance_dt"))
            .withColumn("_as_of_dt", parse_date(as_of_col))
            .withColumn("_open_dt",  parse_date("open_dt"))
            .withColumn("_closed_dt", parse_date("closed_dt"))
        )

        # ── STEP 2: month_diff ────────────────────────────────────────────────
        # Uses balance_dt (not last_reporting_pymt_dt) as per spec for cat13
        df = df.withColumn(
            "_md",
            F.round(F.months_between(
                F.col("_as_of_dt"), F.col("_bal_dt")
            )).cast("int")
        )

        # ── STEP 3: Active flag (point-in-time, no leakage) ───────────────────
        df = df.withColumn(
            "_is_active",
            F.when(
                (F.col("_open_dt") <= F.col("_as_of_dt")) &
                (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt"))),
                F.lit(True)
            ).otherwise(F.lit(False))
        )

        # ── STEP 4: Product flags ─────────────────────────────────────────────
        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))
        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )
        df = (
            df
            .withColumn("_is_cc",   F.col("_acct").isin(CC_CODES))
            .withColumn("_is_pl",   F.col("_acct") == PL_CODE)
            .withColumn("_is_stpl", F.col("_acct") == STPL_CODE)
            # PL with orig_loan_am > 30,000
            .withColumn("_is_pl_gt30k",
                (F.col("_acct") == PL_CODE) &
                F.col("_loan_am").isNotNull() & (F.col("_loan_am") > 30000))
        )

        # ── STEP 5: Build window slot arrays ──────────────────────────────────
        active_cc_f     = F.col("_is_active") & F.col("_is_cc")
        active_pl_f     = F.col("_is_active") & F.col("_is_pl")
        active_stpl_f   = F.col("_is_active") & F.col("_is_stpl")
        active_pl_gt30k = F.col("_is_active") & F.col("_is_pl_gt30k")

        # All-accounts windows
        w3   = self._window(3)
        w6   = self._window(6)
        w12  = self._window(12)

        # Active product-specific windows — 6m, 12m, 36m
        w3_cc        = self._window(3,  active_filter=active_cc_f)
        w6_cc        = self._window(6,  active_filter=active_cc_f)
        w12_cc       = self._window(12, active_filter=active_cc_f)
        w36_cc       = self._window(36, active_filter=active_cc_f)
        w3_pl        = self._window(3,  active_filter=active_pl_f)
        w6_pl        = self._window(6,  active_filter=active_pl_f)
        w12_pl       = self._window(12, active_filter=active_pl_f)
        w36_pl       = self._window(36, active_filter=active_pl_f)
        w3_stpl      = self._window(3,  active_filter=active_stpl_f)
        w6_stpl      = self._window(6,  active_filter=active_stpl_f)
        w12_stpl     = self._window(12, active_filter=active_stpl_f)
        w3_pl_gt30k  = self._window(3,  active_filter=active_pl_gt30k)
        w6_pl_gt30k  = self._window(6,  active_filter=active_pl_gt30k)
        w12_pl_gt30k = self._window(12, active_filter=active_pl_gt30k)

        # ── helper: missed payment count over a window ────────────────────────
        def missed(slots):
            return F.sum(F.when(
                F.coalesce(*slots).isNotNull() & (F.coalesce(*slots) <= 0),
                F.lit(1)).otherwise(F.lit(0))
            )

        def flag_missed(slots):
            return F.max(F.when(
                F.coalesce(*slots).isNotNull() & (F.coalesce(*slots) <= 0),
                F.lit(1)).otherwise(F.lit(0))
            )

        # ── STEP 6: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # ── Windowed payment sums — all accounts ─────────────────────────
            F.sum(F.coalesce(*w3,  F.lit(0.0))).alias("payments_3_month"),
            F.sum(F.coalesce(*w6,  F.lit(0.0))).alias("payments_6_month"),
            F.sum(F.coalesce(*w12, F.lit(0.0))).alias("payments_12_month"),

            # ── Active CC repayment features ──────────────────────────────────
            # MAX/MIN across all available payment slots in window
            # greatest(*slots) per row → then MAX/MIN across tradelines
            F.max(F.greatest(*w36_cc)).alias("MaxRepaymentOnActiveCreditCardAccounts"),
            F.min(F.least(*[F.coalesce(s, F.lit(None).cast("double")) for s in w36_cc])
            ).alias("MinRepaymentOnActiveCreditCardAccounts"),
            F.max(F.greatest(*w12_cc)).alias("MaxRepaymentOnActiveCreditCardAccounts_12m"),
            F.min(F.least(*[F.coalesce(s, F.lit(None).cast("double")) for s in w12_cc])
            ).alias("MinRepaymentOnActiveCreditCardAccounts_12m"),
            F.max(F.greatest(*w6_cc)).alias("MaxRepaymentOnActiveCreditCardAccounts_6m"),
            F.min(F.least(*[F.coalesce(s, F.lit(None).cast("double")) for s in w6_cc])
            ).alias("MinRepaymentOnActiveCreditCardAccounts_6m"),

            # ── Active PL repayment features ──────────────────────────────────
            F.max(F.greatest(*w36_pl)).alias("MaxRepaymentOnActivePersonalLoanAccounts"),
            F.min(F.least(*[F.coalesce(s, F.lit(None).cast("double")) for s in w36_pl])
            ).alias("MinRepaymentOnActivePersonalLoanAccounts"),
            F.max(F.greatest(*w12_pl)).alias("MaxRepaymentOnActivePersonalLoanAccounts_12m"),
            F.min(F.least(*[F.coalesce(s, F.lit(None).cast("double")) for s in w12_pl])
            ).alias("MinRepaymentOnActivePersonalLoanAccounts_12m"),
            F.max(F.greatest(*w6_pl)).alias("MaxRepaymentOnActivePersonalLoanAccounts_6m"),
            F.min(F.least(*[F.coalesce(s, F.lit(None).cast("double")) for s in w6_pl])
            ).alias("MinRepaymentOnActivePersonalLoanAccounts_6m"),

            # ── Missed payments — all accounts ────────────────────────────────
            missed(w3).alias("no_of_missed_payments_last_3m"),
            missed(w6).alias("no_of_missed_payments_last_6m"),
            missed(w12).alias("no_of_missed_payments_last_12m"),

            # ── Missed payments — CC ──────────────────────────────────────────
            missed(w3_cc).alias("no_of_missed_payments_cc_last_3m"),
            missed(w6_cc).alias("no_of_missed_payments_cc_last_6m"),
            missed(w12_cc).alias("no_of_missed_payments_cc_last_12m"),

            # ── Missed payments — PL (all sizes) ─────────────────────────────
            missed(w3_pl).alias("no_of_missed_payments_pl_last_3m"),
            missed(w6_pl).alias("no_of_missed_payments_pl_last_6m"),
            missed(w12_pl).alias("no_of_missed_payments_pl_last_12m"),

            # ── Missed payments — STPL (acct_type='242') ─────────────────────
            missed(w3_stpl).alias("no_of_missed_payments_stpl_last_3m"),
            missed(w6_stpl).alias("no_of_missed_payments_stpl_last_6m"),
            missed(w12_stpl).alias("no_of_missed_payments_stpl_last_12m"),

            # ── Missed payments — PL with loan_amt > 30K ─────────────────────
            missed(w3_pl_gt30k).alias("no_of_missed_payments_pl_gt30k_last_3m"),
            missed(w6_pl_gt30k).alias("no_of_missed_payments_pl_gt30k_last_6m"),
            missed(w12_pl_gt30k).alias("no_of_missed_payments_pl_gt30k_last_12m"),

            # ── Binary missed payment flags ───────────────────────────────────
            flag_missed(w3).alias("flag_missed_payment_last_3m"),
            flag_missed(w12).alias("flag_missed_payment_last_12m"),
            flag_missed(w3_stpl).alias("flag_missed_stpl_last_3m"),
            flag_missed(w12_stpl).alias("flag_missed_stpl_last_12m"),
            flag_missed(w3_pl_gt30k).alias("flag_missed_pl_gt30k_last_3m"),
            flag_missed(w12_pl_gt30k).alias("flag_missed_pl_gt30k_last_12m"),

            # ── Additional payment quality features ───────────────────────────
            # Max single payment in last 12m — ability-to-pay signal
            F.max(F.greatest(*w12)).alias("max_payment_last_12m"),

            # Mean payment in last 12m — consistency signal
            F.mean(F.coalesce(*w12)).alias("mean_payment_last_12m"),

            # For ratio feature
            F.sum(F.coalesce(*w3,  F.lit(None).cast("double"))).alias("_pmt_sum_3m"),
            F.sum(F.coalesce(*w12, F.lit(None).cast("double"))).alias("_pmt_sum_12m"),
        )

        # ── STEP 7: Derived ratio features ───────────────────────────────────

        # ratio_payment_3m_to_12m
        # Low → payments declining recently (risk signal)
        feature_df = feature_df.withColumn(
            "ratio_payment_3m_to_12m",
            F.when(
                F.col("_pmt_sum_12m").isNotNull() & (F.col("_pmt_sum_12m") > 0),
                F.col("_pmt_sum_3m") / F.col("_pmt_sum_12m")
            ).otherwise(F.lit(None).cast("double"))
        )

        feature_df = feature_df.drop("_pmt_sum_3m", "_pmt_sum_12m")

        self._log_done(feature_df)
        return feature_df


class RepaymentRatioFeatures(TradelineFeatureBase):
    """
    Category 15: Repayment Ratio / Balance Reduction

    repayment_pct = 1 - (balance_at_asof / orig_loan_am)

    balance_at_asof resolved via month_diff slot lookup (same as cat05/cat12).
    Aggregated as AVG across tradelines per customer.

    ── Requested ────────────────────────────────────────────────────────────
    RepaymentPercentageTotalSTPL    Avg repayment % across ALL STPL accounts
    RepaymentPercentageActiveSTPL   Avg repayment % across ACTIVE STPL only
    Paid_ratio_unsecured            Avg repayment % across ALL unsecured accounts

    ── Added for risk and analytics ─────────────────────────────────────────
    RepaymentPercentageTotalPL      Avg repayment % across ALL PL accounts
    RepaymentPercentageActivePL     Avg repayment % across ACTIVE PL only
    repayment_pct_all               Avg repayment % across ALL accounts
    repayment_pct_active            Avg repayment % across ACTIVE accounts only
    min_repayment_pct_active        Min repayment % — most under-repaid account
    max_repayment_pct_active        Max repayment % — most repaid account
    flag_fully_repaid_pl            1 if any PL is fully repaid (repayment_pct >= 1)
    flag_balance_grown_stpl         1 if any STPL has balance > orig_loan_am (< 0 pct)
    flag_balance_grown_pl           1 if any PL has balance > orig_loan_am
    count_stpl_fully_repaid         Count of STPL accounts with repayment_pct >= 1
    count_pl_fully_repaid           Count of PL accounts with repayment_pct >= 1
    ratio_repayment_pl_vs_stpl      RepaymentPercentageActivePL / RepaymentPercentageActiveSTPL
                                    High = PL repaid better than STPL → lower PL risk
    """

    CATEGORY = "cat15_repayment_ratio"

    @staticmethod
    def _balance_at_asof() -> F.Column:
        """
        Resolve BALANCE_AM at as_of_dt using month_diff slot lookup.

        month_diff = round(months_between(as_of_dt, balance_dt))
          +ve → as_of AHEAD of balance_dt → gap → use BALANCE_AM (best available)
          0   → as_of == balance_dt       → use BALANCE_AM
          -ve → as_of BEHIND balance_dt   → use BALANCE_AM_NN where NN = -month_diff
          > 36 → beyond history           → NULL
        """
        expr = F.lit(None).cast("double")

        for m in range(-N_HISTORY, N_HISTORY + 1):
            slot = -m   # slot index = -month_diff (how far back from balance_dt)

            if m > 0:
                # as_of is ahead of balance_dt (gap) — use BALANCE_AM directly
                raw = F.when(F.col("balance_am").cast("double") >= 0,
                             F.col("balance_am").cast("double")) \
                       .otherwise(F.lit(None).cast("double"))
            elif slot == 0:
                # as_of == balance_dt
                raw = F.when(F.col("balance_am").cast("double") >= 0,
                             F.col("balance_am").cast("double")) \
                       .otherwise(F.lit(None).cast("double"))
            elif 1 <= slot <= N_HISTORY:
                raw = F.when(F.col(BAL_COLS[slot - 1]) >= 0,
                             F.col(BAL_COLS[slot - 1]).cast("double")) \
                       .otherwise(F.lit(None).cast("double"))
            else:
                raw = F.lit(None).cast("double")

            expr = F.when(F.col("_md") == m, raw).otherwise(expr)

        return expr

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
            .withColumn("_bal_dt",    parse_date("balance_dt"))
            .withColumn("_as_of_dt",  parse_date(as_of_col))
            .withColumn("_open_dt",   parse_date("open_dt"))
            .withColumn("_closed_dt", parse_date("closed_dt"))
        )

        # ── STEP 2: month_diff ────────────────────────────────────────────────
        df = df.withColumn(
            "_md",
            F.round(F.months_between(F.col("_as_of_dt"), F.col("_bal_dt"))).cast("int")
        )

        # ── STEP 3: Active flag ───────────────────────────────────────────────
        df = df.withColumn(
            "_is_active",
            F.when(
                (F.col("_open_dt") <= F.col("_as_of_dt")) &
                (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt"))),
                F.lit(True)
            ).otherwise(F.lit(False))
        )

        # ── STEP 4: Product flags ─────────────────────────────────────────────
        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))
        df = (
            df
            .withColumn("_is_pl",   F.col("_acct") == PL_CODE)
            .withColumn("_is_stpl", F.col("_acct") == STPL_CODE)
            .withColumn("_is_usl",  ~F.col("_acct").isin(SECURED_CODES))
        )

        # ── STEP 5: Clean orig_loan_am ────────────────────────────────────────
        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 6: Resolve balance at as_of_dt ──────────────────────────────
        df = df.withColumn("_bal_asof", self._balance_at_asof())

        # ── STEP 7: Compute repayment percentage per row ──────────────────────
        # repayment_pct = 1 - (balance_at_asof / orig_loan_am)
        # NULL if either is NULL
        df = df.withColumn(
            "_repay_pct",
            F.when(
                F.col("_loan_am").isNotNull() &
                F.col("_bal_asof").isNotNull(),
                F.lit(1.0) - (F.col("_bal_asof") / F.col("_loan_am"))
            ).otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 8: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # ── Requested features ────────────────────────────────────────────

            # STPL — all accounts (active + closed)
            F.avg(F.when(F.col("_is_stpl"), F.col("_repay_pct"))
            ).alias("RepaymentPercentageTotalSTPL"),

            # STPL — active only
            F.avg(F.when(F.col("_is_stpl") & F.col("_is_active"), F.col("_repay_pct"))
            ).alias("RepaymentPercentageActiveSTPL"),

            # Unsecured — all accounts
            F.avg(F.when(F.col("_is_usl"), F.col("_repay_pct"))
            ).alias("Paid_ratio_unsecured"),

            # ── PL variants ───────────────────────────────────────────────────

            F.avg(F.when(F.col("_is_pl"), F.col("_repay_pct"))
            ).alias("RepaymentPercentageTotalPL"),

            F.avg(F.when(F.col("_is_pl") & F.col("_is_active"), F.col("_repay_pct"))
            ).alias("RepaymentPercentageActivePL"),

            # ── Portfolio-level repayment ─────────────────────────────────────

            F.avg("_repay_pct").alias("repayment_pct_all"),

            F.avg(F.when(F.col("_is_active"), F.col("_repay_pct"))
            ).alias("repayment_pct_active"),

            F.min(F.when(F.col("_is_active"), F.col("_repay_pct"))
            ).alias("min_repayment_pct_active"),

            F.max(F.when(F.col("_is_active"), F.col("_repay_pct"))
            ).alias("max_repayment_pct_active"),

            # ── Risk flags ────────────────────────────────────────────────────

            # Flag: any PL fully repaid (repayment_pct >= 1.0)
            F.max(F.when(F.col("_is_pl") & (F.col("_repay_pct") >= 1.0), F.lit(1))
                   .otherwise(F.lit(0))
            ).alias("flag_fully_repaid_pl"),

            # Flag: any STPL with balance grown beyond orig (repayment_pct < 0)
            # Indicates penalty/rollover accumulation
            F.max(F.when(F.col("_is_stpl") & (F.col("_repay_pct") < 0), F.lit(1))
                   .otherwise(F.lit(0))
            ).alias("flag_balance_grown_stpl"),

            # Flag: any PL with balance grown beyond orig
            F.max(F.when(F.col("_is_pl") & (F.col("_repay_pct") < 0), F.lit(1))
                   .otherwise(F.lit(0))
            ).alias("flag_balance_grown_pl"),

            # Count of STPL fully repaid
            F.sum(F.when(F.col("_is_stpl") & (F.col("_repay_pct") >= 1.0), F.lit(1))
                   .otherwise(F.lit(0))
            ).alias("count_stpl_fully_repaid"),

            # Count of PL fully repaid
            F.sum(F.when(F.col("_is_pl") & (F.col("_repay_pct") >= 1.0), F.lit(1))
                   .otherwise(F.lit(0))
            ).alias("count_pl_fully_repaid"),

            # For ratio feature
            F.avg(F.when(F.col("_is_pl") & F.col("_is_active"), F.col("_repay_pct"))
            ).alias("_rp_pl"),
            F.avg(F.when(F.col("_is_stpl") & F.col("_is_active"), F.col("_repay_pct"))
            ).alias("_rp_stpl"),
        )

        # ── STEP 9: Derived ratio ─────────────────────────────────────────────
        # ratio_repayment_pl_vs_stpl
        # > 1 = PL being repaid faster/better than STPL → lower PL risk relative
        # < 1 = STPL being repaid better than PL → PL stress signal
        feature_df = feature_df.withColumn(
            "ratio_repayment_pl_vs_stpl",
            F.when(
                F.col("_rp_stpl").isNotNull() & (F.col("_rp_stpl") > 0),
                F.col("_rp_pl") / F.col("_rp_stpl")
            ).otherwise(F.lit(None).cast("double"))
        ).drop("_rp_pl", "_rp_stpl")

        self._log_done(feature_df)
        return feature_df
