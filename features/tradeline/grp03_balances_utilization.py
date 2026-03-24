# features/tradeline/grp03_balances_utilization.py  [OPTIMISED]
# =============================================================================
# Group 03 — Outstanding Balances & Credit Utilization
# =============================================================================
# OPTIMISATION:
#   grp03a: replaced F.greatest(*36_when_cols) history_selector with
#           F.array() + resolve_slot_at_asof() — eliminates large plan node count.
#   grp03b: replaced per-slot CASE A/B loops over 37 slots with
#           F.array() + element_at() for both pdu and util arrays.
#
# Semantics preserved exactly (validated against grp03a/b outputs).
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger
from core.date_utils import parse_date
from core.utils import build_history_array, resolve_slot_at_asof, resolve_slot

logger = get_logger(__name__)

CC_CODES  = {"5","213","214","220","224","225"}
PL_CODE   = "123"
HL_CODE   = "195"
CL_CODE   = "189"
STPL_CODE = "242"
SECURED_CODES = {
    "47","58","168","172","173","175","181","184","185","191","195",
    "197","198","199","200","219","221","222","223","240","241","243","246","248",
}

N_HISTORY = 36
PDU_HIST  = [f"past_due_am_{str(i).zfill(2)}"    for i in range(1, N_HISTORY + 1)]
BAL_HIST  = [f"balance_am_{str(i).zfill(2)}"     for i in range(1, N_HISTORY + 1)]
CL_HIST   = [f"credit_limit_am_{str(i).zfill(2)}" for i in range(1, N_HISTORY + 1)]


# =============================================================================
# grp03a — Outstanding Balance
# =============================================================================

class OutstandingBalanceFeatures(TradelineFeatureBase):
    """Group 03a: Outstanding Balance (array-optimised)."""

    CATEGORY = "grp03a_outstanding_balance"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        df = (
            df
            .withColumn("_closed_dt",  parse_date("closed_dt"))
            .withColumn("_open_dt",    parse_date("open_dt"))
            .withColumn("_as_of_dt", F.col(as_of_col).cast("date"))
            .withColumn("_balance_dt", parse_date("balance_dt"))
        )

        df = df.withColumn(
            "_is_active",
            (F.col("_open_dt") <= F.col("_as_of_dt")) &
            (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt")))
        )

        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am").cast("double") > 0, F.col("orig_loan_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )

        # grp03a: months_between(balance_dt, as_of_dt) — balance_dt FIRST
        # negative → use slot 0; positive → use history slot
        df = df.withColumn(
            "_month_diff",
            F.ceil(F.months_between(F.col("_balance_dt"), F.col("_as_of_dt"))).cast("int")
        )

        # ── Build balance array ONCE (replaces F.greatest(*36 WHEN cols)) ────
        df = build_history_array(df, "balance_am", BAL_HIST, "_bal_arr", clean_negative=False)

        # Resolve balance at as_of_dt
        df = df.withColumn("_resolved_bal",
            F.when(
                resolve_slot_at_asof("_bal_arr", "_month_diff") > 0,
                resolve_slot_at_asof("_bal_arr", "_month_diff")
            ).otherwise(F.lit(0.0))
        )

        df = df.withColumn("_acct_type", F.trim(F.col("acct_type_cd").cast("string")))
        df = (
            df
            .withColumn("_is_cl",   F.col("_acct_type") == CL_CODE)
            .withColumn("_is_pl",   F.col("_acct_type") == PL_CODE)
            .withColumn("_is_hl",   F.col("_acct_type") == HL_CODE)
            .withColumn("_is_cc",   F.col("_acct_type").isin(CC_CODES))
            .withColumn("_is_stpl", F.col("_acct_type") == STPL_CODE)
            .withColumn("_is_stpl_qualified",
                (F.col("_acct_type") == STPL_CODE) &
                F.col("_loan_am").isNotNull() & (F.col("_loan_am") <= 30000))
        )

        def _i(cond):
            return F.when(cond, F.lit(1)).otherwise(F.lit(0))

        feature_df = df.groupBy(group_cols).agg(
            F.sum(F.when(F.col("_is_active"), F.col("_resolved_bal"))).alias("total_active_outstanding_debt"),
            F.sum("_resolved_bal").alias("currentbalancedue"),
            F.sum(F.when(F.col("_is_active") & F.col("_is_stpl_qualified"), F.col("_resolved_bal"))).alias("TotalOutstandingOnActiveSTPL"),
            F.sum(F.when(F.col("_is_active") & F.col("_is_cl"),   F.col("_resolved_bal"))).alias("curr_CL_agg_bal"),
            F.sum(F.when(F.col("_is_active") & F.col("_is_pl"),   F.col("_resolved_bal"))).alias("curr_PL_agg_bal"),
            F.sum(F.when(F.col("_is_active") & F.col("_is_hl"),   F.col("_resolved_bal"))).alias("curr_HL_agg_bal"),
            F.sum(F.when(F.col("_is_active") & F.col("_is_cc"),   F.col("_resolved_bal"))).alias("curr_CC_agg_bal"),
            F.max(F.when(F.col("_is_cc"),                          F.col("_resolved_bal"))).alias("max_agg_card_bal"),
        )

        self._log_done(feature_df)
        return feature_df


# =============================================================================
# grp03b — Credit Utilization
# =============================================================================

class CreditUtilizationFeatures(TradelineFeatureBase):
    """Group 03b: Credit Utilization (array-optimised)."""

    CATEGORY = "grp03b_credit_utilization"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        df = (
            df
            .withColumn("_bal_dt",   parse_date("balance_dt"))
            .withColumn("_as_of_dt", F.col(as_of_col).cast("date"))
            .withColumn("_open_dt",  parse_date("open_dt"))
            .withColumn("_closed_dt",parse_date("closed_dt"))
        )

        # grp03b: months_between(as_of_dt, balance_dt) — as_of FIRST → positive
        df = df.withColumn(
            "_md",
            F.ceil(F.months_between(F.col("_as_of_dt"), F.col("_bal_dt"))).cast("int")
        )

        df = df.withColumn(
            "_is_active",
            (F.col("_open_dt") <= F.col("_as_of_dt")) &
            (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt")))
        )

        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))
        df = (
            df
            .withColumn("_is_cc",        F.col("_acct").isin(CC_CODES))
            .withColumn("_is_pl",        F.col("_acct") == PL_CODE)
            .withColumn("_is_stpl",      F.col("_acct") == STPL_CODE)
            .withColumn("_is_usl",       ~F.col("_acct").isin(SECURED_CODES))
            .withColumn("_is_active_cc", F.col("_is_active") & F.col("_is_cc"))
            .withColumn("_is_active_pl", F.col("_is_active") & F.col("_is_pl"))
        )

        df = df.withColumn(
            "_pdu_am",
            F.when(
                F.col("past_due_am").isNotNull() & (F.col("past_due_am").cast("double") >= 0),
                F.col("past_due_am").cast("double")
            ).otherwise(F.lit(None).cast("double"))
        )

        # ── Build pdu array ONCE ──────────────────────────────────────────────
        df = build_history_array(df, "past_due_am", PDU_HIST, "_pdu_arr", clean_negative=False)

        # ── Build balance and credit-limit arrays for CC util ─────────────────
        df = build_history_array(df, "balance_am",      BAL_HIST, "_bal_arr", clean_negative=False)
        df = build_history_array(df, "credit_limit_am", CL_HIST,  "_cl_arr",  clean_negative=False)

        # ── Missed payment frequency — per-row column using CASE A/B logic ────
        # CASE A (_md <= 0): window slots start at -_md
        # CASE B (_md > 0):  valid slots [0 .. W-_md-1]
        # Optimised: precompute indicators via array + transform, not 37 WHEN chains

        def _mpf_col(window: int, product_filter=None) -> F.Column:
            """
            Per-row missed payment frequency ratio using array transform.
            Transform the pdu_arr into indicator array, then sum valid slots.
            Avoids 37×W WHEN chains — uses O(37) element_at calls instead.
            """
            # For each slot idx 0..N_HISTORY, compute indicator if slot is in window
            # CASE A: in window when -_md <= idx <= -_md + W - 1  (_md <= 0)
            # CASE B: in window when 0 <= idx <= W - _md - 1       (_md > 0, _md < W)

            # Build a sequence of (in_window: bool, pdu: double) and sum
            # Using transform over sequence avoids Python loop unrolling into plan
            indicators = F.transform(
                F.sequence(F.lit(0), F.lit(N_HISTORY)),
                lambda idx: F.when(
                    (
                        # CASE A
                        (F.col("_md") <= 0) &
                        (idx >= (-F.col("_md"))) &
                        (idx <= (-F.col("_md") + F.lit(window - 1)))
                    ) | (
                        # CASE B
                        (F.col("_md") > 0) & (F.col("_md") < F.lit(window)) &
                        (idx >= 0) & (idx <= (F.lit(window) - F.col("_md") - F.lit(1)))
                    ),
                    # pdu indicator for this slot
                    F.when(
                        F.element_at(F.col("_pdu_arr"), (idx + F.lit(1)).cast("int")).cast("double") > 0,
                        F.lit(1.0)
                    ).otherwise(F.lit(0.0))
                ).otherwise(F.lit(None).cast("double"))
            )

            valid_n = F.size(F.filter(indicators, lambda x: x.isNotNull()))
            total   = F.aggregate(
                F.filter(indicators, lambda x: x.isNotNull()),
                F.lit(0.0),
                lambda acc, x: acc + x
            )
            result  = F.when(valid_n > 0, (total / valid_n.cast("double")).cast("double"))

            if product_filter is not None:
                result = F.when(product_filter, result).otherwise(F.lit(None).cast("double"))
            return result

        # ── CC util per slot (array-indexed) ─────────────────────────────────
        def _util_col(window: int) -> F.Column:
            """
            Per-row CC utilization (avg over valid window slots).
            Uses transform over sequence — avoids 37 WHEN chains per window.
            """
            utils = F.transform(
                F.sequence(F.lit(0), F.lit(N_HISTORY)),
                lambda idx: F.when(
                    (
                        (F.col("_md") <= 0) &
                        (idx >= (-F.col("_md"))) &
                        (idx <= (-F.col("_md") + F.lit(window - 1)))
                    ) | (
                        (F.col("_md") > 0) & (F.col("_md") < F.lit(window)) &
                        (idx >= 0) & (idx <= (F.lit(window) - F.col("_md") - F.lit(1)))
                    ),
                    F.when(
                        F.col("_is_cc"),
                        F.when(
                            F.element_at(F.col("_cl_arr"), (idx + F.lit(1)).cast("int")).cast("double") > 0,
                            (
                                F.greatest(F.element_at(F.col("_bal_arr"), (idx + F.lit(1)).cast("int")).cast("double"), F.lit(0.0)) /
                                F.element_at(F.col("_cl_arr"), (idx + F.lit(1)).cast("int")).cast("double")
                            )
                        )
                    )
                ).otherwise(F.lit(None).cast("double"))
            )
            valid   = F.filter(utils, lambda x: x.isNotNull())
            n       = F.size(valid)
            total   = F.aggregate(valid, F.lit(0.0), lambda acc, x: acc + x)
            avg_val = F.when(n > 0, total / n)
            max_val = F.array_max(valid)
            return avg_val, max_val

        # Compute mpf and util per-row columns
        df = (
            df
            .withColumn("_mpf_3m",       _mpf_col(3))
            .withColumn("_mpf_6m",       _mpf_col(6))
            .withColumn("_mpf_12m",      _mpf_col(12))
            .withColumn("_mpf_3m_cc",    _mpf_col(3,  F.col("_is_cc")))
            .withColumn("_mpf_6m_cc",    _mpf_col(6,  F.col("_is_cc")))
            .withColumn("_mpf_12m_cc",   _mpf_col(12, F.col("_is_cc")))
            .withColumn("_mpf_3m_pl",    _mpf_col(3,  F.col("_is_active_pl")))
            .withColumn("_mpf_6m_pl",    _mpf_col(6,  F.col("_is_active_pl")))
            .withColumn("_mpf_12m_pl",   _mpf_col(12, F.col("_is_active_pl")))
            .withColumn("_mpf_12m_stpl", _mpf_col(12, F.col("_is_stpl")))
            .withColumn("_mpf_12m_usl",  _mpf_col(12, F.col("_is_usl")))
        )

        # CC util columns (tuple → unpack)
        u3avg,  u3max  = _util_col(3)
        u6avg,  u6max  = _util_col(6)
        u12avg, u12max = _util_col(12)
        u36avg, u36max = _util_col(36)
        df = (
            df
            .withColumn("_util_avg_3m",  u3avg)
            .withColumn("_util_max_3m",  u3max)
            .withColumn("_util_avg_6m",  u6avg)
            .withColumn("_util_max_6m",  u6max)
            .withColumn("_util_avg_12m", u12avg)
            .withColumn("_util_max_12m", u12max)
            .withColumn("_util_avg_36m", u36avg)
            .withColumn("_util_max_36m", u36max)
        )

        def _i(cond):
            return F.when(cond, F.lit(1)).otherwise(F.lit(0))

        feature_df = df.groupBy(group_cols).agg(
            # Past due freq
            F.avg("_mpf_3m").alias("past_due_freq_last_3m"),
            F.avg("_mpf_6m").alias("past_due_freq_last_6m"),
            F.avg("_mpf_12m").alias("past_due_freq_last_12m"),
            F.avg("_mpf_3m_cc").alias("past_due_freq_cc_last_3m"),
            F.avg("_mpf_6m_cc").alias("past_due_freq_cc_last_6m"),
            F.avg("_mpf_12m_cc").alias("past_due_freq_cc_last_12m"),
            F.avg("_mpf_3m_pl").alias("past_due_freq_pl_last_3m"),
            F.avg("_mpf_6m_pl").alias("past_due_freq_pl_last_6m"),
            F.avg("_mpf_12m_pl").alias("past_due_freq_pl_last_12m"),
            F.avg("_mpf_12m_stpl").alias("past_due_freq_stpl_last_12m"),
            F.avg("_mpf_12m_usl").alias("past_due_freq_usl_last_12m"),
            # CC util
            F.avg("_util_avg_3m").alias("avg_cc_utilization_last_3m"),
            F.avg("_util_avg_6m").alias("avg_cc_utilization_last_6m"),
            F.avg("_util_avg_12m").alias("avg_cc_utilization_last_12m"),
            F.avg("_util_avg_36m").alias("avg_cc_utilization_last_36m"),
            F.max("_util_max_3m").alias("max_cc_utilization_last_3m"),
            F.max("_util_max_6m").alias("max_cc_utilization_last_6m"),
            F.max("_util_max_12m").alias("max_cc_utilization_last_12m"),
            F.max("_util_max_36m").alias("max_cc_utilization_last_36m"),
            # Past due
            F.sum(F.when(F.col("_is_active"), F.col("_pdu_am"))).alias("total_past_due_active"),
            F.sum("_pdu_am").alias("total_past_due_all"),
            F.max("_pdu_am").alias("max_past_due_single_account"),
            F.sum(_i(F.col("_pdu_am") > 0)).alias("count_accounts_with_past_due"),
            F.sum(_i(F.col("_is_active_cc") & (F.col("_pdu_am") > 0))).alias("count_cc_with_past_due"),
            F.max(_i(F.col("_pdu_am") > 0)).alias("flag_any_past_due"),
            F.avg("_mpf_12m").alias("_mpf_12m_avg"),
        )

        feature_df = feature_df.withColumn(
            "flag_consistent_revolver",
            F.when(F.col("_mpf_12m_avg").isNotNull() & (F.col("_mpf_12m_avg") > 0.5), F.lit(1))
             .otherwise(F.lit(0))
        ).drop("_mpf_12m_avg")

        self._log_done(feature_df)
        return feature_df
