# features/tradeline/cat05_outstanding_balance.py
# =============================================================================
# Category 05 — Outstanding Balance / Current Debt
# =============================================================================
# Source table  : experian_tradeline_segment
# Granularity   : One row per (customer_scrub_key, scrub_output_date)  [scrub mode]
#                 One row per (party_code, open_dt)                    [retro mode]
#
# Reference     : Experian Bureau Products v3.2 — Appendix A (new acct_type_cd)
#
# ── BALANCE RESOLUTION LOGIC ─────────────────────────────────────────────────
#
# Experian provides two ways to get balance:
#   BALANCE_AM           → Balance as of the LAST REPORTED DATE (balance_dt)
#   BALANCE_AM_01..36    → Monthly history going BACK from balance_dt
#                          _01 = 1 month before balance_dt
#                          _02 = 2 months before balance_dt, etc.
#                          _36 = 36 months before balance_dt
#
# For SCRUB mode: as_of_dt ≈ balance_dt → month_diff ≈ 0 → use BALANCE_AM directly
# For RETRO mode: as_of_dt < balance_dt → need to walk back into BALANCE_AM_NN
#
# month_diff = ceil(months_between(balance_dt, as_of_dt))
#            = how many months BACK from balance_dt we need to go
#
# Resolution:
#   month_diff <= 0  → as_of_dt is on or after balance_dt → BALANCE_AM is current
#   1 <= month_diff <= 36 → pick BALANCE_AM_{month_diff} from history columns
#   month_diff > 36  → beyond history window → NULL
#                      (unified: same code path handles both scrub and retro modes
#
# The resolved per-row balance is then summed/maxed per PK group per product type.
#
# Source columns:
#   balance_am        → Balance as of last reported date (-1 = NULL)
#   balance_dt        → Last reported date
#   balance_am_01..36 → Monthly history columns
#   orig_loan_am      → For STPL qualifier (orig_loan_am <= 30,000)
#   closed_dt         → For active flag
#   acct_type_cd      → Product type
#
# Product filters (new acct_type_cd — Appendix A):
#   CL  : '189'  Loan, Consumer  (collateral/consumer loan)
#   PL  : '123'  Loan, Personal Cash
#   HL  : '195'  Loan, Property
#   CC  : '5','213','214','220','225'
#   STPL: '242'  Short Term Personal Loan [Unsecured], orig_loan_am <= 30K
#
# NOTE on CL filter:
#   The reference implementation uses acct_type_cd = '189' (Loan, Consumer) for CL.
#   This is kept exactly as specified. If broader secured loan coverage is needed,
#   extend CL_CODES to include SPL_CODES from cat02/cat03.
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger

logger = get_logger(__name__)


CC_CODES  = {"5", "213", "214", "220", "225"}

PL_CODE   = "123"
HL_CODE   = "195"
CL_CODE   = "189"
STPL_CODE = "242"




class OutstandingBalanceFeatures(TradelineFeatureBase):
    """
    Category 05: Outstanding Balance / Current Debt

    Uses time-series BALANCE_AM_01..36 to resolve the correct balance
    as of the as-of date, not just the last reported date.

    Features
    --------
    total_outstanding_debt
        Sum of resolved balance across ALL accounts

    total_active_outstanding_debt
        Sum of resolved balance across ACTIVE accounts only

    currentbalancedue
        Sum of resolved balance across ALL accounts
        (reporting alias of total_outstanding_debt)

    TotalOutstandingOnActiveSTPL
        Sum of resolved balance for active STPL accounts (acct_type='242', loan<=30K)

    TotalOutstandingOnActivePLAccounts
        Sum of resolved balance for active PL accounts (acct_type='123')

    curr_CL_agg_bal
        Sum of resolved balance for active Consumer Loan accounts (acct_type='189')

    curr_PL_agg_bal
        Sum of resolved balance for active PL accounts (acct_type='123')

    curr_HL_agg_bal
        Sum of resolved balance for active HL accounts (acct_type='195')

    curr_CC_agg_bal
        Sum of resolved balance for active CC accounts

    max_agg_card_bal
        Max resolved balance across ALL CC accounts (any status)
    """

    CATEGORY = "cat05_outstanding_balance"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        # ── STEP 1: Parse date columns ────────────────────────────────────────
        def parse_date(col_name: str) -> F.Column:
            return F.coalesce(
                F.to_date(F.col(col_name), "dd/MM/yyyy"),
                F.to_date(F.col(col_name), "yyyy-MM-dd"),
                F.to_date(F.col(col_name), "MM/dd/yyyy"),
            )

        df = (
            df
            .withColumn("_closed_dt", parse_date("closed_dt"))
            .withColumn("_as_of_dt",  parse_date(as_of_col))
        )

        # ── STEP 2: Active flag ───────────────────────────────────────────────
        df = df.withColumn(
            "_is_active",
            F.when(
                F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt")),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        # ── STEP 3: Parse balance_dt ──────────────────────────────────────────
        df = df.withColumn("_balance_dt", parse_date("balance_dt"))

        # ── STEP 4: Clean orig_loan_am (-1 = NULL) ────────────────────────────
        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 5: Compute month_diff ────────────────────────────────────────
        # month_diff = how many months back from balance_dt is our as_of_dt
        #
        # Scrub mode : as_of_dt = scrub_output_date ≈ balance_dt → month_diff ≈ 0
        # Retro mode : as_of_dt = open_dt (past)               → month_diff > 0
        #
        # months_between(balance_dt, as_of_dt) is positive when as_of_dt < balance_dt
        # ceil() so a partial month counts as a full step back
        df = df.withColumn(
            "_month_diff",
            F.ceil(
                F.months_between(F.col("_balance_dt"), F.col("_as_of_dt"))
            ).cast("int")
        )

        # ── STEP 6: Resolve balance as of as_of_dt ────────────────────────────
        #
        # month_diff <= 0  → as_of_dt is on or after balance_dt
        #                    BALANCE_AM is already the current/future value → use it
        #
        # 1 <= month_diff <= 36 → as_of_dt is N months before balance_dt
        #                         pick BALANCE_AM_NN from history columns
        #
        # month_diff > 36  → beyond history window → NULL
        #
        # This is identical logic whether as_of_dt is scrub_output_date or open_dt.
        # The same code path handles both modes correctly.

        # Build selector: only the branch where i == month_diff returns non-null,
        # greatest() picks that single non-null value
        N_HISTORY = 36
        history_cols = [f"balance_am_{str(i).zfill(2)}" for i in range(1, N_HISTORY + 1)]

        history_selector = F.greatest(*[
            F.when(
                F.col("_month_diff") == i,
                F.col(history_cols[i - 1]).cast("double")
            )
            for i in range(1, N_HISTORY + 1)
        ])

        df = df.withColumn(
            "_resolved_bal",
            F.when(
                F.col("_month_diff") <= 0,
                F.col("balance_am").cast("double")          # use BALANCE_AM directly
            ).when(
                (F.col("_month_diff") >= 1) & (F.col("_month_diff") <= N_HISTORY),
                history_selector                             # pick BALANCE_AM_NN
            ).otherwise(
                F.lit(None).cast("double")                  # beyond 36m → NULL
            )
        )

        # Clamp: negative balances (credit positions) → 0
        df = df.withColumn(
            "_resolved_bal",
            F.when(F.col("_resolved_bal") > 0, F.col("_resolved_bal"))
             .otherwise(F.lit(0.0))
        )

        # ── STEP 7: Normalise acct_type_cd ───────────────────────────────────
        df = df.withColumn("_acct_type", F.trim(F.col("acct_type_cd").cast("string")))

        # ── STEP 8: Product type flags ────────────────────────────────────────
        df = (
            df
            .withColumn("_is_cl",
                F.col("_acct_type") == CL_CODE)
            .withColumn("_is_pl",
                F.col("_acct_type") == PL_CODE)
            .withColumn("_is_hl",
                F.col("_acct_type") == HL_CODE)
            .withColumn("_is_cc",
                F.col("_acct_type").isin(CC_CODES))
            .withColumn("_is_stpl",
                F.col("_acct_type") == STPL_CODE)
        )

        # STPL qualifier: acct_type='242' AND orig_loan_am <= 30K
        df = df.withColumn(
            "_is_stpl_qualified",
            F.col("_is_stpl") &
            F.col("_loan_am").isNotNull() &
            (F.col("_loan_am") <= 30000)
        )

        # ── STEP 8: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # Active accounts only
            F.sum(
                F.when(F.col("_is_active") == 1, F.col("_resolved_bal"))
            ).alias("total_active_outstanding_debt"),

            # Total outstanding — all accounts (all statuses)
            F.sum("_resolved_bal").alias("currentbalancedue"),

            # Active STPL (acct_type='242', orig_loan_am <= 30K)
            F.sum(
                F.when(
                    (F.col("_is_active") == 1) & F.col("_is_stpl_qualified"),
                    F.col("_resolved_bal")
                )
            ).alias("TotalOutstandingOnActiveSTPL"),

            # Active Consumer Loans (CL = acct_type '189')
            F.sum(
                F.when(
                    (F.col("_is_active") == 1) & F.col("_is_cl"),
                    F.col("_resolved_bal")
                )
            ).alias("curr_CL_agg_bal"),

            # Active Personal Loans
            F.sum(
                F.when(
                    (F.col("_is_active") == 1) & F.col("_is_pl"),
                    F.col("_resolved_bal")
                )
            ).alias("curr_PL_agg_bal"),

            # Active Home Loans
            F.sum(
                F.when(
                    (F.col("_is_active") == 1) & F.col("_is_hl"),
                    F.col("_resolved_bal")
                )
            ).alias("curr_HL_agg_bal"),

            # Active Credit Cards
            F.sum(
                F.when(
                    (F.col("_is_active") == 1) & F.col("_is_cc"),
                    F.col("_resolved_bal")
                )
            ).alias("curr_CC_agg_bal"),

            # Max balance across ALL CC accounts (any status)
            F.max(
                F.when(F.col("_is_cc"), F.col("_resolved_bal"))
            ).alias("max_agg_card_bal"),
        )

        self._log_done(feature_df)
        return feature_df


class CreditUtilizationFeatures(TradelineFeatureBase):
    """
    Category 14: Credit Utilization / Revolving Behaviour

    A. revolved_ratio_last_Wm  — delinquency frequency, all/product accounts
    B. utilization_*           — CC balance / credit limit, windowed avg/max
    """

    CATEGORY = "cat14_credit_utilization"

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER A: revolved ratio column (per row)
    # count(past_due > 0 in valid slots) / W
    # NULL if no valid slots in window
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _revolved_col(window: int, product_filter=None) -> F.Column:
        """Per-row revolved ratio for window W."""
        indicators = []
        for idx, col in enumerate(ALL_PDU_COLS):
            in_window = (
                (F.col("_md") <= idx) &
                (F.col("_md") + F.lit(window - 1) >= idx)
            )
            flag = F.when(in_window, F.when(F.col(col).cast("double") > 0, F.lit(1.0))
                           .otherwise(F.lit(0.0))
                  ).otherwise(F.lit(None).cast("double"))
            if product_filter is not None:
                flag = F.when(product_filter, flag).otherwise(F.lit(None).cast("double"))
            indicators.append(flag)

        total_past_due = sum(F.coalesce(c, F.lit(0.0)) for c in indicators)
        valid_count    = sum(F.when(c.isNotNull(), F.lit(1.0)).otherwise(F.lit(0.0))
                             for c in indicators)

        return F.when(
            valid_count > 0,
            (total_past_due / F.lit(float(window))).cast("double")
        ).otherwise(F.lit(None).cast("double"))

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER B: utilization per slot (per row) = BALANCE_AM_NN / ORIG_LOAN_AM
    # Only for CC. Returns list of per-slot utilization values for the window.
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _util_slots(window: int) -> List[F.Column]:
        """
        Per-slot utilization = balance_am_NN / orig_loan_am for each slot in window.
        Valid slots: month_diff <= idx <= month_diff + W - 1
        NULL if slot is gap, beyond history, balance=-1, or orig_loan_am invalid.
        """
        slots = []
        for idx, col in enumerate(ALL_BAL_COLS):
            in_window = (
                (F.col("_md") <= idx) &
                (F.col("_md") + F.lit(window - 1) >= idx)
            )
            bal = F.when(F.col(col) >= 0, F.col(col).cast("double")).otherwise(F.lit(None).cast("double"))
            util = F.when(
                in_window & bal.isNotNull() & F.col("_credit_limit").isNotNull(),
                (bal / F.col("_credit_limit")).cast("double")
            ).otherwise(F.lit(None).cast("double"))
            slots.append(util)
        return slots

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
            .withColumn("_is_cc",   F.col("_acct").isin(CC_CODES))
            .withColumn("_is_pl",   F.col("_acct") == PL_CODE)
            .withColumn("_is_stpl", F.col("_acct") == STPL_CODE)
            .withColumn("_is_usl",  ~F.col("_acct").isin(SECURED_CODES))
            .withColumn("_is_active_cc", F.col("_is_active") & F.col("_is_cc"))
            .withColumn("_is_active_pl", F.col("_is_active") & F.col("_is_pl"))
        )

        # ── STEP 5: Credit limit (denominator for utilization) ────────────────
        # Use ORIG_LOAN_AM as the sanctioned CC limit (-1 = NULL)
        df = df.withColumn(
            "_credit_limit",
            F.when(
                F.col("_is_cc") &
                F.col("orig_loan_am").isNotNull() &
                (F.col("orig_loan_am") > 0),
                F.col("orig_loan_am").cast("double")
            ).otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 6: Current past due ──────────────────────────────────────────
        df = df.withColumn(
            "_pdu_am",
            F.when(F.col("past_due_am") > 0, F.col("past_due_am").cast("double"))
             .otherwise(F.lit(0.0))
        )

        # ── STEP 7: Per-row revolved ratio columns ────────────────────────────
        df = (
            df
            .withColumn("_rev_3m",       self._revolved_col(3))
            .withColumn("_rev_6m",       self._revolved_col(6))
            .withColumn("_rev_12m",      self._revolved_col(12))
            .withColumn("_rev_3m_cc",    self._revolved_col(3,  F.col("_is_cc")))
            .withColumn("_rev_6m_cc",    self._revolved_col(6,  F.col("_is_cc")))
            .withColumn("_rev_12m_cc",   self._revolved_col(12, F.col("_is_cc")))
            .withColumn("_rev_3m_pl",    self._revolved_col(3,  F.col("_is_active_pl")))
            .withColumn("_rev_6m_pl",    self._revolved_col(6,  F.col("_is_active_pl")))
            .withColumn("_rev_12m_pl",   self._revolved_col(12, F.col("_is_active_pl")))
            .withColumn("_rev_12m_stpl", self._revolved_col(12, F.col("_is_stpl")))
            .withColumn("_rev_12m_usl",  self._revolved_col(12, F.col("_is_usl")))
        )

        # ── STEP 8: Per-slot utilization for CC ───────────────────────────────
        # Utilization = balance_am_NN / orig_loan_am (CC only)
        util_3m  = self._util_slots(3)
        util_6m  = self._util_slots(6)
        util_12m = self._util_slots(12)
        util_36m = self._util_slots(36)

        # Apply CC filter to all utilization slots
        util_3m_cc  = [F.when(F.col("_is_cc"), s).otherwise(F.lit(None).cast("double")) for s in util_3m]
        util_6m_cc  = [F.when(F.col("_is_cc"), s).otherwise(F.lit(None).cast("double")) for s in util_6m]
        util_12m_cc = [F.when(F.col("_is_cc"), s).otherwise(F.lit(None).cast("double")) for s in util_12m]
        util_36m_cc = [F.when(F.col("_is_cc"), s).otherwise(F.lit(None).cast("double")) for s in util_36m]

        # Per-row avg and max utilization within window
        def row_avg_util(slots):
            non_null = [F.coalesce(s, F.lit(None).cast("double")) for s in slots]
            total    = sum(F.coalesce(s, F.lit(0.0)) for s in slots)
            count    = sum(F.when(s.isNotNull(), F.lit(1.0)).otherwise(F.lit(0.0)) for s in slots)
            return F.when(count > 0, total / count).otherwise(F.lit(None).cast("double"))

        def row_max_util(slots):
            valid = [s for s in slots]
            return F.greatest(*valid)

        df = (
            df
            .withColumn("_util_avg_3m",  row_avg_util(util_3m_cc))
            .withColumn("_util_avg_6m",  row_avg_util(util_6m_cc))
            .withColumn("_util_avg_12m", row_avg_util(util_12m_cc))
            .withColumn("_util_avg_36m", row_avg_util(util_36m_cc))
            .withColumn("_util_max_3m",  row_max_util(util_3m_cc))
            .withColumn("_util_max_6m",  row_max_util(util_6m_cc))
            .withColumn("_util_max_12m", row_max_util(util_12m_cc))
            .withColumn("_util_max_36m", row_max_util(util_36m_cc))
        )

        # ── STEP 9: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # ── A. Revolved ratio — all accounts ─────────────────────────────
            F.avg("_rev_3m").alias("revolved_ratio_last_3m"),
            F.avg("_rev_6m").alias("revolved_ratio_last_6m"),
            F.avg("_rev_12m").alias("revolved_ratio_last_12m"),

            # Revolved ratio — CC only
            F.avg("_rev_3m_cc").alias("revolved_ratio_cc_last_3m"),
            F.avg("_rev_6m_cc").alias("revolved_ratio_cc_last_6m"),
            F.avg("_rev_12m_cc").alias("revolved_ratio_cc_last_12m"),

            # Revolved ratio — active PL
            F.avg("_rev_3m_pl").alias("revolved_ratio_pl_last_3m"),
            F.avg("_rev_6m_pl").alias("revolved_ratio_pl_last_6m"),
            F.avg("_rev_12m_pl").alias("revolved_ratio_pl_last_12m"),

            # Revolved ratio — STPL and USL
            F.avg("_rev_12m_stpl").alias("revolved_ratio_stpl_last_12m"),
            F.avg("_rev_12m_usl").alias("revolved_ratio_usl_last_12m"),

            # ── B. CC Utilization — AVG(balance/limit) across window ─────────
            # AVG across CC tradelines of per-row avg utilization in window
            F.avg("_util_avg_3m").alias("avg_cc_utilization_last_3m"),
            F.avg("_util_avg_6m").alias("avg_cc_utilization_last_6m"),
            F.avg("_util_avg_12m").alias("avg_cc_utilization_last_12m"),
            F.avg("_util_avg_36m").alias("avg_cc_utilization_last_36m"),

            # MAX utilization — peak credit usage across window (stress signal)
            F.max("_util_max_3m").alias("max_cc_utilization_last_3m"),
            F.max("_util_max_6m").alias("max_cc_utilization_last_6m"),
            F.max("_util_max_12m").alias("max_cc_utilization_last_12m"),
            F.max("_util_max_36m").alias("max_cc_utilization_last_36m"),

            # ── Supplementary past due features ──────────────────────────────
            F.sum(F.when(F.col("_is_active"), F.col("_pdu_am"))).alias("total_past_due_active"),
            F.sum("_pdu_am").alias("total_past_due_all"),
            F.max("_pdu_am").alias("max_past_due_single_account"),
            F.sum(F.when(F.col("_pdu_am") > 0, F.lit(1)).otherwise(F.lit(0))
            ).alias("count_accounts_with_past_due"),
            F.sum(F.when(F.col("_is_active_cc") & (F.col("_pdu_am") > 0), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_cc_with_past_due"),
            F.max(F.when(F.col("_pdu_am") > 0, F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_any_past_due"),

            # For derived flag
            F.avg("_rev_12m").alias("_rev_12m_avg"),
        )

        # ── STEP 10: Derived flag ─────────────────────────────────────────────
        # flag_consistent_revolver: past due in > 50% of last 12m months
        feature_df = feature_df.withColumn(
            "flag_consistent_revolver",
            F.when(
                F.col("_rev_12m_avg").isNotNull() & (F.col("_rev_12m_avg") > 0.5),
                F.lit(1)
            ).otherwise(F.lit(0))
        ).drop("_rev_12m_avg")

        self._log_done(feature_df)
        return feature_df
