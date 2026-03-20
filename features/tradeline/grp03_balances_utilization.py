# features/tradeline/grp03_balances_utilization.py
# =============================================================================
# Group 03 — Outstanding Balances & Credit Utilization
# =============================================================================
# Merged from: cat05 (outstanding balance), cat14 (credit utilization)
#
# Sections:
#   A. OutstandingBalanceFeatures  — resolved balance per product at as_of_dt
#   B. CreditUtilizationFeatures   — missed payment frequency ratio + CC utilization windowed
#
# Balance resolution (point-in-time):
#   month_diff = ceil(months_between(balance_dt, as_of_dt))
#   <= 0       → use BALANCE_AM directly
#   1..36      → use BALANCE_AM_NN history column
#   > 36       → NULL (beyond window)
#
# Secured vs Unsecured — Appendix A verified:
#   SECURED  = collateral-backed: Automobile, Mortgage, Property, Gold,
#              Shares/Securities, FD, Commercial Vehicle, Two-Wheeler,
#              Three-Wheeler, Used Car, Construction Equipment, P2P Auto,
#              MFI Housing, GECL Secured, Non-Funded Credit Facilities,
#              Business Loan Against Deposits, Leasing
#   UNSECURED = PL, Consumer Loan, CC (all codes), Student, Professional,
#              MFI (Business/Personal/Other), P2P (Personal/Education),
#              Business Loans (general/priority), STPL, USL, GECL Unsecured
#   NOTE: CC (5,213,214,220,224,225) = CC/unsecured category (all CC codes incl. 220 treated as CC)
#   NOTE: Education loan (130,247) = UNSECURED
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger
from core.date_utils import parse_date

logger = get_logger(__name__)

# =============================================================================
# CODE SETS — verified against Appendix A (Experian Bureau Products v3.2)
# =============================================================================

CC_CODES  = {"5", "213", "214", "220", "224", "225"}   # All CCs incl. 220 (Secured CC — treated as CC/unsecured)

PL_CODE   = "123"   # Loan, Personal Cash
HL_CODE   = "195"   # Loan, Property
CL_CODE   = "189"   # Loan, Consumer
STPL_CODE = "242"   # Short Term Personal Loan

# SECURED = physical or financial collateral backing the loan
# Verified per Appendix A — NOTE: "220" (Secured CC) moved to CC_CODES; "181","197-200","240","246","248" added
SECURED_CODES = {
    "47",   # Instalment Loan, Automobile
    "58",   # Instalment Loan, Mortgage
    "168",  # Microfinance, Housing
    "172",  # Instalment Loan, Commercial Vehicle
    "173",  # Instalment Loan, Two-Wheeler
    "175",  # Business Loan Against Bank Deposits
    "181",  # Credit Facility, Non-Funded
    "184",  # Loan Against Bank Deposits
    "185",  # Loan Against Shares/Securities
    "191",  # Loan, Gold
    "195",  # Loan, Property
    "197",  # Non-Funded Credit Facility, General
    "198",  # Non-Funded Credit Facility, Priority Sector - Small Business
    "199",  # Non-Funded Credit Facility, Priority Sector - Agriculture
    "200",  # Non-Funded Credit Facility, Priority Sector - Others
    "219",  # Leasing, Other
    # 220 (Secured Credit Card) removed — CCs (incl. 220) are treated as CC/unsecured category
    "221",  # Used Car Loan
    "222",  # Construction Equipment Loan
    "223",  # Tractor Loan
    "240",  # Pradhan Mantri Awas Yojna  (housing scheme)
    "241",  # Business Loan – Secured
    "243",  # Priority Sector Gold Loan
    "246",  # P2P Auto Loan
    "248",  # GECL Loan Secured
}

# UNSECURED — for reference (anything not in SECURED_CODES and not OTHER)
# Includes: PL(123), Consumer(189), CC(all), Student(130), Professional(187),
#           MFI Business/Personal/Other(167,169,170), P2P PL/Education(245,247),
#           Business Loans(176-179), STPL(242,244), USL(226-228), GECL Unsec(249)

N_HISTORY    = 36
PDU_COLS     = [f"past_due_am_{str(i).zfill(2)}" for i in range(1, N_HISTORY + 1)]
BAL_COLS     = [f"balance_am_{str(i).zfill(2)}"  for i in range(1, N_HISTORY + 1)]
ALL_PDU_COLS = ["past_due_am"] + PDU_COLS    # idx 0 = current, 1-36 = history
ALL_BAL_COLS = ["balance_am"]  + BAL_COLS    # idx 0 = current, 1-36 = history
CL_COLS      = [f"credit_limit_am_{str(i).zfill(2)}" for i in range(1, N_HISTORY + 1)]
ALL_CL_COLS  = ["credit_limit_am"] + CL_COLS     # idx 0 = current, 1-36 = history


class OutstandingBalanceFeatures(TradelineFeatureBase):
    """
    Category 05: Outstanding Balance / Current Debt

    Resolves balance at as_of_dt using month_diff slot resolution.
    Features: total/active debt, product-level balances (PL/CL/HL/CC/STPL).
    """

    CATEGORY = "grp03a_outstanding_balance"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]


        df = (
            df
            .withColumn("_closed_dt",  parse_date("closed_dt"))
            .withColumn("_open_dt",    parse_date("open_dt"))
            .withColumn("_as_of_dt",   parse_date(as_of_col))
            .withColumn("_balance_dt", parse_date("balance_dt"))
        )

        # Active flag — PIT
        df = df.withColumn(
            "_is_active",
            (F.col("_open_dt") <= F.col("_as_of_dt")) &
            (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt")))
        )

        # Clean orig_loan_am
        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am").cast("double") > 0,
                   F.col("orig_loan_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )

        # month_diff = how many months back from balance_dt is as_of_dt
        df = df.withColumn(
            "_month_diff",
            F.ceil(F.months_between(F.col("_balance_dt"), F.col("_as_of_dt"))).cast("int")
        )

        # Resolve balance at as_of_dt via slot selection
        history_selector = F.greatest(*[
            F.when(F.col("_month_diff") == i,
                   F.col(BAL_COLS[i - 1]).cast("double"))
            for i in range(1, N_HISTORY + 1)
        ])

        df = df.withColumn(
            "_resolved_bal",
            F.when(F.col("_month_diff") <= 0,
                   F.col("balance_am").cast("double"))
             .when((F.col("_month_diff") >= 1) & (F.col("_month_diff") <= N_HISTORY),
                   history_selector)
             .otherwise(F.lit(None).cast("double"))
        )

        # Clamp negatives to 0
        df = df.withColumn(
            "_resolved_bal",
            F.when(F.col("_resolved_bal") > 0, F.col("_resolved_bal"))
             .otherwise(F.lit(0.0))
        )

        df = df.withColumn("_acct_type", F.trim(F.col("acct_type_cd").cast("string")))

        # Product flags — all boolean
        df = (
            df
            .withColumn("_is_cl",   F.col("_acct_type") == CL_CODE)
            .withColumn("_is_pl",   F.col("_acct_type") == PL_CODE)
            .withColumn("_is_hl",   F.col("_acct_type") == HL_CODE)
            .withColumn("_is_cc",   F.col("_acct_type").isin(CC_CODES))
            .withColumn("_is_stpl", F.col("_acct_type") == STPL_CODE)
            .withColumn("_is_stpl_qualified",
                (F.col("_acct_type") == STPL_CODE) &
                F.col("_loan_am").isNotNull() &
                (F.col("_loan_am") <= 30000))
        )

        def _i(cond):
            return F.when(cond, F.lit(1)).otherwise(F.lit(0))

        feature_df = df.groupBy(group_cols).agg(

            # Total active balance
            F.sum(F.when(F.col("_is_active"), F.col("_resolved_bal"))
            ).alias("total_active_outstanding_debt"),

            # Total balance — all accounts
            F.sum("_resolved_bal").alias("currentbalancedue"),

            # Active STPL (orig <= 30K)
            F.sum(
                F.when(F.col("_is_active") & F.col("_is_stpl_qualified"),
                       F.col("_resolved_bal"))
            ).alias("TotalOutstandingOnActiveSTPL"),

            # Active Consumer Loans (189)
            F.sum(
                F.when(F.col("_is_active") & F.col("_is_cl"),
                       F.col("_resolved_bal"))
            ).alias("curr_CL_agg_bal"),

            # Active Personal Loans (123)
            F.sum(
                F.when(F.col("_is_active") & F.col("_is_pl"),
                       F.col("_resolved_bal"))
            ).alias("curr_PL_agg_bal"),

            # Active Home Loans (195)
            F.sum(
                F.when(F.col("_is_active") & F.col("_is_hl"),
                       F.col("_resolved_bal"))
            ).alias("curr_HL_agg_bal"),

            # Active Credit Cards — UNSECURED revolving
            F.sum(
                F.when(F.col("_is_active") & F.col("_is_cc"),
                       F.col("_resolved_bal"))
            ).alias("curr_CC_agg_bal"),

            # Max CC balance (any status)
            F.max(
                F.when(F.col("_is_cc"), F.col("_resolved_bal"))
            ).alias("max_agg_card_bal"),
        )

        self._log_done(feature_df)
        return feature_df


class CreditUtilizationFeatures(TradelineFeatureBase):
    """
    Category 14: Credit Utilization / Revolving Behaviour

    A. missed_payment_freq — past_due frequency over window (all/CC/PL/STPL/USL)
    B. CC utilization  — balance / credit limit, windowed avg and max
    C. Past due summary features
    """

    CATEGORY = "grp03b_credit_utilization"

    @staticmethod
    def _missed_payment_freq_col(window: int, product_filter=None) -> F.Column:
        """Per-row missed payment frequency ratio for window W using past_due_am history."""
        indicators = []
        for idx in range(1, N_HISTORY + 1):
            col = PDU_COLS[idx - 1]
            in_window = (F.col("_md") <= F.lit(idx)) & (F.col("_md") + F.lit(window - 1) >= F.lit(idx))
            flag = F.when(
                in_window,
                F.when(F.col(col).cast("double") > 0, F.lit(1.0)).otherwise(F.lit(0.0))
            ).otherwise(F.lit(None).cast("double"))
            if product_filter is not None:
                flag = F.when(product_filter, flag).otherwise(F.lit(None).cast("double"))
            indicators.append(flag)

        total   = sum(F.coalesce(c, F.lit(0.0)) for c in indicators)
        valid_n = sum(F.when(c.isNotNull(), F.lit(1.0)).otherwise(F.lit(0.0)) for c in indicators)
        return F.when(valid_n > 0, (total / F.lit(float(window))).cast("double")
               ).otherwise(F.lit(None).cast("double"))

    @staticmethod
    def _util_slots(window: int) -> List[F.Column]:
        """
        Per-slot CC utilization = balance_am_NN / credit_limit_am_NN.

        Uses matching history columns for BOTH balance AND credit limit,
        so utilization is computed at the same point-in-time as the balance.
        Slot is only valid when:
          - slot is within the window for this row (as_of relative to balance_dt)
          - balance_am_NN is non-null and >= 0
          - credit_limit_am_NN is non-null and > 0
        NULL → no data for that slot (gap or beyond window)
        """
        slots = []
        for idx in range(1, N_HISTORY + 1):
            bal_col = BAL_COLS[idx - 1]
            cl_col  = CL_COLS[idx - 1]
            in_window = (F.col("_md") <= F.lit(idx)) & (F.col("_md") + F.lit(window - 1) >= F.lit(idx))
            bal = F.when(F.col(bal_col).cast("double") >= 0,
                         F.col(bal_col).cast("double")).otherwise(F.lit(None).cast("double"))
            cl  = F.when(F.col(cl_col).cast("double") > 0,
                         F.col(cl_col).cast("double")).otherwise(F.lit(None).cast("double"))
            util = F.when(
                in_window & bal.isNotNull() & cl.isNotNull(),
                (bal / cl).cast("double")
            ).otherwise(F.lit(None).cast("double"))
            slots.append(util)
        return slots

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]


        df = (
            df
            .withColumn("_bal_dt",   parse_date("balance_dt"))
            .withColumn("_as_of_dt", parse_date(as_of_col))
            .withColumn("_open_dt",  parse_date("open_dt"))
            .withColumn("_closed_dt",parse_date("closed_dt"))
        )

        # month_diff: as_of relative to balance_dt
        df = df.withColumn(
            "_md",
            F.ceil(F.months_between(F.col("_as_of_dt"), F.col("_bal_dt"))).cast("int")
        )

        # Active flag
        df = df.withColumn(
            "_is_active",
            (F.col("_open_dt") <= F.col("_as_of_dt")) &
            (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt")))
        )

        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))

        # Product flags — all boolean, consistent types
        df = (
            df
            .withColumn("_is_cc",        F.col("_acct").isin(CC_CODES))
            .withColumn("_is_pl",        F.col("_acct") == PL_CODE)
            .withColumn("_is_stpl",      F.col("_acct") == STPL_CODE)
            # UNSECURED = ~SECURED (includes CC, PL, Student, Education P2P, MFI, BL, STPL)
            .withColumn("_is_usl",       ~F.col("_acct").isin(SECURED_CODES))
            .withColumn("_is_active_cc", F.col("_is_active") & F.col("_is_cc"))
            .withColumn("_is_active_pl", F.col("_is_active") & F.col("_is_pl"))
        )

        # Current credit limit for CC (credit_limit_am column — Experian variable 13)
        # Used as scalar denominator for current-month utilization
        # Per-slot utilization uses credit_limit_am_NN history columns directly
        df = df.withColumn(
            "_credit_limit_now",
            F.when(
                F.col("_is_cc") &
                F.col("credit_limit_am").isNotNull() &
                (F.col("credit_limit_am").cast("double") > 0),
                F.col("credit_limit_am").cast("double")
            ).otherwise(F.lit(None).cast("double"))
        )

        # Current past due amount — NULL if source is NULL/negative (no reporting data)
        # 0 only when there IS reporting data and past_due = 0
        df = df.withColumn(
            "_pdu_am",
            F.when(
                F.col("past_due_am").isNotNull() & (F.col("past_due_am").cast("double") >= 0),
                F.col("past_due_am").cast("double")
            ).otherwise(F.lit(None).cast("double"))
        )

        # Missed payment frequency columns (per row)
        df = (
            df
            .withColumn("_mpf_3m",       self._missed_payment_freq_col(3))
            .withColumn("_mpf_6m",       self._missed_payment_freq_col(6))
            .withColumn("_mpf_12m",      self._missed_payment_freq_col(12))
            .withColumn("_mpf_3m_cc",    self._missed_payment_freq_col(3,  F.col("_is_cc")))
            .withColumn("_mpf_6m_cc",    self._missed_payment_freq_col(6,  F.col("_is_cc")))
            .withColumn("_mpf_12m_cc",   self._missed_payment_freq_col(12, F.col("_is_cc")))
            .withColumn("_mpf_3m_pl",    self._missed_payment_freq_col(3,  F.col("_is_active_pl")))
            .withColumn("_mpf_6m_pl",    self._missed_payment_freq_col(6,  F.col("_is_active_pl")))
            .withColumn("_mpf_12m_pl",   self._missed_payment_freq_col(12, F.col("_is_active_pl")))
            .withColumn("_mpf_12m_stpl", self._missed_payment_freq_col(12, F.col("_is_stpl")))
            .withColumn("_mpf_12m_usl",  self._missed_payment_freq_col(12, F.col("_is_usl")))
        )

        # CC utilization slots
        def apply_cc_filter(slots):
            return [F.when(F.col("_is_cc"), s).otherwise(F.lit(None).cast("double"))
                    for s in slots]

        util_3m_cc  = apply_cc_filter(self._util_slots(3))
        util_6m_cc  = apply_cc_filter(self._util_slots(6))
        util_12m_cc = apply_cc_filter(self._util_slots(12))
        util_36m_cc = apply_cc_filter(self._util_slots(36))

        def row_avg(slots):
            total = sum(F.coalesce(s, F.lit(0.0)) for s in slots)
            count = sum(F.when(s.isNotNull(), F.lit(1.0)).otherwise(F.lit(0.0)) for s in slots)
            return F.when(count > 0, total / count).otherwise(F.lit(None).cast("double"))

        def row_max(slots):
            return F.greatest(*slots)

        df = (
            df
            .withColumn("_util_avg_3m",  row_avg(util_3m_cc))
            .withColumn("_util_avg_6m",  row_avg(util_6m_cc))
            .withColumn("_util_avg_12m", row_avg(util_12m_cc))
            .withColumn("_util_avg_36m", row_avg(util_36m_cc))
            .withColumn("_util_max_3m",  row_max(util_3m_cc))
            .withColumn("_util_max_6m",  row_max(util_6m_cc))
            .withColumn("_util_max_12m", row_max(util_12m_cc))
            .withColumn("_util_max_36m", row_max(util_36m_cc))
        )

        def _i(cond):
            return F.when(cond, F.lit(1)).otherwise(F.lit(0))

        feature_df = df.groupBy(group_cols).agg(

            # A. Missed payment frequency — all accounts
            F.avg("_mpf_3m").alias("missed_payment_freq_last_3m"),
            F.avg("_mpf_6m").alias("missed_payment_freq_last_6m"),
            F.avg("_mpf_12m").alias("missed_payment_freq_last_12m"),

            # CC only
            F.avg("_mpf_3m_cc").alias("missed_payment_freq_cc_last_3m"),
            F.avg("_mpf_6m_cc").alias("missed_payment_freq_cc_last_6m"),
            F.avg("_mpf_12m_cc").alias("missed_payment_freq_cc_last_12m"),

            # Active PL only
            F.avg("_mpf_3m_pl").alias("missed_payment_freq_pl_last_3m"),
            F.avg("_mpf_6m_pl").alias("missed_payment_freq_pl_last_6m"),
            F.avg("_mpf_12m_pl").alias("missed_payment_freq_pl_last_12m"),

            # STPL and USL (unsecured = ~SECURED, includes CC+PL+education+MFI etc.)
            F.avg("_mpf_12m_stpl").alias("missed_payment_freq_stpl_last_12m"),
            F.avg("_mpf_12m_usl").alias("missed_payment_freq_usl_last_12m"),

            # B. CC Utilization (balance/limit)
            F.avg("_util_avg_3m").alias("avg_cc_utilization_last_3m"),
            F.avg("_util_avg_6m").alias("avg_cc_utilization_last_6m"),
            F.avg("_util_avg_12m").alias("avg_cc_utilization_last_12m"),
            F.avg("_util_avg_36m").alias("avg_cc_utilization_last_36m"),
            F.max("_util_max_3m").alias("max_cc_utilization_last_3m"),
            F.max("_util_max_6m").alias("max_cc_utilization_last_6m"),
            F.max("_util_max_12m").alias("max_cc_utilization_last_12m"),
            F.max("_util_max_36m").alias("max_cc_utilization_last_36m"),

            # C. Past due summary
            F.sum(F.when(F.col("_is_active"), F.col("_pdu_am"))
            ).alias("total_past_due_active"),

            F.sum("_pdu_am").alias("total_past_due_all"),
            F.max("_pdu_am").alias("max_past_due_single_account"),

            F.sum(_i(F.col("_pdu_am") > 0)).alias("count_accounts_with_past_due"),
            F.sum(_i(F.col("_is_active_cc") & (F.col("_pdu_am") > 0))
            ).alias("count_cc_with_past_due"),
            F.max(_i(F.col("_pdu_am") > 0)).alias("flag_any_past_due"),

            # For derived flag
            F.avg("_mpf_12m").alias("_mpf_12m_avg"),
        )

        # flag_consistent_revolver: past due > 50% of last 12 months
        feature_df = feature_df.withColumn(
            "flag_consistent_revolver",
            F.when(
                F.col("_mpf_12m_avg").isNotNull() & (F.col("_mpf_12m_avg") > 0.5),
                F.lit(1)
            ).otherwise(F.lit(0))
        ).drop("_mpf_12m_avg")

        self._log_done(feature_df)
        return feature_df
