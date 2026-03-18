# features/tradeline/cat03_loan_amount_exposure.py
# =============================================================================
# Category 03 — Loan Amount / Exposure Features
# =============================================================================
# Source table  : experian_tradeline_segment
# Granularity   : One row per (customer_scrub_key, scrub_output_date)  [scrub mode]
#                 One row per (party_code, open_dt)                    [retro mode]
#
# Reference     : Experian Bureau Products v3.2 — Appendix A (new acct_type_cd)
#
# Source columns used:
#   orig_loan_am     → Disbursed loan amount (Experian uses -1 as null placeholder)
#   credit_limit_am  → Credit limit for credit cards (-1 = null)
#   closed_dt        → Trade line closed date (null = still open)
#   acct_type_cd     → Account type (new codes from Appendix A)
#
# Key rules:
#   • orig_loan_am = -1  → treat as NULL (not available)
#   • credit_limit_am = -1 → treat as NULL
#   • Active  : closed_dt IS NULL OR closed_dt > as_of_dt
#   • Inactive: closed_dt IS NOT NULL AND closed_dt <= as_of_dt
#   • STPL    : acct_type_cd = '242'  AND  orig_loan_am <= 30,000
#   • PL      : acct_type_cd = '123'
#   • GL      : acct_type_cd IN ('191','243')
#   • CC      : acct_type_cd IN ('5','213','214','220','224','225')
#   • HL      : acct_type_cd IN ('58','195','168','240')
#   • AL      : acct_type_cd IN ('47','173','172','221','222','223','246')
#   • SPL     : acct_type_cd IN ('184','185','175','241','248','181','197','198','199','200')
#   • USL     : acct_type_cd IN USL_CODES (same as cat02)
#   • "Others": acct_type_cd NOT in any named group
#   • For max_loanamount_cc / max_credit_limit → use credit_limit_am when
#     orig_loan_am is null, as CC products report limit not disbursed amount
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# CODE SETS  (reuse same definitions as cat02 for consistency)
# =============================================================================

GL_CODES   = {"191", "243"}
AL_CODES   = {"47", "173", "172", "221", "222", "223", "246"}
HL_CODES   = {"58", "195", "168", "240"}
CC_CODES   = {"5", "213", "214", "220", "224", "225"}
SPL_CODES  = {"184", "185", "175", "241", "248", "181", "197", "198", "199", "200"}
USL_CODES = {
    "123", "189", "187", "130",
    "242", "244", "245", "247",
    "167", "169", "170",
    "176", "177", "178", "179", "228",
    "227", "226", "249",
}

PL_CODE   = "123"
STPL_CODE = "242"

ALL_NAMED_CODES = GL_CODES | AL_CODES | HL_CODES | CC_CODES | SPL_CODES | USL_CODES




# =============================================================================

class LoanAmountExposureFeatures(TradelineFeatureBase):
    """
    Category 03: Loan Amount / Exposure Features

    Features
    --------
    maxLoanAmountAllAccountsExceptGLandCC
        Max orig_loan_am across all accounts excluding Gold and Credit Cards

    MaxLoanAmountInactiveSTPL
        Max orig_loan_am for CLOSED STPL accounts (acct_type='242', amount<=30K)

    MaxLoanAmountActiveSTPL
        Max orig_loan_am for ACTIVE STPL accounts (acct_type='242', amount<=30K)

    MaxLoanAmountTotalSTPL
        Max orig_loan_am across ALL STPL accounts (acct_type='242', amount<=30K)

    MaxLoanAmountActivePersonalLoan
        Max orig_loan_am for ACTIVE PL accounts (acct_type='123')

    TotalLoanAmtOnActivePLAccounts
        Sum of orig_loan_am for ACTIVE PL accounts (acct_type='123')

    sumloanamount_personalloan
        Sum of orig_loan_am for ALL PL accounts (acct_type='123')

    max_loanamount_sl
        Max orig_loan_am across secured loan accounts (SPL_CODES)

    max_loanamount_usl
        Max orig_loan_am across unsecured loan accounts (USL_CODES)

    max_loanamount_cc
        Max credit_limit_am for CC accounts (falls back to orig_loan_am if null)

    max_loanamount_gl
        Max orig_loan_am for gold loan accounts

    max_loanamount_hl
        Max orig_loan_am for home loan accounts

    max_loanamount_al
        Max orig_loan_am for auto / vehicle loan accounts

    max_loanamount_others
        Max orig_loan_am for accounts not in any named product group

    max_loanamount
        Max orig_loan_am across ALL accounts

    max_credit_limit
        Max credit_limit_am across all active CC accounts

    active_total_cc_credit_limit
        Sum of credit_limit_am across all ACTIVE CC accounts
        (derived companion feature as total revolving credit capacity)
    """

    CATEGORY = "cat03_loan_amount_exposure"  # grp02_loan_amounts

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

        # ── STEP 2: Active / Inactive flags ──────────────────────────────────
        df = (
            df
            .withColumn(
                "_is_active",
                F.when(
                    F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt")),
                    F.lit(1)
                ).otherwise(F.lit(0))
            )
            .withColumn(
                "_is_inactive",
                F.when(
                    F.col("_closed_dt").isNotNull() & (F.col("_closed_dt") <= F.col("_as_of_dt")),
                    F.lit(1)
                ).otherwise(F.lit(0))
            )
        )

        # ── STEP 3: Clean amount columns — treat -1 as null ───────────────────
        # Experian uses -1 as a placeholder for "information not available"
        df = (
            df
            .withColumn(
                "_loan_am",
                F.when(F.col("orig_loan_am").cast("double") > 0, F.col("orig_loan_am").cast("double"))
                 .otherwise(F.lit(None).cast("double"))
            )
            .withColumn(
                "_credit_limit_am",
                F.when(F.col("credit_limit_am").cast("double") > 0, F.col("credit_limit_am").cast("double"))
                 .otherwise(F.lit(None).cast("double"))
            )
        )

        # ── STEP 4: Normalise acct_type_cd ───────────────────────────────────
        df = df.withColumn("_acct_type", F.trim(F.col("acct_type_cd").cast("string")))

        # ── STEP 5: Product type flags ────────────────────────────────────────
        df = (
            df
            .withColumn("_is_pl",    F.col("_acct_type") == PL_CODE)
            .withColumn("_is_stpl",  F.col("_acct_type") == STPL_CODE)
            .withColumn("_is_gl",    F.col("_acct_type").isin(GL_CODES))
            .withColumn("_is_al",    F.col("_acct_type").isin(AL_CODES))
            .withColumn("_is_hl",    F.col("_acct_type").isin(HL_CODES))
            .withColumn("_is_cc",    F.col("_acct_type").isin(CC_CODES))
            .withColumn("_is_spl",   F.col("_acct_type").isin(SPL_CODES))
            .withColumn("_is_usl",   F.col("_acct_type").isin(USL_CODES))
            .withColumn("_is_other", ~F.col("_acct_type").isin(ALL_NAMED_CODES))
        )

        # ── STEP 6: STPL qualifier — acct_type='242' AND amount <= 30,000 ────
        # Note: only rows that satisfy the amount threshold are considered STPL
        # for the MaxLoanAmount STPL features
        df = df.withColumn(
            "_is_stpl_qualified",
            F.col("_is_stpl") & (F.col("_loan_am") <= 30000) & F.col("_loan_am").isNotNull()
        )

        # ── STEP 7: For CC — use orig_loan_am (sanctioned/credit limit amount) ─
        # orig_loan_am is the authoritative source for CC credit limit values
        df = df.withColumn(
            "_cc_am",
            F.when(F.col("_is_cc"), F.col("_loan_am"))
             .otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 8: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # Max loan amount — all accounts EXCEPT GL and CC
            F.max(
                F.when(~F.col("_is_gl") & ~F.col("_is_cc"), F.col("_loan_am"))
            ).alias("maxLoanAmountAllAccountsExceptGLandCC"),

            # STPL — inactive (closed), amount <= 30K
            F.max(
                F.when(F.col("_is_stpl_qualified") & (F.col("_is_inactive") == 1), F.col("_loan_am"))
            ).alias("MaxLoanAmountInactiveSTPL"),

            # STPL — active (open), amount <= 30K
            F.max(
                F.when(F.col("_is_stpl_qualified") & (F.col("_is_active") == 1), F.col("_loan_am"))
            ).alias("MaxLoanAmountActiveSTPL"),

            # STPL — all (active + inactive), amount <= 30K
            F.max(
                F.when(F.col("_is_stpl_qualified"), F.col("_loan_am"))
            ).alias("MaxLoanAmountTotalSTPL"),

            # Personal Loan — active accounts only
            F.max(
                F.when(F.col("_is_pl") & (F.col("_is_active") == 1), F.col("_loan_am"))
            ).alias("MaxLoanAmountActivePersonalLoan"),

            # Personal Loan — sum of active accounts
            F.sum(
                F.when(F.col("_is_pl") & (F.col("_is_active") == 1), F.col("_loan_am"))
            ).alias("TotalLoanAmtOnActivePLAccounts"),

            # Personal Loan — sum of ALL accounts (active + inactive)
            F.sum(
                F.when(F.col("_is_pl"), F.col("_loan_am"))
            ).alias("sumloanamount_personalloan"),

            # Max by product group
            F.max(F.when(F.col("_is_spl"),   F.col("_loan_am"))).alias("max_loanamount_sl"),
            F.max(F.when(F.col("_is_usl"),   F.col("_loan_am"))).alias("max_loanamount_usl"),
            F.max(F.when(F.col("_is_cc"),    F.col("_cc_am")  )).alias("max_loanamount_cc"),
            F.max(F.when(F.col("_is_gl"),    F.col("_loan_am"))).alias("max_loanamount_gl"),
            F.max(F.when(F.col("_is_hl"),    F.col("_loan_am"))).alias("max_loanamount_hl"),
            F.max(F.when(F.col("_is_al"),    F.col("_loan_am"))).alias("max_loanamount_al"),
            F.max(F.when(F.col("_is_other"), F.col("_loan_am"))).alias("max_loanamount_others"),

            # Global max across ALL accounts
            F.max(F.col("_loan_am")).alias("max_loanamount"),

            # CC credit limit features — use orig_loan_am as credit limit source
            # (orig_loan_am carries the sanctioned/credit limit for CC products)
            # max_credit_limit  → highest orig_loan_am across ALL active CC accounts
            F.max(
                F.when(F.col("_is_cc") & (F.col("_is_active") == 1), F.col("_loan_am"))
            ).alias("max_credit_limit"),

            # active_total_cc_credit_limit → sum of orig_loan_am across ALL active CC accounts
            # = total revolving credit capacity available to the customer
            F.sum(
                F.when(F.col("_is_cc") & (F.col("_is_active") == 1), F.col("_loan_am"))
            ).alias("active_total_cc_credit_limit"),
        )

        # ── STEP 9: Derived / ratio features (computed on the aggregated row) ──

        # 1. ratio_max_pl_to_max_loanamount
        #    How dominant PL is in overall credit exposure.
        #    High ratio in high-amount bands → concentrated unsecured risk.
        feature_df = feature_df.withColumn(
            "ratio_max_pl_to_max_loanamount",
            F.when(
                F.col("max_loanamount").isNotNull() & (F.col("max_loanamount") > 0),
                F.col("MaxLoanAmountActivePersonalLoan") / F.col("max_loanamount")
            ).otherwise(F.lit(None).cast("double"))
        )

        # 2. ratio_active_pl_sum_to_max_loan
        #    Active PL debt load relative to customer's peak loan exposure.
        feature_df = feature_df.withColumn(
            "ratio_active_pl_sum_to_max_loan",
            F.when(
                F.col("max_loanamount").isNotNull() & (F.col("max_loanamount") > 0),
                F.col("TotalLoanAmtOnActivePLAccounts") / F.col("max_loanamount")
            ).otherwise(F.lit(None).cast("double"))
        )

        # 3. flag_stpl_only
        #    Customer's entire credit history is sub-30K STPLs → thin/NTC segment.
        #    1 = stpl seen AND overall max loan is also within 30K band
        feature_df = feature_df.withColumn(
            "flag_stpl_only",
            F.when(
                F.col("MaxLoanAmountTotalSTPL").isNotNull() &
                (F.col("max_loanamount") <= 30000),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        # 4. flag_high_value_hl
        #    Home loan ≥ 25 lakh → higher income/asset segment.
        #    Strong negative (protective) risk signal in PL/STPL models.
        feature_df = feature_df.withColumn(
            "flag_high_value_hl",
            F.when(
                F.col("max_loanamount_hl").isNotNull() & (F.col("max_loanamount_hl") >= 2500000),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        # 5. flag_has_large_pl
        #    Active PL ≥ 5 lakh → high income OR over-leverage signal.
        feature_df = feature_df.withColumn(
            "flag_has_large_pl",
            F.when(
                F.col("MaxLoanAmountActivePersonalLoan").isNotNull() &
                (F.col("MaxLoanAmountActivePersonalLoan") >= 500000),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        # 6. total_active_exposure
        #    Sum of orig_loan_am across ALL active accounts.
        #    Computed here as a derived aggregate from the row-level data
        #    already grouped — requires a second pass on the pre-agg df.
        #    Implemented inline using a sub-aggregation on the flagged df.
        active_exposure_df = df.groupBy(group_cols).agg(
            F.sum(
                F.when(F.col("_is_active") == 1, F.col("_loan_am"))
            ).alias("total_active_exposure")
        )
        feature_df = feature_df.join(active_exposure_df, on=group_cols, how="left")

        # 7. sumloanamount_personalloan_lt_50K
        #    Sum of PL accounts where orig_loan_am <= 50,000.
        #    Granular small-ticket PL exposure for STPL/microfinance models.
        pl_lt50k_df = df.groupBy(group_cols).agg(
            F.sum(
                F.when(F.col("_is_pl") & F.col("_loan_am").isNotNull() & (F.col("_loan_am") <= 50000),
                       F.col("_loan_am"))
            ).alias("sumloanamount_personalloan_lt_50K")
        )
        feature_df = feature_df.join(pl_lt50k_df, on=group_cols, how="left")

        # 8. max_loanamount_usl_vs_sl_ratio
        #    Relative unsecured vs secured peak exposure.
        #    Null when no secured credit — pure unsecured customers flagged separately.
        feature_df = feature_df.withColumn(
            "max_loanamount_usl_vs_sl_ratio",
            F.when(
                F.col("max_loanamount_sl").isNotNull() & (F.col("max_loanamount_sl") > 0),
                F.col("max_loanamount_usl") / F.col("max_loanamount_sl")
            ).otherwise(F.lit(None).cast("double"))
        )

        # 9. flag_stpl_active_and_inactive
        #    1 = customer has both open AND closed STPLs → experienced STPL borrower.
        #    0 = either never had STPL, or only active, or only closed ones.
        feature_df = feature_df.withColumn(
            "flag_stpl_active_and_inactive",
            F.when(
                F.col("MaxLoanAmountActiveSTPL").isNotNull() &
                F.col("MaxLoanAmountInactiveSTPL").isNotNull(),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        # 10. flag_pure_unsecured_customer
        #     1 = has unsecured loans AND no secured / HL / AL / GL exposure at all.
        #     Warrants separate scorecard treatment in risk models.
        feature_df = feature_df.withColumn(
            "flag_pure_unsecured_customer",
            F.when(
                F.col("max_loanamount_usl").isNotNull() &
                F.col("max_loanamount_sl").isNull() &
                F.col("max_loanamount_hl").isNull() &
                F.col("max_loanamount_al").isNull() &
                F.col("max_loanamount_gl").isNull(),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        self._log_done(feature_df)
        return feature_df


class CreditCardLimitsFeatures(TradelineFeatureBase):
    """
    Category 04: Credit Card Limits / Utilization Indicators

    All features are SUM of orig_loan_am for CC accounts satisfying:
        orig_loan_am >= threshold  AND  account opened within vintage window

    Features
    --------
    CreditLimit_25000_Vintage_24   CC accounts with limit >= 25K, opened <= 24m ago
    CreditLimit_50000_Vintage_24   CC accounts with limit >= 50K, opened <= 24m ago
    CreditLimit_100000_Vintage_24  CC accounts with limit >= 1L,  opened <= 24m ago
    CreditLimit_100000_Vintage_12  CC accounts with limit >= 1L,  opened <= 12m ago
    CreditLimit_100000_Vintage_6   CC accounts with limit >= 1L,  opened <= 6m ago
    CreditLimit_10000_Vintage_6    CC accounts with limit >= 10K, opened <= 6m ago
    CreditLimit_50000_Vintage_12   CC accounts with limit >= 50K, opened <= 12m ago
    """

    CATEGORY = "cat04_credit_card_limits"  # grp02_loan_amounts

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
            .withColumn("_open_dt",  parse_date("open_dt"))
            .withColumn("_as_of_dt", parse_date(as_of_col))
        )

        # ── STEP 2: Months since account was opened ───────────────────────────
        # months_between(later, earlier) → positive when open_dt < as_of_dt
        df = df.withColumn(
            "_vintage_months",
            F.months_between(F.col("_as_of_dt"), F.col("_open_dt"))
        )

        # ── STEP 3: Clean orig_loan_am — treat -1 as null ─────────────────────
        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am").cast("double") > 0, F.col("orig_loan_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 4: CC flag ───────────────────────────────────────────────────
        df = df.withColumn(
            "_is_cc",
            F.col("acct_type_cd").cast("string").isin(CC_CODES)
        )

        # ── STEP 5: Helper — qualifying flag per (amount_threshold, vintage) ─
        # A row qualifies for a band if:
        #   _is_cc = True
        #   AND _loan_am >= amount_threshold
        #   AND 0 <= _vintage_months <= vintage_window
        def cc_limit_sum(amount_threshold: int, vintage_months: int) -> F.Column:
            return F.sum(
                F.when(
                    F.col("_is_cc") &
                    F.col("_loan_am").isNotNull() &
                    (F.col("_loan_am") >= amount_threshold) &
                    (F.col("_vintage_months") >= 0) &
                    (F.col("_vintage_months") <= vintage_months),
                    F.col("_loan_am")
                ).otherwise(F.lit(None).cast("double"))
            )

        # ── STEP 6: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            cc_limit_sum(25000,  24).alias("CreditLimit_25000_Vintage_24"),
            cc_limit_sum(50000,  24).alias("CreditLimit_50000_Vintage_24"),
            cc_limit_sum(100000, 24).alias("CreditLimit_100000_Vintage_24"),
            cc_limit_sum(100000, 12).alias("CreditLimit_100000_Vintage_12"),
            cc_limit_sum(100000,  6).alias("CreditLimit_100000_Vintage_6"),
            cc_limit_sum(10000,   6).alias("CreditLimit_10000_Vintage_6"),
            cc_limit_sum(50000,  12).alias("CreditLimit_50000_Vintage_12"),
        )

        self._log_done(feature_df)
        return feature_df


class HighestCreditSignalsFeatures(TradelineFeatureBase):
    """
    Category 07: Highest Credit / Loan History Signals

    All features are MAX of orig_loan_am within product type and time window.
    Only accounts where open_dt <= as_of_dt are eligible (no leakage).

    ── Requested features ────────────────────────────────────────────────────

    highest_credit_gold_loan_5_years
        Max orig_loan_am for Gold Loan accounts opened in last 60 months

    highest_credit_consumer_loan_3_years
        Max orig_loan_am for Consumer Loan (acct_type='189') opened in last 36 months

    highest_credit_personal_loan_3_years
        Max orig_loan_am for Personal Loan (acct_type='123') opened in last 36 months

    highest_credit_housing_loan
        Max orig_loan_am for Housing/Property Loan accounts (all-time)

    highest_credit_unsecured_3_years
        Max orig_loan_am for unsecured accounts opened in last 36 months

    ── Additional risk features ──────────────────────────────────────────────

    highest_credit_al_3_years
        Max orig_loan_am for Auto/Vehicle Loan accounts opened in last 36 months.
        AL sanction size is a strong income proxy.

    highest_credit_cc_3_years
        Max orig_loan_am for Credit Card accounts opened in last 36 months.
        Highest CC limit granted — lender confidence signal.

    highest_credit_pl_5_years
        Max orig_loan_am for PL accounts opened in last 60 months.
        Longer window captures peak PL capacity over recent credit cycle.

    highest_credit_any_5_years
        Max orig_loan_am across ALL products opened in last 60 months.
        Overall peak credit capacity signal.

    highest_credit_any_alltime
        Max orig_loan_am across ALL products, all-time.
        Already in cat03 as max_loanamount — NOT repeated here.

    highest_credit_gl_3_years
        Max orig_loan_am for Gold Loan accounts opened in last 36 months.
        Used as denominator in ratio below.

    ratio_highest_pl_3y_to_gl_3y
        highest_credit_personal_loan_3_years / highest_credit_gl_3_years.
        Same 3-year window for both — apples-to-apples comparison.
        High ratio → shifting from secured (GL) to unsecured (PL) credit.
        NULL when no GL in 3y window.

    flag_ever_had_hl
        1 if customer has any Housing Loan (all-time), else 0.
        Strong negative risk signal — property ownership proxy.

    flag_ever_had_gl_gt_1L
        1 if customer has any Gold Loan > 1,00,000 (all-time), else 0.
        High-value gold loans indicate asset ownership and lender trust.
    """

    CATEGORY = "cat07_highest_credit_signals"  # grp02_loan_amounts

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
            .withColumn("_open_dt",  parse_date("open_dt"))
            .withColumn("_as_of_dt", parse_date(as_of_col))
        )

        # ── STEP 2: Months since open — no leakage guard ──────────────────────
        # NULL for future accounts (open_dt > as_of_dt)
        df = df.withColumn(
            "_months_since_open",
            F.when(
                F.col("_open_dt") <= F.col("_as_of_dt"),
                F.months_between(F.col("_as_of_dt"), F.col("_open_dt"))
            ).otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 3: Clean orig_loan_am (-1 = NULL) ────────────────────────────
        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 4: Normalise acct_type_cd ───────────────────────────────────
        df = df.withColumn("_acct_type", F.trim(F.col("acct_type_cd").cast("string")))

        # ── STEP 5: Product type flags ────────────────────────────────────────
        df = (
            df
            .withColumn("_is_gl",       F.col("_acct_type").isin(GL_CODES))
            .withColumn("_is_consumer",  F.col("_acct_type") == CONSUMER_CODE)
            .withColumn("_is_pl",        F.col("_acct_type") == PL_CODE)
            .withColumn("_is_hl",        F.col("_acct_type").isin(HL_CODES))
            .withColumn("_is_al",        F.col("_acct_type").isin(AL_CODES))
            .withColumn("_is_cc",        F.col("_acct_type").isin(CC_CODES))
            .withColumn("_is_unsecured", ~F.col("_acct_type").isin(SECURED_CODES))
        )

        # ── STEP 6: Window flags ──────────────────────────────────────────────
        df = (
            df
            .withColumn("_in_36m",
                F.col("_months_since_open").isNotNull() &
                (F.col("_months_since_open") <= 36))
            .withColumn("_in_60m",
                F.col("_months_since_open").isNotNull() &
                (F.col("_months_since_open") <= 60))
            .withColumn("_alltime",
                F.col("_months_since_open").isNotNull())   # open_dt <= as_of_dt
        )

        # ── STEP 7: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # ── Requested features ────────────────────────────────────────────

            # Gold Loan — max sanction in last 5 years
            F.max(
                F.when(F.col("_is_gl") & F.col("_in_60m"), F.col("_loan_am"))
            ).alias("highest_credit_gold_loan_5_years"),

            # Consumer Loan — max sanction in last 3 years
            F.max(
                F.when(F.col("_is_consumer") & F.col("_in_36m"), F.col("_loan_am"))
            ).alias("highest_credit_consumer_loan_3_years"),

            # Personal Loan — max sanction in last 3 years
            F.max(
                F.when(F.col("_is_pl") & F.col("_in_36m"), F.col("_loan_am"))
            ).alias("highest_credit_personal_loan_3_years"),

            # Housing Loan — max sanction all-time
            F.max(
                F.when(F.col("_is_hl") & F.col("_alltime"), F.col("_loan_am"))
            ).alias("highest_credit_housing_loan"),

            # Unsecured — max sanction in last 3 years
            F.max(
                F.when(F.col("_is_unsecured") & F.col("_in_36m"), F.col("_loan_am"))
            ).alias("highest_credit_unsecured_3_years"),

            # ── Additional risk features ──────────────────────────────────────

            # Auto/Vehicle Loan — max sanction in last 3 years
            F.max(
                F.when(F.col("_is_al") & F.col("_in_36m"), F.col("_loan_am"))
            ).alias("highest_credit_al_3_years"),

            # Credit Card — max credit limit granted in last 3 years
            F.max(
                F.when(F.col("_is_cc") & F.col("_in_36m"), F.col("_loan_am"))
            ).alias("highest_credit_cc_3_years"),

            # Personal Loan — max sanction in last 5 years
            F.max(
                F.when(F.col("_is_pl") & F.col("_in_60m"), F.col("_loan_am"))
            ).alias("highest_credit_pl_5_years"),

            # Any product — max sanction in last 5 years
            F.max(
                F.when(F.col("_in_60m"), F.col("_loan_am"))
            ).alias("highest_credit_any_5_years"),

            # Gold Loan — max sanction in last 3 years (for ratio with PL)
            F.max(
                F.when(F.col("_is_gl") & F.col("_in_36m"), F.col("_loan_am"))
            ).alias("highest_credit_gl_3_years"),

            # Flag: customer has ever had a Housing Loan
            F.max(
                F.when(F.col("_is_hl") & F.col("_alltime"), F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_ever_had_hl"),

            # Flag: customer has ever had a Gold Loan > 1 Lakh
            F.max(
                F.when(
                    F.col("_is_gl") & F.col("_alltime") &
                    F.col("_loan_am").isNotNull() & (F.col("_loan_am") > 100000),
                    F.lit(1)
                ).otherwise(F.lit(0))
            ).alias("flag_ever_had_gl_gt_1L"),
        )

        # ── STEP 8: Derived ratio features (post-aggregation) ─────────────────

        # ratio_highest_pl_3y_to_gl_3y
        # High PL relative to GL in the same 3-year window
        # → shifting from secured to unsecured credit within same period
        # NULL when no GL in 3y window
        feature_df = feature_df.withColumn(
            "ratio_highest_pl_3y_to_gl_3y",
            F.when(
                F.col("highest_credit_gl_3_years").isNotNull() &
                (F.col("highest_credit_gl_3_years") > 0),
                F.col("highest_credit_personal_loan_3_years") /
                F.col("highest_credit_gl_3_years")
            ).otherwise(F.lit(None).cast("double"))
        )

        self._log_done(feature_df)
        return feature_df


class LoanVolumeOverTimeFeatures(TradelineFeatureBase):
    """
    Category 08: Loan Volume Over Time

    All windowed features use open_dt <= as_of_dt (no leakage).
    Windows: 12 months, 36 months.

    ── Requested features ────────────────────────────────────────────────────

    total_loan_amt_in_last12_mon
        Sum of orig_loan_am across ALL products opened in last 12 months.
        Captures short-term borrowing velocity.

    total_loan_amt_3_years
        Sum of orig_loan_am across ALL products opened in last 36 months.
        Medium-term total credit absorbed.

    mean_credit_unsecured_in_3_years
        Mean orig_loan_am of unsecured accounts opened in last 36 months.
        Typical ticket size for unsecured credit — income proxy.

    ── Additional risk features ──────────────────────────────────────────────

    total_loan_amt_pl_12m
        Sum of orig_loan_am for PL accounts opened in last 12 months.
        Short-window PL absorption — key credit hunger signal.

    total_loan_amt_pl_3_years
        Sum of orig_loan_am for PL accounts opened in last 36 months.
        Distinct from cat03.sumloanamount_personalloan (all-time, no window).

    total_loan_amt_unsecured_12m
        Sum of orig_loan_am for unsecured accounts opened in last 12 months.
        Recent unsecured exposure velocity.

    total_loan_amt_secured_3_years
        Sum of orig_loan_am for secured accounts opened in last 36 months.
        Secured credit absorption — lower risk than unsecured equivalent.

    mean_credit_pl_in_3_years
        Mean orig_loan_am of PL accounts opened in last 36 months.
        Typical PL ticket size — shifts over credit lifecycle.

    mean_credit_all_in_12m
        Mean orig_loan_am across ALL products opened in last 12 months.
        Recent average ticket size — rising mean with rising count = stress.

    ratio_unsecured_to_total_vol_3y
        total_loan_amt_unsecured_3y / total_loan_amt_3_years.
        Share of unsecured in total 3y borrowing.
        Rising ratio → unsecured drift — key risk escalation signal.

    ratio_12m_to_3y_volume
        total_loan_amt_in_last12_mon / total_loan_amt_3_years.
        Recent borrowing as fraction of 3-year total.
        High ratio → borrowing concentrated in recent 12 months = velocity spike.
    """

    CATEGORY = "cat08_loan_volume_over_time"  # grp02_loan_amounts

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
            .withColumn("_open_dt",  parse_date("open_dt"))
            .withColumn("_as_of_dt", parse_date(as_of_col))
        )

        # ── STEP 2: Months since open — NULL for future accounts (no leakage) ─
        df = df.withColumn(
            "_months_since_open",
            F.when(
                F.col("_open_dt") <= F.col("_as_of_dt"),
                F.months_between(F.col("_as_of_dt"), F.col("_open_dt"))
            ).otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 3: Clean orig_loan_am (-1 = NULL) ────────────────────────────
        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 4: Normalise acct_type_cd ───────────────────────────────────
        df = df.withColumn("_acct_type", F.trim(F.col("acct_type_cd").cast("string")))

        # ── STEP 5: Product flags ─────────────────────────────────────────────
        df = (
            df
            .withColumn("_is_pl",       F.col("_acct_type") == PL_CODE)
            .withColumn("_is_unsecured", ~F.col("_acct_type").isin(SECURED_CODES))
            .withColumn("_is_secured",   F.col("_acct_type").isin(SECURED_CODES))
        )

        # ── STEP 6: Window flags ──────────────────────────────────────────────
        df = (
            df
            .withColumn("_in_12m",
                F.col("_months_since_open").isNotNull() &
                (F.col("_months_since_open") <= 12))
            .withColumn("_in_36m",
                F.col("_months_since_open").isNotNull() &
                (F.col("_months_since_open") <= 36))
        )

        # ── STEP 7: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # ── Requested ─────────────────────────────────────────────────────

            # Total loan amount — all products, last 12 months
            F.sum(
                F.when(F.col("_in_12m"), F.col("_loan_am"))
            ).alias("total_loan_amt_in_last12_mon"),

            # Total loan amount — all products, last 36 months
            F.sum(
                F.when(F.col("_in_36m"), F.col("_loan_am"))
            ).alias("total_loan_amt_3_years"),

            # Mean unsecured loan amount — last 36 months
            F.mean(
                F.when(F.col("_is_unsecured") & F.col("_in_36m"), F.col("_loan_am"))
            ).alias("mean_credit_unsecured_in_3_years"),

            # ── Additional ────────────────────────────────────────────────────

            # PL sum — last 12 months (windowed; cat03 has all-time)
            F.sum(
                F.when(F.col("_is_pl") & F.col("_in_12m"), F.col("_loan_am"))
            ).alias("total_loan_amt_pl_12m"),

            # PL sum — last 36 months (windowed; cat03 has all-time)
            F.sum(
                F.when(F.col("_is_pl") & F.col("_in_36m"), F.col("_loan_am"))
            ).alias("total_loan_amt_pl_3_years"),

            # Unsecured sum — last 12 months
            F.sum(
                F.when(F.col("_is_unsecured") & F.col("_in_12m"), F.col("_loan_am"))
            ).alias("total_loan_amt_unsecured_12m"),

            # Unsecured sum — last 36 months (denominator for ratio below)
            F.sum(
                F.when(F.col("_is_unsecured") & F.col("_in_36m"), F.col("_loan_am"))
            ).alias("total_loan_amt_unsecured_3y"),

            # Secured sum — last 36 months
            F.sum(
                F.when(F.col("_is_secured") & F.col("_in_36m"), F.col("_loan_am"))
            ).alias("total_loan_amt_secured_3_years"),

            # Mean PL loan amount — last 36 months
            F.mean(
                F.when(F.col("_is_pl") & F.col("_in_36m"), F.col("_loan_am"))
            ).alias("mean_credit_pl_in_3_years"),

            # Mean loan amount — all products, last 12 months
            F.mean(
                F.when(F.col("_in_12m"), F.col("_loan_am"))
            ).alias("mean_credit_all_in_12m"),
        )

        # ── STEP 8: Derived ratio features (post-aggregation) ─────────────────

        # ratio_unsecured_to_total_vol_3y
        # Share of unsecured in total 3y loan volume
        # Rising ratio = unsecured drift = risk escalation signal
        feature_df = feature_df.withColumn(
            "ratio_unsecured_to_total_vol_3y",
            F.when(
                F.col("total_loan_amt_3_years").isNotNull() &
                (F.col("total_loan_amt_3_years") > 0),
                F.col("total_loan_amt_unsecured_3y") / F.col("total_loan_amt_3_years")
            ).otherwise(F.lit(None).cast("double"))
        )

        # ratio_12m_to_3y_volume
        # Recent 12m borrowing as fraction of 3y total
        # High ratio = borrowing concentrated in last year = velocity spike
        feature_df = feature_df.withColumn(
            "ratio_12m_to_3y_volume",
            F.when(
                F.col("total_loan_amt_3_years").isNotNull() &
                (F.col("total_loan_amt_3_years") > 0),
                F.col("total_loan_amt_in_last12_mon") / F.col("total_loan_amt_3_years")
            ).otherwise(F.lit(None).cast("double"))
        )

        self._log_done(feature_df)
        return feature_df


