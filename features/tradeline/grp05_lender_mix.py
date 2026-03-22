# features/tradeline/cat09_lender_type_mix.py
# =============================================================================
# Category 09 — Lender Type / Source Mix
# =============================================================================
# Source table  : experian_tradeline_segment
# Granularity   : One row per (customer_scrub_key, scrub_output_date)  [scrub mode]
#                 One row per (party_code, open_dt)                    [retro mode]
#
# Reference     : Experian Bureau Products v3.2 — Appendix B (Type of institute/Entity)
#
# Purpose:
#   Captures the distribution of credit across lender types.
#   NBFC exposure, private bank penetration, and lender mix are strong
#   risk signals — customers who exhaust bank credit often shift to NBFCs,
#   and heavy NBFC reliance is a known predictor of stress.
#
# Source columns used:
#   m_sub_id       → Type of institute (Appendix B) — primary lender type field
#   orig_loan_am   → Disbursed/sanctioned loan amount (-1 = NULL)
#   open_dt        → Trade line open date (window anchor)
#   closed_dt      → Trade line closed date (for active flag)
#   acct_type_cd   → Account type (for product filtering)
#
# Appendix B — M_SUB_ID codes (Experian v3.2):
#   COB  Co-operative Bank
#   FOR  Foreign Bank
#   HFC  Housing Finance Company
#   NBF  Non-Banking Financial Institution (NBFC)
#   PUB  Public Sector Bank
#   PVT  Private Sector Bank
#   RRB  Regional Rural Bank
#   TEL  Telecom
#   SRC  Securities Firm
#   MFI  Mutual Fund Institutions (Microfinance)
#   INS  Insurance Sector
#   CCS  Cooperative Credit Society
#   BRO  Brokerage Firm
#   CRA  Credit Rating Agency
#   SFB  Small Finance Bank
#   SFI  State Financial Institution
#
# Lender groupings used:
#   NBFC    : 'NBF'
#   PRB     : 'PVT'  (Private Sector Bank)
#   PUB     : 'PUB'  (Public Sector Bank)
#   BANK    : 'PVT','PUB','FOR','RRB','SFB','COB'  (all bank types)
#   MFI     : 'MFI'
#   HFC     : 'HFC'
#
# Key rules:
#   • Only accounts where open_dt <= as_of_dt are eligible (no leakage)
#   • Window: months_between(as_of_dt, open_dt) <= N
#   • Active: open_dt <= as_of_dt AND (closed_dt IS NULL OR closed_dt > as_of_dt)
#   • m_sub_id must be trimmed (may have whitespace)
#   • orig_loan_am = -1 → NULL
#
# Product filters:
#   PL  : '123'
#   CC  : '5','213','214','220','224','225'  — all CCs incl. Secured CC (220)
#   CL  : '189'  (Consumer Loan)
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger
from core.date_utils import parse_date

logger = get_logger(__name__)


# =============================================================================
# LENDER TYPE SETS  (Appendix B)
# =============================================================================

NBFC_CODE = "NBF"
PRB_CODE  = "PVT"   # Private sector bank
PUB_CODE  = "PUB"   # Public sector bank
MFI_CODE  = "MFI"
HFC_CODE  = "HFC"
SFB_CODE  = "SFB"   # Small Finance Bank

# All regulated bank types (excludes NBFC, MFI, HFC, TEL etc.)
BANK_CODES = {"PVT", "PUB", "FOR", "RRB", "SFB", "COB"}

# Product codes
PL_CODE = "123"
CC_CODES = {"5", "213", "214", "220", "224", "225"}    # All CCs incl. 220 (Secured CC — treated as CC/unsecured)
CL_CODE  = "189"

# Secured account type codes (same set as cat01/cat03/cat06)
SECURED_ACCT_CODES = {
    # 220 (Secured Credit Card) removed — CCs (incl. 220) are treated as CC/unsecured category
    "47", "58", "195", "168", "173", "221",
    "175", "222", "172", "219", "184", "185", "191",
    "181", "197", "198", "199", "200",
    "223", "240", "241", "243", "246", "248",
}


class LenderTypeMixFeatures(TradelineFeatureBase):
    """
    Category 09: Lender Type / Source Mix

    Distribution of credit between NBFCs, private banks, public banks and others.
    All windowed features use open_dt <= as_of_dt (no leakage).
    Windows: 12 months, 24 months, 36 months, 60 months.

    ── Requested features ────────────────────────────────────────────────────

    perc_PL_taken_from_nbfc_5_years
        % of PL accounts (by count) from NBFC in last 60 months
        = NBFC PL count / total PL count in 60m

    no_of_CL_taken_from_nbfc_5_years
        Count of Consumer Loan accounts from NBFC in last 60 months

    perc_CC_taken_from_prb_5_years
        % of CC accounts (by count) from Private Banks in last 60 months

    perc_PL_taken_from_nbfc_3_years
        % of PL accounts (by count) from NBFC in last 36 months

    no_of_CL_taken_from_nbfc_3_years
        Count of Consumer Loan accounts from NBFC in last 36 months

    perc_CC_taken_from_prb_3_years
        % of CC accounts (by count) from Private Banks in last 36 months

    no_of_nbfc_lt_10k_12m
        Count of NBFC accounts with orig_loan_am < 10,000 in last 12 months
        Small-ticket NBFC loans = high-risk lending indicator

    count_accounts_Pvt_bank
        Count of all accounts from Private Banks (all-time, open_dt <= as_of_dt)

    count_accounts_Pvt_bank_last24m
        Count of all accounts from Private Banks opened in last 24 months

    count_pl_Pvt_bank
        Count of PL accounts from Private Banks (all-time)

    count_pl_Pvt_bank_last24m
        Count of PL accounts from Private Banks opened in last 24 months

    ── Additional risk features ──────────────────────────────────────────────

    count_accounts_nbfc_alltime
        Total NBFC account count (all-time).
        Baseline NBFC exposure.

    count_accounts_nbfc_last24m
        NBFC account count opened in last 24 months.
        Recent NBFC reliance.

    count_pl_nbfc_last24m
        PL accounts from NBFC in last 24 months.
        Most predictive NBFC signal for PL risk models.

    count_accounts_pub_bank
        Count of all accounts from Public Sector Banks (all-time).
        Public bank presence = formal credit access indicator.

    count_accounts_mfi
        Count of MFI accounts (all-time).
        MFI exposure = microfinance / rural segment signal.

    count_accounts_sfb
        Count of Small Finance Bank accounts (all-time).
        SFB lending often to thin-file / NTC customers.

    perc_accounts_from_nbfc_3y
        NBFC account count in 3y / total account count in 3y.
        Overall NBFC share — not just PL or CL.

    perc_accounts_from_bank_3y
        Bank account count in 3y / total account count in 3y.
        Bank share — complement to NBFC share.

    ratio_nbfc_to_bank_count_3y
        NBFC count / bank count in last 36 months.
        > 1 = more NBFC than bank accounts → high risk flag.
        NULL when no bank accounts in window.

    flag_any_nbfc_last12m
        1 if customer has any NBFC account opened in last 12 months.
        Binary trigger for NBFC recency risk.

    flag_nbfc_only_3y
        1 if ALL accounts in last 3 years are from NBFC (no bank).
        Extreme signal — customer unable to access formal bank credit.
    """

    CATEGORY = "grp05_lender_mix"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]


        df = (
            df
            .withColumn("_open_dt",   parse_date("open_dt"))
            .withColumn("_closed_dt", parse_date("closed_dt"))
            .withColumn("_as_of_dt",  parse_date(as_of_col))
        )

        # ── STEP 2: Months since open — NULL for future accounts (no leakage) ─
        df = df.withColumn(
            "_months_since_open",
            F.when(
                F.col("_open_dt") <= F.col("_as_of_dt"),
                F.months_between(F.col("_as_of_dt"), F.col("_open_dt"))
            ).otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 3: Active flag (point-in-time, no leakage) ───────────────────
        df = df.withColumn(
            "_is_active",
            F.when(
                (F.col("_open_dt") <= F.col("_as_of_dt")) &
                (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt"))),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        # ── STEP 4: Clean orig_loan_am (-1 = NULL) ────────────────────────────
        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 5: Normalise m_sub_id and acct_type_cd ───────────────────────
        df = (
            df
            .withColumn("_lender", F.upper(F.trim(F.col("m_sub_id").cast("string"))))
            .withColumn("_acct_type", F.trim(F.col("acct_type_cd").cast("string")))
        )

        # ── STEP 6: Lender type flags ─────────────────────────────────────────
        df = (
            df
            .withColumn("_is_nbfc",     F.col("_lender") == NBFC_CODE)
            .withColumn("_is_prb",      F.col("_lender") == PRB_CODE)
            .withColumn("_is_pub",      F.col("_lender") == PUB_CODE)
            .withColumn("_is_mfi",      F.col("_lender") == MFI_CODE)
            .withColumn("_is_sfb",      F.col("_lender") == SFB_CODE)
            .withColumn("_is_bank",     F.col("_lender").isin(BANK_CODES))
        )

        # ── STEP 7: Product flags ─────────────────────────────────────────────
        df = (
            df
            .withColumn("_is_pl", F.col("_acct_type") == PL_CODE)
            .withColumn("_is_cc", F.col("_acct_type").isin(CC_CODES))
            .withColumn("_is_cl", F.col("_acct_type") == CL_CODE)
            .withColumn("_is_secured_prod",   F.col("_acct_type").isin(SECURED_ACCT_CODES))
            .withColumn("_is_unsecured_prod", ~F.col("_acct_type").isin(SECURED_ACCT_CODES))
        )

        # ── STEP 8: Window flags ──────────────────────────────────────────────
        df = (
            df
            .withColumn("_in_12m",
                F.col("_months_since_open").isNotNull() &
                (F.col("_months_since_open") <= 12))
            .withColumn("_in_24m",
                F.col("_months_since_open").isNotNull() &
                (F.col("_months_since_open") <= 24))
            .withColumn("_in_36m",
                F.col("_months_since_open").isNotNull() &
                (F.col("_months_since_open") <= 36))
            .withColumn("_in_60m",
                F.col("_months_since_open").isNotNull() &
                (F.col("_months_since_open") <= 60))
            .withColumn("_alltime",
                F.col("_months_since_open").isNotNull())
        )

        # ── STEP 9: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # ── Requested features ────────────────────────────────────────────

            # NBFC PL count in 60m — numerator for percentage
            F.sum(F.when(F.col("_is_pl") & F.col("_is_nbfc") & F.col("_in_60m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_nbfc_pl_60m"),

            # Total PL count in 60m — denominator
            F.sum(F.when(F.col("_is_pl") & F.col("_in_60m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_total_pl_60m"),

            # CL from NBFC in 60m
            F.sum(F.when(F.col("_is_cl") & F.col("_is_nbfc") & F.col("_in_60m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("no_of_CL_taken_from_nbfc_5_years"),

            # Private bank CC count in 60m — numerator
            F.sum(F.when(F.col("_is_cc") & F.col("_is_prb") & F.col("_in_60m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_prb_cc_60m"),

            # Total CC count in 60m — denominator
            F.sum(F.when(F.col("_is_cc") & F.col("_in_60m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_total_cc_60m"),

            # NBFC PL count in 36m — numerator
            F.sum(F.when(F.col("_is_pl") & F.col("_is_nbfc") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_nbfc_pl_36m"),

            # Total PL count in 36m — denominator
            F.sum(F.when(F.col("_is_pl") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_total_pl_36m"),

            # CL from NBFC in 36m
            F.sum(F.when(F.col("_is_cl") & F.col("_is_nbfc") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("no_of_CL_taken_from_nbfc_3_years"),

            # Private bank CC count in 36m — numerator
            F.sum(F.when(F.col("_is_cc") & F.col("_is_prb") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_prb_cc_36m"),

            # Total CC count in 36m — denominator
            F.sum(F.when(F.col("_is_cc") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_total_cc_36m"),

            # NBFC accounts < 10K in last 12m
            F.sum(F.when(
                F.col("_is_nbfc") & F.col("_in_12m") &
                F.col("_loan_am").isNotNull() & (F.col("_loan_am") < 10000),
                F.lit(1)).otherwise(F.lit(0))
            ).alias("no_of_nbfc_lt_10k_12m"),

            # Private bank — all accounts, all-time
            F.sum(F.when(F.col("_is_prb") & F.col("_alltime"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_accounts_Pvt_bank"),

            # Private bank — all accounts, last 24m
            F.sum(F.when(F.col("_is_prb") & F.col("_in_24m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_accounts_Pvt_bank_last24m"),

            # Private bank — PL accounts, all-time
            F.sum(F.when(F.col("_is_pl") & F.col("_is_prb") & F.col("_alltime"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_pl_Pvt_bank"),

            # Private bank — PL accounts, last 24m
            F.sum(F.when(F.col("_is_pl") & F.col("_is_prb") & F.col("_in_24m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_pl_Pvt_bank_last24m"),

            # ── Additional risk features ──────────────────────────────────────

            # NBFC all accounts — all-time and 24m
            F.sum(F.when(F.col("_is_nbfc") & F.col("_alltime"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_accounts_nbfc_alltime"),

            F.sum(F.when(F.col("_is_nbfc") & F.col("_in_24m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_accounts_nbfc_last24m"),

            # NBFC PL in 24m
            F.sum(F.when(F.col("_is_pl") & F.col("_is_nbfc") & F.col("_in_24m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_pl_nbfc_last24m"),

            # Public sector bank — all accounts, all-time
            F.sum(F.when(F.col("_is_pub") & F.col("_alltime"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_accounts_pub_bank"),

            # MFI — all accounts, all-time
            F.sum(F.when(F.col("_is_mfi") & F.col("_alltime"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_accounts_mfi"),

            # Small Finance Bank — all accounts, all-time
            F.sum(F.when(F.col("_is_sfb") & F.col("_alltime"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_accounts_sfb"),

            # Total accounts in 36m — denominator for share features
            F.sum(F.when(F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_total_accts_36m"),

            # NBFC accounts in 36m — numerator for share
            F.sum(F.when(F.col("_is_nbfc") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_nbfc_accts_36m"),

            # Bank accounts in 36m — numerator for share
            F.sum(F.when(F.col("_is_bank") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_bank_accts_36m"),

            # Flag: any NBFC in last 12m
            F.max(F.when(F.col("_is_nbfc") & F.col("_in_12m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_any_nbfc_last12m"),

            # ── Secured / Unsecured lender institute features ─────────────────
            # Secured loans from NBFCs (NBFC + secured product) — 3y
            F.sum(F.when(F.col("_is_nbfc") & F.col("_is_secured_prod") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_secured_loans_from_nbfc_3y"),

            # Unsecured loans from NBFCs (NBFC + unsecured product) — 3y
            F.sum(F.when(F.col("_is_nbfc") & F.col("_is_unsecured_prod") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_unsecured_loans_from_nbfc_3y"),

            # Secured loans from banks — 3y
            F.sum(F.when(F.col("_is_bank") & F.col("_is_secured_prod") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_secured_loans_from_bank_3y"),

            # Unsecured loans from banks — 3y
            F.sum(F.when(F.col("_is_bank") & F.col("_is_unsecured_prod") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_unsecured_loans_from_bank_3y"),

            # Total secured accounts — 3y (denominator)
            F.sum(F.when(F.col("_is_secured_prod") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_total_secured_36m"),

            # Total unsecured accounts — 3y (denominator)
            F.sum(F.when(F.col("_is_unsecured_prod") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_total_unsecured_36m"),

            # NBFC unsecured — 3y (numerator for perc)
            F.sum(F.when(F.col("_is_nbfc") & F.col("_is_unsecured_prod") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_nbfc_unsecured_36m"),

            # Bank secured — 3y (numerator for perc)
            F.sum(F.when(F.col("_is_bank") & F.col("_is_secured_prod") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_bank_secured_36m"),

            # ── pvt / pub bank % features — numerators ────────────────────────
            # PVT bank accounts — 3y (numerator)
            F.sum(F.when(F.col("_is_prb") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_prb_accts_36m"),

            # PUB bank accounts — 3y (numerator)
            F.sum(F.when(F.col("_is_pub") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_pub_accts_36m"),

            # PVT bank PL — 3y (numerator for % PL from pvt)
            F.sum(F.when(F.col("_is_prb") & F.col("_is_pl") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_prb_pl_36m"),

            # PUB bank PL — 3y (numerator for % PL from pub)
            F.sum(F.when(F.col("_is_pub") & F.col("_is_pl") & F.col("_in_36m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_pub_pl_36m"),
        )

        # ── STEP 10: Derived percentage and ratio features ────────────────────

        # perc_PL_taken_from_nbfc_5_years
        feature_df = feature_df.withColumn(
            "perc_PL_taken_from_nbfc_5_years",
            F.when(
                F.col("_total_pl_60m") > 0,
                F.col("_nbfc_pl_60m") / F.col("_total_pl_60m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # perc_CC_taken_from_prb_5_years
        feature_df = feature_df.withColumn(
            "perc_CC_taken_from_prb_5_years",
            F.when(
                F.col("_total_cc_60m") > 0,
                F.col("_prb_cc_60m") / F.col("_total_cc_60m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # perc_PL_taken_from_nbfc_3_years
        feature_df = feature_df.withColumn(
            "perc_PL_taken_from_nbfc_3_years",
            F.when(
                F.col("_total_pl_36m") > 0,
                F.col("_nbfc_pl_36m") / F.col("_total_pl_36m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # perc_CC_taken_from_prb_3_years
        feature_df = feature_df.withColumn(
            "perc_CC_taken_from_prb_3_years",
            F.when(
                F.col("_total_cc_36m") > 0,
                F.col("_prb_cc_36m") / F.col("_total_cc_36m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # perc_accounts_from_nbfc_3y
        feature_df = feature_df.withColumn(
            "perc_accounts_from_nbfc_3y",
            F.when(
                F.col("_total_accts_36m") > 0,
                F.col("_nbfc_accts_36m") / F.col("_total_accts_36m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # perc_accounts_from_bank_3y
        feature_df = feature_df.withColumn(
            "perc_accounts_from_bank_3y",
            F.when(
                F.col("_total_accts_36m") > 0,
                F.col("_bank_accts_36m") / F.col("_total_accts_36m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # ratio_nbfc_to_bank_count_3y
        feature_df = feature_df.withColumn(
            "ratio_nbfc_to_bank_count_3y",
            F.when(
                F.col("_bank_accts_36m") > 0,
                F.col("_nbfc_accts_36m") / F.col("_bank_accts_36m")
            ).otherwise(F.lit(None).cast("double"))
        )

        # flag_nbfc_only_3y — all 3y accounts are NBFC, no bank at all
        feature_df = feature_df.withColumn(
            "flag_nbfc_only_3y",
            F.when(
                (F.col("_total_accts_36m") > 0) &
                (F.col("_nbfc_accts_36m") == F.col("_total_accts_36m")),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        # ── Secured / Unsecured institute derived features ────────────────────

        # % of unsecured accounts sourced from NBFCs in 3y
        # High = NBFC is the primary unsecured lender → risk signal
        feature_df = feature_df.withColumn(
            "perc_unsecured_from_nbfc_3y",
            F.when(
                F.col("_total_unsecured_36m") > 0,
                F.col("_nbfc_unsecured_36m") / F.col("_total_unsecured_36m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # % of secured accounts sourced from banks in 3y
        # High = customer gets secured credit from formal banks → lower risk
        feature_df = feature_df.withColumn(
            "perc_secured_from_bank_3y",
            F.when(
                F.col("_total_secured_36m") > 0,
                F.col("_bank_secured_36m") / F.col("_total_secured_36m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # ── pvt / pub bank percentage features ───────────────────────────────

        # % of all accounts (3y) from private sector banks
        # Higher = more formal private bank penetration → lower risk signal
        feature_df = feature_df.withColumn(
            "perc_accounts_from_pvt_bank_3y",
            F.when(
                F.col("_total_accts_36m") > 0,
                F.col("_prb_accts_36m") / F.col("_total_accts_36m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # % of all accounts (3y) from public sector banks
        # Higher = formal public bank penetration → lower risk signal
        feature_df = feature_df.withColumn(
            "perc_accounts_from_pub_bank_3y",
            F.when(
                F.col("_total_accts_36m") > 0,
                F.col("_pub_accts_36m") / F.col("_total_accts_36m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # % of PL accounts (3y) from private sector banks
        # Private banks lend PL to lower-risk customers → strong quality signal
        feature_df = feature_df.withColumn(
            "perc_pl_from_pvt_bank_3y",
            F.when(
                F.col("_total_pl_36m") > 0,
                F.col("_prb_pl_36m") / F.col("_total_pl_36m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # % of PL accounts (3y) from public sector banks
        feature_df = feature_df.withColumn(
            "perc_pl_from_pub_bank_3y",
            F.when(
                F.col("_total_pl_36m") > 0,
                F.col("_pub_pl_36m") / F.col("_total_pl_36m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # Combined pvt + pub bank % — overall formal bank share of PL
        # = perc_pl_from_pvt_bank_3y + perc_pl_from_pub_bank_3y
        # Complement is NBFC/MFI/SFB share
        feature_df = feature_df.withColumn(
            "perc_pl_from_formal_bank_3y",
            F.when(
                F.col("_total_pl_36m") > 0,
                (F.col("_prb_pl_36m") + F.col("_pub_pl_36m")) / F.col("_total_pl_36m") * 100
            ).otherwise(F.lit(None).cast("double"))
        )

        # Drop all intermediate underscore columns
        drop_cols = [
            "_nbfc_pl_60m", "_total_pl_60m", "_prb_cc_60m", "_total_cc_60m",
            "_nbfc_pl_36m", "_total_pl_36m", "_prb_cc_36m", "_total_cc_36m",
            "_total_accts_36m", "_nbfc_accts_36m", "_bank_accts_36m",
            "_total_secured_36m", "_total_unsecured_36m",
            "_nbfc_unsecured_36m", "_bank_secured_36m",
            "_prb_accts_36m", "_pub_accts_36m",
            "_prb_pl_36m", "_pub_pl_36m",
        ]
        feature_df = feature_df.drop(*drop_cols)

        self._log_done(feature_df)
        return feature_df