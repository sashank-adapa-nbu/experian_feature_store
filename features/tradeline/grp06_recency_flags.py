# features/tradeline/cat10_recency_credit_activity.py
# =============================================================================
# Category 10 — Recency of Credit Activity
# =============================================================================
# Source table  : experian_tradeline_segment
# Granularity   : One row per (customer_scrub_key, scrub_output_date)  [scrub mode]
#                 One row per (party_code, open_dt)                    [retro mode]
#
# Reference     : Experian Bureau Products v3.2 — Appendix A (new acct_type_cd)
#
# Purpose:
#   Time in months since the customer last OPENED a credit account of each
#   product type, measured back from as_of_dt.
#   Lower value = more recent credit-seeking activity.
#   NULL = customer has never had that product type as of as_of_dt.
#
# Definition:
#   months_from_last_<product> =
#       MIN( months_between(as_of_dt, open_dt) )
#       WHERE acct_type_cd IN <product_codes>
#         AND open_dt <= as_of_dt        ← no leakage
#
#   MIN of months_since_open = MOST RECENTLY opened account of that type
#   (smallest gap from as_of_dt = opened latest).
#
# Source columns used:
#   open_dt        → Trade line open date
#   closed_dt      → For active flag on active-specific features
#   acct_type_cd   → Account type
#   m_sub_id       → Lender type (for NBFC recency)
#
# Non-redundancy vs cat06:
#   cat06.minbureauage       = min age, ALL products, all accounts
#   cat06.minbureauage_active = min age, ALL products, active only
#   cat10 = product-specific recency (CL/CC/PL/GL/HL/AL/STPL/NBFC/any-active)
#
# Product filters (Appendix A):
#   CL  : '189'
#   CC  : '5','213','214','220','224','225'  — all CCs incl. Secured CC (220)
#   PL  : '123'
#   GL  : '191','243'
#   HL  : '58','195','168','240'
#   AL  : '47','173','172','221','222','223','246'
#   STPL: '242'
#   SPL : '184','185','175','241','248','181','197','198','199','200'
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger
from core.date_utils import parse_date

logger = get_logger(__name__)


# =============================================================================
# CODE SETS
# =============================================================================

GL_CODES  = {"191", "243"}
AL_CODES  = {"47", "173", "172", "221", "222", "223", "246"}
HL_CODES  = {"58", "195", "168", "240"}
CC_CODES  = {"5", "213", "214", "220", "224", "225"}   
MCF_CODES = {"167", "169", "170"}          # Microfinance — Business, Personal, Other
P2P_CODES = {"245", "246", "247"}          # P2P Personal, Auto, Education

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

PL_CODE   = "123"
CL_CODE   = "189"
STPL_CODE = "242"
NBFC_CODE = "NBF"




class RecencyCreditActivityFeatures(TradelineFeatureBase):
    """
    Category 10: Recency of Credit Activity

    All features = months between as_of_dt and the most recently OPENED
    account of that product type (MIN months_since_open per product).
    Only accounts where open_dt <= as_of_dt are eligible (no leakage).
    NULL when no account of that type exists as of as_of_dt.

    ── Requested features ────────────────────────────────────────────────────

    months_from_last_cl
        Months since most recent Consumer Loan (acct='189') opened

    months_from_last_cc
        Months since most recent Credit Card opened

    months_from_last_pl
        Months since most recent Personal Loan (acct='123') opened

    ── Additional risk features ──────────────────────────────────────────────

    months_from_last_gl
        Months since most recent Gold Loan opened.
        Very recent GL = possible financial stress / asset liquidation signal.

    months_from_last_hl
        Months since most recent Home Loan opened.

    months_from_last_al
        Months since most recent Auto/Vehicle Loan opened.

    months_from_last_stpl
        Months since most recent STPL opened.
        Very recent STPL (< 3m) = active short-term borrower.

    months_from_last_spl
        Months since most recent Secured Personal / Business Loan opened.

    months_from_last_any
        Months since ANY account opened (across all products).
        Distinct from cat06.minbureauage which is all accounts incl active.

    months_from_last_any_active
        Months since most recent ACTIVE account opened.
        Active = open_dt <= as_of_dt AND not yet closed.

    months_from_last_nbfc
        Months since most recent NBFC account opened.
        Recent NBFC = possible bank credit exhaustion signal.

    months_from_last_pl_or_stpl
        Months since most recent PL or STPL opened (combined unsecured retail).
        Key recency signal for PL/STPL risk models.

    flag_pl_opened_last_3m
        1 if any PL was opened in last 3 months from as_of_dt.
        Very recent PL = fresh obligation, high early delinquency risk.

    flag_cc_opened_last_6m
        1 if any CC was opened in last 6 months.
        Recent CC = revolving credit appetite signal.

    flag_any_opened_last_1m
        1 if any account opened in last 1 month.
        Extremely fresh credit = highest early stress risk.

    count_products_ever
        Count of distinct product types customer has ever used (as of as_of_dt).
        Breadth of credit product experience.
    """

    CATEGORY = "grp06a_recency_credit_activity"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        # ── STEP 1: Parse date columns ────────────────────────────────────────

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

        # ── STEP 4: Normalise acct_type_cd and m_sub_id ───────────────────────
        df = (
            df
            .withColumn("_acct_type", F.trim(F.col("acct_type_cd").cast("string")))
            .withColumn("_lender",    F.upper(F.trim(F.col("m_sub_id").cast("string"))))
        )

        # ── STEP 5: Product flags ─────────────────────────────────────────────
        df = (
            df
            .withColumn("_is_cl",   F.col("_acct_type") == CL_CODE)
            .withColumn("_is_cc",   F.col("_acct_type").isin(CC_CODES))
            .withColumn("_is_pl",   F.col("_acct_type") == PL_CODE)
            .withColumn("_is_gl",   F.col("_acct_type").isin(GL_CODES))
            .withColumn("_is_hl",   F.col("_acct_type").isin(HL_CODES))
            .withColumn("_is_al",   F.col("_acct_type").isin(AL_CODES))
            .withColumn("_is_stpl", F.col("_acct_type") == STPL_CODE)
            .withColumn("_is_nbfc", F.col("_lender") == NBFC_CODE)
            .withColumn("_is_pl_or_stpl",
                (F.col("_acct_type") == PL_CODE) | (F.col("_acct_type") == STPL_CODE))
        )

        # ── STEP 6: Aggregate ─────────────────────────────────────────────────
        # MIN of months_since_open per product = most recently opened account
        # NULL propagation: if no eligible row, MIN returns NULL naturally

        feature_df = df.groupBy(group_cols).agg(

            # ── Requested ─────────────────────────────────────────────────────

            F.min(
                F.when(F.col("_is_cl"),   F.col("_months_since_open"))
            ).alias("months_from_last_cl"),

            F.min(
                F.when(F.col("_is_cc"),   F.col("_months_since_open"))
            ).alias("months_from_last_cc"),

            F.min(
                F.when(F.col("_is_pl"),   F.col("_months_since_open"))
            ).alias("months_from_last_pl"),

            # ── Additional ────────────────────────────────────────────────────

            F.min(
                F.when(F.col("_is_gl"),   F.col("_months_since_open"))
            ).alias("months_from_last_gl"),

            F.min(
                F.when(F.col("_is_hl"),   F.col("_months_since_open"))
            ).alias("months_from_last_hl"),

            F.min(
                F.when(F.col("_is_al"),   F.col("_months_since_open"))
            ).alias("months_from_last_al"),

            F.min(
                F.when(F.col("_is_stpl"), F.col("_months_since_open"))
            ).alias("months_from_last_stpl"),

            # Any product — most recently opened account of any type
            F.min(
                F.col("_months_since_open")
            ).alias("months_from_last_any"),

            # Active accounts only — most recently opened still-open account
            F.min(
                F.when(F.col("_is_active") == 1, F.col("_months_since_open"))
            ).alias("months_from_last_any_active"),

            # NBFC — most recently opened NBFC account
            F.min(
                F.when(F.col("_is_nbfc"), F.col("_months_since_open"))
            ).alias("months_from_last_nbfc"),

            # PL or STPL combined — unsecured retail recency
            F.min(
                F.when(F.col("_is_pl_or_stpl"), F.col("_months_since_open"))
            ).alias("months_from_last_pl_or_stpl"),

            # ── Binary recency flags ──────────────────────────────────────────

            # PL opened in last 3 months
            F.max(
                F.when(
                    F.col("_is_pl") &
                    F.col("_months_since_open").isNotNull() &
                    (F.col("_months_since_open") <= 3),
                    F.lit(1)
                ).otherwise(F.lit(0))
            ).alias("flag_pl_opened_last_3m"),

            # CC opened in last 6 months
            F.max(
                F.when(
                    F.col("_is_cc") &
                    F.col("_months_since_open").isNotNull() &
                    (F.col("_months_since_open") <= 6),
                    F.lit(1)
                ).otherwise(F.lit(0))
            ).alias("flag_cc_opened_last_6m"),

            # Any account opened in last 1 month
            F.max(
                F.when(
                    F.col("_months_since_open").isNotNull() &
                    (F.col("_months_since_open") <= 1),
                    F.lit(1)
                ).otherwise(F.lit(0))
            ).alias("flag_any_opened_last_1m"),

            # Count of distinct product types ever used (as of as_of_dt)
            F.countDistinct(
                F.when(
                    F.col("_months_since_open").isNotNull(),
                    F.col("_acct_type")
                )
            ).alias("count_products_ever"),
        )

        self._log_done(feature_df)
        return feature_df


class CreditBehaviourFlagsFeatures(TradelineFeatureBase):
    """
    Category 11: Credit Behaviour Flags

    All features are binary (0/1).
    Only accounts where open_dt <= as_of_dt are considered (no leakage).

    ── Requested features ────────────────────────────────────────────────────

    hasActiveTradeLineInLast12Months
        1 if any account opened in last 12m is still active at as_of_dt

    hasActiveTradeLineInLast24Months
        1 if any account opened in last 24m is still active at as_of_dt

    has_taken_PL_ever
        1 if customer has any PL account ever (open_dt <= as_of_dt)

    has_taken_pl_5yrs
        1 if customer has any PL account opened in last 60 months

    has_active_credit_card_flag
        1 if customer has any active CC account at as_of_dt

    mcf_user
        1 if customer has any Microfinance account (acct_type in 167/168/169/170)
        ever, as of as_of_dt

    ── Additional risk features ──────────────────────────────────────────────

    has_active_pl_flag
        1 if customer has any active PL at as_of_dt

    has_active_stpl_flag
        1 if customer has any active STPL (acct='242') at as_of_dt

    has_active_gl_flag
        1 if customer has any active Gold Loan at as_of_dt

    has_active_hl_flag
        1 if customer has any active Home Loan at as_of_dt

    has_taken_hl_ever
        1 if customer has ever had a Home Loan (strong negative risk signal)

    has_taken_gl_ever
        1 if customer has ever had a Gold Loan

    has_taken_cc_ever
        1 if customer has ever had a Credit Card

    has_taken_stpl_ever
        1 if customer has ever had a STPL

    mcf_active_user
        1 if customer has any ACTIVE Microfinance account at as_of_dt
        Distinct from mcf_user (ever); active = current MFI borrower

    has_p2p_loan_ever
        1 if customer has any P2P loan (acct 245/246/247) ever
        P2P = alternative lending, associated with bank-rejected profiles

    has_only_secured_credit
        1 if ALL accounts ever are secured products and no unsecured exists
        Pure secured profile = lower risk segment

    has_both_pl_and_cc
        1 if customer has both PL and CC ever
        Indicates broader credit product access and usage

    has_taken_pl_last_12m
        1 if any PL opened in last 12 months
        Very recent PL = fresh obligation signal

    has_multiple_active_pl
        1 if customer has 2 or more active PL accounts at as_of_dt
        Multiple active PLs = over-leverage signal
    """

    CATEGORY = "grp06b_credit_behaviour_flags"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        # ── STEP 1: Parse date columns ────────────────────────────────────────

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

        # ── STEP 4: Clean orig_loan_am ────────────────────────────────────────
        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 5: Normalise acct_type_cd ───────────────────────────────────
        df = df.withColumn("_acct_type", F.trim(F.col("acct_type_cd").cast("string")))

        # ── STEP 6: Product flags ─────────────────────────────────────────────
        df = (
            df
            .withColumn("_is_pl",       F.col("_acct_type") == PL_CODE)
            .withColumn("_is_cc",       F.col("_acct_type").isin(CC_CODES))
            .withColumn("_is_gl",       F.col("_acct_type").isin(GL_CODES))
            .withColumn("_is_al",       F.col("_acct_type").isin(AL_CODES))
            .withColumn("_is_hl",       F.col("_acct_type").isin(HL_CODES))
            .withColumn("_is_stpl",     F.col("_acct_type") == STPL_CODE)
            .withColumn("_is_mcf",      F.col("_acct_type").isin(MCF_CODES))
            .withColumn("_is_p2p",      F.col("_acct_type").isin(P2P_CODES))
            .withColumn("_is_secured",  F.col("_acct_type").isin(SECURED_CODES))
            .withColumn("_is_unsecured", ~F.col("_acct_type").isin(SECURED_CODES))
        )

        # ── STEP 7: Window flags ──────────────────────────────────────────────
        df = (
            df
            .withColumn("_in_12m",
                F.col("_months_since_open").isNotNull() &
                (F.col("_months_since_open") <= 12))
            .withColumn("_in_24m",
                F.col("_months_since_open").isNotNull() &
                (F.col("_months_since_open") <= 24))
            .withColumn("_in_60m",
                F.col("_months_since_open").isNotNull() &
                (F.col("_months_since_open") <= 60))
            .withColumn("_alltime",
                F.col("_months_since_open").isNotNull())
        )

        # ── STEP 8: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # ── Requested ─────────────────────────────────────────────────────

            # Active tradeline opened in last 12m
            F.max(
                F.when(F.col("_in_12m") & (F.col("_is_active") == 1), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("hasActiveTradeLineInLast12Months"),

            # Active tradeline opened in last 24m
            F.max(
                F.when(F.col("_in_24m") & (F.col("_is_active") == 1), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("hasActiveTradeLineInLast24Months"),

            # Ever had PL
            F.max(
                F.when(F.col("_is_pl") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_taken_PL_ever"),

            # PL in last 5 years
            F.max(
                F.when(F.col("_is_pl") & F.col("_in_60m"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_taken_pl_5yrs"),

            # Active CC at as_of_dt
            F.max(
                F.when(F.col("_is_cc") & (F.col("_is_active") == 1), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_active_credit_card_flag"),

            # Microfinance user — ever
            F.max(
                F.when(F.col("_is_mcf") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("mcf_user"),

            # ── Additional ────────────────────────────────────────────────────

            # Active PL at as_of_dt
            F.max(
                F.when(F.col("_is_pl") & (F.col("_is_active") == 1), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_active_pl_flag"),

            # Active STPL at as_of_dt
            F.max(
                F.when(F.col("_is_stpl") & (F.col("_is_active") == 1), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_active_stpl_flag"),

            # Active GL at as_of_dt
            F.max(
                F.when(F.col("_is_gl") & (F.col("_is_active") == 1), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_active_gl_flag"),

            # Active HL at as_of_dt
            F.max(
                F.when(F.col("_is_hl") & (F.col("_is_active") == 1), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_active_hl_flag"),

            # Ever had HL
            F.max(
                F.when(F.col("_is_hl") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_taken_hl_ever"),

            # Ever had GL
            F.max(
                F.when(F.col("_is_gl") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_taken_gl_ever"),

            # Ever had CC
            F.max(
                F.when(F.col("_is_cc") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_taken_cc_ever"),

            # Ever had STPL
            F.max(
                F.when(F.col("_is_stpl") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_taken_stpl_ever"),

            # Active MFI account at as_of_dt
            F.max(
                F.when(F.col("_is_mcf") & (F.col("_is_active") == 1), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("mcf_active_user"),

            # Ever had P2P loan
            F.max(
                F.when(F.col("_is_p2p") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_p2p_loan_ever"),

            # PL opened in last 12m
            F.max(
                F.when(F.col("_is_pl") & F.col("_in_12m"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_taken_pl_last_12m"),

            # Count of active PL accounts — used for has_multiple_active_pl below
            F.sum(
                F.when(F.col("_is_pl") & (F.col("_is_active") == 1), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("_active_pl_count"),

            # Count of secured accounts ever — used for has_only_secured_credit
            F.sum(
                F.when(F.col("_is_secured") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("_secured_count_ever"),

            # Count of unsecured accounts ever — used for has_only_secured_credit
            F.sum(
                F.when(F.col("_is_unsecured") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("_unsecured_count_ever"),

            # Count of PL ever — used for has_both_pl_and_cc
            F.sum(
                F.when(F.col("_is_pl") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("_pl_count_ever"),

            # Count of CC ever — used for has_both_pl_and_cc
            F.sum(
                F.when(F.col("_is_cc") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("_cc_count_ever"),

            # ── AL flags ─────────────────────────────────────────────────────

            # Active AL at as_of_dt
            F.max(
                F.when(F.col("_is_al") & (F.col("_is_active") == 1), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_active_al_flag"),

            # Ever had AL
            F.max(
                F.when(F.col("_is_al") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_taken_al_ever"),


            # ── USL flags ─────────────────────────────────────────────────────
            # USL = ~SECURED_CODES (all unsecured products: PL, CC, consumer, MFI, etc.)

            # Active USL at as_of_dt
            F.max(
                F.when(F.col("_is_unsecured") & (F.col("_is_active") == 1), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_active_usl_flag"),

            # Ever had USL
            F.max(
                F.when(F.col("_is_unsecured") & F.col("_alltime"), F.lit(1))
                 .otherwise(F.lit(0))
            ).alias("has_taken_usl_ever"),
        )

        # ── STEP 9: Derived flags (post-aggregation) ──────────────────────────

        # has_only_secured_credit
        # 1 = has secured accounts AND zero unsecured accounts ever
        feature_df = feature_df.withColumn(
            "has_only_secured_credit",
            F.when(
                (F.col("_secured_count_ever") > 0) &
                (F.col("_unsecured_count_ever") == 0),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        # has_both_pl_and_cc
        # 1 = has taken both PL and CC ever → broader credit product access
        feature_df = feature_df.withColumn(
            "has_both_pl_and_cc",
            F.when(
                (F.col("_pl_count_ever") > 0) & (F.col("_cc_count_ever") > 0),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        # has_multiple_active_pl
        # 1 = 2 or more active PL accounts → over-leverage signal
        feature_df = feature_df.withColumn(
            "has_multiple_active_pl",
            F.when(F.col("_active_pl_count") >= 2, F.lit(1))
             .otherwise(F.lit(0))
        )

        # Drop intermediate columns
        feature_df = feature_df.drop(
            "_active_pl_count", "_secured_count_ever",
            "_unsecured_count_ever", "_pl_count_ever", "_cc_count_ever"
        )

        self._log_done(feature_df)
        return feature_df
