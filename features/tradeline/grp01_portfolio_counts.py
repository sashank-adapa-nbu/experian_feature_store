# features/tradeline/grp01_portfolio_counts.py
# =============================================================================
# Group 01 — Portfolio Counts & Active Portfolio & Bureau Segments
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# CODE SETS
# =============================================================================

CC_CODES  = {"5", "214", "220", "213", "224", "225"}
PL_CODE   = "123"
STPL_CODE = "242"
GOLD_CODE = "191"

GL_CODES  = {"191", "243"}
AL_CODES  = {"47", "173", "172", "221", "222", "223", "246"}
HL_CODES  = {"58", "195", "168", "240"}
SPL_CODES = {"184", "185", "175", "241", "248", "181", "197", "198", "199", "200"}
USPL_CODES = {
    "123", "189", "187", "130", "242", "244", "245", "247",
    "167", "169", "170", "176", "177", "178", "179",
    "228", "227", "226", "249",
}
OTHER_CODES = {"999", "121", "219", "196", "215", "216", "217"}
ALL_NAMED   = GL_CODES | AL_CODES | HL_CODES | CC_CODES | SPL_CODES | USPL_CODES

UNSECURED_EXCLUDE = {
    "47", "58", "195", "168", "220", "173", "221",
    "175", "222", "172", "219", "184", "185", "191", "223",
}


def _i(cond):
    """Convert any condition/column to int 0/1 for safe summing."""
    return F.when(cond, F.lit(1)).otherwise(F.lit(0))


class PortfolioCountsFeatures(TradelineFeatureBase):
    """
    Group 01: Portfolio Counts & Active Portfolio & Bureau Segments
    All flags stored as BOOLEAN internally.
    Aggregations use _i(flag) to convert to int only at sum time.
    """

    CATEGORY = "grp01_portfolio_counts"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        def parse_date(c):
            return F.coalesce(
                F.to_date(F.col(c), "dd/MM/yyyy"),
                F.to_date(F.col(c), "yyyy-MM-dd"),
                F.to_date(F.col(c), "MM/dd/yyyy"),
            )

        df = (
            df
            .withColumn("_open_dt",   parse_date("open_dt"))
            .withColumn("_closed_dt", parse_date("closed_dt"))
            .withColumn("_as_of_dt",  parse_date(as_of_col))
        )

        # Months since open
        df = df.withColumn(
            "_months_since_open",
            F.when(F.col("_open_dt") <= F.col("_as_of_dt"),
                   F.months_between(F.col("_as_of_dt"), F.col("_open_dt")))
             .otherwise(F.lit(None).cast("double"))
        )

        # Bureau age — same as months_since_open (max gives oldest account age)
        df = df.withColumn(
            "_age_months",
            F.when(F.col("_open_dt") <= F.col("_as_of_dt"),
                   F.months_between(F.col("_as_of_dt"), F.col("_open_dt")))
             .otherwise(F.lit(None).cast("double"))
        )

        # Clean loan amount — cast string to double
        df = df.withColumn(
            "_loan_am",
            F.when(
                F.col("orig_loan_am").isNotNull() &
                (F.col("orig_loan_am").cast("double") > 0),
                F.col("orig_loan_am").cast("double")
            ).otherwise(F.lit(None).cast("double"))
        )

        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))

        # ── ALL FLAGS AS BOOLEAN ──────────────────────────────────────────────
        # Active flag (boolean)
        df = df.withColumn(
            "_active",
            (F.col("_open_dt") <= F.col("_as_of_dt")) &
            (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt")))
        )

        # Window flags (boolean)
        df = (
            df
            .withColumn("_in_6m",    F.col("_months_since_open").isNotNull() & (F.col("_months_since_open") <= 6))
            .withColumn("_in_12m",   F.col("_months_since_open").isNotNull() & (F.col("_months_since_open") <= 12))
            .withColumn("_in_24m",   F.col("_months_since_open").isNotNull() & (F.col("_months_since_open") <= 24))
            .withColumn("_in_36m",   F.col("_months_since_open").isNotNull() & (F.col("_months_since_open") <= 36))
            .withColumn("_eligible", F.col("_months_since_open").isNotNull())
        )

        # Product flags (boolean — all consistent, no int mixing)
        df = (
            df
            .withColumn("_is_pl",        F.col("_acct") == PL_CODE)
            .withColumn("_is_cc",        F.col("_acct").isin(CC_CODES))
            .withColumn("_is_gold",      F.col("_acct") == GOLD_CODE)
            .withColumn("_is_stpl",      F.col("_acct") == STPL_CODE)
            .withColumn("_is_gl",        F.col("_acct").isin(GL_CODES))
            .withColumn("_is_al",        F.col("_acct").isin(AL_CODES))
            .withColumn("_is_hl",        F.col("_acct").isin(HL_CODES))
            .withColumn("_is_spl",       F.col("_acct").isin(SPL_CODES))
            .withColumn("_is_usl",       F.col("_acct").isin(USPL_CODES))
            .withColumn("_is_other",     ~F.col("_acct").isin(ALL_NAMED))
            .withColumn("_is_unsecured", ~F.col("_acct").isin(UNSECURED_EXCLUDE))
        )

        # loan_flag row condition (boolean)
        df = df.withColumn(
            "_loan_flag",
            F.col("_eligible") & (
                (F.col("_acct").isin({"168", "195"}) &
                 (F.col("orig_loan_am").cast("double") >= 200000)) |
                ((F.col("_acct") == "5") &
                 (F.col("orig_loan_am").cast("double") >= 50000)) |
                ((F.col("_acct") == "123") &
                 (F.col("orig_loan_am").cast("double") >= 200000)) |
                ((F.col("_acct") == "221") &
                 (F.col("orig_loan_am").cast("double") >= 300000))
            )
        )

        # ── AGGREGATE — use _i() to convert boolean → int only at sum time ───
        feature_df = df.groupBy(group_cols).agg(

            # A. Tradeline counts
            F.sum(_i(F.col("_in_6m"))).alias("count_of_tradelines_opened_last_6m"),
            F.sum(_i(F.col("_in_12m"))).alias("count_of_tradelines_opened_last_12m"),

            F.sum(_i(F.col("_in_6m")  & F.col("_loan_am").isNotNull() & (F.col("_loan_am") < 10000))).alias("count_of_tradelines_opened_last_6m_lt_10000"),
            F.sum(_i(F.col("_in_12m") & F.col("_loan_am").isNotNull() & (F.col("_loan_am") < 20000))).alias("count_of_tradelines_opened_last_12m_lt_20000"),

            F.sum(_i(F.col("_in_6m")  & F.col("_is_pl"))).alias("count_of_tradelines_opened_last_6m_personal_loan"),
            F.sum(_i(F.col("_in_12m") & F.col("_is_pl"))).alias("count_of_tradelines_opened_last_1y_personal_loan"),
            F.sum(_i(F.col("_in_36m") & F.col("_is_pl"))).alias("count_of_tradelines_opened_last_3y_personal_loan"),
            F.sum(_i(F.col("_in_36m") & F.col("_is_cc"))).alias("count_of_tradelines_opened_last_3y_cc"),

            F.sum(_i(F.col("_active") & F.col("_is_pl"))).alias("CountBureauActivePersonalLoanAccounts"),
            F.sum(_i(F.col("_is_unsecured"))).alias("countoftradelines_unsecured"),

            F.sum(_i(F.col("_in_24m") & F.col("_is_pl") & F.col("_loan_am").isNotNull() & (F.col("_loan_am") <= 30000))).alias("countlessthan30KPlinLast24Months"),
            F.sum(_i(F.col("_is_gold") & F.col("_loan_am").isNotNull() & (F.col("_loan_am") > 200000))).alias("countBureauGoldLoansGreaterThan2L"),
            F.sum(_i(F.col("_active") & F.col("_is_stpl") & F.col("_loan_am").isNotNull() & (F.col("_loan_am") <= 30000))).alias("CountBureauActiveSTPLAccounts"),

            # B. Active portfolio counts by product
            F.sum(_i(F.col("_active"))).alias("active_loans"),
            F.sum(_i(F.col("_active") & F.col("_is_usl"))).alias("active_usl"),
            F.sum(_i(F.col("_active") & F.col("_is_spl"))).alias("active_spl"),
            F.sum(_i(F.col("_active") & F.col("_is_cc"))).alias("active_cc"),
            F.sum(_i(F.col("_active") & F.col("_is_hl"))).alias("active_hl"),
            F.sum(_i(F.col("_active") & F.col("_is_al"))).alias("active_al"),
            F.sum(_i(F.col("_active") & F.col("_is_gl"))).alias("active_gl"),
            F.sum(_i(F.col("_active") & F.col("_is_other"))).alias("active_other_loans"),

            # C. Bureau segment intermediates
            F.max(_i(F.col("_loan_flag"))).alias("loan_flag"),
            F.sum(_i(F.col("_eligible") & F.col("_is_pl"))).alias("tot_pls"),
            F.max(F.when(F.col("_eligible") & F.col("_is_pl"), F.col("_loan_am"))).alias("max_pl_loan_amount"),
            F.sum(_i(F.col("_eligible") & F.col("_is_cc"))).alias("tot_cc"),
            F.max(F.when(F.col("_eligible") & F.col("_is_cc"), F.col("_loan_am"))).alias("max_cc_loan_amount"),
            F.sum(_i(F.col("_eligible"))).alias("tot_tradelines"),
            F.max("_age_months").alias("bureaage_months"),
        )

        # Segment labels
        feature_df = (
            feature_df
            .withColumn("pl_thickness",
                F.when(F.coalesce(F.col("tot_pls"), F.lit(0)) == 0,                                          F.lit("No PL"))
                 .when((F.col("tot_pls") <= 2) & (F.col("max_pl_loan_amount") < 100000),                     F.lit("Thin PL"))
                 .when((F.col("tot_pls") >  2) & (F.col("max_pl_loan_amount") >= 100000),                    F.lit("Thick PL"))
                 .otherwise(                                                                                   F.lit("Medium PL")))
            .withColumn("cc_thickness",
                F.when(F.coalesce(F.col("tot_cc"), F.lit(0)) == 0,                                           F.lit("No CC"))
                 .when((F.col("tot_cc") == 1) & (F.col("max_cc_loan_amount") < 100000),                      F.lit("Thin CC"))
                 .when((F.col("tot_cc") >= 2) & (F.col("max_cc_loan_amount") >= 100000),                     F.lit("Thick CC"))
                 .otherwise(                                                                                   F.lit("Medium CC")))
            .withColumn("bureaage",
                F.when(F.col("bureaage_months").isNotNull() & (F.col("bureaage_months") > 0)
                                                             & (F.col("bureaage_months") <= 12),              F.lit("<1Y"))
                 .when((F.col("bureaage_months") > 12) & (F.col("bureaage_months") <= 24),                   F.lit("1-2Y"))
                 .when((F.col("bureaage_months") > 24) & (F.col("bureaage_months") <= 60),                   F.lit("2-5Y"))
                 .when( F.col("bureaage_months") > 60,                                                        F.lit(">5Y"))
                 .otherwise(                                                                                   F.lit("Unknown")))
            .withColumn("bur_cohort",
                F.when((F.col("bureaage_months") >= 18) & (F.col("tot_tradelines") >= 3),                    F.lit("Thick"))
                 .otherwise(                                                                                   F.lit("Thin")))
        )

        self._log_done(feature_df)
        return feature_df
