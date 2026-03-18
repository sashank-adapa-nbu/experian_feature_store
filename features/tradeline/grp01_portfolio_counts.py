# features/tradeline/grp01_portfolio_counts.py
# =============================================================================
# Group 01 — Portfolio Counts & Active Portfolio
# =============================================================================
# Sections:
#   A. Tradeline opened counts (windowed, by product, by amount)
#   B. Active portfolio counts by product type
#   C. Bureau segment labels (thickness, cohort)
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
#unsecured codes
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

SECURED_CODES = {
    "47", "58", "195", "168", "220", "173", "221",
    "175", "222", "172", "219", "184", "185", "191",
    "223", "243", "241",
}


class PortfolioCountsFeatures(TradelineFeatureBase):
    """
    Group 01: Portfolio Counts & Active Portfolio & Bureau Segments

    A. Tradeline counts — opened within windows, by product/amount
    B. Active portfolio counts — by product group (cat02)
    C. Bureau segment labels — pl_thickness, cc_thickness, bureaage, bur_cohort
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

        # Months since open — NULL for future accounts
        df = df.withColumn(
            "_months_since_open",
            F.when(F.col("_open_dt") <= F.col("_as_of_dt"),
                   F.months_between(F.col("_as_of_dt"), F.col("_open_dt")))
             .otherwise(F.lit(None).cast("double"))
        )

        # Bureau age (for segments)
        df = df.withColumn(
            "_age_months",
            F.when(F.col("_open_dt") <= F.col("_as_of_dt"),
                   F.months_between(F.col("_as_of_dt"), F.col("_open_dt")))
             .otherwise(F.lit(None).cast("double"))
        )

        # Active flag — point-in-time
        df = df.withColumn(
            "_is_active",
            F.when(
                (F.col("_open_dt") <= F.col("_as_of_dt")) &
                (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt"))),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        # Clean loan amount
        df = df.withColumn(
            "_loan_am",
            F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am"))
             .otherwise(F.lit(None))
        )

        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))

        # Window flags
        df = (
            df
            .withColumn("_in_6m",  F.col("_months_since_open").isNotNull() & (F.col("_months_since_open") <= 6))
            .withColumn("_in_12m", F.col("_months_since_open").isNotNull() & (F.col("_months_since_open") <= 12))
            .withColumn("_in_24m", F.col("_months_since_open").isNotNull() & (F.col("_months_since_open") <= 24))
            .withColumn("_in_36m", F.col("_months_since_open").isNotNull() & (F.col("_months_since_open") <= 36))
            .withColumn("_eligible", F.col("_months_since_open").isNotNull())
        )

        # Product flags
        df = (
            df
            .withColumn("_is_pl",       F.col("_acct") == PL_CODE)
            .withColumn("_is_cc",       F.col("_acct").isin(CC_CODES))
            .withColumn("_is_gold",     F.col("_acct") == GOLD_CODE)
            .withColumn("_is_stpl",     F.col("_acct") == STPL_CODE)
            .withColumn("_is_gl",       F.col("_acct").isin(GL_CODES))
            .withColumn("_is_al",       F.col("_acct").isin(AL_CODES))
            .withColumn("_is_hl",       F.col("_acct").isin(HL_CODES))
            .withColumn("_is_spl",      F.col("_acct").isin(SPL_CODES))
            .withColumn("_is_usl",      F.col("_acct").isin(USPL_CODES))
            .withColumn("_is_other",    ~F.col("_acct").isin(ALL_NAMED))
            .withColumn("_is_unsecured", ~F.col("_acct").isin(UNSECURED_EXCLUDE))
        )

        # loan_flag row condition
        df = df.withColumn(
            "_loan_flag_row",
            F.when(
                F.col("_eligible") & (
                    (F.col("_acct").isin({"168", "195"}) & F.col("_loan_am").isNotNull() & (F.col("_loan_am") >= 200000)) |
                    ((F.col("_acct") == "5") & (F.col("orig_loan_am").cast("double") >= 50000)) |
                    ((F.col("_acct") == "123") & F.col("_loan_am").isNotNull() & (F.col("_loan_am") >= 200000)) |
                    ((F.col("_acct") == "221") & F.col("_loan_am").isNotNull() & (F.col("_loan_am") >= 300000))
                ), F.lit(1)
            ).otherwise(F.lit(0))
        )

        feature_df = df.groupBy(group_cols).agg(

            # ── A. Tradeline counts (cat01) ───────────────────────────────────
            F.sum(F.when(F.col("_in_6m"),  F.lit(1)).otherwise(F.lit(0))).alias("count_of_tradelines_opened_last_6m"),
            F.sum(F.when(F.col("_in_12m"), F.lit(1)).otherwise(F.lit(0))).alias("count_of_tradelines_opened_last_12m"),

            F.sum(F.when(F.col("_in_6m")  & F.col("_loan_am").isNotNull() & (F.col("_loan_am") < 10000), F.lit(1)).otherwise(F.lit(0))).alias("count_of_tradelines_opened_last_6m_lt_10000"),
            F.sum(F.when(F.col("_in_12m") & F.col("_loan_am").isNotNull() & (F.col("_loan_am") < 20000), F.lit(1)).otherwise(F.lit(0))).alias("count_of_tradelines_opened_last_12m_lt_20000"),

            F.sum(F.when(F.col("_in_6m")  & F.col("_is_pl"), F.lit(1)).otherwise(F.lit(0))).alias("count_of_tradelines_opened_last_6m_personal_loan"),
            F.sum(F.when(F.col("_in_12m") & F.col("_is_pl"), F.lit(1)).otherwise(F.lit(0))).alias("count_of_tradelines_opened_last_1y_personal_loan"),
            F.sum(F.when(F.col("_in_36m") & F.col("_is_pl"), F.lit(1)).otherwise(F.lit(0))).alias("count_of_tradelines_opened_last_3y_personal_loan"),
            F.sum(F.when(F.col("_in_36m") & F.col("_is_cc"), F.lit(1)).otherwise(F.lit(0))).alias("count_of_tradelines_opened_last_3y_cc"),

            F.sum(F.when((F.col("_is_active") == 1) & F.col("_is_pl"), F.lit(1)).otherwise(F.lit(0))).alias("CountBureauActivePersonalLoanAccounts"),
            F.sum("_is_unsecured").alias("countoftradelines_unsecured"),

            F.sum(F.when(F.col("_in_24m") & F.col("_is_pl") & F.col("_loan_am").isNotNull() & (F.col("_loan_am") <= 30000), F.lit(1)).otherwise(F.lit(0))).alias("countlessthan30KPlinLast24Months"),
            F.sum(F.when(F.col("_is_gold") & F.col("_loan_am").isNotNull() & (F.col("_loan_am") > 200000), F.lit(1)).otherwise(F.lit(0))).alias("countBureauGoldLoansGreaterThan2L"),
            F.sum(F.when((F.col("_is_active") == 1) & F.col("_is_stpl") & F.col("_loan_am").isNotNull() & (F.col("_loan_am") <= 30000), F.lit(1)).otherwise(F.lit(0))).alias("CountBureauActiveSTPLAccounts"),

            # ── B. Active portfolio counts by product (cat02) ───────────────── # all active (replaces CountBureauActiveAccounts)
            F.sum(F.col("_is_active")).alias("CountBureauActiveAccounts"),                                                         
            F.sum(F.when((F.col("_is_active") == 1) & F.col("_is_usl"),   F.lit(1)).otherwise(F.lit(0))).alias("active_usl"),
            F.sum(F.when((F.col("_is_active") == 1) & F.col("_is_spl"),   F.lit(1)).otherwise(F.lit(0))).alias("active_spl"),
            F.sum(F.when((F.col("_is_active") == 1) & F.col("_is_cc"),    F.lit(1)).otherwise(F.lit(0))).alias("active_cc"),
            F.sum(F.when((F.col("_is_active") == 1) & F.col("_is_hl"),    F.lit(1)).otherwise(F.lit(0))).alias("active_hl"),
            F.sum(F.when((F.col("_is_active") == 1) & F.col("_is_al"),    F.lit(1)).otherwise(F.lit(0))).alias("active_al"),
            F.sum(F.when((F.col("_is_active") == 1) & F.col("_is_gl"),    F.lit(1)).otherwise(F.lit(0))).alias("active_gl"),
            F.sum(F.when((F.col("_is_active") == 1) & F.col("_is_other"), F.lit(1)).otherwise(F.lit(0))).alias("active_other_loans"),

            # ── C. Bureau segment intermediates (cat19) ───────────────────────
            F.max("_loan_flag_row").alias("loan_flag"),
            F.sum(F.when(F.col("_eligible") & F.col("_is_pl"), F.lit(1)).otherwise(F.lit(0))).alias("tot_pls"),
            F.max(F.when(F.col("_eligible") & F.col("_is_pl"), F.col("_loan_am"))).alias("max_pl_loan_amount"),
            F.sum(F.when(F.col("_eligible") & F.col("_is_cc"), F.lit(1)).otherwise(F.lit(0))).alias("tot_cc"),
            F.max(F.when(F.col("_eligible") & F.col("_is_cc"), F.col("_loan_am"))).alias("max_cc_loan_amount"),
            F.sum(F.when(F.col("_eligible"), F.lit(1)).otherwise(F.lit(0))).alias("tot_tradelines"),
            F.max("_age_months").alias("bureaage_months"),
        )

        # Segment labels
        feature_df = (
            feature_df
            .withColumn("pl_thickness",
                F.when(F.coalesce(F.col("tot_pls"), F.lit(0)) == 0, F.lit("No PL"))
                 .when((F.col("tot_pls") <= 2) & (F.col("max_pl_loan_amount") < 100000), F.lit("Thin PL"))
                 .when((F.col("tot_pls") > 2)  & (F.col("max_pl_loan_amount") >= 100000), F.lit("Thick PL"))
                 .otherwise(F.lit("Medium PL")))
            .withColumn("cc_thickness",
                F.when(F.coalesce(F.col("tot_cc"), F.lit(0)) == 0, F.lit("No CC"))
                 .when((F.col("tot_cc") == 1) & (F.col("max_cc_loan_amount") < 100000), F.lit("Thin CC"))
                 .when((F.col("tot_cc") >= 2) & (F.col("max_cc_loan_amount") >= 100000), F.lit("Thick CC"))
                 .otherwise(F.lit("Medium CC")))
            .withColumn("bureaage",
                F.when(F.col("bureaage_months").isNotNull() & (F.col("bureaage_months") > 0)  & (F.col("bureaage_months") <= 12), F.lit("<1Y"))
                 .when((F.col("bureaage_months") > 12) & (F.col("bureaage_months") <= 24), F.lit("1-2Y"))
                 .when((F.col("bureaage_months") > 24) & (F.col("bureaage_months") <= 60), F.lit("2-5Y"))
                 .when(F.col("bureaage_months") > 60, F.lit(">5Y"))
                 .otherwise(F.lit("Unknown")))
            .withColumn("bur_cohort",
                F.when((F.col("bureaage_months") >= 18) & (F.col("tot_tradelines") >= 3), F.lit("Thick"))
                 .otherwise(F.lit("Thin")))
        )

        self._log_done(feature_df)
        return feature_df
