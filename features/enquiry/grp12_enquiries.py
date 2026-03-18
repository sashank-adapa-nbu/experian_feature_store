# features/enquiry/cat16_credit_enquiries.py
# =============================================================================
# Category 16 — Credit Enquiries / Credit Hunger
# =============================================================================
# SOURCE TABLE : uc_dataorg_prod.l1_experian.experian_enquiry_segment
#
# ── ACTUAL COLUMNS ────────────────────────────────────────────────────────────
#   customer_scrub_key  → PK (scrub_id + customer_id)
#   inq_purp_cd         → Inquiry purpose code (Appendix I)
#   inq_date            → Date of enquiry (string — parsed below)
#   m_sub_id            → Lender type (Appendix B)
#   amount              → Enquiry amount
#   scrub_output_date   → Scrub reference date (as_of in scrub mode)
#
# ── ENQUIRY PURPOSE CODES (Appendix I) ───────────────────────────────────────
#   1  Agriculture         11 Microfinance         (unsecured)
#   2  Auto Loan           12 Non-Funded Facility
#   3  Business Loan       13 Personal Loan        (unsecured)
#   4  Commercial Vehicle  14 Property Loan        (secured)
#   5  Construction Equip  15 Telecom
#   6  Consumer Search     16 Two/Three Wheeler    (secured)
#   7  Credit Card         17 Working Capital
#   8  Education Loan      18 Consumer Loan        (unsecured)
#   9  Leasing             19 Credit Review
#  10  Loan vs Collateral  99 Others
#
# ── WINDOW LOGIC ─────────────────────────────────────────────────────────────
#   months_since_enq = months_between(as_of_dt, inq_date)
#   Only enquiries where inq_date <= as_of_dt counted (no leakage).
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.enquiry.base import EnquiryFeatureBase
from core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# ENQUIRY PURPOSE CODE GROUPS (Appendix I)
# =============================================================================

PL_ENQ_CODES   = {"13"}
CC_ENQ_CODES   = {"7"}
UNSECURED_ENQ  = {"13", "11", "18", "8"}
SECURED_ENQ    = {"2", "4", "5", "10", "14", "16", "1"}
BUSINESS_ENQ   = {"3", "17", "12"}

NBFC_CODE = "NBF"
PRV_CODE  = "PVT"
PUB_CODE  = "PUB"


class CreditEnquiriesFeatures(EnquiryFeatureBase):
    """
    Category 16: Credit Enquiries / Credit Hunger

    Source: experian_enquiry_segment
    Key columns: inq_purp_cd, inq_date, m_sub_id, amount
    """

    CATEGORY = "grp12_enquiries"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        # ── STEP 1: Parse inq_date ────────────────────────────────────────────
        def parse_date(c):
            return F.coalesce(
                F.to_date(F.col(c), "dd/MM/yyyy"),
                F.to_date(F.col(c), "yyyy-MM-dd"),
                F.to_date(F.col(c), "MM/dd/yyyy"),
            )

        df = (
            df
            .withColumn("_inq_dt",   parse_date("inq_date"))
            .withColumn("_as_of_dt", parse_date(as_of_col))
        )

        # ── STEP 2: months_since_enq — no leakage guard ───────────────────────
        # Only count enquiries where inq_date <= as_of_dt
        df = df.withColumn(
            "_months_since_enq",
            F.when(
                F.col("_inq_dt") <= F.col("_as_of_dt"),
                F.months_between(F.col("_as_of_dt"), F.col("_inq_dt"))
            ).otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 3: Normalise codes ───────────────────────────────────────────
        df = (
            df
            .withColumn("_purp",   F.trim(F.col("inq_purp_cd").cast("string")))
            .withColumn("_lender", F.upper(F.trim(F.col("m_sub_id").cast("string"))))
        )

        # ── STEP 4: Product and lender flags ─────────────────────────────────
        df = (
            df
            .withColumn("_is_pl",        F.col("_purp").isin(PL_ENQ_CODES))
            .withColumn("_is_cc",        F.col("_purp").isin(CC_ENQ_CODES))
            .withColumn("_is_unsecured", F.col("_purp").isin(UNSECURED_ENQ))
            .withColumn("_is_secured",   F.col("_purp").isin(SECURED_ENQ))
            .withColumn("_is_nbfc",      F.col("_lender") == NBFC_CODE)
            .withColumn("_is_pvt",       F.col("_lender") == PRV_CODE)
        )

        # ── STEP 5: Window flags ──────────────────────────────────────────────
        df = (
            df
            .withColumn("_alltime",
                F.col("_months_since_enq").isNotNull())
            .withColumn("_in_1m",
                F.col("_months_since_enq").isNotNull() &
                (F.col("_months_since_enq") <= 1))
            .withColumn("_in_3m",
                F.col("_months_since_enq").isNotNull() &
                (F.col("_months_since_enq") <= 3))
            .withColumn("_in_6m",
                F.col("_months_since_enq").isNotNull() &
                (F.col("_months_since_enq") <= 6))
            .withColumn("_in_12m",
                F.col("_months_since_enq").isNotNull() &
                (F.col("_months_since_enq") <= 12))
            .withColumn("_in_24m",
                F.col("_months_since_enq").isNotNull() &
                (F.col("_months_since_enq") <= 24))
        )

        # ── STEP 6: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # ── Requested ─────────────────────────────────────────────────────
            F.sum(F.when(F.col("_alltime"), F.lit(1)).otherwise(F.lit(0))
            ).alias("total_enquiries"),

            F.sum(F.when(F.col("_in_3m"),  F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_in_last_3m"),

            F.sum(F.when(F.col("_in_6m"),  F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_in_last_6m"),

            F.sum(F.when(F.col("_in_12m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_in_last_12m"),

            # ── Additional windowed counts ────────────────────────────────────
            F.sum(F.when(F.col("_in_24m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_in_last_24m"),

            # PL enquiries
            F.sum(F.when(F.col("_is_pl") & F.col("_in_6m"),  F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_pl_last_6m"),
            F.sum(F.when(F.col("_is_pl") & F.col("_in_12m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_pl_last_12m"),

            # CC enquiries
            F.sum(F.when(F.col("_is_cc") & F.col("_in_6m"),  F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_cc_last_6m"),
            F.sum(F.when(F.col("_is_cc") & F.col("_in_12m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_cc_last_12m"),

            # Unsecured and secured
            F.sum(F.when(F.col("_is_unsecured") & F.col("_in_6m"),  F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_unsecured_last_6m"),
            F.sum(F.when(F.col("_is_unsecured") & F.col("_in_12m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_unsecured_last_12m"),
            F.sum(F.when(F.col("_is_secured")   & F.col("_in_12m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_secured_last_12m"),

            # NBFC and private bank
            F.sum(F.when(F.col("_is_nbfc") & F.col("_in_6m"),  F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_nbfc_last_6m"),
            F.sum(F.when(F.col("_is_nbfc") & F.col("_in_12m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_nbfc_last_12m"),
            F.sum(F.when(F.col("_is_pvt")  & F.col("_in_12m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("enquiry_pvt_bank_last_12m"),

            # ── Recency ───────────────────────────────────────────────────────
            F.min(F.when(F.col("_alltime"),                          F.col("_months_since_enq"))
            ).alias("months_since_last_enquiry"),
            F.min(F.when(F.col("_is_pl")   & F.col("_alltime"),     F.col("_months_since_enq"))
            ).alias("months_since_last_pl_enquiry"),
            F.min(F.when(F.col("_is_cc")   & F.col("_alltime"),     F.col("_months_since_enq"))
            ).alias("months_since_last_cc_enquiry"),

            # ── Flags ─────────────────────────────────────────────────────────
            F.max(F.when(F.col("_in_1m"),                            F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_enquiry_last_1m"),
            F.max(F.when(F.col("_is_pl")   & F.col("_in_3m"),       F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_pl_enquiry_last_3m"),
            F.max(F.when(F.col("_is_nbfc") & F.col("_in_6m"),       F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_nbfc_enquiry_last_6m"),

            # For ratio features
            F.sum(F.when(F.col("_is_unsecured") & F.col("_in_12m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_unsec_12m"),
            F.sum(F.when(F.col("_is_nbfc")      & F.col("_in_12m"), F.lit(1)).otherwise(F.lit(0))
            ).alias("_nbfc_12m"),
            F.sum(F.when(F.col("_in_12m"),                           F.lit(1)).otherwise(F.lit(0))
            ).alias("_total_12m"),
        )

        # ── STEP 7: Derived ratios ────────────────────────────────────────────
        feature_df = (
            feature_df
            .withColumn(
                "ratio_unsecured_to_total_enq",
                F.when(F.col("_total_12m") > 0,
                       F.col("_unsec_12m") / F.col("_total_12m"))
                 .otherwise(F.lit(None).cast("double"))
            )
            .withColumn(
                "ratio_nbfc_to_total_enq",
                F.when(F.col("_total_12m") > 0,
                       F.col("_nbfc_12m") / F.col("_total_12m"))
                 .otherwise(F.lit(None).cast("double"))
            )
            .drop("_unsec_12m", "_nbfc_12m", "_total_12m")
        )

        self._log_done(feature_df)
        return feature_df
