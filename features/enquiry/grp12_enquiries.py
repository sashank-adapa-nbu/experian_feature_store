# features/enquiry/grp12_enquiries.py
# =============================================================================
# Group 12 — Credit Enquiries / Credit Hunger
# =============================================================================
# Source : experian_enquiry_segment
# PK     : customer_scrub_key + scrub_output_date  (scrub mode)
#          party_code         + reference_dt        (retro mode)
#
# Key columns:
#   customer_scrub_key  → join key
#   inq_purp_cd         → Enquiry purpose (Appendix I)
#   inq_date            → Date of enquiry (various string formats in data)
#   m_sub_id            → Lender type (Appendix B: NBF / PVT / PUB)
#   amount              → Enquiry amount
#
# Enquiry Purpose Codes — Appendix I (used as inq_purp_cd):
#   1=Agriculture  2=Auto  3=Business  4=Commercial Vehicle  5=Construction Equip
#   6=Consumer Search  7=Credit Card  8=Education  9=Leasing  10=Loan vs Collateral
#   11=Microfinance  12=Non-Funded Credit  13=Personal Loan  14=Property  15=Telecom
#   16=Two/Three Wheeler  17=Working Capital  18=Consumer Loan  19=Credit Review  99=Others
#
# Classification (Appendix I):
#   UNSECURED  : 13=PL, 11=MFI, 18=Consumer, 8=Education       (no collateral)
#   SECURED    : 1=Agri, 2=Auto, 4=CV, 5=Equip, 10=Collateral, 14=Property, 16=2W/3W
#   CC         : 7=Credit Card                                  (kept separate — also unsecured)
#   BUSINESS   : 3=Business, 17=Working Capital, 12=Non-Funded
#   UNCLASSIFIED: 6,9,15,19,99
#
# Date parsing:
#   Uses F.try_to_date (tolerates invalid/unparseable strings → NULL)
#   rather than F.to_date which raises an exception on bad input.
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.enquiry.base import EnquiryFeatureBase
from core.logger import get_logger
from core.date_utils import parse_date

logger = get_logger(__name__)


# =============================================================================
# ENQUIRY PURPOSE CODE GROUPS — Appendix I
# =============================================================================

PL_ENQ_CODES   = {"13"}                          # Personal Loan enquiry
CC_ENQ_CODES   = {"7"}                           # Credit Card enquiry
UNSECURED_ENQ  = {"13", "11", "18", "8","7"}         # PL + MFI + Consumer + Education
SECURED_ENQ    = {"1", "2", "4", "5", "10", "14", "16"}  # Agri/Auto/CV/Equip/Collateral/Property/2W3W
BUSINESS_ENQ   = {"3", "17", "12"}               # Business/Working Capital/Non-Funded

NBFC_CODE = "NBF"
PRV_CODE  = "PVT"
PUB_CODE  = "PUB"



class CreditEnquiriesFeatures(EnquiryFeatureBase):
    """
    Group 12: Credit Enquiries / Credit Hunger

    Source: experian_enquiry_segment
    Key columns: inq_purp_cd, inq_date, m_sub_id, amount

    Features
    --------
    total_enquiries
        Count of all enquiries where inq_date <= as_of_dt

    enquiry_in_last_3m / 6m / 12m / 24m
        Windowed enquiry counts

    enquiry_pl_last_6m / 12m
        Personal Loan enquiries (inq_purp_cd=13)

    enquiry_cc_last_6m / 12m
        Credit Card enquiries (inq_purp_cd=7)

    enquiry_unsecured_last_6m / 12m
        Unsecured enquiries (PL + MFI + Consumer + Education)

    enquiry_secured_last_12m
        Secured enquiries (Auto/CV/Equip/Collateral/Property/2W3W)

    enquiry_nbfc_last_6m / 12m
        Enquiries from NBFC lenders

    enquiry_pvt_bank_last_12m
        Enquiries from private bank lenders

    months_since_last_enquiry
        Months since most recent enquiry

    months_since_last_pl_enquiry / cc_enquiry
        Product-specific recency

    flag_enquiry_last_1m
        1 if any enquiry in last 1 month

    flag_pl_enquiry_last_3m
        1 if any PL enquiry in last 3 months

    flag_nbfc_enquiry_last_6m
        1 if any NBFC enquiry in last 6 months

    ratio_unsecured_to_total_enq
        Unsecured enquiries (12m) / Total enquiries (12m)

    ratio_nbfc_to_total_enq
        NBFC enquiries (12m) / Total enquiries (12m)
    """

    CATEGORY = "grp12_enquiries"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        # ── STEP 1: Parse dates — try_to_date tolerates bad formats → NULL ────
        df = (
            df
            .withColumn("_inq_dt",   parse_date("inq_date"))
            .withColumn("_as_of_dt", F.col(as_of_col).cast("date"))
        )

        # ── STEP 2: Months since enquiry (no leakage — only inq_date <= as_of) ─
        df = df.withColumn(
            "_months_since_enq",
            F.when(
                F.col("_inq_dt").isNotNull() &
                F.col("_as_of_dt").isNotNull() &
                (F.col("_inq_dt") <= F.col("_as_of_dt")),
                F.months_between(F.col("_as_of_dt"), F.col("_inq_dt"))
            ).otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 3: Normalise codes ───────────────────────────────────────────
        df = (
            df
            .withColumn("_purp",   F.trim(F.col("inq_purp_cd").cast("string")))
            .withColumn("_lender", F.upper(F.trim(F.col("m_sub_id").cast("string"))))
        )

        # ── STEP 4: Product and lender flags (all boolean) ───────────────────
        df = (
            df
            .withColumn("_is_pl",        F.col("_purp").isin(PL_ENQ_CODES))
            .withColumn("_is_cc",        F.col("_purp").isin(CC_ENQ_CODES))
            .withColumn("_is_unsecured", F.col("_purp").isin(UNSECURED_ENQ))
            .withColumn("_is_secured",   F.col("_purp").isin(SECURED_ENQ))
            .withColumn("_is_nbfc",      F.col("_lender") == NBFC_CODE)
            .withColumn("_is_pvt",       F.col("_lender") == PRV_CODE)
        )

        # ── STEP 5: Window flags (boolean) ───────────────────────────────────
        df = (
            df
            .withColumn("_alltime", F.col("_months_since_enq").isNotNull())
            .withColumn("_in_1m",   F.col("_months_since_enq").isNotNull() & (F.col("_months_since_enq") <= 1))
            .withColumn("_in_3m",   F.col("_months_since_enq").isNotNull() & (F.col("_months_since_enq") <= 3))
            .withColumn("_in_6m",   F.col("_months_since_enq").isNotNull() & (F.col("_months_since_enq") <= 6))
            .withColumn("_in_12m",  F.col("_months_since_enq").isNotNull() & (F.col("_months_since_enq") <= 12))
            .withColumn("_in_24m",  F.col("_months_since_enq").isNotNull() & (F.col("_months_since_enq") <= 24))
        )

        def _i(cond):
            return F.when(cond, F.lit(1)).otherwise(F.lit(0))

        # ── STEP 6: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # All-time and windowed counts
            F.sum(_i(F.col("_alltime"))).alias("total_enquiries"),
            F.sum(_i(F.col("_in_3m"))).alias("enquiry_in_last_3m"),
            F.sum(_i(F.col("_in_6m"))).alias("enquiry_in_last_6m"),
            F.sum(_i(F.col("_in_12m"))).alias("enquiry_in_last_12m"),
            F.sum(_i(F.col("_in_24m"))).alias("enquiry_in_last_24m"),

            # PL enquiries (inq_purp_cd=13)
            F.sum(_i(F.col("_is_pl") & F.col("_in_6m"))).alias("enquiry_pl_last_6m"),
            F.sum(_i(F.col("_is_pl") & F.col("_in_12m"))).alias("enquiry_pl_last_12m"),

            # CC enquiries (inq_purp_cd=7)
            F.sum(_i(F.col("_is_cc") & F.col("_in_6m"))).alias("enquiry_cc_last_6m"),
            F.sum(_i(F.col("_is_cc") & F.col("_in_12m"))).alias("enquiry_cc_last_12m"),

            # Unsecured enquiries (PL + MFI + Consumer + Education)
            F.sum(_i(F.col("_is_unsecured") & F.col("_in_6m"))).alias("enquiry_unsecured_last_6m"),
            F.sum(_i(F.col("_is_unsecured") & F.col("_in_12m"))).alias("enquiry_unsecured_last_12m"),

            # Secured enquiries
            F.sum(_i(F.col("_is_secured") & F.col("_in_12m"))).alias("enquiry_secured_last_12m"),

            # Lender type
            F.sum(_i(F.col("_is_nbfc") & F.col("_in_6m"))).alias("enquiry_nbfc_last_6m"),
            F.sum(_i(F.col("_is_nbfc") & F.col("_in_12m"))).alias("enquiry_nbfc_last_12m"),
            F.sum(_i(F.col("_is_pvt")  & F.col("_in_12m"))).alias("enquiry_pvt_bank_last_12m"),

            # Recency
            F.min(F.when(F.col("_alltime"), F.col("_months_since_enq"))
            ).alias("months_since_last_enquiry"),

            F.min(F.when(F.col("_is_pl") & F.col("_alltime"), F.col("_months_since_enq"))
            ).alias("months_since_last_pl_enquiry"),

            F.min(F.when(F.col("_is_cc") & F.col("_alltime"), F.col("_months_since_enq"))
            ).alias("months_since_last_cc_enquiry"),

            # Flags
            F.max(_i(F.col("_in_1m"))).alias("flag_enquiry_last_1m"),
            F.max(_i(F.col("_is_pl") & F.col("_in_3m"))).alias("flag_pl_enquiry_last_3m"),
            F.max(_i(F.col("_is_nbfc") & F.col("_in_6m"))).alias("flag_nbfc_enquiry_last_6m"),

            # Intermediates for ratio computation
            F.sum(_i(F.col("_is_unsecured") & F.col("_in_12m"))).alias("_unsec_12m"),
            F.sum(_i(F.col("_is_nbfc")      & F.col("_in_12m"))).alias("_nbfc_12m"),
            F.sum(_i(F.col("_in_12m"))).alias("_total_12m"),
        )

        # ── STEP 7: Derived ratios ────────────────────────────────────────────
        feature_df = (
            feature_df
            .withColumn(
                "ratio_unsecured_to_total_enq",
                F.when(F.col("_total_12m") > 0,
                       (F.col("_unsec_12m") / F.col("_total_12m")).cast("double"))
                 .otherwise(F.lit(None).cast("double"))
            )
            .withColumn(
                "ratio_nbfc_to_total_enq",
                F.when(F.col("_total_12m") > 0,
                       (F.col("_nbfc_12m") / F.col("_total_12m")).cast("double"))
                 .otherwise(F.lit(None).cast("double"))
            )
            .drop("_unsec_12m", "_nbfc_12m", "_total_12m")
        )

        self._log_done(feature_df)
        return feature_df
