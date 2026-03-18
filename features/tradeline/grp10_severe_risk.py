# features/tradeline/cat17_writeoffs_severe_risk.py
# =============================================================================
# Category 17 — Write-offs / Severe Risk Indicators
# =============================================================================
# Source table  : experian_tradeline_segment
#
# ── SOURCE COLUMNS ────────────────────────────────────────────────────────────
#   suit_filed_willful_dflt        → Suit filed / Wilful default status (Appendix E)
#   written_off_and_settled_status → Write-off / Settled status (Appendix F)
#   write_off_status_dt            → Date of write-off status
#   dflt_status_dt                 → Date account entered default status
#   charge_off_am                  → Charge-off amount
#   open_dt, closed_dt             → For active flag and vintage
#   acct_type_cd                   → Product type
#
# ── APPENDIX E — SUIT_FILED_WILLFUL_DFLT (new codes) ─────────────────────────
#   200 Restructured
#   201 Suit Filed
#   202 Wilful Default
#   203 Suit Filed (Wilful Default)
#   204 Written Off
#   205 Suit Filed & Written Off
#   206 Wilful Default & Written Off
#   207 Suit Filed (Wilful Default) & Written Off
#   208 Settled
#   209 Post (WO) Settled
#   Old: 01/1=Suit, 02/2=Wilful, 03/3=Suit+Wilful (also supported)
#
# ── APPENDIX F — WRITTEN_OFF_AND_SETTLED_STATUS ───────────────────────────────
#   02  Written-off
#   03  Settled
#   04  Post (WO) Settled
#   05  Account Sold
#   06  Written Off and Account Sold
#   08  Account Purchased and Written Off
#   13  Post Write Off Closed
#   (00,01,11,12 = restructured variants — not write-offs)
#
# ── SEVERITY TIERS ────────────────────────────────────────────────────────────
#   Tier 1 — Wilful Default (highest severity):
#     suit=202/203/206/207 OR suit=02/2/03/3 (old codes)
#   Tier 2 — Written Off:
#     suit=204/205/206/207 OR wo=02/06/08/13
#   Tier 3 — Suit Filed (without WO):
#     suit=201/203/205/207 OR suit=01/1/03/3 (old codes)
#   Tier 4 — Settled:
#     suit=208/209 OR wo=03/04
#   Tier 5 — Restructured:
#     suit=200 OR wo=00/01/11/12
#
# ── WINDOW LOGIC ─────────────────────────────────────────────────────────────
#   wo_5_years / wo_3_years use write_off_status_dt or dflt_status_dt
#   Only events where event_dt <= as_of_dt are counted (no leakage)
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# STATUS CODE SETS
# =============================================================================

# Appendix E — SUIT_FILED_WILLFUL_DFLT
SUIT_CODES         = {"201", "203", "205", "207", "01", "1", "03", "3"}
WILFUL_CODES       = {"202", "203", "206", "207", "02", "2", "03", "3"}
WO_SUIT_CODES      = {"204", "205", "206", "207"}          # written off via suit col
SETTLED_SUIT_CODES = {"208", "209"}
RESTRUCTURED_SUIT  = {"200"}

# Appendix F — WRITTEN_OFF_AND_SETTLED_STATUS
WO_STATUS_CODES      = {"02", "06", "08", "13"}            # written off
SETTLED_STATUS_CODES = {"03", "04"}                        # settled
RESTRUCTURED_STATUS  = {"00", "01", "11", "12"}            # restructured

# Combined write-off indicator (either column)
ALL_WO_SUIT = WO_SUIT_CODES | WILFUL_CODES   # wilful default treated as WO-equivalent

# Product codes
PL_CODE   = "123"
STPL_CODE = "242"
USPL_CODES = {
    "123", "189", "187", "130", "242", "244", "245", "247",
    "167", "169", "170", "176", "177", "178", "179",
    "228", "227", "226", "249",
}


class WriteoffsSevereRiskFeatures(TradelineFeatureBase):
    """
    Category 17: Write-offs / Severe Risk Indicators

    ── Requested ────────────────────────────────────────────────────────────
    suit_filed_willful_dflt     1 if any account has suit filed or wilful default ever
    wo                          1 if any account has write-off status ever
    wo_5_years                  1 if any write-off event in last 60 months
    wo_3_years                  1 if any write-off event in last 36 months

    ── Additional risk features ──────────────────────────────────────────────
    flag_wilful_default         1 if any wilful default (codes 202/203/206/207)
    flag_suit_filed             1 if any suit filed (without requiring WO)
    flag_settled                1 if any account settled (Appendix E/F)
    flag_restructured           1 if any account restructured
    flag_wo_pl                  1 if any PL account written off
    flag_wo_stpl                1 if any STPL account written off
    flag_wo_uspl                1 if any USL account written off
    flag_wo_active              1 if any CURRENTLY ACTIVE account has WO flag
                                (lender still reporting as active despite WO)
    count_wo_accounts           Count of written-off accounts ever
    count_suit_accounts         Count of accounts with suit filed ever
    total_charge_off_am         Sum of charge-off amounts across all WO accounts
    max_charge_off_am           Max single account charge-off amount
    months_since_last_wo        Months since most recent write-off event (from write_off_status_dt)
    months_since_last_suit      Months since most recent suit filed event (from dflt_status_dt)
    wo_1_year                   1 if any write-off in last 12 months (most severe recency)
    flag_post_wo_settled        1 if any account has Post (WO) Settled (code 209/04)
                                Shows borrower resolved a prior write-off
    """

    CATEGORY = "grp10_severe_risk"

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
            .withColumn("_as_of_dt",    parse_date(as_of_col))
            .withColumn("_open_dt",     parse_date("open_dt"))
            .withColumn("_closed_dt",   parse_date("closed_dt"))
            .withColumn("_wo_dt",       parse_date("write_off_status_dt"))
            .withColumn("_dflt_dt",     parse_date("dflt_status_dt"))
        )

        # ── STEP 2: Active flag ───────────────────────────────────────────────
        df = df.withColumn(
            "_is_active",
            F.when(
                (F.col("_open_dt") <= F.col("_as_of_dt")) &
                (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt"))),
                F.lit(True)
            ).otherwise(F.lit(False))
        )

        # ── STEP 3: Normalise status codes ───────────────────────────────────
        df = (
            df
            .withColumn("_suit", F.trim(F.col("suit_filed_willful_dflt").cast("string")))
            .withColumn("_wo",   F.trim(F.col("written_off_and_settled_status").cast("string")))
        )

        # ── STEP 4: Status flags ──────────────────────────────────────────────
        df = (
            df
            .withColumn("_is_suit",
                F.col("_suit").isin(SUIT_CODES))
            .withColumn("_is_wilful",
                F.col("_suit").isin(WILFUL_CODES))
            .withColumn("_is_wo",
                F.col("_suit").isin(WO_SUIT_CODES) |
                F.col("_wo").isin(WO_STATUS_CODES))
            .withColumn("_is_settled",
                F.col("_suit").isin(SETTLED_SUIT_CODES) |
                F.col("_wo").isin(SETTLED_STATUS_CODES))
            .withColumn("_is_restructured",
                F.col("_suit").isin(RESTRUCTURED_SUIT) |
                F.col("_wo").isin(RESTRUCTURED_STATUS))
            .withColumn("_is_post_wo_settled",
                (F.col("_suit") == "209") |
                (F.col("_wo")   == "04"))
        )

        # ── STEP 5: Product flags ─────────────────────────────────────────────
        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))
        df = (
            df
            .withColumn("_is_pl",   F.col("_acct") == PL_CODE)
            .withColumn("_is_stpl", F.col("_acct") == STPL_CODE)
            .withColumn("_is_uspl", F.col("_acct").isin(USPL_CODES))
        )

        # ── STEP 6: Clean charge_off_am ───────────────────────────────────────
        df = df.withColumn(
            "_co_am",
            F.when(F.col("charge_off_am") > 0, F.col("charge_off_am").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 7: Months since WO/default event ─────────────────────────────
        # Use write_off_status_dt for WO, dflt_status_dt for suit
        # Only events that occurred on or before as_of_dt (no leakage)
        df = (
            df
            .withColumn("_months_since_wo",
                F.when(
                    F.col("_wo_dt").isNotNull() & (F.col("_wo_dt") <= F.col("_as_of_dt")),
                    F.months_between(F.col("_as_of_dt"), F.col("_wo_dt"))
                ).otherwise(F.lit(None).cast("double")))
            .withColumn("_months_since_suit",
                F.when(
                    F.col("_dflt_dt").isNotNull() & (F.col("_dflt_dt") <= F.col("_as_of_dt")),
                    F.months_between(F.col("_as_of_dt"), F.col("_dflt_dt"))
                ).otherwise(F.lit(None).cast("double")))
        )

        # ── STEP 8: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # ── Requested ─────────────────────────────────────────────────────

            # suit_filed_willful_dflt: 1 if any suit OR wilful default ever
            F.max(F.when(F.col("_is_suit") | F.col("_is_wilful"), F.lit(1)).otherwise(F.lit(0))
            ).alias("suit_filed_willful_dflt"),

            # wo: 1 if any write-off ever
            F.max(F.when(F.col("_is_wo"), F.lit(1)).otherwise(F.lit(0))
            ).alias("wo"),

            # wo_5_years: WO event within last 60 months
            F.max(F.when(
                F.col("_is_wo") & F.col("_months_since_wo").isNotNull() &
                (F.col("_months_since_wo") <= 60),
                F.lit(1)).otherwise(F.lit(0))
            ).alias("wo_5_years"),

            # wo_3_years: WO event within last 36 months
            F.max(F.when(
                F.col("_is_wo") & F.col("_months_since_wo").isNotNull() &
                (F.col("_months_since_wo") <= 36),
                F.lit(1)).otherwise(F.lit(0))
            ).alias("wo_3_years"),

            # ── Additional severity flags ─────────────────────────────────────

            F.max(F.when(F.col("_is_wilful"),      F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_wilful_default"),

            F.max(F.when(F.col("_is_suit"),        F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_suit_filed"),

            F.max(F.when(F.col("_is_settled"),     F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_settled"),

            F.max(F.when(F.col("_is_restructured"), F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_restructured"),

            # Product-specific WO
            F.max(F.when(F.col("_is_wo") & F.col("_is_pl"),   F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_wo_pl"),

            F.max(F.when(F.col("_is_wo") & F.col("_is_stpl"), F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_wo_stpl"),

            F.max(F.when(F.col("_is_wo") & F.col("_is_uspl"), F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_wo_uspl"),

            # Active account with WO status (lender still reporting as live)
            F.max(F.when(F.col("_is_wo") & F.col("_is_active"), F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_wo_active"),

            # Post-WO Settled (resolved a prior write-off)
            F.max(F.when(F.col("_is_post_wo_settled"), F.lit(1)).otherwise(F.lit(0))
            ).alias("flag_post_wo_settled"),

            # WO in last 12 months (most severe recency)
            F.max(F.when(
                F.col("_is_wo") & F.col("_months_since_wo").isNotNull() &
                (F.col("_months_since_wo") <= 12),
                F.lit(1)).otherwise(F.lit(0))
            ).alias("wo_1_year"),

            # Count features
            F.sum(F.when(F.col("_is_wo"),   F.lit(1)).otherwise(F.lit(0))
            ).alias("count_wo_accounts"),

            F.sum(F.when(F.col("_is_suit"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_suit_accounts"),

            # Charge-off amounts
            F.sum(F.when(F.col("_is_wo"), F.col("_co_am"))
            ).alias("total_charge_off_am"),

            F.max(F.when(F.col("_is_wo"), F.col("_co_am"))
            ).alias("max_charge_off_am"),

            # Recency of WO and suit events
            F.min(F.when(F.col("_is_wo"),   F.col("_months_since_wo"))
            ).alias("months_since_last_wo"),

            F.min(F.when(F.col("_is_suit"), F.col("_months_since_suit"))
            ).alias("months_since_last_suit"),
        )

        self._log_done(feature_df)
        return feature_df
