# features/tradeline/grp10_severe_risk.py
# =============================================================================
# Group 10 — Write-offs / Severe Risk Indicators
# =============================================================================
# Source : experian_tradeline_segment
#
# ── SOURCE COLUMNS ────────────────────────────────────────────────────────────
#   suit_filed_willful_dflt        → Suit filed / Wilful default status (Appendix E)
#   written_off_and_settled_status → Write-off / Settled status (Appendix F)
#   write_off_status_dt            → Date of write-off event
#   dflt_status_dt                 → Date account entered default / suit status
#   total_write_off_am_4in         → Total write-off amount (string, -1/null = no data)
#   principal_write_off_am_4in     → Principal portion of write-off (string)
#   settled_am_4in                 → Settled / recovery amount (string)
#   open_dt, closed_dt             → For active flag
#   acct_type_cd                   → Product type
#
# ── APPENDIX E — SUIT_FILED_WILLFUL_DFLT ─────────────────────────────────────
#   New codes:
#     200 Restructured
#     201 Suit Filed
#     202 Wilful Default
#     203 Suit Filed (Wilful Default)
#     204 Written Off
#     205 Suit Filed & Written Off
#     206 Wilful Default & Written Off
#     207 Suit Filed (Wilful Default) & Written Off
#     208 Settled
#     209 Post (WO) Settled
#   Old codes (still present in data):
#     00/0  No Suit Filed
#     01/1  Suit Filed
#     02/2  Wilful Default
#     03/3  Suit Filed (Wilful Default)
#
# ── APPENDIX F — WRITTEN_OFF_AND_SETTLED_STATUS ──────────────────────────────
#   00  Restructure Loan
#   01  Restructure Loan (Govt. Mandated)
#   02  Written-off
#   03  Settled
#   04  Post (WO) Settled
#   05  Account Sold
#   06  Written Off and Account Sold
#   07  Account Purchased
#   08  Account Purchased and Written Off
#   09  Account Purchased and Settled
#   10  Account Purchased and Restructured
#   11  Restructured due to Natural Calamity
#   12  Restructured due to COVID-19
#   13  Post Write Off Closed
#   14  Restructured & Closed
#   15  Auctioned & Settled
#   16  Repossessed & Settled
#   17  Guarantee Invoked
#   99  Clear existing status
#
# ── SEVERITY TIERS ────────────────────────────────────────────────────────────
#   Tier 1 — Wilful Default  : suit = 202/203/206/207 or old 02/2/03/3
#   Tier 2 — Written Off     : suit = 204/205/206/207 or wo = 02/05/06/08/13
#   Tier 3 — Suit Filed      : suit = 201/203/205/207 or old 01/1/03/3
#   Tier 4 — Settled         : suit = 208/209 or wo = 03/04/09/15/16
#   Tier 5 — Restructured    : suit = 200 or wo = 00/01/10/11/12/14
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger
from core.date_utils import parse_date

logger = get_logger(__name__)


# =============================================================================
# STATUS CODE SETS — verified against Appendix E & F (Experian v3.2)
# =============================================================================

# Appendix E — SUIT_FILED_WILLFUL_DFLT
SUIT_CODES          = {"201", "203", "205", "207", "01", "1", "03", "3"}
WILFUL_CODES        = {"202", "203", "206", "207", "02", "2", "03", "3"}
WO_SUIT_CODES       = {"204", "205", "206", "207"}
SETTLED_SUIT_CODES  = {"208", "209"}
RESTRUCTURED_SUIT   = {"200"}

# Appendix F — WRITTEN_OFF_AND_SETTLED_STATUS

WO_STATUS_CODES      = {"02", "05", "06", "08", "13"}            # written off
SETTLED_STATUS_CODES = {"03", "04", "09", "15", "16"}                        # settled
RESTRUCTURED_STATUS  = {"00", "01", "10", "11", "12", "14"}            # restructured

SETTLED_STATUS_CODES  = {"03", "04", "09", "15", "16"}
# 03=Settled, 04=Post(WO)Settled, 09=Account Purchased and Settled,
# 15=Auctioned & Settled, 16=Repossessed & Settled

RESTRUCTURED_STATUS   = {"00", "01", "10", "11", "12", "14"}
# 00=Restructure Loan, 01=Restructure(Govt), 10=Account Purchased and Restructured,
# 11=Restructured Natural Calamity, 12=Restructured COVID-19, 14=Restructured & Closed

# Product codes
PL_CODE   = "123"
STPL_CODE = "242"
USL_CODES = {
    "123", "189", "187", "130", "242", "244", "245", "247",
    "167", "169", "170", "176", "177", "178", "179",
    "228", "227", "226", "249",
}

UNSECURED_EXCLUDE = {
    # All secured codes — excluded when counting unsecured tradelines
    # Must match SECURED_CODES exactly (Appendix A verified)
    # NOTE: 220 (Secured CC) removed — CCs are treated as unsecured/CC category
    "47", "58", "168", "172", "173", "175", "181", "184", "185",
    "191", "195", "197", "198", "199", "200", "219", "221",
    "222", "223", "240", "241", "243", "246", "248",
}


class WriteoffsSevereRiskFeatures(TradelineFeatureBase):
    """
    Group 10: Write-offs / Severe Risk Indicators

    Derives derogatory severity flags and write-off amount features from
    suit_filed_willful_dflt (Appendix E) and written_off_and_settled_status
    (Appendix F), with recovery analysis from amount columns.

    ── Core flags ───────────────────────────────────────────────────────────
    suit_filed_willful_dflt     1 if any account has suit filed or wilful default ever
    wo                          1 if any account has write-off status ever
    wo_5_years                  1 if any write-off event in last 60 months
    wo_3_years                  1 if any write-off event in last 36 months

    ── Severity flags ───────────────────────────────────────────────────────
    flag_wilful_default         1 if any wilful default (codes 202/203/206/207)
    flag_suit_filed             1 if any suit filed (without requiring WO)
    flag_settled                1 if any account settled (Appendix E/F)
    flag_restructured           1 if any account restructured
    flag_wo_active              1 if any CURRENTLY ACTIVE account has WO flag
    flag_post_wo_settled        1 if any account Post (WO) Settled (code 209/04)
    wo_1_year                   1 if any write-off in last 12 months

    ── Product-specific WO flags ────────────────────────────────────────────
    flag_wo_pl                  1 if any PL account written off
    flag_wo_stpl                1 if any STPL account written off
    flag_wo_usl                 1 if any unsecured loan account written off

    ── Count features ───────────────────────────────────────────────────────
    count_wo_accounts           Count of written-off accounts ever
    count_suit_accounts         Count of accounts with suit filed ever

    ── Total write-off amount (total_write_off_am_4in) ──────────────────────
    total_charge_off_am         Sum of total write-off amounts across all WO accounts
    max_charge_off_am           Max single account total write-off amount

    ── Principal write-off amount (principal_write_off_am_4in) ──────────────
    total_principal_wo_am       Sum of principal written off across all WO accounts
    max_principal_wo_am         Max single account principal write-off amount
    ratio_principal_to_total_wo total_principal_wo_am / total_charge_off_am
                                Proportion of WO that is pure principal loss.
                                High (~1.0) = mostly principal, low interest accrual.
                                Low = large interest/fee component in WO.

    ── Settled / recovery amount (settled_am_4in) ───────────────────────────
    total_settled_am            Sum of settled amounts across all settled accounts
    max_settled_am              Max single account settled amount
    ratio_settled_to_wo         total_settled_am / total_charge_off_am
                                Recovery rate: 1.0 = full recovery, 0.0 = no recovery.
                                Low ratio = poor recovery behaviour.
    flag_partial_recovery       1 if settled > 0 but < total WO amount
                                Borrower settled partially but did not clear full WO.
    net_loss_am                 total_charge_off_am - total_settled_am
                                Actual net credit loss after recovery.

    ── Recency ──────────────────────────────────────────────────────────────
    months_since_last_wo        Months since most recent write-off event
    months_since_last_suit      Months since most recent suit filed event
    """

    CATEGORY = "grp10_severe_risk"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        df = (
            df
            .withColumn("_as_of_dt", F.col(as_of_col).cast("date"))
            .withColumn("_open_dt",   parse_date("open_dt"))
            .withColumn("_closed_dt", parse_date("closed_dt"))
            .withColumn("_wo_dt",     parse_date("write_off_status_dt"))
            .withColumn("_dflt_dt",   parse_date("dflt_status_dt"))
        )

        # ── STEP 2: Active flag (PIT) ─────────────────────────────────────────
        df = df.withColumn(
            "_is_active",
            (F.col("_open_dt") <= F.col("_as_of_dt")) &
            (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt")))
        )

        # ── STEP 3: Normalise status codes ────────────────────────────────────
        df = (
            df
            .withColumn("_suit", F.trim(F.col("suit_filed_willful_dflt").cast("string")))
            .withColumn("_wo",   F.trim(F.col("written_off_and_settled_status").cast("string")))
        )

        # ── STEP 4: Status flags (all boolean) ───────────────────────────────
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
                (F.col("_suit") == "209") | (F.col("_wo") == "04"))
        )

        # ── STEP 5: Product flags (all boolean) ──────────────────────────────
        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))
        df = (
            df
            .withColumn("_is_pl",   F.col("_acct") == PL_CODE)
            .withColumn("_is_stpl", F.col("_acct") == STPL_CODE)
            .withColumn("_is_usl", ~F.col("_acct").isin(UNSECURED_EXCLUDE))
        )

        # ── STEP 6: Write-off and settled amounts ────────────────────────────
        # All amount columns are string type; -1 and NULL both mean no data.
        def _clean_am(col_name: str) -> F.Column:
            c = F.col(col_name).cast("double")
            return F.when(c > 0, c).otherwise(F.lit(None).cast("double"))

        df = (
            df
            .withColumn("_wo_total_am",    _clean_am("total_write_off_am_4in"))
            .withColumn("_wo_princ_am",    _clean_am("principal_write_off_am_4in"))
            .withColumn("_settled_am",     _clean_am("settled_am_4in"))
        )

        # ── STEP 7: Months since WO and suit events (PIT — no leakage) ───────
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
        def _i(cond):
            return F.when(cond, F.lit(1)).otherwise(F.lit(0))

        feature_df = df.groupBy(group_cols).agg(

            # ── Core flags ────────────────────────────────────────────────────
            F.max(_i(F.col("_is_suit") | F.col("_is_wilful"))
            ).alias("suit_filed_willful_dflt"),

            F.max(_i(F.col("_is_wo"))
            ).alias("wo"),

            F.max(_i(F.col("_is_wo") & F.col("_months_since_wo").isNotNull() &
                     (F.col("_months_since_wo") <= 60))
            ).alias("wo_5_years"),

            F.max(_i(F.col("_is_wo") & F.col("_months_since_wo").isNotNull() &
                     (F.col("_months_since_wo") <= 36))
            ).alias("wo_3_years"),

            # ── Severity flags ────────────────────────────────────────────────
            F.max(_i(F.col("_is_wilful"))).alias("flag_wilful_default"),
            F.max(_i(F.col("_is_suit"))).alias("flag_suit_filed"),
            F.max(_i(F.col("_is_settled"))).alias("flag_settled"),
            F.max(_i(F.col("_is_restructured"))).alias("flag_restructured"),
            F.max(_i(F.col("_is_post_wo_settled"))).alias("flag_post_wo_settled"),

            F.max(_i(F.col("_is_wo") & F.col("_months_since_wo").isNotNull() &
                     (F.col("_months_since_wo") <= 12))
            ).alias("wo_1_year"),

            # Active account with WO status (zombie signal — lender still reporting live)
            F.max(_i(F.col("_is_wo") & F.col("_is_active"))).alias("flag_wo_active"),

            # ── Product-specific WO flags ─────────────────────────────────────
            F.max(_i(F.col("_is_wo") & F.col("_is_pl"))).alias("flag_wo_pl"),
            F.max(_i(F.col("_is_wo") & F.col("_is_stpl"))).alias("flag_wo_stpl"),
            F.max(_i(F.col("_is_wo") & F.col("_is_usl"))).alias("flag_wo_usl"),

            # ── Count features ────────────────────────────────────────────────
            F.sum(_i(F.col("_is_wo"))).alias("count_wo_accounts"),
            F.sum(_i(F.col("_is_suit"))).alias("count_suit_accounts"),

            # ── Total write-off amount (total_write_off_am_4in) ───────────────
            F.sum(F.when(F.col("_is_wo"), F.col("_wo_total_am"))
            ).alias("total_charge_off_am"),

            F.max(F.when(F.col("_is_wo"), F.col("_wo_total_am"))
            ).alias("max_charge_off_am"),

            # ── Principal write-off amount (principal_write_off_am_4in) ───────
            F.sum(F.when(F.col("_is_wo"), F.col("_wo_princ_am"))
            ).alias("total_principal_wo_am"),

            F.max(F.when(F.col("_is_wo"), F.col("_wo_princ_am"))
            ).alias("max_principal_wo_am"),

            # Intermediates for ratios (dropped after derived features)
            F.sum(F.when(F.col("_is_wo"),      F.col("_wo_total_am"))).alias("_sum_wo"),
            F.sum(F.when(F.col("_is_wo"),      F.col("_wo_princ_am"))).alias("_sum_princ"),
            F.sum(F.when(F.col("_is_settled"), F.col("_settled_am"))).alias("_sum_settled"),

            # ── Settled / recovery amount (settled_am_4in) ────────────────────
            F.sum(F.when(F.col("_is_settled"), F.col("_settled_am"))
            ).alias("total_settled_am"),

            F.max(F.when(F.col("_is_settled"), F.col("_settled_am"))
            ).alias("max_settled_am"),

            # Partial recovery flag — needs per-account comparison
            # An account is partial if: is_settled AND settled_am < total_write_off_am
            F.max(_i(
                F.col("_is_settled") &
                F.col("_settled_am").isNotNull() &
                F.col("_wo_total_am").isNotNull() &
                (F.col("_settled_am") < F.col("_wo_total_am"))
            )).alias("flag_partial_recovery"),

            # ── Recency ───────────────────────────────────────────────────────
            F.min(F.when(F.col("_is_wo"),   F.col("_months_since_wo"))
            ).alias("months_since_last_wo"),

            F.min(F.when(F.col("_is_suit"), F.col("_months_since_suit"))
            ).alias("months_since_last_suit"),
        )

        # ── STEP 9: Derived ratio features ───────────────────────────────────
        feature_df = (
            feature_df

            # ratio_principal_to_total_wo = principal_wo / total_wo
            # High (~1.0) = mostly principal loss, minimal interest/fee
            # Low = large interest/fee component in written-off amount
            .withColumn(
                "ratio_principal_to_total_wo",
                F.when(
                    F.col("_sum_wo").isNotNull() & (F.col("_sum_wo") > 0),
                    (F.col("_sum_princ") / F.col("_sum_wo")).cast("double")
                ).otherwise(F.lit(None).cast("double"))
            )

            # ratio_settled_to_wo = settled / total_wo
            # Recovery rate: 1.0 = full recovery, 0.0 = no recovery
            # Low ratio = poor recovery behaviour (borrower did not pay back)
            .withColumn(
                "ratio_settled_to_wo",
                F.when(
                    F.col("_sum_wo").isNotNull() & (F.col("_sum_wo") > 0),
                    (F.col("_sum_settled") / F.col("_sum_wo")).cast("double")
                ).otherwise(F.lit(None).cast("double"))
            )

            # net_loss_am = total_wo - settled
            # Actual net credit loss after recovery
            .withColumn(
                "net_loss_am",
                F.when(
                    F.col("_sum_wo").isNotNull(),
                    (F.col("_sum_wo") - F.coalesce(F.col("_sum_settled"), F.lit(0.0))).cast("double")
                ).otherwise(F.lit(None).cast("double"))
            )

            .drop("_sum_wo", "_sum_princ", "_sum_settled")
        )

        self._log_done(feature_df)
        return feature_df