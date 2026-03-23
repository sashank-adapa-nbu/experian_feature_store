# features/tradeline/grp04_bureau_vintage.py
# Merged from: cat06
# Removed: count_active_accounts (= active_loans in grp01)

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

PL_CODE  = "123"
CC_CODES = {"5", "213", "214", "220", "224", "225"}    # All CCs incl. 220 (Secured CC — treated as CC/unsecured)

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


class BureauVintageFeatures(TradelineFeatureBase):
    """
    Category 06: Bureau Vintage / Credit History Age

    All ages are in MONTHS from open_dt to as_of_dt.
    Only accounts where open_dt <= as_of_dt are considered (no leakage).
    Active = open_dt <= as_of_dt AND (closed_dt IS NULL OR closed_dt > as_of_dt)

    ── Requested features ────────────────────────────────────────────────────

    meanbureauage
        Mean age (months) across ALL tradelines ever opened

    meanbureauageonactive
        Mean age (months) across ACTIVE tradelines only

    maxbureauage_gt_100k
        Max age (months) of accounts where orig_loan_am > 100K

    meanbureauage_gt_100k
        Mean age (months) of accounts where orig_loan_am > 100K

    meanbureauage_unsecured
        Mean age (months) of unsecured accounts

    meanbureauage_pl
        Mean age (months) of Personal Loan accounts

    meanbureauage_cc
        Mean age (months) of Credit Card accounts

    bureage_overall
        Max age (months) across ALL tradelines — proxy for credit history length

    bureage_pl
        Max age (months) of Personal Loan accounts

    bureage_cc
        Max age (months) of Credit Card accounts

    burage_unsecured
        Max age (months) of unsecured accounts

    bureaugae_gt_100k
        Max age (months) of accounts with orig_loan_am > 100K

    ── Additional risk features ──────────────────────────────────────────────

    minbureauage
        Min age (months) across ALL tradelines — age of the newest account.
        Captures recent credit-seeking behaviour.

    minbureauage_active
        Min age (months) across ACTIVE tradelines — newest live account.
        Strong recency signal.

    count_accounts_gt_60m
        Count of accounts older than 60 months (5 years).
        Thick-file / seasoned borrower indicator.

    count_accounts_gt_24m
        Count of accounts older than 24 months.
        Establishes minimum credit history threshold.

    meanbureauage_active_pl
        Mean age of ACTIVE PL accounts.
        Distinguishes seasoned PL borrowers from new ones.

    age_range_months
        age_oldest_account - age_newest_account.
        Width of credit history window.
        Narrow range = credit concentrated in a short period.

    ratio_mean_to_max_age
        meanbureauage / bureage_overall.
        Close to 1 = accounts opened at similar time (thin history).
        Close to 0 = long history with many recent accounts.

    count_active_accounts
        Count of active accounts (open_dt <= as_of_dt, not yet closed).
        Denominator for active-based ratios.
    """

    CATEGORY = "grp04_bureau_vintage"

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

        # ── STEP 2: Age in months ─────────────────────────────────────────────
        # months_between(as_of_dt, open_dt) → positive when open_dt < as_of_dt
        # Set to NULL for future accounts (open_dt > as_of_dt) — no leakage
        df = df.withColumn(
            "_age_months",
            F.when(
                F.col("_open_dt") <= F.col("_as_of_dt"),
                F.months_between(F.col("_as_of_dt"), F.col("_open_dt"))
            ).otherwise(F.lit(None).cast("double"))
        )

        # ── STEP 3: Active flag — point-in-time, no leakage ──────────────────
        # Account must have OPENED before as_of_dt AND not yet closed at as_of_dt
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

        # ── STEP 5: Normalise acct_type_cd ───────────────────────────────────
        df = df.withColumn("_acct_type", F.trim(F.col("acct_type_cd").cast("string")))

        # ── STEP 6: Product type flags ────────────────────────────────────────
        df = (
            df
            .withColumn("_is_pl",
                F.col("_acct_type") == PL_CODE)
            .withColumn("_is_cc",
                F.col("_acct_type").isin(CC_CODES))
            .withColumn("_is_unsecured",
                ~F.col("_acct_type").isin(SECURED_CODES))
            .withColumn("_is_gt_100k",
                F.col("_loan_am").isNotNull() & (F.col("_loan_am") > 100000))
        )

        # ── STEP 7: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # ── Mean age features ─────────────────────────────────────────────

            # Mean age — all accounts (open_dt <= as_of_dt)
            F.mean(F.col("_age_months")).alias("meanbureauage"),

            # Mean age — active accounts only
            F.mean(
                F.when(F.col("_is_active") == 1, F.col("_age_months"))
            ).alias("meanbureauageonactive"),

            # Mean age — accounts with loan amount > 100K
            F.mean(
                F.when(F.col("_is_gt_100k"), F.col("_age_months"))
            ).alias("meanbureauage_gt_100k"),

            # Mean age — unsecured accounts
            F.mean(
                F.when(F.col("_is_unsecured"), F.col("_age_months"))
            ).alias("meanbureauage_unsecured"),

            # Mean age — PL accounts
            F.mean(
                F.when(F.col("_is_pl"), F.col("_age_months"))
            ).alias("meanbureauage_pl"),

            # Mean age — CC accounts
            F.mean(
                F.when(F.col("_is_cc"), F.col("_age_months"))
            ).alias("meanbureauage_cc"),

            # Mean age — active PL accounts
            F.mean(
                F.when(F.col("_is_pl") & (F.col("_is_active") == 1), F.col("_age_months"))
            ).alias("meanbureauage_active_pl"),

            # ── Max age features (bureau history length) ──────────────────────

            # Max age — all accounts (= length of credit history)
            F.max(F.col("_age_months")).alias("bureage_overall"),

            # Max age — PL accounts
            F.max(
                F.when(F.col("_is_pl"), F.col("_age_months"))
            ).alias("bureage_pl"),

            # Max age — CC accounts
            F.max(
                F.when(F.col("_is_cc"), F.col("_age_months"))
            ).alias("bureage_cc"),

            # Max age — unsecured accounts
            F.max(
                F.when(F.col("_is_unsecured"), F.col("_age_months"))
            ).alias("burage_unsecured"),

            # Max age — accounts with loan amount > 100K
            F.max(
                F.when(F.col("_is_gt_100k"), F.col("_age_months"))
            ).alias("maxbureauage_gt_100k"),



            # ── Min age features (recency signals) ────────────────────────────

            # Min age — all accounts (age of newest account)
            F.min(F.col("_age_months")).alias("minbureauage"),

            # Min age — active accounts only
            F.min(
                F.when(F.col("_is_active") == 1, F.col("_age_months"))
            ).alias("minbureauage_active"),

            # ── Count features ────────────────────────────────────────────────

            # Count of seasoned accounts (> 60 months = 5 years)
            F.sum(
                F.when(F.col("_age_months") > 60, F.lit(1)).otherwise(F.lit(0))
            ).alias("count_accounts_gt_60m"),

            # Count of established accounts (> 24 months)
            F.sum(
                F.when(F.col("_age_months") > 24, F.lit(1)).otherwise(F.lit(0))
            ).alias("count_accounts_gt_24m"),

            # count_active_accounts removed — use active_loans in grp01_portfolio_counts
        )

        # ── STEP 8: Derived ratio features (post-aggregation) ─────────────────

        # age_range_months — width of credit history window
        # Narrow = credit concentrated in short period (higher risk)
        # Wide   = long-standing borrower with varied history
        feature_df = feature_df.withColumn(
            "age_range_months",
            F.when(
                F.col("bureage_overall").isNotNull() & F.col("minbureauage").isNotNull(),
                F.col("bureage_overall") - F.col("minbureauage")
            ).otherwise(F.lit(None).cast("double"))
        )

        # ratio_mean_to_max_age
        # Close to 1 = accounts all opened around same time (thin/concentrated history)
        # Close to 0 = long history with many recent accounts
        feature_df = feature_df.withColumn(
            "ratio_mean_to_max_age",
            F.when(
                F.col("bureage_overall").isNotNull() & (F.col("bureage_overall") > 0),
                F.col("meanbureauage") / F.col("bureage_overall")
            ).otherwise(F.lit(None).cast("double"))
        )

        self._log_done(feature_df)
        return feature_df