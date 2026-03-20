# features/tradeline/cat18_obligations.py
# =============================================================================
# Category 18 — Obligations
# =============================================================================
# Source table  : experian_tradeline_segment
#
# ── IMPUTED EMI (final_emi) ───────────────────────────────────────────────────
#   When EMI column is missing/zero, we impute using product-specific rate × balance:
#
#   acct_type_cd = 123 (PL)           → 0.03  × orig_loan_am
#   acct_type_cd IN (5,213,214,220,224,225) (CC) → 0.05  × orig_loan_am
#   acct_type_cd IN (58,195) (HL)     → 0.01  × orig_loan_am
#   acct_type_cd IN (173,189) (2W/CL) → 0.05  × orig_loan_am
#   acct_type_cd = 47  (AL)           → 0.025 × orig_loan_am
#   acct_type_cd IN (177,199) (BL)    → (0.13/12) × balance_am  (revolving — use current balance)
#   else                              → 0.01  × orig_loan_am
#
# ── CURRENT OBLIGATION FILTER ────────────────────────────────────────────────
#   Account is "still obligating" when:
#     balance_am >= 0.08 × orig_loan_am   (at least 8% of loan still outstanding)
#     AND closed_dt IS NULL               (account not closed)
#
#   This filters out near-fully-repaid or closed accounts from obligation sum.
#
# ── WINDOWED OBLIGATION (time-based) ─────────────────────────────────────────
#   Uses balance_am slot resolution (same as cat15) via month_diff from balance_dt.
#   For past windows: balance_at_slot / orig_loan_am >= 0.08 AND account was open
#   at that point in time.
#
#   month_diff = round(months_between(as_of_dt, balance_dt))
#   balance_at_slot = BALANCE_AM (slot 0) or BALANCE_AM_NN (slots 1-36)
#
# ── COLUMNS USED ─────────────────────────────────────────────────────────────
#   acct_type_cd, orig_loan_am, balance_am, balance_am_01..36
#   closed_dt, open_dt, balance_dt
#   actual_payment_am, actual_payment_am_01..12 (for Max_obligations_paid)
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger
from core.date_utils import parse_date

logger = get_logger(__name__)


N_BAL_HISTORY = 36
BAL_COLS = [f"balance_am_{str(i).zfill(2)}" for i in range(1, N_BAL_HISTORY + 1)]

N_PMT_HISTORY = 36
PMT_COLS = [f"actual_payment_am_{str(i).zfill(2)}" for i in range(1, N_PMT_HISTORY + 1)]


class ObligationsFeatures(TradelineFeatureBase):
    """
    Category 18: Obligations

    final_emi = imputed EMI per product type × orig_loan_am or balance_am
    obligations = SUM(final_emi) for accounts where balance >= 8% of orig_loan_am
                  AND closed_dt IS NULL, at as_of_dt

    Windowed obligations use balance slot resolution (same as cat15).
    """

    CATEGORY = "grp09_obligations"

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER: resolve balance at offset k from as_of_dt
    # Same slot logic as cat15: actual_slot = -month_diff + k (inverted sign)
    # month_diff = round(months_between(as_of_dt, balance_dt))
    # +ve → as_of ahead → gap → use BALANCE_AM
    # -ve → as_of behind → use BALANCE_AM_{-month_diff}
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _bal_slot(k: int = 0) -> F.Column:
        """Resolve balance_am at window offset k (0 = as_of_dt)."""
        expr = F.lit(None).cast("double")
        for m in range(-N_BAL_HISTORY, N_BAL_HISTORY + 1):
            slot = k - m   # actual slot index (0=balance_am, 1=balance_am_01, ...)
            if m > 0:
                # gap: as_of ahead of balance_dt → use BALANCE_AM as best available
                raw = F.when(F.col("balance_am").cast("double") >= 0,
                             F.col("balance_am").cast("double")) \
                       .otherwise(F.lit(None).cast("double"))
            elif slot == 0:
                raw = F.when(F.col("balance_am").cast("double") >= 0,
                             F.col("balance_am").cast("double")) \
                       .otherwise(F.lit(None).cast("double"))
            elif 1 <= slot <= N_BAL_HISTORY:
                raw = F.when(F.col(BAL_COLS[slot - 1]) >= 0,
                             F.col(BAL_COLS[slot - 1]).cast("double")) \
                       .otherwise(F.lit(None).cast("double"))
            else:
                raw = F.lit(None).cast("double")
            expr = F.when(F.col("_md") == m, raw).otherwise(expr)
        return expr

    @staticmethod
    def _pmt_slot(k: int = 0) -> F.Column:
        """Resolve actual_payment_am at window offset k from as_of_dt."""
        expr = F.lit(None).cast("double")
        for m in range(-N_PMT_HISTORY, N_PMT_HISTORY + 1):
            slot = k - m
            if m > 0:
                raw = F.when(F.col("actual_payment_am").cast("double") >= 0,
                             F.col("actual_payment_am").cast("double")) \
                       .otherwise(F.lit(None).cast("double"))
            elif slot == 0:
                raw = F.when(F.col("actual_payment_am").cast("double") >= 0,
                             F.col("actual_payment_am").cast("double")) \
                       .otherwise(F.lit(None).cast("double"))
            elif 1 <= slot <= N_PMT_HISTORY:
                raw = F.when(F.col(PMT_COLS[slot - 1]) >= 0,
                             F.col(PMT_COLS[slot - 1]).cast("double")) \
                       .otherwise(F.lit(None).cast("double"))
            else:
                raw = F.lit(None).cast("double")
            expr = F.when(F.col("_md") == m, raw).otherwise(expr)
        return expr

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        # ── STEP 1: Parse dates ───────────────────────────────────────────────

        df = (
            df
            .withColumn("_bal_dt",    parse_date("balance_dt"))
            .withColumn("_as_of_dt",  parse_date(as_of_col))
            .withColumn("_open_dt",   parse_date("open_dt"))
            .withColumn("_closed_dt", parse_date("closed_dt"))
        )

        # ── STEP 2: month_diff ────────────────────────────────────────────────
        df = df.withColumn(
            "_md",
            F.ceil(F.months_between(F.col("_as_of_dt"), F.col("_bal_dt"))).cast("int")
        )

        # ── STEP 3: Normalise acct_type_cd and clean amounts ─────────────────
        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))
        df = (
            df
            .withColumn("_orig_am",
                F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
                 .otherwise(F.lit(None).cast("double")))
            .withColumn("_bal_am",
                F.when(F.col("balance_am") >= 0, F.col("balance_am").cast("double"))
                 .otherwise(F.lit(None).cast("double")))
        )

        # ── STEP 4: Imputed EMI (final_emi) ───────────────────────────────────
        # Precedence: use actual EMI column if > 0, else impute by product rule
        imputed_emi = (
            F.when(F.col("_acct") == "123",
                   F.lit(0.03)         * F.col("_orig_am"))          # PL
             .when(F.col("_acct").isin({"5", "213", "214", "220", "224", "225"}),
                   F.lit(0.05)         * F.col("_orig_am"))          # CC (all codes incl. 220)
             .when(F.col("_acct").isin({"58", "195"}),
                   F.lit(0.01)         * F.col("_orig_am"))          # HL
             .when(F.col("_acct").isin({"173", "189"}),
                   F.lit(0.05)         * F.col("_orig_am"))          # 2W / Consumer
             .when(F.col("_acct") == "47",
                   F.lit(0.025)        * F.col("_orig_am"))          # Auto
             .when(F.col("_acct").isin({"177", "199"}),
                   F.lit(0.13 / 12)    * F.col("_bal_am"))           # BL / Agri (revolving)
             .otherwise(F.lit(0.01)   * F.col("_orig_am"))           # default
        )

        df = df.withColumn(
            "_final_emi",
            F.when(
                F.col("emi").isNotNull() & (F.col("emi").cast("double") > 0),
                F.col("emi").cast("double")
            ).otherwise(imputed_emi)
        )

        # ── STEP 5: Resolve balance at as_of_dt ──────────────────────────────
        df = df.withColumn("_bal_asof", self._bal_slot(k=0))

        # ── STEP 6: Obligation filter at as_of_dt ────────────────────────────
        # Account is "obligating" when:
        #   balance_at_asof >= 8% of orig_loan_am  (still significantly outstanding)
        #   AND account was open at as_of_dt (not yet closed)
        #   Point-in-time: closed_dt IS NULL OR closed_dt > as_of_dt (retro safe)
        df = df.withColumn(
            "_is_obligating",
            F.when(
                F.col("_orig_am").isNotNull() &
                F.col("_bal_asof").isNotNull() &
                (F.col("_bal_asof") >= F.lit(0.08) * F.col("_orig_am")) &
                (F.col("_open_dt") <= F.col("_as_of_dt")) &
                (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt"))),
                F.lit(True)
            ).otherwise(F.lit(False))
        )

        # ── STEP 7: Windowed obligation filter ───────────────────────────────
        # For past time windows: check if account was obligating at that point.
        # Point-in-time closed_dt: account was open at (as_of_dt - k months)
        # = closed_dt IS NULL OR closed_dt > (as_of_dt - k months)
        # Approximated as: closed_dt > as_of_dt (conservative — if closed after
        # as_of it was open at all prior slots too; if closed before as_of it was
        # already excluded at as_of level via _is_obligating).
        # For strictness at each slot: use add_months(as_of_dt, -k) as cutoff.
        def obligating_at_slot(k: int) -> F.Column:
            bal_k = self._bal_slot(k)
            slot_dt = F.add_months(F.col("_as_of_dt"), -k)   # date k months before as_of
            was_open = (
                (F.col("_open_dt") <= slot_dt) &
                (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > slot_dt))
            )
            return F.when(
                F.col("_orig_am").isNotNull() &
                bal_k.isNotNull() &
                (bal_k >= F.lit(0.08) * F.col("_orig_am")) &
                was_open,
                F.col("_final_emi")
            ).otherwise(F.lit(0.0))

        # ── STEP 8: Resolve payment at as_of for Max_obligations_paid ────────
        # Max payment across last 12m window
        pmt_slots_12m = [self._pmt_slot(k) for k in range(12)]

        # ── STEP 9: Aggregate ─────────────────────────────────────────────────
        feature_df = df.groupBy(group_cols).agg(

            # ── Current obligations (at as_of_dt) ────────────────────────────
            # obligations = SUM(final_emi) for obligating accounts
            F.sum(
                F.when(F.col("_is_obligating"), F.col("_final_emi"))
                 .otherwise(F.lit(0.0))
            ).alias("obligations"),

            # Count of obligating accounts
            F.sum(F.when(F.col("_is_obligating"), F.lit(1)).otherwise(F.lit(0))
            ).alias("count_obligating_accounts"),

            # ── Windowed obligations ──────────────────────────────────────────
            # obligations_6m: SUM of final_emi for accounts obligating 6m ago
            F.sum(obligating_at_slot(6)).alias("obligations_6m"),

            # obligations_12m: SUM of final_emi for accounts obligating 12m ago
            F.sum(obligating_at_slot(12)).alias("obligations_12m"),

            # obligations_24m
            F.sum(obligating_at_slot(24)).alias("obligations_24m"),

            # ── Max obligations paid in last 12m ─────────────────────────────
            # Max single-month total payment across all tradelines in last 12m
            # For each month k, sum payments across all tradelines, then take max
            # Here we take MAX of per-tradeline payment slots → customer-level max
            F.max(
                F.greatest(*pmt_slots_12m)
            ).alias("Max_obligations_paid_last12months"),

            # Mean payment in last 12m (all tradelines)
            F.mean(
                F.coalesce(*pmt_slots_12m)
            ).alias("mean_obligations_paid_last12months"),

            # ── Obligation trend features ─────────────────────────────────────
            # For ratio: current vs 12m ago
            F.sum(
                F.when(F.col("_is_obligating"), F.col("_final_emi")).otherwise(F.lit(0.0))
            ).alias("_oblig_now"),

            F.sum(obligating_at_slot(12)).alias("_oblig_12m"),

            # ── Product-level obligations (point-in-time via _is_obligating) ──
            # PL obligations at as_of
            F.sum(
                F.when(F.col("_is_obligating") & (F.col("_acct") == "123"),
                       F.col("_final_emi")).otherwise(F.lit(0.0))
            ).alias("obligations_pl"),

            # HL obligations at as_of
            F.sum(
                F.when(F.col("_is_obligating") & F.col("_acct").isin({"58", "195"}),
                       F.col("_final_emi")).otherwise(F.lit(0.0))
            ).alias("obligations_hl"),

            # CC obligations at as_of
            F.sum(
                F.when(F.col("_is_obligating") & F.col("_acct").isin(
                    {"5", "213", "214", "220", "224", "225"}),  # All CCs incl. 220 (Secured CC — treated as CC/unsecured)
                       F.col("_final_emi")).otherwise(F.lit(0.0))
            ).alias("obligations_cc"),
        )

        # ── STEP 10: Derived features ─────────────────────────────────────────

        # ratio_obligations_change_12m
        # obligations_now / obligations_12m_ago
        # > 1 → obligations have grown (more debt taken on) → risk signal
        # < 1 → obligations reduced (repayments progressing)
        feature_df = feature_df.withColumn(
            "ratio_obligations_change_12m",
            F.when(
                F.col("_oblig_12m").isNotNull() & (F.col("_oblig_12m") > 0),
                F.col("_oblig_now") / F.col("_oblig_12m")
            ).otherwise(F.lit(None).cast("double"))
        ).drop("_oblig_now", "_oblig_12m")

        self._log_done(feature_df)
        return feature_df
