# features/tradeline/grp09_obligations.py  [OPTIMISED]
# =============================================================================
# Group 09 — Obligations
# =============================================================================
# CRITICAL FIX: replaced 73-WHEN nested _bal_slot() / _pmt_slot() chains
# with F.array() + element_at() indexing. Same plan fix as grp07/08.
# =============================================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List

from features.tradeline.base import TradelineFeatureBase
from core.logger import get_logger
from core.date_utils import parse_date
from core.utils import build_history_array, resolve_slot, resolve_slot_at_asof

logger = get_logger(__name__)

N_BAL = 36
N_PMT = 12    # obligations only needs 12m payment history

BAL_HIST = [f"balance_am_{str(i).zfill(2)}"        for i in range(1, N_BAL + 1)]
PMT_HIST = [f"actual_payment_am_{str(i).zfill(2)}" for i in range(1, N_PMT + 1)]


class ObligationsFeatures(TradelineFeatureBase):
    """Group 09: Obligations (array-optimised)."""

    CATEGORY = "grp09_obligations"

    def compute(self, df: DataFrame, pk_cols: List[str], as_of_col: str) -> DataFrame:
        self._log_start(mode="dynamic", date="batch")
        group_cols = pk_cols + [as_of_col]

        df = (
            df
            .withColumn("_bal_dt",    parse_date("balance_dt"))
            .withColumn("_as_of_dt",  parse_date(as_of_col))
            .withColumn("_open_dt",   parse_date("open_dt"))
            .withColumn("_closed_dt", parse_date("closed_dt"))
        )

        # grp09 uses grp03a-style month_diff: months_between(balance_dt, as_of_dt)
        # _month_diff <= 0 when as_of >= balance_dt (normal case) → use slot 0
        df = df.withColumn(
            "_month_diff",
            F.ceil(F.months_between(F.col("_bal_dt"), F.col("_as_of_dt"))).cast("int")
        )

        # For windowed obligations at offset k months:
        # _md (as_of - balance_dt direction, positive) used by resolve_slot
        df = df.withColumn(
            "_md",
            (-F.col("_month_diff")).cast("int")
        )

        df = df.withColumn("_acct", F.trim(F.col("acct_type_cd").cast("string")))
        df = (
            df
            .withColumn("_orig_am",
                F.when(F.col("orig_loan_am") > 0, F.col("orig_loan_am").cast("double"))
                 .otherwise(F.lit(None).cast("double")))
            .withColumn("_bal_am",
                F.when(F.col("balance_am").cast("double") >= 0, F.col("balance_am").cast("double"))
                 .otherwise(F.lit(None).cast("double")))
        )

        # ── Imputed EMI ───────────────────────────────────────────────────────
        imputed_emi = (
            F.when(F.col("_acct") == "123",                                  F.lit(0.03)      * F.col("_orig_am"))
             .when(F.col("_acct").isin({"5","213","214","220","224","225"}),  F.lit(0.05)      * F.col("_orig_am"))
             .when(F.col("_acct").isin({"58","195"}),                         F.lit(0.01)      * F.col("_orig_am"))
             .when(F.col("_acct").isin({"173","189"}),                        F.lit(0.05)      * F.col("_orig_am"))
             .when(F.col("_acct") == "47",                                    F.lit(0.025)     * F.col("_orig_am"))
             .when(F.col("_acct").isin({"177","199"}),                        F.lit(0.13/12)   * F.col("_bal_am"))
             .otherwise(                                                       F.lit(0.01)      * F.col("_orig_am"))
        )
        # ── EMI sanity check ─────────────────────────────────────────────────
        # Accept raw emi only when it is present, positive, AND plausible:
        #   emi <= 50% of orig_loan_am
        # Cases where emi > 50% of orig_loan_am indicate a data quality issue
        # (e.g. balance mis-stored in the emi field). Fall back to imputed_emi
        # for those rows so that obligations are not distorted by bad data.
        # When orig_loan_am is null we cannot validate, so accept raw emi as-is.
        df = df.withColumn(
            "_final_emi",
            F.when(
                F.col("emi").isNotNull() & (F.col("emi").cast("double") > 0) &
                (
                    F.col("_orig_am").isNull() |
                    (F.col("emi").cast("double") <= F.lit(0.9) * F.col("_orig_am"))
                ),
                F.col("emi").cast("double")
            ).otherwise(imputed_emi)
        )

        # ── Build balance array ONCE ──────────────────────────────────────────
        df = build_history_array(df, "balance_am", BAL_HIST, "_bal_arr", clean_negative=False)

        # ── Build payment array ONCE (12m only) ───────────────────────────────
        df = build_history_array(df, "actual_payment_am", PMT_HIST, "_pmt_arr", clean_negative=True)

        # ── Resolve balance at as_of_dt ───────────────────────────────────────
        df = df.withColumn("_bal_asof", resolve_slot_at_asof("_bal_arr", "_month_diff"))

        # ── Current obligation filter ─────────────────────────────────────────
        df = df.withColumn(
            "_is_obligating",
            F.when(
                F.col("_orig_am").isNotNull() & F.col("_bal_asof").isNotNull() &
                (F.col("_bal_asof") >= F.lit(0.08) * F.col("_orig_am")) &
                (F.col("_open_dt") <= F.col("_as_of_dt")) &
                (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > F.col("_as_of_dt"))),
                F.lit(True)
            ).otherwise(F.lit(False))
        )

        # ── Obligation at offset k months (array-indexed) ─────────────────────
        def obligating_at_offset(k: int) -> F.Column:
            """Balance k months before as_of_dt, using array indexing."""
            # slot_idx for offset k from as_of: slot_idx = k - _md
            # where _md = ceil(months_between(as_of, bal_dt)) = -_month_diff
            bal_k     = resolve_slot("_bal_arr", k, "_md")
            slot_dt   = F.add_months(F.col("_as_of_dt"), -k)
            was_open  = (
                (F.col("_open_dt") <= slot_dt) &
                (F.col("_closed_dt").isNull() | (F.col("_closed_dt") > slot_dt))
            )
            return F.when(
                F.col("_orig_am").isNotNull() & bal_k.isNotNull() &
                (bal_k >= F.lit(0.08) * F.col("_orig_am")) & was_open,
                F.col("_final_emi")
            ).otherwise(F.lit(0.0))

        # ── Payment slots (12m, array-indexed) ───────────────────────────────
        pmt_slots_12m = [resolve_slot("_pmt_arr", k) for k in range(12)]

        feature_df = df.groupBy(group_cols).agg(

            F.sum(F.when(F.col("_is_obligating"), F.col("_final_emi")).otherwise(F.lit(0.0))).alias("obligations"),
            F.sum(F.when(F.col("_is_obligating"), F.lit(1)).otherwise(F.lit(0))).alias("count_obligating_accounts"),

            F.sum(obligating_at_offset(6)).alias("obligations_6m"),
            F.sum(obligating_at_offset(12)).alias("obligations_12m"),
            F.sum(obligating_at_offset(24)).alias("obligations_24m"),

            F.max(F.greatest(*pmt_slots_12m)).alias("Max_obligations_paid_last12months"),
            F.mean(F.coalesce(*pmt_slots_12m)).alias("mean_obligations_paid_last12months"),

            # Intermediates for ratio
            F.sum(F.when(F.col("_is_obligating"), F.col("_final_emi")).otherwise(F.lit(0.0))).alias("_oblig_now"),
            F.sum(obligating_at_offset(12)).alias("_oblig_12m"),

            # Product-level obligations
            F.sum(F.when(F.col("_is_obligating") & (F.col("_acct") == "123"),       F.col("_final_emi")).otherwise(F.lit(0.0))).alias("obligations_pl"),
            F.sum(F.when(F.col("_is_obligating") & F.col("_acct").isin({"58","195"}), F.col("_final_emi")).otherwise(F.lit(0.0))).alias("obligations_hl"),
            F.sum(F.when(F.col("_is_obligating") & F.col("_acct").isin({"5","213","214","220","224","225"}), F.col("_final_emi")).otherwise(F.lit(0.0))).alias("obligations_cc"),
        )

        feature_df = feature_df.withColumn(
            "ratio_obligations_change_12m",
            F.when(
                F.col("_oblig_12m").isNotNull() & (F.col("_oblig_12m") > 0),
                F.col("_oblig_now") / F.col("_oblig_12m")
            ).otherwise(F.lit(None).cast("double"))
        ).drop("_oblig_now", "_oblig_12m")

        self._log_done(feature_df)
        return feature_df