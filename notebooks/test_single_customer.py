# Databricks notebook source
# notebooks/test_single_customer.py
# =============================================================================
# Single Customer Feature Validation Test
# =============================================================================
# Purpose:
#   Validate all feature groups for ONE customer_scrub_key before running the
#   full pipeline as a job. Each group runs in its own cell so you can pinpoint
#   exactly which group/feature breaks.
#
# customer_scrub_key is unique per customer per scrub — no date needed.
#
# How to use:
#   1. Set customer_scrub_key widget
#   2. Run all cells top to bottom (or one cell at a time to debug)
#   3. Each group shows a transposed feature=value listing
#   4. Summary at the end shows PASS/FAIL per group
#
# READ-ONLY — nothing is written to Delta.
# =============================================================================

# COMMAND ----------
import sys
REPO_ROOT = "/Workspace/Repos/<your-username>/experian_feature_store"  # ← update
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

# COMMAND ----------
# ── Widget ────────────────────────────────────────────────────────────────────
dbutils.widgets.text("customer_scrub_key", "", "customer_scrub_key to test (required)")
TEST_KEY = dbutils.widgets.get("customer_scrub_key").strip()
assert TEST_KEY, "Set customer_scrub_key widget before running"
print(f"Testing: customer_scrub_key = '{TEST_KEY}'")

# COMMAND ----------
# ── Load raw tradeline rows ───────────────────────────────────────────────────
# customer_scrub_key is unique per customer per scrub — filter directly, no date needed
from config import config

tl_raw = (
    spark.table(config.TRADELINE_TABLE)
    .filter(F.col("customer_scrub_key") == TEST_KEY)
)

row_count = tl_raw.count()
print(f"Raw tradeline rows : {row_count}")
assert row_count > 0, f"No rows found for customer_scrub_key='{TEST_KEY}'"

# Read scrub_output_date from the data itself
SCRUB_DATE = tl_raw.select("scrub_output_date").first()["scrub_output_date"]
print(f"scrub_output_date  : {SCRUB_DATE}")

# Add party_code from master table
master = (
    spark.table(config.MASTER_TABLE)
    .select("customer_scrub_key", "party_code")
    .filter(F.col("customer_scrub_key") == TEST_KEY)
    .limit(1)
)
party_row = master.collect()
PARTY_CODE = party_row[0]["party_code"] if party_row else None
print(f"party_code         : {PARTY_CODE}")

tl_df = tl_raw.join(master, on="customer_scrub_key", how="left")

PK_COLS   = ["customer_scrub_key", "party_code", "scrub_output_date"]
AS_OF_COL = "scrub_output_date"

# COMMAND ----------
# ── Show raw tradeline rows ───────────────────────────────────────────────────
print(f"=== RAW TRADELINE ROWS for {TEST_KEY} ===")
tl_df.select(
    "acct_type_cd", "open_dt", "closed_dt",
    "orig_loan_am", "balance_am", "days_past_due",
    "m_sub_id", "suit_filed_willful_dflt", "written_off_and_settled_status"
).show(50, truncate=False)

# COMMAND ----------
# ── Helper: display features transposed ──────────────────────────────────────
def show_features(df: DataFrame, group_name: str):
    rows = df.collect()
    if not rows:
        print(f"  ⚠  {group_name}: no output rows")
        return
    data = rows[0].asDict()
    feat = {k: v for k, v in data.items() if k not in PK_COLS + [AS_OF_COL]}
    print(f"\n{'─'*70}")
    print(f"  {group_name}  ({len(feat)} features)")
    print(f"{'─'*70}")
    for k, v in feat.items():
        print(f"  {k:<55} = {v}")

def run_group(cls, label: str):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    try:
        result = cls().compute(tl_df, PK_COLS, AS_OF_COL)
        result = result.filter(F.col("customer_scrub_key") == TEST_KEY)
        show_features(result, label)
        print(f"\n  ✅ {label} OK")
        return result
    except Exception as e:
        import traceback
        print(f"  ❌ {label} FAILED — {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

# COMMAND ----------
from features.tradeline.grp01_portfolio_counts import PortfolioCountsFeatures
r01 = run_group(PortfolioCountsFeatures, "grp01_portfolio_counts")

# COMMAND ----------
from features.tradeline.grp02_loan_amounts import LoanAmountExposureFeatures
r02a = run_group(LoanAmountExposureFeatures, "grp02a_loan_amount_exposure")

# COMMAND ----------
from features.tradeline.grp02_loan_amounts import CreditCardLimitsFeatures
r02b = run_group(CreditCardLimitsFeatures, "grp02b_credit_card_limits")

# COMMAND ----------
from features.tradeline.grp02_loan_amounts import HighestCreditSignalsFeatures
r02c = run_group(HighestCreditSignalsFeatures, "grp02c_highest_credit_signals")

# COMMAND ----------
from features.tradeline.grp02_loan_amounts import LoanVolumeOverTimeFeatures
r02d = run_group(LoanVolumeOverTimeFeatures, "grp02d_loan_volume_over_time")

# COMMAND ----------
from features.tradeline.grp03_balances_utilization import OutstandingBalanceFeatures
r03a = run_group(OutstandingBalanceFeatures, "grp03a_outstanding_balance")

# COMMAND ----------
from features.tradeline.grp03_balances_utilization import CreditUtilizationFeatures
r03b = run_group(CreditUtilizationFeatures, "grp03b_credit_utilization")

# COMMAND ----------
from features.tradeline.grp04_bureau_vintage import BureauVintageFeatures
r04 = run_group(BureauVintageFeatures, "grp04_bureau_vintage")

# COMMAND ----------
from features.tradeline.grp05_lender_mix import LenderTypeMixFeatures
r05 = run_group(LenderTypeMixFeatures, "grp05_lender_mix")

# COMMAND ----------
from features.tradeline.grp06_recency_flags import RecencyCreditActivityFeatures
r06a = run_group(RecencyCreditActivityFeatures, "grp06a_recency_credit_activity")

# COMMAND ----------
from features.tradeline.grp06_recency_flags import CreditBehaviourFlagsFeatures
r06b = run_group(CreditBehaviourFlagsFeatures, "grp06b_credit_behaviour_flags")

# COMMAND ----------
from features.tradeline.grp07_delinquency import DelinquencyDPDFeatures
r07 = run_group(DelinquencyDPDFeatures, "grp07_delinquency_dpd")

# COMMAND ----------
from features.tradeline.grp08_payment_repayment import PaymentBehaviourFeatures
r08a = run_group(PaymentBehaviourFeatures, "grp08a_payment_behaviour")

# COMMAND ----------
from features.tradeline.grp08_payment_repayment import RepaymentRatioFeatures
r08b = run_group(RepaymentRatioFeatures, "grp08b_repayment_ratio")

# COMMAND ----------
from features.tradeline.grp09_obligations import ObligationsFeatures
r09 = run_group(ObligationsFeatures, "grp09_obligations")

# COMMAND ----------
from features.tradeline.grp10_severe_risk import WriteoffsSevereRiskFeatures
r10 = run_group(WriteoffsSevereRiskFeatures, "grp10_severe_risk")

# COMMAND ----------
# ── Enquiries ─────────────────────────────────────────────────────────────────
# Enquiry table is filtered by customer_scrub_key + inq_date <= scrub_output_date
from features.enquiry.grp12_enquiries import CreditEnquiriesFeatures

print(f"\n{'='*70}")
print("  grp12_enquiries")
print(f"{'='*70}")
try:
    enq_df = (
        spark.table(config.ENQUIRY_TABLE)
        .filter(F.col("customer_scrub_key") == TEST_KEY)
        .filter(F.to_date(F.col("inq_date")) <= F.lit(str(SCRUB_DATE)).cast("date"))
        .join(master, on="customer_scrub_key", how="left")
        .withColumn("scrub_output_date", F.lit(str(SCRUB_DATE)).cast("date"))
    )
    enq_count = enq_df.count()
    print(f"  Enquiry rows: {enq_count}")
    if enq_count > 0:
        print("  Raw enquiry rows:")
        enq_df.select("inq_date", "inq_purp_cd", "m_sub_id", "amount").show(20, False)
        r12 = CreditEnquiriesFeatures().compute(enq_df, PK_COLS, AS_OF_COL)
        r12 = r12.filter(F.col("customer_scrub_key") == TEST_KEY)
        show_features(r12, "grp12_enquiries")
        print("\n  ✅ grp12_enquiries OK")
    else:
        r12 = None
        print("  ℹ No enquiry rows — enquiry features will be NULL")
except Exception as e:
    import traceback
    r12 = None
    print(f"  ❌ grp12_enquiries FAILED — {type(e).__name__}: {e}")
    traceback.print_exc()

# COMMAND ----------
# ── Summary ───────────────────────────────────────────────────────────────────
results = {
    "grp01_portfolio_counts":        r01,
    "grp02a_loan_amount_exposure":   r02a,
    "grp02b_credit_card_limits":     r02b,
    "grp02c_highest_credit_signals": r02c,
    "grp02d_loan_volume_over_time":  r02d,
    "grp03a_outstanding_balance":    r03a,
    "grp03b_credit_utilization":     r03b,
    "grp04_bureau_vintage":          r04,
    "grp05_lender_mix":              r05,
    "grp06a_recency_credit_activity":r06a,
    "grp06b_credit_behaviour_flags": r06b,
    "grp07_delinquency_dpd":         r07,
    "grp08a_payment_behaviour":      r08a,
    "grp08b_repayment_ratio":        r08b,
    "grp09_obligations":             r09,
    "grp10_severe_risk":             r10,
    "grp12_enquiries":               r12,
}

passed = sum(1 for v in results.values() if v is not None)
failed = [k for k, v in results.items() if v is None]

print("\n" + "="*70)
print("  SUMMARY")
print("="*70)
print(f"  customer_scrub_key : {TEST_KEY}")
print(f"  party_code         : {PARTY_CODE}")
print(f"  scrub_output_date  : {SCRUB_DATE}")
print(f"  tradeline rows     : {row_count}")
print()
for name, result in results.items():
    print(f"  {'✅ PASS' if result is not None else '❌ FAIL'}  {name}")
print()
print(f"  Passed : {passed}/{len(results)}")
if failed:
    print(f"  Failed : {len(failed)}/{len(results)}")
    print(f"  Fix    : {failed}")
else:
    print("  All groups passed ✅ — safe to run the pipeline job")

# COMMAND ----------
# ── Optional: wide join of all passing groups ─────────────────────────────────
# Uncomment to join everything into one wide row for cross-checking
#
# from functools import reduce
# passing = [r for r in results.values() if r is not None]
# if passing:
#     join_cols = PK_COLS + [AS_OF_COL]
#     wide = reduce(lambda a, b: a.join(b, on=join_cols, how="left"), passing)
#     print(f"Wide row: {len(wide.columns)} features")
#     display(wide.limit(1))
