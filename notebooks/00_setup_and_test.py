# Databricks notebook source
# notebooks/00_setup_and_test.py
# =============================================================================
# Setup & Import Validation
# Run this first to confirm the repo is correctly installed and importable.
# =============================================================================

# COMMAND ----------
import sys
REPO_ROOT = "/Workspace/Repos/<your-username>/experian_feature_store"  # ← update
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print(f"REPO_ROOT: {REPO_ROOT}")

# COMMAND ----------
# ── 1. Config ─────────────────────────────────────────────────────────────────
print("Testing config...")
from config import config
print(f"  TRADELINE_TABLE : {config.TRADELINE_TABLE}")
print(f"  ENQUIRY_TABLE   : {config.ENQUIRY_TABLE}")
print(f"  MASTER_TABLE    : {config.MASTER_TABLE}")
print(f"  SCRUB_PK_COLS   : {config.SCRUB_PK_COLS}")
print(f"  RETRO_PK_COLS   : {config.RETRO_PK_COLS}")
print("  ✅ config OK")

# COMMAND ----------
# ── 2. Core utilities ─────────────────────────────────────────────────────────
print("Testing core...")
from core.logger import get_logger
logger = get_logger("setup_test")
logger.info("Logger works")
print("  ✅ core OK")

# COMMAND ----------
# ── 3. Feature base classes ───────────────────────────────────────────────────
print("Testing base classes...")
from features.tradeline.base import TradelineFeatureBase
from features.enquiry.base   import EnquiryFeatureBase
print(f"  TradelineFeatureBase : {TradelineFeatureBase}")
print(f"  EnquiryFeatureBase   : {EnquiryFeatureBase}")
print("  ✅ base classes OK")

# COMMAND ----------
# ── 4. Feature groups ─────────────────────────────────────────────────────────
print("Testing feature group imports...")

from features.tradeline.grp01_portfolio_counts      import PortfolioCountsFeatures
from features.tradeline.grp02_loan_amounts          import (LoanAmountExposureFeatures,
                                                             CreditCardLimitsFeatures,
                                                             HighestCreditSignalsFeatures,
                                                             LoanVolumeOverTimeFeatures)
from features.tradeline.grp03_balances_utilization  import (OutstandingBalanceFeatures,
                                                             CreditUtilizationFeatures)
from features.tradeline.grp04_bureau_vintage        import BureauVintageFeatures
from features.tradeline.grp05_lender_mix            import LenderTypeMixFeatures
from features.tradeline.grp06_recency_flags         import (RecencyCreditActivityFeatures,
                                                             CreditBehaviourFlagsFeatures)
from features.tradeline.grp07_delinquency           import DelinquencyDPDFeatures
from features.tradeline.grp08_payment_repayment     import (PaymentBehaviourFeatures,
                                                             RepaymentRatioFeatures)
from features.tradeline.grp09_obligations           import ObligationsFeatures
from features.tradeline.grp10_severe_risk           import WriteoffsSevereRiskFeatures
from features.enquiry.grp12_enquiries               import CreditEnquiriesFeatures

print("  ✅ all 11 group imports OK")

# COMMAND ----------
# ── 5. Registry ───────────────────────────────────────────────────────────────
print("Testing registry...")
from features.registry import TRADELINE_FEATURE_CLASSES, ENQUIRY_FEATURE_CLASSES

print(f"  TRADELINE_FEATURE_CLASSES : {len(TRADELINE_FEATURE_CLASSES)} classes")
for cls in TRADELINE_FEATURE_CLASSES:
    print(f"    - {cls().CATEGORY}")

print(f"  ENQUIRY_FEATURE_CLASSES   : {len(ENQUIRY_FEATURE_CLASSES)} classes")
for cls in ENQUIRY_FEATURE_CLASSES:
    print(f"    - {cls().CATEGORY}")

print("  ✅ registry OK")

# COMMAND ----------
# ── 6. Pipeline classes ───────────────────────────────────────────────────────
print("Testing pipeline imports...")
from pipeline.scrub_pipeline import ScrubPipeline
from pipeline.retro_pipeline import RetroPipeline
print(f"  ScrubPipeline : {ScrubPipeline}")
print(f"  RetroPipeline : {RetroPipeline}")
print("  ✅ pipeline OK")

# COMMAND ----------
# ── 7. Source table access ────────────────────────────────────────────────────
print("Testing source table access...")
try:
    tl  = spark.table(config.TRADELINE_TABLE)
    enq = spark.table(config.ENQUIRY_TABLE)
    mst = spark.table(config.MASTER_TABLE)
    print(f"  Tradeline  : {tl.count():,} rows | {len(tl.columns)} cols")
    print(f"  Enquiry    : {enq.count():,} rows | {len(enq.columns)} cols")
    print(f"  Master     : {mst.count():,} rows | {len(mst.columns)} cols")
    print("  ✅ source tables accessible")
except Exception as e:
    print(f"  ❌ Table access failed: {e}")

# COMMAND ----------
print("\n" + "="*50)
print("  ALL CHECKS PASSED — repo is ready ✅")
print("  Next: run test_single_customer.py to validate")
print("        features for one customer before scheduling")
print("        the full pipeline job.")
print("="*50)
