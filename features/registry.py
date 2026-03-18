# features/registry.py
# =============================================================================
# Central registry of all feature category classes.
#
# ➕ TO ADD A NEW CATEGORY (when logic is implemented):
#   1. Implement compute() in the relevant cat*.py file
#   2. That's it — it's already imported and registered below.
#
# Execution order = order in TRADELINE_FEATURE_REGISTRY / ENQUIRY_FEATURE_REGISTRY.
# All outputs are left-joined on PK cols into one wide feature table.
# =============================================================================

# ── Tradeline Feature Modules (cat01–cat15, cat17–cat19) ─────────────────────
from features.tradeline.cat01_loan_account_counts     import LoanAccountCountsFeatures
from features.tradeline.cat02_active_loan_portfolio   import ActiveLoanPortfolioFeatures
from features.tradeline.cat03_loan_amount_exposure    import LoanAmountExposureFeatures
from features.tradeline.cat04_credit_card_limits      import CreditCardLimitsFeatures
from features.tradeline.cat05_outstanding_balance     import OutstandingBalanceFeatures
from features.tradeline.cat06_bureau_vintage          import BureauVintageFeatures
from features.tradeline.cat07_highest_credit_signals  import HighestCreditSignalsFeatures
from features.tradeline.cat08_loan_volume_over_time   import LoanVolumeOverTimeFeatures
from features.tradeline.cat09_lender_type_mix         import LenderTypeMixFeatures
from features.tradeline.cat10_recency_credit_activity import RecencyCreditActivityFeatures
from features.tradeline.cat11_credit_behaviour_flags  import CreditBehaviourFlagsFeatures
from features.tradeline.cat12_delinquency_dpd         import DelinquencyDPDFeatures
from features.tradeline.cat13_payment_behaviour       import PaymentBehaviourFeatures
from features.tradeline.cat14_credit_utilization      import CreditUtilizationFeatures
from features.tradeline.cat15_repayment_ratio         import RepaymentRatioFeatures
from features.tradeline.cat17_writeoffs_severe_risk   import WriteoffsSevereRiskFeatures
from features.tradeline.cat18_credit_score_risk       import CreditScoreRiskFeatures
from features.tradeline.cat19_credit_thickness        import CreditThicknessFeatures

# ── Enquiry Feature Modules (cat16) ──────────────────────────────────────────
from features.enquiry.cat16_credit_enquiries import CreditEnquiriesFeatures


# =============================================================================
# TRADELINE REGISTRY
# Order = join sequence. Base table is the first entry.
# =============================================================================
TRADELINE_FEATURE_REGISTRY = [
    LoanAccountCountsFeatures(),        # cat01 — Tradeline / Loan Account Counts
    ActiveLoanPortfolioFeatures(),      # cat02 — Active Loan Portfolio Composition
    LoanAmountExposureFeatures(),       # cat03 — Loan Amount / Exposure Features
    CreditCardLimitsFeatures(),         # cat04 — Credit Card Limits / Capacity Indicators
    OutstandingBalanceFeatures(),       # cat05 — Outstanding Balance / Current Debt
    BureauVintageFeatures(),            # cat06 — Bureau Vintage / Credit History Age
    HighestCreditSignalsFeatures(),     # cat07 — Highest Credit / Loan History Signals
    LoanVolumeOverTimeFeatures(),       # cat08 — Loan Volume Over Time
    LenderTypeMixFeatures(),            # cat09 — Lender Type / Source Mix
    RecencyCreditActivityFeatures(),    # cat10 — Recency of Credit Activity
    CreditBehaviourFlagsFeatures(),     # cat11 — Credit Behaviour Flags
    DelinquencyDPDFeatures(),           # cat12 — Delinquency / DPD Behaviour
    PaymentBehaviourFeatures(),         # cat13 — Payment Behaviour
    CreditUtilizationFeatures(),        # cat14 — Credit Utilization / Revolving Behaviour
    RepaymentRatioFeatures(),           # cat15 — Repayment Ratio / Balance Reduction
    WriteoffsSevereRiskFeatures(),      # cat17 — Write-offs / Severe Risk Indicators
    CreditScoreRiskFeatures(),          # cat18 — Credit Score / Risk Quality Indicators
    CreditThicknessFeatures(),          # cat19 — Credit Thickness Segmentation
]

# =============================================================================
# ENQUIRY REGISTRY
# =============================================================================
ENQUIRY_FEATURE_REGISTRY = [
    CreditEnquiriesFeatures(),          # cat16 — Credit Enquiries / Credit Hunger
]
