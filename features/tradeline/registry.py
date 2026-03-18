# features/tradeline/registry.py
# =============================================================================
# Feature Registry — Restructured Groups
# =============================================================================
# 11 groups total (10 tradeline + 1 enquiry)
# Each group file may contain 1 or more feature classes.
# To register a class: add it to TRADELINE_FEATURE_CLASSES or ENQUIRY_FEATURE_CLASSES.
# =============================================================================

from features.tradeline.grp01_portfolio_counts   import PortfolioCountsFeatures
from features.tradeline.grp02_loan_amounts       import (
    LoanAmountExposureFeatures,
    CreditCardLimitsFeatures,
    HighestCreditSignalsFeatures,
    LoanVolumeOverTimeFeatures,
)
from features.tradeline.grp03_balances_utilization import (
    OutstandingBalanceFeatures,
    CreditUtilizationFeatures,
)
from features.tradeline.grp04_bureau_vintage     import BureauVintageFeatures
from features.tradeline.grp05_lender_mix         import LenderTypeMixFeatures
from features.tradeline.grp06_recency_flags      import (
    RecencyCreditActivityFeatures,
    CreditBehaviourFlagsFeatures,
)
from features.tradeline.grp07_delinquency        import DelinquencyDPDFeatures
from features.tradeline.grp08_payment_repayment  import (
    PaymentBehaviourFeatures,
    RepaymentRatioFeatures,
)
from features.tradeline.grp09_obligations        import ObligationsFeatures
from features.tradeline.grp10_severe_risk        import WriteoffsSevereRiskFeatures

from features.enquiry.grp12_enquiries            import CreditEnquiriesFeatures

# =============================================================================
# TRADELINE FEATURE CLASSES
# Order matters — features are computed and joined in this sequence.
# =============================================================================

TRADELINE_FEATURE_CLASSES = [
    PortfolioCountsFeatures,        # grp01 — counts, active portfolio, segments
    LoanAmountExposureFeatures,     # grp02 — max/sum loan amounts
    CreditCardLimitsFeatures,       # grp02 — CC credit limit vintage
    HighestCreditSignalsFeatures,   # grp02 — highest credit by product/window
    LoanVolumeOverTimeFeatures,     # grp02 — loan volume over time
    OutstandingBalanceFeatures,     # grp03 — outstanding balances
    CreditUtilizationFeatures,      # grp03 — revolved ratio, CC utilization
    BureauVintageFeatures,          # grp04 — bureau age features
    LenderTypeMixFeatures,          # grp05 — NBFC/bank distribution
    RecencyCreditActivityFeatures,  # grp06 — months since last event
    CreditBehaviourFlagsFeatures,   # grp06 — binary usage flags
    DelinquencyDPDFeatures,         # grp07 — DPD behaviour
    PaymentBehaviourFeatures,       # grp08 — payment amounts, missed
    RepaymentRatioFeatures,         # grp08 — repayment %, balance reduction
    ObligationsFeatures,            # grp09 — EMI obligations
    WriteoffsSevereRiskFeatures,    # grp10 — write-offs, suits
]

# =============================================================================
# ENQUIRY FEATURE CLASSES
# =============================================================================

ENQUIRY_FEATURE_CLASSES = [
    CreditEnquiriesFeatures,        # grp12 — enquiry counts, recency, ratios
]
