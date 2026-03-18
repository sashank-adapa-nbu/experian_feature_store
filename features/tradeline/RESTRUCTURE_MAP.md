# Feature Restructure Map
# Old cat01-cat19 → New grouped files

grp01_portfolio_counts.py
  ← cat01 (tradeline counts, minus CountBureauActiveAccounts — redundant with cat02.active_loans)
  ← cat02 (active portfolio by product)
  ← cat19: tot_pls, tot_cc, tot_tradelines, loan_flag, bur_cohort, pl_thickness, cc_thickness

grp02_loan_amounts.py
  ← cat03 (max/sum loan amounts, flags)
  ← cat04 (CC credit limit vintage features)
  ← cat07 (highest credit by product/window)
  ← cat08 (loan volume over time)

grp03_balances_utilization.py
  ← cat05 (outstanding balance, past due amounts)
  ← cat14 (revolved ratio, CC utilization, past due features)

grp04_bureau_vintage.py
  ← cat06 (minus count_active_accounts — redundant), keep bureaage_months
  ← cat19: bureaage, bureaage_months (reference from cat06)

grp05_lender_mix.py
  ← cat09 (unchanged — clean standalone)

grp06_recency_flags.py
  ← cat10 (months_from_last_X)
  ← cat11 (has_X_ever, has_active_X, binary behaviour flags)

grp07_delinquency.py
  ← cat12 (DPD — unchanged, clean standalone)

grp08_payment_repayment.py
  ← cat13 (payment amounts, missed payments)
  ← cat15 (repayment percentage, balance reduction)

grp09_obligations.py
  ← cat18 (obligations, imputed EMI)

grp10_severe_risk.py
  ← cat17 (write-offs, suits)

grp11_bureau_segments.py
  ← cat19: pl_thickness, cc_thickness, bureaage, bur_cohort, loan_flag
  (pure segmentation labels — separate from numeric features)

enquiry/grp12_enquiries.py
  ← cat16 (unchanged — different source table)
