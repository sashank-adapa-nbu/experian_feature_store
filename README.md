# Experian Feature Store — Databricks Project

A modular, extensible feature engineering pipeline for Experian tradeline and enquiry data.

---

## 🏗️ Architecture Overview

```
experian_feature_store/
├── config/           → Table paths, global constants
├── core/             → Logging, utilities
├── pipeline/         → Two execution modes (scrub / retro)
├── features/
│   ├── tradeline/    → 18 feature category files (cat01–cat15, cat17–cat19)
│   ├── enquiry/      → 1 feature category file (cat16)
│   └── registry.py  → Single file that wires all categories into the pipeline
├── output/           → Delta table writer
└── notebooks/        → Entry-point notebooks
```

---

## ⚙️ Two Execution Modes

### Mode 1 — Scrub Pipeline (`01_run_scrub_pipeline`)
- **PK:** `customer_scrub_key`, `scrub_id`
- **As-of date:** `scrub_output_date`
- **Use case:** Scrub-on-scrub analysis, model inferencing

### Mode 2 — Retro Pipeline (`02_run_retro_pipeline`)
- **PK:** `party_code`, `open_dt`
- **As-of date:** `open_dt` (loan open date)
- **Use case:** Retro feature derivation for model training

---

## 📋 Feature Categories — Implementation Status

### Tradeline Source (`experian_tradeline_segment`)

| # | Category | File | Status |
|---|---|---|---|
| 01 | Tradeline / Loan Account Counts | `cat01_loan_account_counts.py` | ⏳ Pending |
| 02 | Active Loan Portfolio Composition | `cat02_active_loan_portfolio.py` | ⏳ Pending |
| 03 | Loan Amount / Exposure Features | `cat03_loan_amount_exposure.py` | ⏳ Pending |
| 04 | Credit Card Limits / Capacity Indicators | `cat04_credit_card_limits.py` | ⏳ Pending |
| 05 | Outstanding Balance / Current Debt | `cat05_outstanding_balance.py` | ⏳ Pending |
| 06 | Bureau Vintage / Credit History Age | `cat06_bureau_vintage.py` | ⏳ Pending |
| 07 | Highest Credit / Loan History Signals | `cat07_highest_credit_signals.py` | ⏳ Pending |
| 08 | Loan Volume Over Time | `cat08_loan_volume_over_time.py` | ⏳ Pending |
| 09 | Lender Type / Source Mix | `cat09_lender_type_mix.py` | ⏳ Pending |
| 10 | Recency of Credit Activity | `cat10_recency_credit_activity.py` | ⏳ Pending |
| 11 | Credit Behaviour Flags | `cat11_credit_behaviour_flags.py` | ⏳ Pending |
| 12 | Delinquency / DPD Behaviour | `cat12_delinquency_dpd.py` | ⏳ Pending |
| 13 | Payment Behaviour | `cat13_payment_behaviour.py` | ⏳ Pending |
| 14 | Credit Utilization / Revolving Behaviour | `cat14_credit_utilization.py` | ⏳ Pending |
| 15 | Repayment Ratio / Balance Reduction | `cat15_repayment_ratio.py` | ⏳ Pending |
| 17 | Write-offs / Severe Risk Indicators | `cat17_writeoffs_severe_risk.py` | ⏳ Pending |
| 18 | Credit Score / Risk Quality Indicators | `cat18_credit_score_risk.py` | ⏳ Pending |
| 19 | Credit Thickness Segmentation | `cat19_credit_thickness.py` | ⏳ Pending |

### Enquiry Source (`experian_enquiry_segment`)

| # | Category | File | Status |
|---|---|---|---|
| 16 | Credit Enquiries / Credit Hunger | `cat16_credit_enquiries.py` | ⏳ Pending |

---

## ➕ How to Implement a Category (when rules are provided)

1. Open the relevant `cat*.py` file
2. Replace the placeholder block inside `compute()`:
   ```python
   # ── FEATURE LOGIC TO BE ADDED ─────────────────────────────
   feature_df = df.select(group_cols).distinct()
   ```
   with actual PySpark feature derivation logic.
3. Return a DataFrame: `pk_cols` + `as_of_col` + feature columns — **one row per PK**.
4. Nothing else to change — registry already wires it in automatically.

---

## 📦 Output Tables

| Table | Mode |
|---|---|
| `uc_dataorg_prod.l3_nbu.experian_tradeline_features_scrub` | Scrub |
| `uc_dataorg_prod.l3_nbu.experian_enquiry_features_scrub` | Scrub |
| `uc_dataorg_prod.l3_nbu.experian_tradeline_features_retro` | Retro |
| `uc_dataorg_prod.l3_nbu.experian_enquiry_features_retro` | Retro |

---

## 🗓️ Scheduling as Databricks Jobs

- Widget: **`scrub_output_date`** — pass a date (`2024-04-18`) or `ALL`
- Output partitioned by `scrub_output_date` — safe to rerun any date
