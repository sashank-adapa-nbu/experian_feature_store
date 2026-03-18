# Experian Bureau Feature Store
## Project Description

---

## 1. Overview

This project builds a **modular, production-grade feature engineering pipeline** for Experian credit bureau data on Databricks. It transforms raw bureau tradeline and enquiry data into a wide set of risk-relevant features used for credit underwriting, scorecard development, and model training.

The pipeline supports two execution modes — **Scrub** (live scoring) and **Retro** (historical model training) — from a single codebase, with all features computed point-in-time to prevent data leakage.

---

## 2. Business Problem

When a customer applies for a loan, the lender needs to assess their creditworthiness using external bureau data. Experian delivers this data as raw tradeline segments — one row per account per customer per scrub date. This raw data is:

- **High dimensional** — 200+ columns per row, 1–50 rows per customer
- **Time-anchored** — historical columns (`_01` to `_36`) anchored to the last reporting date, not the current date
- **Heterogeneous** — different account types (PL, CC, HL, GL, STPL) require different feature logic
- **Leakage-prone** — naive feature computation can accidentally include future information into training data

The goal is to convert this raw bureau data into a **single wide feature row per customer** that can be directly consumed by ML models and scorecards.

---

## 3. Data Sources

| Table | Description | Key Columns |
|---|---|---|
| `experian_tradeline_segment` | One row per account per customer per scrub | `customer_scrub_key`, `acct_type_cd`, `balance_am_01..36`, `days_past_due_01..36`, etc. |
| `experian_enquiry_segment` | One row per credit enquiry event | `customer_scrub_key`, `inq_purp_cd`, `inq_date`, `m_sub_id` |
| `experian_master_table` | Maps `customer_scrub_key` → `party_code` | `customer_scrub_key`, `party_code`, `scrub_output_date` |

---

## 4. Execution Modes

### Mode 1 — Scrub Pipeline
**Use case:** Live scoring, periodic bureau refresh

- **PK:** `customer_scrub_key`, `party_code`, `scrub_output_date`
- **As-of date:** `scrub_output_date` (date Experian delivered the bureau data)
- **Trigger:** Scheduled Databricks job — runs for a specific scrub date or incrementally processes all pending dates
- **Output:** `experian_tradeline_features_scrub`, `experian_enquiry_features_scrub`

### Mode 2 — Retro Pipeline
**Use case:** Model training, historical feature backfill

- **PK:** `party_code`, `reference_dt` (the loan application or event date)
- **As-of date:** `reference_dt`
- **Input:** Any Delta table with `[party_code, reference_dt]` columns
- **Scrub selection:** For each `(party_code, reference_dt)`, finds the latest `scrub_output_date` within a configurable window (default ±12 months after `reference_dt`)
- **Output:** User-specified table names — each model/use case writes to its own tables

---

## 5. Point-in-Time Correctness (No Leakage)

All features are computed strictly as of the `as_of_dt` (either `scrub_output_date` or `reference_dt`). Three mechanisms enforce this:

**1. Open/close date guard**
Accounts opened after `as_of_dt` are excluded. Active flag = `open_dt <= as_of_dt AND (closed_dt IS NULL OR closed_dt > as_of_dt)`.

**2. History column slot resolution**
Columns like `BALANCE_AM`, `DAYS_PAST_DUE`, `ACTUAL_PAYMENT_AM` are anchored to `balance_dt` / `last_reporting_pymt_dt`, not `as_of_dt`. A `month_diff` calculation resolves the correct historical slot:

```
month_diff = round(months_between(as_of_dt, reporting_dt))
actual_slot = k - month_diff

slot < 0   → gap (data not yet reported) → NULL
slot == 0  → current column (e.g. BALANCE_AM)
slot 1–36  → history column (e.g. BALANCE_AM_03)
slot > 36  → beyond history → NULL
```

**3. Enquiry date filter**
Enquiry features only count events where `inq_date <= as_of_dt`.

---

## 6. Feature Groups (365 features across 11 groups)

| Group | File | Features | Description |
|---|---|---|---|
| grp01 | `grp01_portfolio_counts.py` | 32 | Account counts, active portfolio by product, bureau segments |
| grp02 | `grp02_loan_amounts.py` | 59 | Max/sum loan amounts, CC limits, highest credit, loan volumes |
| grp03 | `grp03_balances_utilization.py` | 34 | Outstanding balances, revolving ratio, CC utilization |
| grp04 | `grp04_bureau_vintage.py` | 18 | Bureau age, history length, mean/max age by product |
| grp05 | `grp05_lender_mix.py` | 34 | NBFC vs bank distribution, % from formal lenders |
| grp06 | `grp06_recency_flags.py` | 36 | Months since last event, binary credit behaviour flags |
| grp07 | `grp07_delinquency.py` | 44 | DPD max/count/bucket/recency, severity flags |
| grp08 | `grp08_payment_repayment.py` | 54 | Payment amounts, missed payments, repayment ratios |
| grp09 | `grp09_obligations.py` | 11 | Imputed EMI, current and windowed obligations |
| grp10 | `grp10_severe_risk.py` | 20 | Write-offs, suits, wilful defaults, charge-off amounts |
| grp12 | `grp12_enquiries.py` | 23 | Enquiry counts, recency, NBFC/bank split (enquiry table) |

### Feature Categories Covered
- **Tradeline counts** — accounts opened, active by product type, thickness segments
- **Loan exposure** — max/sum sanctioned amounts, credit limits, highest credit by product
- **Outstanding debt** — balance at as-of date, product-level outstanding, utilization
- **Credit history** — bureau age, vintage bands, product experience breadth
- **Lender mix** — NBFC reliance, bank penetration, secured vs unsecured sources
- **Recency** — months since last PL/CC/GL/STPL opened, recency flags
- **Delinquency** — DPD history, bucket flags (10-30, 30-60, 60-90), severity indicators
- **Payment behaviour** — payments made, missed counts by product, repayment percentage
- **Obligations** — imputed monthly EMI, current and historical obligation burden
- **Severe risk** — write-offs (3Y/5Y), suit filed, wilful default, post-WO settlement
- **Credit hunger** — enquiry velocity, product intent, lender type of enquiry

---

## 7. Repository Structure

```
experian_feature_store/
│
├── config/
│   └── config.py                    ← All table names, PK cols, window settings
│
├── core/
│   ├── logger.py
│   └── utils.py
│
├── features/
│   ├── tradeline/
│   │   ├── base.py                  ← Abstract base class for all tradeline groups
│   │   ├── registry.py              ← Registers all feature classes in order
│   │   ├── grp01_portfolio_counts.py
│   │   ├── grp02_loan_amounts.py
│   │   ├── grp03_balances_utilization.py
│   │   ├── grp04_bureau_vintage.py
│   │   ├── grp05_lender_mix.py
│   │   ├── grp06_recency_flags.py
│   │   ├── grp07_delinquency.py
│   │   ├── grp08_payment_repayment.py
│   │   ├── grp09_obligations.py
│   │   └── grp10_severe_risk.py
│   │
│   └── enquiry/
│       ├── base.py                  ← Abstract base class for enquiry groups
│       └── grp12_enquiries.py
│
├── pipeline/
│   ├── base_pipeline.py             ← Shared run/join/write logic
│   ├── scrub_pipeline.py            ← Scrub mode implementation
│   └── retro_pipeline.py            ← Retro mode with scrub window selection
│
├── output/
│   └── writer.py                    ← Delta write wrapper
│
└── notebooks/
    ├── 01_run_scrub_pipeline.py     ← Databricks job: scrub mode
    ├── 02_run_retro_pipeline.py     ← Databricks job: retro mode
    └── test_single_customer.py      ← Pre-run validation for one customer
```

---

## 8. Extensibility

Adding a new feature group requires three steps:

1. **Create** `features/tradeline/grp_new_category.py` with a class extending `TradelineFeatureBase`
2. **Register** it in `features/tradeline/registry.py` by adding to `TRADELINE_FEATURE_CLASSES`
3. **Test** it in `test_single_customer.py` by adding one `run_group(NewClass, "label")` cell

No changes needed to the pipeline, writer, or notebooks.

---

## 9. Key Design Decisions

| Decision | Rationale |
|---|---|
| One row per feature class per `compute()` call | Clean separation — each group is independently testable and replaceable |
| `month_diff` slot resolution | Correct point-in-time balance/DPD without data leakage in retro mode |
| `balance_dt` vs `last_reporting_pymt_dt` | Balance history uses `balance_dt`; DPD/payment uses `last_reporting_pymt_dt` for accuracy |
| Scrub window for retro (`ref_dt` to `ref_dt + 12m`) | Ensures scrub is close enough to reference date; configurable via `RETRO_MAX_SCRUB_MONTHS` |
| `party_code` added in scrub pipeline via master join | Ensures scrub output is joinable downstream by party_code without an extra lookup |
| DPD uses numeric columns only for flags | `is_dpd_30/90/180` flags are deterministic from numeric DPD; payment rating codes only used for max/count/bucket features |

---

## 10. Output Tables

| Mode | Table | Partition |
|---|---|---|
| Scrub — tradeline | `uc_dataorg_prod.l3_nbu.experian_tradeline_features_scrub` | `scrub_output_date` |
| Scrub — enquiry | `uc_dataorg_prod.l3_nbu.experian_enquiry_features_scrub` | `scrub_output_date` |
| Retro — tradeline | User-specified at runtime | None (overwrite) |
| Retro — enquiry | User-specified at runtime | None (overwrite) |
