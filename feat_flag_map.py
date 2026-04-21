FLAG_COMPANION_DICT = {

    # ── grp02a — Product max loan amounts  [F.coalesce(..., F.lit(0))] ────────
    "max_loanamount_sl":                    "has_taken_usl_ever",
    "max_loanamount_usl":                   "has_taken_usl_ever",
    "max_loanamount_cc":                    "has_taken_cc_ever",
    "max_loanamount_hl":                    "flag_ever_had_hl",
    "max_loanamount_al":                    "has_taken_al_ever",

    # ── grp02a — Active PL / CC / all  [F.coalesce(..., F.lit(0))] ───────────
    "MaxLoanAmountActivePersonalLoan":      "has_active_pl_flag",
    "sumloanamount_personalloan":           "has_taken_PL_ever",
    "active_total_cc_credit_limit":         "has_active_credit_card_flag",
    "total_active_exposure":                "BureauActiveTradelines",

    # ── grp02c — Highest credit by product  [F.coalesce(..., F.lit(0))] ──────
    "highest_credit_housing_loan":          "flag_ever_had_hl",
    "highest_credit_unsecured_3_years":     "has_taken_usl_ever",
    "highest_credit_cc_3_years":            "has_taken_cc_ever",
    "highest_credit_pl_5_years":            "has_taken_PL_ever",
    "highest_credit_any_5_years":           "BureauActiveTradelines",
    "highest_credit_personal_loan_3_years": "has_taken_PL_ever",
    "highest_credit_consumer_loan_3_years": "has_taken_PL_ever",

    # ── grp02d — Loan volume over time  [F.coalesce(..., F.lit(0))] ──────────
    "total_loan_amt_3_years":               "count_accounts_gt_24m",
    "total_loan_amt_unsecured_3y":          "has_taken_usl_ever",
    "total_loan_amt_secured_3_years":       "has_taken_hl_ever",

    # ── grp09 — Obligations  [F.sum(...).otherwise(F.lit(0.0))] ─────────────
    "obligations":                          "count_obligating_accounts",
    "obligations_6m":                       "count_obligating_accounts",
    "obligations_12m":                      "count_obligating_accounts",
    "obligations_24m":                      "count_obligating_accounts",
    "obligations_hl":                       "has_active_hl_flag",
    "obligations_cc":                       "has_active_credit_card_flag",

    # ── grp08 — Payments  [F.coalesce(slot, 0) inside _win_sum] ─────────────
    "payments_3_month":                     "frequency_of_payments_3m",
    "payments_6_month":                     "frequency_of_payments_6m",
    "payments_12_month":                    "frequency_of_payments_12m",
}