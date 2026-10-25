"""Fake loans in the raw Lending Club format, for tests and for trying the code without the real data.

The values are written as text the way the raw file writes them, so clean() is tested end to end.
Default risk depends on FICO, DTI, utilisation and inquiries, so later models have a signal to find.
Nothing here is real data.
"""

import numpy as np
import pandas as pd

PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", "small_business"]
GRADES = ["A", "B", "C", "D", "E", "F", "G"]


def synthetic_raw(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    issue = pd.to_datetime("2007-06-01") + pd.to_timedelta(rng.integers(0, 365 * 11, n), unit="D")
    fico = rng.integers(132, 170, n) * 5  # 660 .. 845 in steps of 5, like the real ranges
    dti = rng.gamma(4, 4.5, n).clip(0, 45)
    util = rng.beta(2, 2, n) * 100
    inq = rng.poisson(0.8, n)
    income = rng.lognormal(11, 0.5, n).round(-2)
    amount = (rng.integers(10, 141, n) * 250).astype(float)
    term = rng.choice([36, 60], n, p=[0.75, 0.25])

    score = -2.2 - 0.02 * (fico - 700) + 0.03 * (dti - 18) + 0.01 * (util - 50) + 0.15 * inq + 0.4 * (term == 60)
    bad = rng.random(n) < 1 / (1 + np.exp(-score))
    grade_idx = np.clip(((score + 4) * 1.4).astype(int), 0, 6)
    rate = 6 + grade_idx * 3.5 + rng.normal(0, 0.8, n)
    monthly = amount * (rate / 1200) / (1 - (1 + rate / 1200) ** (-term))

    status = np.where(bad, "Charged Off", "Fully Paid")
    status = np.where(issue > pd.Timestamp("2016-06-01"), rng.choice(["Current", "Fully Paid"], n), status)
    old = issue < pd.Timestamp("2010-11-01")
    status = np.where(old & (rng.random(n) < 0.05), "Does not meet the credit policy. Status:" + status, status)

    paid = np.where(bad, monthly * rng.integers(3, 30, n), monthly * term)
    emp = rng.choice(["< 1 year", "1 year", "3 years", "5 years", "10+ years", None], n)

    def months(dates):
        return pd.Series(dates).dt.strftime("%b-%Y")

    raw = pd.DataFrame(
        {
            "id": [str(100000 + i) for i in range(n)],
            "issue_d": months(issue),
            "term": [f" {t} months" for t in term],
            "funded_amnt": amount,
            "loan_amnt": amount,
            "emp_length": emp,
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
            "annual_inc": income,
            "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], n),
            "purpose": rng.choice(PURPOSES, n),
            "dti": dti.round(2),
            "delinq_2yrs": rng.poisson(0.3, n),
            "fico_range_low": fico,
            "fico_range_high": fico + 4,
            "inq_last_6mths": inq,
            "mths_since_last_delinq": np.where(rng.random(n) < 0.5, rng.integers(1, 80, n).astype(float), np.nan),
            "mths_since_last_record": np.where(rng.random(n) < 0.15, rng.integers(1, 120, n).astype(float), np.nan),
            "open_acc": rng.integers(2, 30, n),
            "pub_rec": rng.poisson(0.2, n),
            "revol_bal": (util * income / 400).round(),
            "revol_util": [f"{u:.1f}%" for u in util],
            "total_acc": rng.integers(5, 60, n),
            "earliest_cr_line": months(issue - pd.to_timedelta(rng.integers(3 * 365, 30 * 365, n), unit="D")),
            "application_type": "Individual",
            "mort_acc": np.where(issue < pd.Timestamp("2010-01-01"), np.nan, rng.poisson(1.5, n)),
            "pub_rec_bankruptcies": rng.poisson(0.1, n),
            "grade": [GRADES[g] for g in grade_idx],
            "sub_grade": [f"{GRADES[g]}{rng.integers(1, 6)}" for g in grade_idx],
            "int_rate": [f" {r:.2f}%" for r in rate],
            "installment": monthly.round(2),
            "loan_status": status,
            "total_pymnt": paid.round(2),
            "total_rec_prncp": np.minimum(paid, amount).round(2),
            "total_rec_int": np.maximum(paid - amount, 0).round(2),
            "recoveries": np.where(bad, amount * rng.uniform(0, 0.1, n), 0).round(2),
            "last_pymnt_d": months(issue + pd.to_timedelta(rng.integers(90, 365 * 3, n), unit="D")),
            "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL"], n),
        }
    )
    summary_line = {c: None for c in raw.columns} | {"id": "Total amount funded in policy code 1: 1000000"}
    return pd.concat([raw, pd.DataFrame([summary_line])], ignore_index=True).astype(str).replace({"None": None, "nan": None})
