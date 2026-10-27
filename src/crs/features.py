"""Model inputs, built only from application-time columns.

This is the one gate between the data and every model: build() reads APPLICATION columns only,
adds a few derived ones, and refuses to return anything that is not on the allow-list.
"""

import numpy as np

from crs import config

DERIVED = {
    "fico": "middle of the FICO range at application",
    "credit_history_months": "months from the first credit line to the application",
    "loan_to_income": "loan amount / annual income",
}
DROPPED = ["fico_range_low", "fico_range_high", "earliest_cr_line"]  # replaced by the derived columns

CATEGORICAL = ["home_ownership", "verification_status", "purpose", "application_type"]
ALLOWED = [c for c in config.APPLICATION if c not in DROPPED] + list(DERIVED)
FORBIDDEN = set(config.BENCHMARK + config.OUTCOME + config.NOT_FOR_MODELS + ["bad", "loan_status"])


def build(loans):
    X = loans[config.APPLICATION].copy()
    X["fico"] = (X["fico_range_low"] + X["fico_range_high"]) / 2
    months = (loans["issue_d"].dt.year - X["earliest_cr_line"].dt.year) * 12 + (
        loans["issue_d"].dt.month - X["earliest_cr_line"].dt.month
    )
    X["credit_history_months"] = months
    X["loan_to_income"] = X["loan_amnt"] / X["annual_inc"].replace(0, np.nan)
    X = X.drop(columns=DROPPED)
    check(X)
    return X[ALLOWED]


def check(X):
    """Fail loudly if a column that is not known at application time reaches a model."""
    not_allowed = sorted((set(X.columns) & FORBIDDEN) | (set(X.columns) - set(ALLOWED)))
    if not_allowed:
        raise ValueError(f"Not allowed as model input: {not_allowed}")
