"""Data checks after cleaning, and the population table (loans and statuses per issue year)."""

import pandas as pd

from crs import config


def check(loans):
    """A list of problems; empty means the data passed."""
    problems = []
    if loans["id"].duplicated().any():
        problems.append("duplicate loan ids")
    if loans["issue_d"].isna().any():
        problems.append("loans without an issue date")
    unknown = set(loans["loan_status"].dropna()) - set(config.STATUSES)
    if unknown:
        problems.append(f"unknown loan statuses: {sorted(unknown)}")
    if not set(loans["term"].dropna()) <= {36, 60}:
        problems.append("terms other than 36 or 60 months")
    if (loans["loan_amnt"] <= 0).any():
        problems.append("loan amounts that are zero or negative")
    if (loans["fico_range_high"] < loans["fico_range_low"]).any():
        problems.append("FICO ranges where high < low")
    too_old = loans["earliest_cr_line"] > loans["issue_d"]
    if too_old.any():
        problems.append(f"{too_old.sum()} loans whose first credit line is after the issue date")
    return problems


def population(loans):
    """Loans per issue year and status, for the README."""
    table = pd.crosstab(loans["issue_d"].dt.year, loans["loan_status"]).reindex(columns=config.STATUSES, fill_value=0)
    table.index.name = "issue year"
    table["total"] = table.sum(axis=1)
    return table


def missing_share(loans):
    """Share of missing values per application column, highest first."""
    return loans[config.APPLICATION].isna().mean().sort_values(ascending=False).round(3)
