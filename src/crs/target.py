"""Who is in the population, what counts as a default, and which years are used for what.

Population: 36-month loans issued 2007-2015. A loan issued in December 2015 reaches the end of its term
in December 2018, when the data ends, so every loan in the population had its full term to default.
Target: bad = "Charged Off" or "Default", good = "Fully Paid". Loans that are still open or late
at the end of the data have no final outcome and are left out (and counted).
"""

import pandas as pd

BAD = ["Charged Off", "Default"]
GOOD = ["Fully Paid"]
TERM = 36
FIRST_YEAR, LAST_YEAR = 2007, 2015

DEV_YEARS = range(2007, 2015)  # 2007-2014: development, with time cross-validation
OOT_YEAR = 2015  # out-of-time test, used once
CV_TEST_YEARS = [2011, 2012, 2013, 2014]  # each fold trains on all earlier years


def exclusion_reason(loans):
    """Why a loan is not in the modelling population ('' means it is in)."""
    year = loans["issue_d"].dt.year
    reason = pd.Series("", index=loans.index)
    reason[~loans["loan_status"].isin(BAD + GOOD)] = "no final outcome (still open or late)"
    reason[year > LAST_YEAR] = "issued after 2015 (term not over): monitoring only"
    reason[loans["term"] != TERM] = "60-month term"
    return reason


def population(loans):
    """The modelling population with the target column `bad` (1 = default)."""
    keep = exclusion_reason(loans) == ""
    out = loans[keep].copy()
    out["bad"] = out["loan_status"].isin(BAD).astype(int)
    out["year"] = out["issue_d"].dt.year
    return out.reset_index(drop=True)


def exclusions(loans):
    """How many loans were left out, and why: the table for the README."""
    reason = exclusion_reason(loans).replace("", "in the population")
    return reason.value_counts().rename("loans")


def bad_rate_by_year(pop):
    return pop.groupby("year")["bad"].agg(loans="size", bad_rate="mean").round({"bad_rate": 4})


def split(pop):
    """Development (2007-2014) and out-of-time test (2015)."""
    return pop[pop["year"].isin(DEV_YEARS)], pop[pop["year"] == OOT_YEAR]


def time_folds(dev):
    """Expanding-window folds: train on every year before the test year, test on that year.

    Yields (test_year, train_index, test_index) with index labels of `dev`.
    """
    for test_year in CV_TEST_YEARS:
        train = dev.index[dev["year"] < test_year]
        test = dev.index[dev["year"] == test_year]
        yield test_year, train, test
