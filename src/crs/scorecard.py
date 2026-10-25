"""The points scorecard: choose characteristics, fit a logistic regression on their WoE values, scale to points.

Scale: 600 points mean odds of 50 goods to 1 bad, and every 20 points double the odds (PDO = 20).
The score is the sum of the points of the applicant's bins, so anyone can add it up from the table.
The same table gives the reasons for a decline: where the applicant lost the most points.
"""

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from crs import binning

BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
FACTOR = PDO / math.log(2)
OFFSET = BASE_SCORE - FACTOR * math.log(BASE_ODDS)

IV_MIN = 0.01  # below this a characteristic hardly separates goods from bads (the usual rule is 0.02; the CV decides)
CORR_MAX = 0.6  # of two characteristics with more correlated WoE values, keep the one with the higher IV

# Known at application time, but not usable as a reason to decline someone.
# Lending Club verified income mostly for applicants it already found risky, so "verified" looks riskier:
# that is Lending Club's own risk judgement leaking in, and "your income was verified" is not a fair reason.
NOT_IN_SCORECARD = {"verification_status"}

DESCRIPTION = {
    "fico": "FICO score",
    "dti": "debt-to-income ratio",
    "revol_util": "revolving credit utilisation (%)",
    "inq_last_6mths": "credit inquiries in the last 6 months",
    "annual_inc": "annual income",
    "loan_to_income": "loan amount / annual income",
    "credit_history_months": "length of credit history (months)",
    "emp_length": "years in current job",
    "home_ownership": "home ownership",
    "purpose": "loan purpose",
    "loan_amnt": "loan amount",
    "mort_acc": "number of mortgage accounts",
    "open_acc": "number of open credit lines",
    "total_acc": "total number of credit lines",
    "revol_bal": "revolving balance",
    "delinq_2yrs": "delinquencies in the last 2 years",
    "pub_rec": "derogatory public records",
    "pub_rec_bankruptcies": "bankruptcies on record",
    "mths_since_last_delinq": "months since last delinquency",
    "mths_since_last_record": "months since last public record",
    "application_type": "individual or joint application",
    "verification_status": "income verification",
}


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def candidates(specs):
    """Characteristics that may enter, strongest first."""
    usable = [s for s in specs.values() if s["iv"] >= IV_MIN and s["name"] not in NOT_IN_SCORECARD]
    return [s["name"] for s in sorted(usable, key=lambda s: -s["iv"])]


def select(specs, Xw, max_features):
    """Greedy by IV: add a characteristic unless its WoE is too correlated with one already chosen."""
    chosen = []
    corr = Xw.corr().abs()
    for name in candidates(specs):
        if all(corr.loc[name, c] <= CORR_MAX for c in chosen):
            chosen.append(name)
        if len(chosen) == max_features:
            break
    return chosen


def _logit(Xw, y):
    return LogisticRegression(C=1e6, max_iter=2000).fit(Xw, y)


def fit(specs, Xw, y, max_features):
    """Select, fit, and drop characteristics with the wrong sign until all signs are right.

    With WoE inputs every coefficient must be negative (safer bin -> lower default odds).
    A positive one means two characteristics overlap, and the scorecard would reward risk.
    """
    chosen = select(specs, Xw, max_features)
    while True:
        model = _logit(Xw[chosen], y)
        wrong = [name for name, b in zip(chosen, model.coef_[0]) if b >= 0]
        if not wrong:
            break
        chosen.remove(min(wrong, key=lambda n: specs[n]["iv"]))
    return build(specs, chosen, model)


def build(specs, chosen, model):
    k = len(chosen)
    intercept = float(model.intercept_[0])
    base = (OFFSET - FACTOR * intercept) / k  # the intercept is spread evenly over the characteristics
    card = {
        "scale": {"base_score": BASE_SCORE, "base_odds": BASE_ODDS, "pdo": PDO},
        "intercept": intercept,
        "characteristics": [],
    }
    for name, coef in zip(chosen, model.coef_[0]):
        spec = dict(specs[name])
        spec["coef"] = float(coef)
        spec["description"] = DESCRIPTION.get(name, name)
        spec["points"] = [int(round(-FACTOR * coef * w + base)) for w in spec["woe"]]
        card["characteristics"].append(spec)
    return card


def points(card, X):
    """Points per characteristic for every applicant (one column per characteristic)."""
    return pd.DataFrame(
        {c["name"]: np.asarray(c["points"])[binning.bin_codes(c, X[c["name"]])] for c in card["characteristics"]},
        index=X.index,
    )


def score(card, X):
    return points(card, X).sum(axis=1)


def pd_from_score(s):
    """Probability of default implied by the scale (before any recalibration)."""
    return 1 / (1 + np.exp((np.asarray(s, dtype=float) - OFFSET) / FACTOR))


def reasons(card, X, top=3):
    """The top reasons per applicant: characteristics where they lost the most points vs the best bin."""
    pts = points(card, X)
    best = pd.Series({c["name"]: max(c["points"]) for c in card["characteristics"]})
    lost = best - pts
    labels = {c["name"]: c for c in card["characteristics"]}
    out = []
    for i, row in lost.iterrows():
        items = []
        for name, gap in row.sort_values(ascending=False).items():
            if gap <= 0 or len(items) == top:
                break
            c = labels[name]
            label = c["labels"][binning.bin_codes(c, X.loc[[i], name])[0]]
            items.append({"characteristic": name, "text": f"{c['description']}: {label}", "points_lost": int(gap)})
        out.append(items)
    return out


def table(card):
    """The scorecard as a long table: characteristic, bin, loans, bad rate, WoE, points."""
    rows = []
    for c in card["characteristics"]:
        for label, n, bad, w, p in zip(c["labels"], c["n"], c["bad_rate"], c["woe"], c["points"]):
            if n == 0 and label == "missing":
                continue
            rows.append(
                {"characteristic": c["description"], "bin": label, "loans": n, "bad_rate": bad, "woe": w, "points": p}
            )
    return pd.DataFrame(rows)


def vif(Xw):
    """Variance inflation factor per column: 1 / (1 - R^2 against the other columns)."""
    inv = np.linalg.inv(np.corrcoef(Xw.to_numpy(), rowvar=False))
    return pd.Series(np.diag(inv), index=Xw.columns).round(2)
