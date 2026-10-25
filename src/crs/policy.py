"""Lending rules that run before the score: a loan outside them is declined, whatever the scorecard says.

Why they exist: the scorecard learned only from loans Lending Club approved, and Lending Club never approved
a loan of 8 times someone's income or a FICO score of 550. For such applications the scorecard has no
evidence: its last bins are open-ended ("33 % or more of income", "664 or less"), so it would score an extreme
case like the worst case it has seen and might approve it. Real lenders apply rules like these first, too.

Each limit sits at the edge of the approved 2007-2014 loans; `make policy` writes reports/policy.md with
the evidence (how many approved loans each rule would have caught).
"""

import json

import pandas as pd

from crs import config, features, target

REPORT_PATH = config.ROOT / "reports" / "policy.md"
# The new loan's monthly payment, for debt-to-income including the new loan, is computed at
# the average Lending Club rate of 2007-2014 (or at the break-even rate, if that is higher).
ASSUMED_RATE = 0.1264

# name, field it reads, limit, how it fails, text shown to the applicant
RULES = [
    ("min_fico", "fico_score", 660, "below", "FICO score below 660"),
    ("min_income", "annual_income", 10_000, "below", "annual income below $10,000"),
    ("max_loan", "loan_amount", 35_000, "above", "loan above $35,000"),
    ("max_loan_to_income", "loan_to_income", 0.50, "above", "loan above 50 % of annual income"),
    ("max_dti", "debt_to_income", 40, "above", "debt-to-income above 40 % before this loan"),
    ("max_dti_with_loan", "dti_with_loan", 50, "above", "debt-to-income above 50 % including this loan's payment"),
    ("max_inquiries", "inquiries_last_6_months", 7, "above", "more than 7 credit inquiries in 6 months"),
    ("max_utilisation", "revolving_utilisation", 100, "above", "revolving balances above the credit limit (over 100 %)"),
]


# Without these four a lender cannot decide at all, so a missing one declines the application.
REQUIRED = {
    "fico_score": "FICO score not given",
    "annual_income": "annual income not given",
    "loan_amount": "loan amount not given",
    "debt_to_income": "debt-to-income not given",
}


def monthly_payment(amount, rate, term=target.TERM):
    m = rate / 12
    return amount * m / (1 - (1 + m) ** -term)


def derived(application, rate=ASSUMED_RATE):
    """The two values the rules need that the form does not ask for directly."""
    amount, income, dti = (application.get(k) for k in ("loan_amount", "annual_income", "debt_to_income"))
    out = {"loan_to_income": None, "dti_with_loan": None}
    if amount is not None and income:
        out["loan_to_income"] = amount / income
        if dti is not None:
            out["dti_with_loan"] = dti + 100 * monthly_payment(amount, rate) / (income / 12)
    return out


def check(application, rate=ASSUMED_RATE):
    """The rules this application breaks (empty: it may be scored). Apart from REQUIRED, a missing value breaks no rule."""
    values = dict(application) | derived(application, max(rate, ASSUMED_RATE))
    broken = [{"rule": "required", "text": text, "value": None, "limit": None}
              for field, text in REQUIRED.items() if application.get(field) is None]
    for name, field, limit, how, text in RULES:
        v = values.get(field)
        if v is None:
            continue
        if (how == "below" and v < limit) or (how == "above" and v > limit):
            broken.append({"rule": name, "text": text, "value": round(float(v), 4), "limit": limit})
    return broken


def as_dict():
    return {
        "assumed_rate": ASSUMED_RATE,
        "required": REQUIRED,
        "rules": [dict(zip(["name", "field", "limit", "how", "text"], r)) for r in RULES],
    }


def evidence(loans):
    """How many approved 2007-2014 loans each rule would have declined: small numbers mean the limit sits
    at the edge of what the lender approved, which is where the scorecard's evidence ends."""
    pop = target.population(loans)
    dev, _ = target.split(pop)
    X = features.build(dev)
    frame = pd.DataFrame(
        {
            "fico_score": X["fico"],
            "annual_income": X["annual_inc"],
            "loan_amount": X["loan_amnt"],
            "loan_to_income": X["loan_to_income"],
            "debt_to_income": X["dti"],
            # the real instalment, at the loan's real rate
            "dti_with_loan": X["dti"] + 1200 * dev["installment"] / X["annual_inc"].where(X["annual_inc"] > 0),
            "inquiries_last_6_months": X["inq_last_6mths"],
            "revolving_utilisation": X["revol_util"],
        }
    )
    rows, any_broken = [], pd.Series(False, index=frame.index)
    for name, field, limit, how, text in RULES:
        broken = frame[field] < limit if how == "below" else frame[field] > limit
        any_broken |= broken.fillna(False)
        rows.append(
            {
                "rule (decline if)": text,
                "approved loans breaking it": int(broken.sum()),
                "share": round(float(broken.mean()), 5),
                "their default rate": round(float(dev.loc[broken.fillna(False), "bad"].mean()), 3) if broken.any() else None,
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "loans": len(dev),
        "breaking any rule": int(any_broken.sum()),
        "share": round(float(any_broken.mean()), 4),
        "default rate of those": round(float(dev.loc[any_broken, "bad"].mean()), 3),
        "default rate of the rest": round(float(dev.loc[~any_broken, "bad"].mean()), 3),
    }
    return table, summary


def run(loans, log=print):
    table, summary = evidence(loans)
    text = "\n\n".join(
        [
            "# Lending rules before the score (generated by `make policy`)",
            "Each rule sits at the edge of the loans Lending Club approved in 2007-2014, where the scorecard's evidence ends. "
            "Debt-to-income including the new loan uses each loan's real instalment here; for a new application it is "
            f"computed at {ASSUMED_RATE:.2%} (the average 2007-2014 rate) or the break-even rate, if higher.",
            table.to_markdown(index=False),
            f"Together the rules would have declined {summary['breaking any rule']:,} of {summary['loans']:,} approved loans "
            f"({summary['share']:.2%}). Their default rate: {summary['default rate of those']:.1%}, "
            f"against {summary['default rate of the rest']:.1%} for the rest.",
        ]
    ) + "\n"
    REPORT_PATH.write_text(text)
    (config.ROOT / "models" / "policy.json").write_text(json.dumps(as_dict() | {"evidence": summary}, indent=1))
    log(text)
    return table, summary
