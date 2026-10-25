"""Score one application: points per characteristic, PD, decline reasons and the break-even interest rate.

The API uses this module, and docs/scorer.js does the same in the browser from docs/scorecard.json
(a test checks that both give the same answer).

Decision, by risk-based pricing: the lowest annual rate at which a 36-month loan breaks even in expectation,
(1 - PD) x (instalments - amount) = PD x loss rate x amount, after the 1 % investor fee.
If even Lending Club's highest rate (30.99 %) does not cover the risk, decline.
Policy limit: also decline above a 40 % chance of default. In 2007-2014 the return on money lent fell with every
PD band and reached zero there (161 loans above 40 %: +0.2 %), and the scorecard has almost no evidence beyond it.
"""

import json

import pandas as pd

from crs import binning, config, scorecard, strategy

CARD_PATH = config.ROOT / "models" / "scorecard.json"
STRATEGY_PATH = config.ROOT / "models" / "strategy.json"
PAGE_CARD_PATH = config.ROOT / "docs" / "scorecard.json"
MAX_RATE = 0.3099  # the highest rate Lending Club charged
MAX_PD = 0.40  # policy limit, see above

# What an application form asks -> the scorecard's characteristic.
FIELDS = {
    "fico_score": "fico",
    "annual_income": "annual_inc",
    "debt_to_income": "dti",
    "mortgage_accounts": "mort_acc",
    "home_ownership": "home_ownership",
    "inquiries_last_6_months": "inq_last_6mths",
    "revolving_utilisation": "revol_util",
    "purpose": "purpose",
}


def load_card(path=CARD_PATH):
    return json.loads(path.read_text())


def loss_rate(path=STRATEGY_PATH):
    return json.loads(path.read_text())["loss_rate_if_default"]


def to_row(application):
    """One application (dict with FIELDS keys plus loan_amount) -> a one-row frame of characteristics."""
    row = {feature: application.get(field) for field, feature in FIELDS.items()}
    amount, income = application.get("loan_amount"), application.get("annual_income")
    row["loan_to_income"] = amount / income if amount is not None and income else None
    frame = pd.DataFrame([row])
    for col in frame.columns:
        if col not in ("home_ownership", "purpose"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    return frame


def annuity_factor(rate, term=strategy.TERM):
    """Monthly instalment per unit lent at an annual rate."""
    m = rate / 12
    return 1 / term if m == 0 else m / (1 - (1 + m) ** -term)


def break_even_rate(pd_, lgd, term=strategy.TERM, fee=strategy.SERVICE_FEE):
    """Lowest annual rate with expected profit >= 0 (bisection), or None above 100 %."""
    if pd_ >= 1:
        return None
    needed = (pd_ * lgd / (1 - pd_) + 1) / (1 - fee)  # instalments x term / amount that breaks even
    low, high = 0.0, 1.0
    if annuity_factor(high, term) * term < needed:
        return None
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if annuity_factor(mid, term) * term < needed else (low, mid)
    return high


def score_application(application, card=None, lgd=None):
    card = card or load_card()
    lgd = lgd if lgd is not None else loss_rate()
    X = to_row(application)
    pts = scorecard.points(card, X).iloc[0]
    total = int(pts.sum())
    pd_ = float(scorecard.pd_from_score(total))
    rate = break_even_rate(pd_, lgd)
    if pd_ > MAX_PD:
        decision, why = "decline", "pd above policy limit"
    elif rate is None or rate > MAX_RATE:
        decision, why = "decline", "no rate covers the expected loss"
    else:
        decision, why = "approve", "rate covers the expected loss"
    lines = []
    for c in card["characteristics"]:
        code = int(binning.bin_codes(c, X[c["name"]])[0])
        lines.append(
            {
                "characteristic": c["name"],
                "description": c["description"],
                "bin": c["labels"][code],
                "points": int(pts[c["name"]]),
                "max_points": max(c["points"]),
            }
        )
    return {
        "score": total,
        "pd": round(pd_, 4),
        "break_even_rate": None if rate is None else round(rate, 4),
        "decision": decision,
        "decision_rule": why,
        "reasons": scorecard.reasons(card, X)[0],
        "points": lines,
    }


def page_card(card=None, lgd=None):
    """The part of the scorecard the browser needs (no counts beyond what the report already shows)."""
    card = card or load_card()
    keep = ["name", "type", "description", "labels", "points", "splits", "keep", "groups"]
    return {
        "scale": card["scale"],
        "offset": scorecard.OFFSET,
        "factor": scorecard.FACTOR,
        "loss_rate": lgd if lgd is not None else loss_rate(),
        "max_rate": MAX_RATE,
        "max_pd": MAX_PD,
        "fee": strategy.SERVICE_FEE,
        "term": strategy.TERM,
        "fitted_on": card.get("fitted_on"),
        "characteristics": [{k: c[k] for k in keep if k in c} for c in card["characteristics"]],
    }


def export_page_card(path=PAGE_CARD_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(page_card(), indent=1))
    return path
