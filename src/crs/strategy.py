"""Approve or decline: a cut-off chosen by expected profit, then checked against the money actually made in 2015.

Expected profit of one loan = (1 - PD) * gain if repaid  -  PD * loss if it defaults.
- gain if repaid: the instalments over 36 months minus the amount lent (the price is known when the loan is offered),
- loss if it defaults: the amount lent times the average loss rate of defaulted loans in 2007-2014
  (what was paid back before the default, plus recoveries, is already in that rate).
Approve when the expected profit is positive. Lending Club's investors paid a 1 % fee on every payment.

Money is counted as paid, not discounted, and there are no funding costs: a simple, stated view.
"""

import numpy as np
import pandas as pd

SERVICE_FEE = 0.01
TERM = 36


def realised_profit(loans):
    """What each loan actually earned or lost (after the fee)."""
    return (1 - SERVICE_FEE) * loans["total_pymnt"] - loans["funded_amnt"]


def gain_if_repaid(loans):
    return (1 - SERVICE_FEE) * loans["installment"] * TERM - loans["funded_amnt"]


def loss_rate(dev):
    """Average share of the amount lent that is lost on a defaulted loan."""
    bad = dev[dev["bad"] == 1]
    return float((-(realised_profit(bad)) / bad["funded_amnt"]).mean())


def expected_profit(loans, pd_, lgd):
    pd_ = np.asarray(pd_)
    return (1 - pd_) * gain_if_repaid(loans).to_numpy() - pd_ * lgd * loans["funded_amnt"].to_numpy()


def outcome(loans, approve):
    """Approval rate, bad rate and realised profit of the approved loans."""
    approve = np.asarray(approve, dtype=bool)
    chosen = loans[approve]
    return {
        "approval_rate": round(float(approve.mean()), 4),
        "bad_rate": round(float(chosen["bad"].mean()), 4),
        "profit_musd": round(float(realised_profit(chosen).sum()) / 1e6, 2),
        "declined_loans": int((~approve).sum()),
        "declined_bad_rate": round(float(loans.loc[~approve, "bad"].mean()), 4) if (~approve).any() else None,
    }


def pick_share(loans, risk, share):
    """An investor who can fund only `share` of the loans and picks the safest by a ranking."""
    order = np.argsort(np.asarray(risk), kind="stable")[: int(share * len(loans))]
    chosen = loans.iloc[order]
    profit = realised_profit(chosen).sum()
    return {
        "profit_musd": round(float(profit) / 1e6, 2),
        "lent_musd": round(float(chosen["funded_amnt"].sum()) / 1e6, 1),
        "return_on_lent": round(float(profit / chosen["funded_amnt"].sum()), 4),
        "bad_rate": round(float(chosen["bad"].mean()), 4),
    }


def bootstrap_return_diff(loans, risk_a, risk_b, share, reps=200, seed=0):
    """Return on lent of ranking a minus ranking b when funding `share` of the loans, with a 95 % interval."""
    rng = np.random.default_rng(seed)
    profit, lent = realised_profit(loans).to_numpy(), loans["funded_amnt"].to_numpy()
    risk_a, risk_b = np.asarray(risk_a), np.asarray(risk_b)

    def ret(i, risk):
        top = i[np.argsort(risk[i], kind="stable")[: int(share * len(i))]]
        return profit[top].sum() / lent[top].sum()

    everyone = np.arange(len(loans))
    diffs = []
    for _ in range(reps):
        i = rng.integers(0, len(loans), len(loans))
        diffs.append(ret(i, risk_a) - ret(i, risk_b))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": round(float(ret(everyone, risk_a) - ret(everyone, risk_b)), 4),
        "low": round(float(low), 4),
        "high": round(float(high), 4),
    }


def curve(loans, risk, steps=100):
    """Realised profit when approving the safest x % of applicants by a model's ranking, for x = 1..100 %."""
    order = np.argsort(np.asarray(risk), kind="stable")
    cum = np.cumsum(realised_profit(loans).to_numpy()[order])
    shares = np.linspace(0.01, 1, steps)
    idx = np.maximum((shares * len(order)).astype(int) - 1, 0)
    return pd.DataFrame({"approval_rate": shares, "profit_musd": cum[idx] / 1e6})


def score_table(loans, score, cutoffs):
    """The classic strategy table: for each cut-off score, who is approved and what it earned."""
    rows = []
    for c in cutoffs:
        o = outcome(loans, np.asarray(score) >= c)
        rows.append({"cut-off score": c} | {k: o[k] for k in ["approval_rate", "bad_rate", "profit_musd"]})
    return pd.DataFrame(rows)
