"""Readable bin labels for people: '660 to 674' instead of '664.5 to < 674.5', 'rents' instead of 'RENT'.

The cut points are unchanged; only how a bin is named. Count characteristics are cut at .5,
so their bins are whole numbers. The last label is always the missing bin.
"""

import math

CATEGORY_NAMES = {
    "MORTGAGE": "owns with a mortgage",
    "OWN": "owns outright",
    "RENT": "rents",
    "OTHER": "other",
    "debt_consolidation": "debt consolidation",
    "credit_card": "paying off credit cards",
    "home_improvement": "home improvement",
    "major_purchase": "major purchase",
    "small_business": "small business",
    "car": "car",
    "medical": "medical",
    "other": "other",
}
COUNTS = {"inq_last_6mths", "mort_acc", "delinq_2yrs", "pub_rec", "pub_rec_bankruptcies", "open_acc", "total_acc"}
WHOLE = {"fico", "credit_history_months", "emp_length"}
PERCENT = {"dti", "revol_util"}
MONEY = {"annual_inc", "loan_amnt", "revol_bal"}
SHARE = {"loan_to_income"}


def _money(x):
    return f"${x:,.0f}"


def _fmt(name, x):
    if name in PERCENT:
        return f"{x:.1f} %"
    if name in SHARE:
        return f"{x * 100:.0f} %"
    if name in MONEY:
        return _money(round(x, -2))
    return f"{x:g}"


def labels(spec):
    name = spec["name"]
    if spec["type"] == "categorical":
        out = []
        for group in spec["groups"]:
            names = list(dict.fromkeys(CATEGORY_NAMES.get(c, c.replace("_", " ")) for c in group))
            names.sort(key=lambda n: n == "other")  # "other" last
            out.append(names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1])
        return out + ["not given"]

    edges = [-math.inf] + list(spec["splits"]) + [math.inf]
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        if name in COUNTS or name in WHOLE:
            lo = None if low == -math.inf else math.ceil(low)  # cut points sit at .5
            hi = None if high == math.inf else math.ceil(high) - 1
            if lo is None:
                out.append(f"{hi}" if name in COUNTS and hi == 0 else f"{hi} or less")
            elif hi is None:
                out.append(f"{lo} or more")
            elif lo == hi:
                out.append(f"{lo}")
            else:
                out.append(f"{lo} to {hi}")
        else:
            if low == -math.inf:
                out.append(f"below {_fmt(name, high)}")
            elif high == math.inf:
                out.append(f"{_fmt(name, low)} or more")
            else:
                out.append(f"{_fmt(name, low)} to {_fmt(name, high)}")
    suffix = " of income" if name in SHARE else ""
    return [label + suffix for label in out] + ["not given"]
