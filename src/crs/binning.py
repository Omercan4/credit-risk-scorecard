"""Weight-of-Evidence binning: cut each characteristic into a few bins and give every bin a WoE value.

WoE of a bin = ln(share of all goods in the bin / share of all bads in the bin).
Positive WoE = safer than average, negative = riskier. Missing values get their own bin.

optbinning finds the cut points (with the default-rate trend forced where it is a business rule,
for example a higher debt-to-income ratio must never look safer). The result is turned into a plain
dict, so scoring later needs only that dict: the API and the public page use the same numbers.
"""

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

from crs import features

# Default rate must rise (ascending) or fall (descending) with the value. The rest: optbinning picks.
TREND = {
    "fico": "descending",
    "annual_inc": "descending",
    "credit_history_months": "descending",
    "mort_acc": "descending",
    "dti": "ascending",
    "revol_util": "ascending",
    "inq_last_6mths": "ascending",
    "loan_to_income": "ascending",
    "delinq_2yrs": "ascending",
    "pub_rec": "ascending",
    "pub_rec_bankruptcies": "ascending",
}
MIN_BIN_SHARE = 0.05  # a bin needs at least 5 % of the loans, so its default rate is stable
MAX_BINS = 8
RARE_SHARE = 0.01  # categories below 1 % are merged into "OTHER"
OTHER = "OTHER"
# A missing bin this small has no reliable default rate (29 loans can look like the safest group by chance).
# It gets the WoE of the riskiest bin instead, so leaving a field empty never earns points.
SMALL_MISSING_SHARE = 0.01


def _bin_stats(codes, y, n_bins):
    """Loans, bad rate and WoE per bin code (0..n_bins-1 = bins, n_bins = missing)."""
    frame = pd.DataFrame({"code": codes, "bad": y})
    stats = frame.groupby("code")["bad"].agg(n="size", bads="sum").reindex(range(n_bins + 1), fill_value=0)
    goods = stats["n"] - stats["bads"]
    total_goods, total_bads = goods.sum(), stats["bads"].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        woe = np.log((goods / total_goods) / (stats["bads"] / total_bads))
        iv = ((goods / total_goods - stats["bads"] / total_bads) * woe).replace([np.inf, -np.inf], np.nan).fillna(0)
    woe = woe.replace([np.inf, -np.inf], np.nan).fillna(0.0)  # an empty bin (e.g. no missing values) is neutral
    return stats, woe, float(iv.sum())


def fit_one(name, x, y):
    if name in features.CATEGORICAL:
        return _fit_categorical(name, x, y)
    return _fit_numerical(name, x, y)


def _fit_numerical(name, x, y):
    if x.notna().mean() < MIN_BIN_SHARE:
        # Barely recorded in these years (mort_acc only exists from 2012): one bin plus missing, nothing to learn.
        return _finish({"name": name, "type": "numerical", "splits": []}, x, y)
    ob = OptimalBinning(
        name=name,
        dtype="numerical",
        monotonic_trend=TREND.get(name, "auto_asc_desc"),
        min_bin_size=MIN_BIN_SHARE,
        max_n_bins=MAX_BINS,
    )
    ob.fit(x.to_numpy(dtype=float), y)
    spec = {"name": name, "type": "numerical", "splits": [float(s) for s in ob.splits]}
    return _finish(spec, x, y)


def _fit_categorical(name, x, y):
    share = x.value_counts(normalize=True)
    keep = sorted(share[share >= RARE_SHARE].index)
    x = x.where(x.isin(keep) | x.isna(), OTHER)
    ob = OptimalBinning(name=name, dtype="categorical", min_bin_size=MIN_BIN_SHARE, max_n_bins=MAX_BINS)
    ob.fit(x.to_numpy(dtype=object), y)
    groups = [sorted(str(c) for c in group) for group in ob.splits]
    spec = {"name": name, "type": "categorical", "keep": keep, "groups": groups}
    return _finish(spec, x, y)


def _finish(spec, x, y):
    codes = bin_codes(spec, x)
    stats, woe, iv = _bin_stats(codes, np.asarray(y), n_bins(spec))
    if stats["n"].iloc[-1] < SMALL_MISSING_SHARE * len(codes):
        woe.iloc[-1] = woe.iloc[:-1][stats["n"].iloc[:-1] > 0].min()
    spec["labels"] = labels(spec)
    spec["woe"] = [round(float(w), 6) for w in woe]
    spec["n"] = [int(n) for n in stats["n"]]
    spec["bad_rate"] = [round(float(b / n), 4) if n else None for b, n in zip(stats["bads"], stats["n"])]
    spec["iv"] = round(iv, 4)
    return spec


def n_bins(spec):
    return len(spec["splits"]) + 1 if spec["type"] == "numerical" else len(spec["groups"])


def bin_codes(spec, x):
    """Bin number per value; the last code (n_bins) is the missing bin.

    Numerical bins are [split_i, split_i+1). An unseen category goes to the OTHER bin if there is one,
    otherwise to the missing bin (WoE 0 when nothing was missing in training).
    """
    x = pd.Series(x)
    missing = n_bins(spec)
    if spec["type"] == "numerical":
        values = x.to_numpy(dtype=float)
        codes = np.digitize(values, spec["splits"], right=False)
        return np.where(np.isnan(values), missing, codes)
    lookup = {cat: i for i, group in enumerate(spec["groups"]) for cat in group}
    mapped = x.where(x.isin(spec["keep"]) | x.isna(), OTHER)
    fallback = lookup.get(OTHER, missing)
    return np.array([missing if pd.isna(v) else lookup.get(str(v), fallback) for v in mapped])


def labels(spec):
    if spec["type"] == "categorical":
        return [", ".join(group) for group in spec["groups"]] + ["missing"]
    edges = [-np.inf] + spec["splits"] + [np.inf]
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        if low == -np.inf:
            out.append(f"< {high:g}")
        elif high == np.inf:
            out.append(f">= {low:g}")
        else:
            out.append(f"{low:g} to < {high:g}")
    return out + ["missing"]


def woe(spec, x):
    return np.asarray(spec["woe"])[bin_codes(spec, x)]


def fit_all(X, y, names=None):
    names = names or list(X.columns)
    return {name: fit_one(name, X[name], np.asarray(y)) for name in names}


def transform(specs, X):
    return pd.DataFrame({name: woe(spec, X[name]) for name, spec in specs.items()}, index=X.index)
