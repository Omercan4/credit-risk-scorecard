"""Monitoring after go-live: the validated scorecard (fitted on 2007-2014) scores every quarter from 2015 Q1 to 2018 Q4.

Two kinds of checks, as in a bank:
1. Before outcomes exist (every quarter, from day one): are the applicants still like the development sample?
   PSI on the score, CSI on each characteristic, with the usual traffic light.
2. As outcomes arrive: early defaults (bad within the first 12 months) observed vs expected, and the Gini on them.
   A full 36-month outcome takes 3+ years; the 12-month view is known about 15 months after issue.

Expected early-default rates come from a 12-month PD fitted on the development years only.
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from crs import binning, config, features, scorecard, target

REPORT_PATH = config.ROOT / "reports" / "monitoring.md"
RESULTS_PATH = config.ROOT / "models" / "monitoring.json"
PLOT_PATH = config.ROOT / "docs" / "img" / "monitoring.png"
VINTAGE_PLOT = config.ROOT / "docs" / "img" / "vintages.png"

LIVE_FROM = pd.Timestamp("2015-01-01")
SNAPSHOT = pd.Timestamp("2019-03-01")  # the last payment date in the data
EARLY_MONTHS = 12
EARLY_OBSERVABLE_LAG = 15  # months after issue before a 12-month default shows up as late or charged off
AMBER, RED = 0.10, 0.25
OE_AMBER, OE_RED = 1.10, 1.25  # observed / expected early defaults (or the same distance below 1)
BAD_NOW = ["Charged Off", "Default", "Late (31-120 days)"]


def psi(expected, actual, eps=1e-4):
    """Population stability index between two distributions given as shares (or counts) over the same bins."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e, a = e / e.sum() + eps, a / a.sum() + eps
    return float(np.sum((a - e) * np.log(a / e)))


def light(value):
    return "red" if value >= RED else "amber" if value >= AMBER else "green"


def oe_light(ratio):
    if pd.isna(ratio):
        return ""
    gap = max(ratio, 1 / ratio)
    return "red" if gap >= OE_RED else "amber" if gap >= OE_AMBER else "green"


def months_between(start, end):
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)


def early_bad(loans):
    """Went bad within the first 12 months: now charged off / defaulted / 31+ days late, and the last payment
    came in the first 12 months (or never)."""
    mob = months_between(loans["issue_d"], loans["last_pymnt_d"])
    return (loans["loan_status"].isin(BAD_NOW) & (mob.isna() | (mob <= EARLY_MONTHS))).astype(int)


def early_observable(loans):
    return loans["issue_d"] <= SNAPSHOT - pd.DateOffset(months=EARLY_OBSERVABLE_LAG)


def score_bins(dev_score, n=10):
    """Cut points for 10 score bands of roughly equal size in the development sample."""
    edges = np.unique(np.quantile(dev_score, np.linspace(0, 1, n + 1)[1:-1]))
    return edges


def shares(codes, n):
    return np.bincount(np.asarray(codes), minlength=n)[:n] / len(codes)


def run(loans, card_path=config.ROOT / "models" / "scorecard.json", log=print):
    card = json.loads(card_path.read_text())
    pop = target.population(loans)
    dev, _ = target.split(pop)
    X_dev = features.build(dev)
    s_dev = scorecard.score(card, X_dev).to_numpy()
    edges = score_bins(s_dev)
    base_score = shares(np.digitize(s_dev, edges), len(edges) + 1)
    base_csi = {
        c["name"]: shares(binning.bin_codes(c, X_dev[c["name"]]), binning.n_bins(c) + 1) for c in card["characteristics"]
    }

    # 12-month PD from the development years only (all of them have a full history)
    pd12 = LogisticRegression().fit(s_dev.reshape(-1, 1), early_bad(dev))

    live = loans[(loans["term"] == target.TERM) & (loans["issue_d"] >= LIVE_FROM)].copy()
    live["quarter"] = live["issue_d"].dt.to_period("Q").astype(str)
    log(f"Scoring {len(live):,} applications, {live['quarter'].min()} to {live['quarter'].max()} ...")
    X_live = features.build(live)
    live["score"] = scorecard.score(card, X_live).to_numpy()
    live["early_bad"] = early_bad(live)
    live["pd12"] = pd12.predict_proba(live[["score"]].to_numpy())[:, 1]

    rows, csi_rows = [], []
    for q, idx in live.groupby("quarter").groups.items():
        part, Xq = live.loc[idx], X_live.loc[idx]
        row = {
            "quarter": q,
            "applications": len(part),
            "mean score": round(float(part["score"].mean()), 1),
            "score PSI": round(psi(base_score, shares(np.digitize(part["score"], edges), len(edges) + 1)), 4),
        }
        for c in card["characteristics"]:
            n = binning.n_bins(c)
            now = shares(binning.bin_codes(c, Xq[c["name"]]), n + 1)
            value = psi(base_csi[c["name"]], now)
            present = psi(base_csi[c["name"]][:n], now[:n])  # the same, ignoring the missing bin
            csi_rows.append({"quarter": q, "characteristic": c["name"], "CSI": round(value, 4), "CSI present": round(present, 4)})
        if early_observable(part).all():
            observed, expected = part["early_bad"].mean(), part["pd12"].mean()
            row |= {
                "early bad observed": round(float(observed), 4),
                "early bad expected": round(float(expected), 4),
                "observed / expected": round(float(observed / expected), 2),
                "Gini on early bad": round(scorecard.gini(part["early_bad"], -part["score"]), 4),
                "known from": str(pd.Period(q, "Q").asfreq("M", "end") + EARLY_OBSERVABLE_LAG),
            }
        rows.append(row)

    table = pd.DataFrame(rows).set_index("quarter")
    table["score light"] = table["score PSI"].map(light)
    table["outcome light"] = table["observed / expected"].map(oe_light)
    long = pd.DataFrame(csi_rows)
    csi = long.pivot(index="quarter", columns="characteristic", values="CSI")[card_order(card)]
    csi_present = long.pivot(index="quarter", columns="characteristic", values="CSI present")[card_order(card)]
    worst = csi.max().sort_values(ascending=False)
    vintages = vintage_curves(loans)
    drivers = shift_drivers(dev, live, X_dev, X_live, worst.index[0])

    results = {
        "live_model": card.get("fitted_on"),
        "quarters": table.reset_index().to_dict(orient="records"),
        "csi": csi.reset_index().to_dict(orient="records"),
        "worst_characteristics": worst.round(4).to_dict(),
        "worst_present_only": csi_present.max().round(4).to_dict(),
        "dev_early_bad_rate": round(float(early_bad(dev).mean()), 4),
        "shift": drivers,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=1, default=float))
    REPORT_PATH.write_text(report(table, csi, worst, csi_present, drivers, results))
    plot(table, csi_present, PLOT_PATH)
    plot_vintages(vintages, VINTAGE_PLOT)
    log(f"Wrote {RESULTS_PATH}, {REPORT_PATH}, {PLOT_PATH} and {VINTAGE_PLOT}")
    return results


def card_order(card):
    return [c["name"] for c in card["characteristics"]]


def shift_drivers(dev, live, X_dev, X_live, name):
    """The most shifted characteristic: its distribution in development vs each year after go-live."""
    years = live["issue_d"].dt.year
    out = {"characteristic": name, "development 2007-2014": describe(X_dev[name])}
    for y in sorted(years.unique()):
        out[str(y)] = describe(X_live.loc[(years == y).to_numpy(), name])
    return out


def describe(x):
    if not pd.api.types.is_numeric_dtype(x):
        return {k: round(float(v), 3) for k, v in x.value_counts(normalize=True).head(5).items()}
    return {
        "median": round(float(x.median()), 2),
        "p10": round(float(x.quantile(0.1)), 2),
        "p90": round(float(x.quantile(0.9)), 2),
        "missing": round(float(x.isna().mean()), 3),
    }


def vintage_curves(loans, years=range(2012, 2018)):  # 2018 loans are too young to show
    """Cumulative share of loans that went bad, by months on book, per issue year (36-month loans)."""
    m = loans[(loans["term"] == target.TERM) & loans["issue_d"].dt.year.isin(years)].copy()
    m["year"] = m["issue_d"].dt.year
    m["bad_now"] = m["loan_status"].isin(BAD_NOW)
    m["mob_bad"] = months_between(m["issue_d"], m["last_pymnt_d"]).fillna(0)
    curves = {}
    for y, part in m.groupby("year"):
        horizon = min(36, int(months_between(pd.Series([pd.Timestamp(f"{y}-12-01")]), pd.Series([SNAPSHOT])).iloc[0]) - 4)
        months = np.arange(1, horizon + 1)
        curves[int(y)] = pd.Series([np.mean(part["bad_now"] & (part["mob_bad"] <= k)) for k in months], index=months)
    return curves


def report(table, csi, worst, csi_present, drivers, results):
    lights = {"green": "green", "amber": "AMBER", "red": "RED"}
    csi_view = csi.copy().round(3)
    return "\n\n".join(
        [
            "# Monitoring after go-live (generated by `make monitor`)",
            f"Live model: the validated scorecard ({results['live_model']}). Every 36-month application from 2015 Q1 "
            f"is scored. Traffic light for PSI / CSI: below {AMBER} green, {AMBER}-{RED} amber, {RED} or more red.",
            "## Every quarter",
            table.assign(
                **{"score light": table["score light"].map(lights), "outcome light": table["outcome light"].map(lambda v: lights.get(v, ""))}
            ).to_markdown(),
            f"Early bad = went bad in the first {EARLY_MONTHS} months; shown once it can be observed "
            f"(issued at least {EARLY_OBSERVABLE_LAG} months before the data ends). "
            f"Development early-bad rate: {results['dev_early_bad_rate']:.2%}. Outcome light: observed / expected "
            f"at least {OE_AMBER} amber, at least {OE_RED} red (or the same distance below 1). 'known from': the month "
            "when that quarter's early defaults can first be counted.",
            "## Characteristic stability (CSI) per quarter",
            csi_view.to_markdown(),
            "## Largest CSI per characteristic over all quarters",
            pd.DataFrame(
                {"max CSI": worst.round(3), "light": worst.map(light), "max CSI, missing bin ignored": csi_present.max()[worst.index].round(3)}
            ).to_markdown(),
            "A red CSI that disappears when the missing bin is ignored is a recording artifact, not a change in applicants.",
            f"## The biggest shift: {drivers['characteristic']}",
            pd.DataFrame({k: v for k, v in drivers.items() if k != "characteristic"}).to_markdown(),
        ]
    ) + "\n"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot(table, csi_present, path):
    """Top: input stability (score PSI and the three most shifted characteristics, missing bin ignored).
    Bottom: observed / expected early defaults, plotted at the month they become known."""
    plt = _plt()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.6, 6.4), dpi=150, sharex=True, height_ratios=[1, 1])
    x = np.arange(len(table))
    worst = csi_present.max().sort_values(ascending=False).index[:3]
    ymax = max(0.3, float(csi_present[worst].max().max()) * 1.15)
    top.axhspan(RED, ymax, color="#c0504d", alpha=0.08)
    top.axhspan(AMBER, RED, color="#e3a33b", alpha=0.10)
    top.plot(x, table["score PSI"], color="#1f2d3d", lw=2.2, marker="o", ms=3.5, label="score PSI")
    for name, color in zip(worst, ["#c0504d", "#e3a33b", "#5b9a4b"]):
        top.plot(x, csi_present[name], color=color, lw=1.4, ls="--", label=f"CSI {name}")
    top.set(ylabel="PSI / CSI", ylim=(0, ymax))
    top.set_title("Inputs: are the applicants still like 2007-2014?", fontsize=10, loc="left")
    top.legend(frameon=False, fontsize=8, loc="upper left")

    oe = table["observed / expected"]
    bottom.axhspan(OE_RED, 1.7, color="#c0504d", alpha=0.08)
    bottom.axhspan(OE_AMBER, OE_RED, color="#e3a33b", alpha=0.10)
    bottom.axhline(1, color="#999999", lw=1, ls="--")
    bottom.plot(x, oe, color="#c0504d", lw=2.2, marker="o", ms=3.5)
    for i, (q, v) in enumerate(oe.items()):
        if pd.notna(v) and i % 3 == 1:
            bottom.annotate(f"known {table.loc[q, 'known from']}", (i, v), xytext=(0, 8), textcoords="offset points",
                            fontsize=6.5, ha="center", color="#555555")
    bottom.set(ylabel="observed / expected", ylim=(0.8, 1.7))
    bottom.set_title("Outcomes: early defaults (first 12 months) vs what the model expected", fontsize=10, loc="left")
    bottom.set_xticks(x[::2], table.index[::2], fontsize=8)
    for ax in (top, bottom):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_vintages(curves, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    cmap = plt.get_cmap("viridis")
    for i, (year, c) in enumerate(curves.items()):
        ax.plot(c.index, c.to_numpy() * 100, lw=1.8, color=cmap(i / max(len(curves) - 1, 1)), label=str(year))
    ax.set(xlabel="months on book", ylabel="loans gone bad (%)", xlim=(0, 36))
    ax.set_title("Vintage curves: share of each year's 36-month loans that went bad", fontsize=11)
    ax.legend(frameon=False, fontsize=8, title="issued", title_fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
