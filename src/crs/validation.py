"""The out-of-time test on 2015: scorecard vs Lending Club's own grade vs the LightGBM challenger.

2015 was never used for any choice. Everything fitted here uses 2007-2014 only:
- the scorecard (models/scorecard.json),
- the benchmark: Lending Club's sub-grade, turned into a PD by its 2007-2014 default rate,
- the challenger: number of trees by early stopping on 2014, then refitted on 2007-2014,
- the recalibration of the scorecard: a logistic fit of default on score, on the 2014 loans.
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from crs import challenger, config, develop, features, scorecard, target

REPORT_PATH = config.ROOT / "reports" / "validation.md"
RESULTS_PATH = config.ROOT / "models" / "validation.json"
PLOT_PATH = config.ROOT / "docs" / "img" / "calibration.png"
BOOTSTRAP_REPS = 200


def gini(y, risk):
    return 2 * roc_auc_score(y, risk) - 1


def ks(y, risk):
    fpr, tpr, _ = roc_curve(y, risk)
    return float(np.max(tpr - fpr))


def summary(y, risk, pd_=None):
    out = {"gini": round(gini(y, risk), 4), "ks": round(ks(y, risk), 4)}
    if pd_ is not None:
        out |= {"brier": round(brier_score_loss(y, pd_), 4), "mean_pd": round(float(np.mean(pd_)), 4)}
    return out


def bootstrap_diff(y, risk_a, risk_b, reps=BOOTSTRAP_REPS, seed=0):
    """Gini(a) - Gini(b) on the same resampled loans: the difference and its 95 % interval."""
    rng = np.random.default_rng(seed)
    y, risk_a, risk_b = map(np.asarray, (y, risk_a, risk_b))
    diffs = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        diffs.append(gini(y[i], risk_a[i]) - gini(y[i], risk_b[i]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(gini(y, risk_a) - gini(y, risk_b), 4), "low": round(low, 4), "high": round(high, 4)}


def calibration(y, pd_, bands=10):
    """Predicted vs observed default rate in 10 equal-sized bands of predicted PD."""
    frame = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_)})
    frame["band"] = pd.qcut(frame["pd"].rank(method="first"), bands, labels=range(1, bands + 1))
    out = frame.groupby("band", observed=True).agg(loans=("y", "size"), predicted=("pd", "mean"), observed=("y", "mean"))
    return out.round(4)


def subgrade_rank(sub_grade):
    """A1 -> 1 ... G5 -> 35: Lending Club's own risk ranking."""
    return sub_grade.map(lambda g: "ABCDEFG".index(g[0]) * 5 + int(g[1]))


def recalibrate(score_fit, y_fit):
    model = LogisticRegression().fit(np.asarray(score_fit, dtype=float).reshape(-1, 1), y_fit)
    return lambda s: model.predict_proba(np.asarray(s, dtype=float).reshape(-1, 1))[:, 1]


def leak_demo(oot):
    """How good a model looks with one column that is only known after the loan: share of principal repaid."""
    repaid = oot["total_rec_prncp"] / oot["funded_amnt"]
    return round(gini(oot["bad"], -repaid), 4)


def history(dev, card, rounds):
    """Gini per development test year: scorecard (from its time CV), challenger (same folds) and sub-grade."""
    X, y = features.build(dev), dev["bad"].to_numpy()
    k = develop.choose(pd.DataFrame(card["cv"]))
    cv = {r["test_year"]: r["gini"] for r in card["cv"] if r["max_features"] == k}
    out = {}
    for year in target.CV_TEST_YEARS:
        train, test = (dev["year"] < year).to_numpy(), (dev["year"] == year).to_numpy()
        model = challenger.fit(X[train], y[train], rounds)
        out[year] = {
            "scorecard": cv.get(year),
            "challenger": round(gini(y[test], challenger.predict(model, X[test])), 4),
            "sub-grade": round(gini(y[test], subgrade_rank(dev.loc[test, "sub_grade"])), 4),
        }
    return out


def run(loans, card_path=config.ROOT / "models" / "scorecard.json", log=print):
    card = json.loads(card_path.read_text())
    pop = target.population(loans)
    dev, oot = target.split(pop)
    X_dev, X_oot = features.build(dev), features.build(oot)
    y_dev, y_oot = dev["bad"].to_numpy(), oot["bad"].to_numpy()

    log("Scorecard and benchmark ...")
    s_dev, s_oot = scorecard.score(card, X_dev), scorecard.score(card, X_oot)
    raw_pd = scorecard.pd_from_score(s_oot)
    last = (dev["year"] == dev["year"].max()).to_numpy()
    recal_pd = recalibrate(s_dev[last], y_dev[last])(s_oot)
    grade_pd = oot["sub_grade"].map(dev.groupby("sub_grade")["bad"].mean()).fillna(y_dev.mean()).to_numpy()
    grade_risk = subgrade_rank(oot["sub_grade"]).to_numpy()

    log("Challenger: early stopping on 2014, then refit on 2007-2014 ...")
    earlier = ~last
    rounds = challenger.best_rounds(X_dev[earlier], y_dev[earlier], X_dev[last], y_dev[last])
    model = challenger.fit(X_dev, y_dev, rounds)
    ch_pd = challenger.predict(model, X_oot)

    log(f"Bootstrap ({BOOTSTRAP_REPS} resamples) ...")
    results = {
        "oot_year": target.OOT_YEAR,
        "oot_loans": int(len(oot)),
        "oot_bad_rate": round(float(y_oot.mean()), 4),
        "dev_bad_rate": round(float(y_dev.mean()), 4),
        "challenger_trees": int(rounds),
        "models": {
            "Lending Club sub-grade (benchmark)": summary(y_oot, grade_risk, grade_pd),
            "Scorecard, raw PD": summary(y_oot, -s_oot, raw_pd),
            "Scorecard, recalibrated on 2014": summary(y_oot, -s_oot, recal_pd),
            "LightGBM challenger": summary(y_oot, ch_pd, ch_pd),
        },
        "gini_diff": {
            "challenger minus scorecard": bootstrap_diff(y_oot, ch_pd, -s_oot),
            "scorecard minus sub-grade": bootstrap_diff(y_oot, -s_oot, grade_risk),
            "challenger minus sub-grade": bootstrap_diff(y_oot, ch_pd, grade_risk),
        },
        "leak_demo_gini": leak_demo(oot),
    }
    log("Same comparison on the development test years ...")
    results["gini_by_year"] = history(dev, card, rounds) | {
        target.OOT_YEAR: {
            "scorecard": results["models"]["Scorecard, raw PD"]["gini"],
            "challenger": results["models"]["LightGBM challenger"]["gini"],
            "sub-grade": results["models"]["Lending Club sub-grade (benchmark)"]["gini"],
        }
    }
    quarter = oot["issue_d"].dt.quarter.to_numpy()
    results["gini_by_quarter"] = {
        f"Q{q}": {
            "scorecard": round(gini(y_oot[quarter == q], -s_oot[quarter == q]), 4),
            "sub-grade": round(gini(y_oot[quarter == q], grade_risk[quarter == q]), 4),
            "challenger": round(gini(y_oot[quarter == q], ch_pd[quarter == q]), 4),
        }
        for q in sorted(set(quarter))
    }
    purposes = oot["purpose"].value_counts().index[:5]
    results["gini_by_purpose"] = {
        p: {
            "loans": int((oot["purpose"] == p).sum()),
            "scorecard": round(gini(y_oot[(oot["purpose"] == p).to_numpy()], -s_oot[(oot["purpose"] == p).to_numpy()]), 4),
        }
        for p in purposes
    }
    tables = {
        "Scorecard, raw PD": calibration(y_oot, raw_pd),
        "Scorecard, recalibrated on 2014": calibration(y_oot, recal_pd),
        "LightGBM challenger": calibration(y_oot, ch_pd),
    }
    importance = challenger.importance(model, X_oot)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=1))
    REPORT_PATH.write_text(report(results, tables, importance))
    plot(tables, PLOT_PATH)
    log(f"Wrote {RESULTS_PATH}, {REPORT_PATH} and {PLOT_PATH}")
    return results


def report(results, tables, importance):
    models = pd.DataFrame(results["models"]).T
    diffs = pd.DataFrame(results["gini_diff"]).T
    quarters = pd.DataFrame(results["gini_by_quarter"]).T
    purposes = pd.DataFrame(results["gini_by_purpose"]).T
    parts = [
        "# Out-of-time validation (generated by `make validate`)",
        f"Test: {results['oot_loans']:,} loans issued in {results['oot_year']}, never used for any choice. "
        f"Default rate {results['oot_bad_rate']:.1%} (development years: {results['dev_bad_rate']:.1%}).",
        "## All models",
        models.to_markdown(),
        "Gini and KS: ranking (higher is better). Brier: accuracy of the PD (lower is better). "
        "mean_pd: average predicted default rate, to compare with the observed one.",
        f"## Gini differences, with 95 % bootstrap intervals ({BOOTSTRAP_REPS} resamples)",
        diffs.to_markdown(),
        "## Gini by test year (2011-2014: time cross-validation folds; 2015: the out-of-time test)",
        pd.DataFrame(results["gini_by_year"]).T.to_markdown(),
        "## Calibration: predicted vs observed default rate in 10 bands of predicted PD",
    ]
    for name, table in tables.items():
        parts += [f"### {name}", table.to_markdown()]
    parts += [
        "## Stability: Gini by quarter of 2015",
        quarters.to_markdown(),
        "## Stability: scorecard Gini by loan purpose (5 largest)",
        purposes.to_markdown(),
        f"## Challenger: what drives it (mean |SHAP|, {results['challenger_trees']} trees)",
        importance.round(4).rename("mean |SHAP|").to_markdown(),
        "## Why the leakage rules matter",
        f"One column that is only known after the loan (share of principal repaid) ranks 2015 loans with "
        f"Gini {results['leak_demo_gini']}. It looks perfect and is useless for deciding on an application.",
    ]
    return "\n\n".join(parts) + "\n"


def plot(tables, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    top = max(float(t[["predicted", "observed"]].to_numpy().max()) for t in tables.values()) * 1.08
    ax.plot([0, top], [0, top], color="#999999", lw=1, ls="--", label="perfect calibration")
    styles = {"Scorecard, raw PD": "#c0504d", "Scorecard, recalibrated on 2014": "#1f6fb2", "LightGBM challenger": "#5b9a4b"}
    for name, table in tables.items():
        ax.plot(table["predicted"], table["observed"], marker="o", ms=4, lw=1.6, color=styles[name], label=name)
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="predicted default rate", ylabel="observed default rate (2015)")
    ax.set_title("Calibration on the out-of-time year 2015 (10 bands)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
