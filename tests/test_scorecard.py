import math

import numpy as np
import pandas as pd
import pytest

from crs import binning, develop, features, scorecard, target
from crs.data import fixture, load


@pytest.fixture(scope="module")
def dev():
    pop = target.population(load.clean(fixture.synthetic_raw(n=8000, seed=3)))
    return target.split(pop)[0]


@pytest.fixture(scope="module")
def fitted(dev):
    card, specs, Xw = develop.fit_final(dev, max_features=6)
    return card, specs, features.build(dev), dev["bad"].to_numpy()


def test_woe_is_higher_for_safer_bins(fitted):
    _, specs, _, _ = fitted
    fico = specs["fico"]
    rates, woe = fico["bad_rate"][:-1], fico["woe"][:-1]  # without the missing bin
    assert np.argsort(rates).tolist() == np.argsort(woe)[::-1].tolist()


def test_forced_trend_holds(fitted):
    _, specs, _, _ = fitted
    for name in ["fico", "dti", "revol_util"]:
        rates = [r for r in specs[name]["bad_rate"][:-1]]
        expected = sorted(rates, reverse=binning.TREND[name] == "descending")
        assert rates == expected, name


def test_numerical_bins_and_missing():
    spec = {"type": "numerical", "splits": [10.0, 20.0]}
    codes = binning.bin_codes(spec, pd.Series([5, 10, 19.9, 20, np.nan]))
    assert codes.tolist() == [0, 1, 1, 2, 3]  # [s_i, s_i+1), missing = last code


def test_unseen_category_goes_to_other_or_missing():
    with_other = {"type": "categorical", "keep": ["a", "b"], "groups": [["a"], ["b", "OTHER"]]}
    assert binning.bin_codes(with_other, pd.Series(["a", "zzz", None])).tolist() == [0, 1, 2]
    without = {"type": "categorical", "keep": ["a", "b"], "groups": [["a"], ["b"]]}
    assert binning.bin_codes(without, pd.Series(["zzz"])).tolist() == [2]


def test_all_coefficients_negative(fitted):
    card, _, _, _ = fitted
    assert all(c["coef"] < 0 for c in card["characteristics"])


def test_verification_status_is_never_used(fitted):
    card, _, _, _ = fitted
    assert "verification_status" not in [c["name"] for c in card["characteristics"]]


def test_score_is_the_sum_of_the_table_points(fitted):
    card, _, X, _ = fitted
    row = X.iloc[[0]]
    by_hand = 0
    for c in card["characteristics"]:
        by_hand += c["points"][binning.bin_codes(c, row[c["name"]])[0]]
    assert scorecard.score(card, row).iloc[0] == by_hand


def test_higher_score_means_lower_risk(fitted):
    card, _, X, y = fitted
    assert scorecard.gini(y, -scorecard.score(card, X)) > 0.3


def test_scale():
    assert scorecard.pd_from_score(600) == pytest.approx(1 / 51)
    odds = lambda s: (1 - scorecard.pd_from_score(s)) / scorecard.pd_from_score(s)  # noqa: E731
    assert odds(620) / odds(600) == pytest.approx(2)
    assert scorecard.OFFSET == pytest.approx(600 - 20 / math.log(2) * math.log(50))


def test_reasons(fitted):
    card, _, X, _ = fitted
    reasons = scorecard.reasons(card, X.iloc[:50])
    pts = scorecard.points(card, X.iloc[:50])
    for i, items in enumerate(reasons):
        lost = [r["points_lost"] for r in items]
        assert lost == sorted(lost, reverse=True) and all(v > 0 for v in lost) and len(items) <= 3
        for r in items:
            best = next(max(c["points"]) for c in card["characteristics"] if c["name"] == r["characteristic"])
            assert best - pts.iloc[i][r["characteristic"]] == r["points_lost"]


def test_choose_prefers_smaller_within_tolerance():
    cv = pd.DataFrame({"max_features": [6, 10, 6, 10], "gini": [0.40, 0.403, 0.42, 0.421]})
    assert develop.choose(cv) == 6
    cv["gini"] = [0.40, 0.45, 0.40, 0.45]
    assert develop.choose(cv) == 10


def test_characteristic_without_values_gets_one_neutral_bin():
    x = pd.Series([np.nan] * 99 + [1.0])
    spec = binning.fit_one("mort_acc", x, np.array([0, 1] * 50))
    assert spec["splits"] == [] and binning.n_bins(spec) == 1


def test_rare_missing_values_get_the_riskiest_bin():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=5000))
    y = (rng.random(5000) < 1 / (1 + np.exp(-x))).astype(int)
    x[:20] = np.nan
    y[:20] = 0  # by chance a perfect missing group: must not become the best bin
    spec = binning.fit_one("dti", x, y)
    assert spec["woe"][-1] == min(spec["woe"][:-1])
