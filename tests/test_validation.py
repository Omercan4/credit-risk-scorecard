import numpy as np
import pandas as pd
import pytest

from crs import challenger, features, target, validation
from crs.data import fixture, load


@pytest.fixture(scope="module")
def dev():
    pop = target.population(load.clean(fixture.synthetic_raw(n=8000, seed=4)))
    return target.split(pop)[0]


def test_gini_and_ks_on_known_cases():
    y = np.array([0, 0, 1, 1])
    assert validation.gini(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1
    assert validation.gini(y, np.array([0.9, 0.8, 0.2, 0.1])) == -1
    assert validation.ks(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1


def test_bootstrap_interval_contains_the_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    good, noise = y + rng.normal(0, 1, 3000), rng.normal(0, 1, 3000)
    out = validation.bootstrap_diff(y, good, noise, reps=50)
    assert out["low"] <= out["diff"] <= out["high"] and out["low"] > 0


def test_calibration_bands():
    rng = np.random.default_rng(1)
    pd_ = rng.uniform(0.01, 0.4, 20000)
    y = (rng.random(20000) < pd_).astype(int)
    table = validation.calibration(y, pd_)
    assert len(table) == 10 and table["loans"].sum() == 20000
    assert np.allclose(table["predicted"], table["observed"], atol=0.03)  # well calibrated by construction


def test_subgrade_rank():
    assert validation.subgrade_rank(pd.Series(["A1", "A5", "B1", "G5"])).tolist() == [1, 5, 6, 35]


def test_challenger_respects_the_monotone_rules(dev):
    X, y = features.build(dev), dev["bad"].to_numpy()
    model = challenger.fit(X, y, 50)
    base = X.iloc[[0] * 5].copy()
    base["fico"] = [660, 700, 740, 780, 820]
    pd_fico = challenger.predict(model, base)
    assert all(np.diff(pd_fico) <= 1e-12)  # a higher FICO score never raises the PD
    base = X.iloc[[0] * 5].copy()
    base["dti"] = [5, 15, 25, 35, 45]
    assert all(np.diff(challenger.predict(model, base)) >= -1e-12)


def test_challenger_never_sees_forbidden_columns(dev):
    assert not set(challenger.columns()) & features.FORBIDDEN
    assert "verification_status" not in challenger.columns()


def test_challenger_handles_unseen_categories(dev):
    X, y = features.build(dev), dev["bad"].to_numpy()
    model = challenger.fit(X, y, 20)
    new = X.iloc[[0]].assign(purpose="wedding")
    assert 0 < challenger.predict(model, new)[0] < 1
