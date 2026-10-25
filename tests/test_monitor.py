import numpy as np
import pandas as pd
import pytest

from crs import monitor


def test_psi_is_zero_for_the_same_distribution_and_grows_with_shift():
    base = [0.25, 0.25, 0.25, 0.25]
    assert monitor.psi(base, base) == pytest.approx(0, abs=1e-9)
    small, large = monitor.psi(base, [0.3, 0.25, 0.25, 0.2]), monitor.psi(base, [0.7, 0.1, 0.1, 0.1])
    assert 0 < small < monitor.AMBER < monitor.RED < large


def test_psi_accepts_counts():
    assert monitor.psi([10, 30], [1, 3]) == pytest.approx(0, abs=1e-9)


def test_traffic_lights():
    assert [monitor.light(v) for v in [0.05, 0.1, 0.3]] == ["green", "amber", "red"]
    assert [monitor.oe_light(v) for v in [1.05, 1.15, 1.3, 0.75, np.nan]] == ["green", "amber", "red", "red", ""]


def test_early_bad():
    loans = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2016-01-01"] * 5),
            "last_pymnt_d": pd.to_datetime(["2016-08-01", "2017-06-01", "2016-05-01", None, "2016-03-01"]),
            "loan_status": ["Charged Off", "Charged Off", "Fully Paid", "Charged Off", "Late (31-120 days)"],
        }
    )
    # bad within 12 months, bad later, repaid, bad with no payment at all, late within 12 months
    assert monitor.early_bad(loans).tolist() == [1, 0, 0, 1, 1]


def test_early_defaults_only_counted_when_observable():
    loans = pd.DataFrame({"issue_d": pd.to_datetime(["2017-12-01", "2018-01-01"])})
    assert monitor.early_observable(loans).tolist() == [True, False]


def test_score_bins_are_roughly_equal():
    score = np.random.default_rng(0).normal(550, 15, 10000).round()
    edges = monitor.score_bins(score)
    shares = monitor.shares(np.digitize(score, edges), len(edges) + 1)
    assert len(edges) >= 8 and shares.max() < 0.16


def test_vintage_curves_rise_and_stop_at_the_horizon():
    n = 400
    rng = np.random.default_rng(1)
    issue = pd.Timestamp("2017-03-01")
    loans = pd.DataFrame(
        {
            "term": 36,
            "issue_d": issue,
            "loan_status": rng.choice(["Charged Off", "Current"], n, p=[0.1, 0.9]),
            "last_pymnt_d": issue + pd.to_timedelta(rng.integers(30, 400, n), unit="D"),
        }
    )
    curve = monitor.vintage_curves(loans, years=[2017])[2017]
    assert curve.is_monotonic_increasing and curve.index.max() <= 15
