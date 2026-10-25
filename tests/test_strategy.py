import numpy as np
import pandas as pd
import pytest

from crs import rejects, strategy


@pytest.fixture
def loans():
    # two repaid loans and one default, 1,000 lent each
    return pd.DataFrame(
        {
            "funded_amnt": [1000.0, 1000.0, 1000.0],
            "installment": [35.0, 40.0, 40.0],
            "total_pymnt": [1260.0, 1440.0, 400.0],
            "bad": [0, 0, 1],
        }
    )


def test_realised_profit_after_fee(loans):
    assert strategy.realised_profit(loans).round(2).tolist() == [247.4, 425.6, -604.0]


def test_gain_if_repaid_uses_the_price(loans):
    assert strategy.gain_if_repaid(loans).round(2).tolist() == [247.4, 425.6, 425.6]


def test_loss_rate_only_from_defaults(loans):
    assert strategy.loss_rate(loans) == pytest.approx(0.604)


def test_expected_profit_turns_negative_with_risk(loans):
    ep = strategy.expected_profit(loans.iloc[[0]], [0.1, 0.5], 0.6)  # same loan, two PDs
    assert ep[0] > 0 > ep[1]


def test_outcome(loans):
    out = strategy.outcome(loans, [True, True, False])
    assert out["approval_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["bad_rate"] == 0 and out["declined_bad_rate"] == 1
    assert out["profit_musd"] == pytest.approx(0.0, abs=0.01)


def test_pick_share_takes_the_safest(loans):
    out = strategy.pick_share(loans, [0.3, 0.1, 0.9], 2 / 3)
    assert out["bad_rate"] == 0 and out["lent_musd"] == pytest.approx(0.0, abs=0.1)


def test_curve_ends_at_approving_everyone(loans):
    c = strategy.curve(loans, [0.3, 0.1, 0.9], steps=3)
    assert c["profit_musd"].iloc[-1] == pytest.approx(strategy.realised_profit(loans).sum() / 1e6)


def test_bootstrap_return_diff_of_a_ranking_with_itself_is_zero():
    rng = np.random.default_rng(0)
    n = 500
    loans = pd.DataFrame({"funded_amnt": 1000.0, "total_pymnt": rng.uniform(300, 1400, n), "bad": 0})
    risk = rng.random(n)
    out = strategy.bootstrap_return_diff(loans, risk, risk, 0.5, reps=20)
    assert out == {"diff": 0.0, "low": 0.0, "high": 0.0}


def test_score_table_higher_cutoff_approves_fewer(loans):
    table = strategy.score_table(loans, np.array([550, 580, 500]), [500, 560, 600])
    assert table["approval_rate"].tolist() == pytest.approx([1, 1 / 3, 0], abs=1e-4)


def test_rejected_columns_are_parsed():
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2014-01-02", "2015-03-04"]),
            "amount": ["1000.0", "25000"],
            "risk_score": ["650.0", None],
            "dti": ["10%", "55.5%"],
            "emp_length": ["< 1 year", "10+ years"],
        }
    )
    out = rejects.clean(raw)
    assert out["dti"].tolist() == [10.0, 55.5]
    assert out["risk_score"].isna().tolist() == [False, True]
    assert out["emp_length"].tolist() == [0, 10]


def test_compare_accepted_and_rejected():
    accepted = pd.DataFrame(
        {"fico_range_low": [700.0, 680.0], "fico_range_high": [704.0, 684.0], "dti": [10.0, 20.0],
         "loan_amnt": [5000.0, 7000.0], "emp_length": [3.0, 5.0]}
    )
    rejected = pd.DataFrame({"risk_score": [600.0, np.nan], "dti": [45.0, 30.0], "amount": [1000.0, 2000.0],
                             "emp_length": [0.0, 1.0]})
    table = rejects.compare(accepted, rejected)
    assert table.loc["share below 660", "rejected"] == 1.0
    assert table.loc["risk score known", "rejected"] == 0.5
    assert table.loc["share DTI above 40 %", "accepted"] == 0.0
