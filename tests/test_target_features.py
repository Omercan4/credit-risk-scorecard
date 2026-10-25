import pytest

from crs import config, features, target
from crs.data import fixture, load


@pytest.fixture(scope="module")
def loans():
    return load.clean(fixture.synthetic_raw(n=4000, seed=2))


@pytest.fixture(scope="module")
def pop(loans):
    return target.population(loans)


def test_population_rules(pop):
    assert set(pop["term"]) == {36}
    assert pop["year"].between(2007, 2015).all()
    assert set(pop["loan_status"]) <= set(target.BAD + target.GOOD)
    assert set(pop["bad"]) == {0, 1}


def test_exclusions_account_for_every_loan(loans, pop):
    table = target.exclusions(loans)
    assert table.sum() == len(loans)
    assert table["in the population"] == len(pop)


def test_outside_policy_loans_stay_in(pop):
    assert (~pop["meets_policy"]).any()


def test_split_years(pop):
    dev, oot = target.split(pop)
    assert dev["year"].max() == 2014
    assert set(oot["year"]) == {2015}
    assert len(dev) + len(oot) == len(pop)


def test_time_folds_never_train_on_the_future(pop):
    dev, _ = target.split(pop)
    folds = list(target.time_folds(dev))
    assert [year for year, _, _ in folds] == target.CV_TEST_YEARS
    for year, train, test in folds:
        assert dev.loc[train, "year"].max() < year
        assert set(dev.loc[test, "year"]) == {year}
    assert len(folds[-1][1]) > len(folds[0][1])  # the window grows


def test_features_use_only_application_columns(pop):
    X = features.build(pop)
    assert list(X.columns) == features.ALLOWED
    assert not set(X.columns) & features.FORBIDDEN
    assert len(X) == len(pop)


def test_outcome_columns_cannot_get_in(pop):
    X = features.build(pop)
    for col in ["total_pymnt", "int_rate", "grade", "addr_state", "bad"]:
        with pytest.raises(ValueError, match=col):
            features.check(X.assign(**{col: 0}))


def test_every_application_column_is_used_or_replaced():
    used = set(features.ALLOWED) | set(features.DROPPED)
    assert set(config.APPLICATION) <= used


def test_derived_values(pop):
    X = features.build(pop)
    assert X["fico"].between(660, 850).all()
    assert (X["credit_history_months"] > 0).all()
    assert (X["loan_to_income"] > 0).all()
