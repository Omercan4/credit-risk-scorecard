import pandas as pd
import pytest

from crs import config
from crs.data import fixture, load, quality


@pytest.fixture(scope="module")
def loans():
    return load.clean(fixture.synthetic_raw(n=2000, seed=1))


def test_summary_line_is_dropped(loans):
    assert len(loans) == 2000
    assert loans["id"].str.isdigit().all()


def test_text_columns_become_numbers_and_dates(loans):
    assert set(loans["term"]) == {36, 60}
    assert loans["int_rate"].between(1, 40).all()
    assert loans["revol_util"].between(0, 100).all()
    assert pd.api.types.is_datetime64_any_dtype(loans["issue_d"])
    assert pd.api.types.is_datetime64_any_dtype(loans["earliest_cr_line"])


def test_emp_length():
    text = pd.Series(["< 1 year", "1 year", "10+ years", "n/a", None])
    assert load.parse_emp_length(text).tolist()[:3] == [0, 1, 10]
    assert load.parse_emp_length(text).isna().tolist()[3:] == [True, True]


def test_outside_policy_status_is_split():
    status, meets = load.parse_status(pd.Series(["Does not meet the credit policy. Status:Charged Off", "Fully Paid"]))
    assert status.tolist() == ["Charged Off", "Fully Paid"]
    assert meets.tolist() == [False, True]


def test_percent_with_and_without_sign():
    raw = fixture.synthetic_raw(n=10)
    raw.loc[0, "int_rate"] = "12.5"
    assert load.clean(raw)["int_rate"].notna().all()


def test_every_loaded_column_has_exactly_one_group():
    groups = [config.KEYS, config.APPLICATION, config.BENCHMARK, config.OUTCOME, config.NOT_FOR_MODELS]
    flat = [c for g in groups for c in g]
    assert len(flat) == len(set(flat))


def test_fixture_passes_the_checks(loans):
    assert quality.check(loans) == []


def test_checks_find_problems(loans):
    broken = loans.copy()
    broken.loc[0, "id"] = broken.loc[1, "id"]
    broken.loc[2, "loan_status"] = "Lost in space"
    broken.loc[3, "term"] = 48
    problems = quality.check(broken)
    assert len(problems) == 3


def test_population_counts_every_loan(loans):
    table = quality.population(loans)
    assert table["total"].sum() == len(loans)


def test_missing_columns_are_reported(tmp_path):
    raw = fixture.synthetic_raw(n=5).drop(columns=["dti"])
    path = tmp_path / "accepted_2007_to_2018Q4.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="dti"):
        load.read_raw(path)


def test_build_from_a_raw_file(tmp_path):
    path = tmp_path / "raw" / "accepted_2007_to_2018Q4.csv.gz"
    path.parent.mkdir()
    fixture.synthetic_raw(n=50).to_csv(path, index=False)
    out = tmp_path / "loans.parquet"
    loans = load.build(load.find_raw_file(tmp_path / "raw"), out)
    assert len(loans) == 50
    assert len(load.load(out)) == 50
