"""The API and the web page must give the same answer for the same application."""

import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from crs import applicant, config
from crs.serve import app

APPLICANTS = [
    {"fico_score": 765, "inquiries_last_6_months": 0, "revolving_utilisation": 15, "mortgage_accounts": 3,
     "annual_income": 110000, "debt_to_income": 9, "home_ownership": "MORTGAGE", "loan_amount": 10000, "purpose": "credit_card"},
    {"fico_score": 695, "inquiries_last_6_months": 1, "revolving_utilisation": 55, "mortgage_accounts": 1,
     "annual_income": 60000, "debt_to_income": 18, "home_ownership": "RENT", "loan_amount": 12000, "purpose": "debt_consolidation"},
    {"fico_score": 662, "inquiries_last_6_months": 3, "revolving_utilisation": 92, "mortgage_accounts": 0,
     "annual_income": 26000, "debt_to_income": 34, "home_ownership": "RENT", "loan_amount": 15000, "purpose": "small_business"},
    {"fico_score": 664.5, "annual_income": 27297.5, "loan_amount": 5000, "purpose": "wedding", "home_ownership": "OTHER"},  # bin edges, unseen purpose
    {},  # nothing given: every characteristic falls in its missing bin
]
client = TestClient(app)


def test_health_and_scorecard():
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/scorecard").json()["characteristics"]) == 9


@pytest.mark.parametrize("app_", APPLICANTS)
def test_api_matches_python(app_):
    assert client.post("/score", json=app_).json() == applicant.score_application(app_)


def test_api_rejects_impossible_values():
    assert client.post("/score", json={"fico_score": 1200}).status_code == 422


def test_better_file_scores_higher():
    strong, typical, stretched = (client.post("/score", json=a).json() for a in APPLICANTS[:3])
    assert strong["score"] > typical["score"] > stretched["score"]
    assert strong["pd"] < typical["pd"] < stretched["pd"]
    assert strong["break_even_rate"] < typical["break_even_rate"]


def test_policy_limit_declines_very_risky_files():
    stretched = applicant.score_application(APPLICANTS[2])
    assert stretched["pd"] > applicant.MAX_PD and stretched["decision"] == "decline"
    assert stretched["decision_rule"] == "pd above policy limit"
    assert applicant.score_application(APPLICANTS[1])["decision"] == "approve"


def test_break_even_rate_covers_the_expected_loss():
    rate = applicant.break_even_rate(0.15, 0.365)
    instalments = applicant.annuity_factor(rate) * 36 * 0.99
    assert 0.85 * (instalments - 1) == pytest.approx(0.15 * 0.365, abs=1e-6)
    assert applicant.break_even_rate(0.0, 0.365) == pytest.approx(0.0065, abs=0.001)  # only the 1 % fee to cover


def test_page_card_is_up_to_date():
    on_disk = json.loads(applicant.PAGE_CARD_PATH.read_text())
    assert on_disk == json.loads(json.dumps(applicant.page_card()))  # run `make page` after `make scorecard`


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the page's scorer")
def test_page_scorer_matches_python():
    script = (
        "const s = require(process.argv[1]); const card = require(process.argv[2]);"
        "const apps = JSON.parse(process.argv[3]);"
        "console.log(JSON.stringify(apps.map(a => s.score(card, a))));"
    )
    out = subprocess.run(
        ["node", "-e", script, str(config.ROOT / "docs" / "scorer.js"), str(applicant.PAGE_CARD_PATH), json.dumps(APPLICANTS)],
        capture_output=True, text=True, check=True,
    )
    from_js = json.loads(out.stdout)
    for app_, js in zip(APPLICANTS, from_js):
        py = applicant.score_application(app_)
        assert js["score"] == py["score"]
        assert js["pd"] == pytest.approx(py["pd"], abs=1e-4)
        assert js["decision"] == py["decision"] and js["decision_rule"] == py["decision_rule"]
        assert (js["break_even_rate"] is None) == (py["break_even_rate"] is None)
        if py["break_even_rate"] is not None:
            assert js["break_even_rate"] == pytest.approx(py["break_even_rate"], abs=1e-4)
        assert js["reasons"] == py["reasons"]
        assert [line["points"] for line in js["points"]] == [line["points"] for line in py["points"]]
