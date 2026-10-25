"""The scoring API.

POST /score       one application -> score, PD, break-even rate, decision, top 3 reasons, points per characteristic
GET  /scorecard   the points table
GET  /health      is the model loaded
GET  /            the same page as on GitHub Pages

Run: uvicorn crs.serve:app   (or make serve). Not for real lending decisions.
"""

from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from crs import applicant, config

CARD = applicant.load_card()
LGD = applicant.loss_rate()

app = FastAPI(
    title="Credit-risk scorecard",
    version="1.0.0",
    description="Scores a 36-month personal loan application with a points scorecard trained on Lending Club "
    "loans 2007-2014. A portfolio project: not for real lending decisions.",
)


class Application(BaseModel):
    """Leave a field out (or null) if it is unknown: it then gets the points of the missing bin."""

    fico_score: float | None = Field(None, ge=300, le=850, examples=[705])
    annual_income: float | None = Field(None, ge=0, examples=[65000])
    loan_amount: float | None = Field(None, gt=0, le=40000, examples=[12000])
    debt_to_income: float | None = Field(None, ge=0, le=100, description="monthly debt payments / income, in %", examples=[18])
    mortgage_accounts: float | None = Field(None, ge=0, examples=[1])
    home_ownership: Literal["MORTGAGE", "OWN", "RENT", "OTHER"] | None = Field(None, examples=["RENT"])
    inquiries_last_6_months: float | None = Field(None, ge=0, examples=[1])
    revolving_utilisation: float | None = Field(None, ge=0, le=200, description="in %", examples=[55])
    purpose: str | None = Field(None, examples=["debt_consolidation"])


@app.post("/score")
def score(application: Application):
    return applicant.score_application(application.model_dump(), CARD, LGD)


@app.get("/scorecard")
def scorecard_table():
    return applicant.page_card(CARD, LGD)


@app.get("/health")
def health():
    return {"status": "ok", "characteristics": len(CARD["characteristics"]), "fitted_on": CARD.get("fitted_on")}


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(config.ROOT / "docs" / "index.html")


app.mount("/", StaticFiles(directory=config.ROOT / "docs"), name="docs")
