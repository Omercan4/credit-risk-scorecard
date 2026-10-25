"""Make docs/img/social-preview.png (1280 x 640): the card GitHub and LinkedIn show when the repo link is shared.

Run: PYTHONPATH=src .venv/bin/python scripts/make_social_preview.py
It reads models/validation.json and the scorecard, so every number on the card is a real one.
The same dark style as the other two portfolio projects, so the three cards look like a set.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crs import applicant, config  # noqa: E402

BG, CARD, LINE = "#0d1117", "#161b22", "#30363d"
FG, MUTED, BLUE, GREEN, RED = "#e6edf3", "#8b949e", "#58a6ff", "#3fb950", "#f85149"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

EXAMPLE = {  # the "stretched file" on the page
    "fico_score": 662, "inquiries_last_6_months": 3, "revolving_utilisation": 92, "mortgage_accounts": 0,
    "annual_income": 26000, "debt_to_income": 34, "home_ownership": "RENT", "loan_amount": 15000, "purpose": "small_business",
}


def main():
    val = json.loads((config.ROOT / "models" / "validation.json").read_text())
    gini = val["models"]["Scorecard, raw PD"]["gini"]
    oot = val["oot_loans"]
    result = applicant.score_application(EXAMPLE)

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280)
    ax.set_ylim(640, 0)
    ax.axis("off")

    ax.text(64, 70, "PORTFOLIO PROJECT  |  CREDIT RISK", color=BLUE, fontsize=13, fontweight="bold")
    ax.text(64, 135, "Credit-Risk Scorecard", color=FG, fontsize=38, fontweight="bold")
    ax.text(64, 178, "A loan scorecard built like a bank would:\nreasons for every decline, tested on a later year,\nmonitored after go-live.",
            color=MUTED, fontsize=17, va="top", linespacing=1.4)

    stats = [
        (f"{gini:.2f}", f"Gini on {oot/1000:.0f}k loans\nfrom a year it never saw", BLUE),
        ("top 3", "reasons given with\nevery decision", GREEN),
        ("1.49x", "early-default drift\ncaught by monitoring", RED),
    ]
    for i, (big, small, color) in enumerate(stats):
        x = 64 + i * 205
        ax.text(x, 372, big, color=color, fontsize=32, fontweight="bold")
        ax.text(x, 392, small, color=MUTED, fontsize=11.5, va="top", linespacing=1.4)

    ax.text(64, 520, "2.26 M Lending Club loans, 2007-2018. Not for real lending decisions.", color=FG, fontsize=13)
    ax.text(64, 590, "WoE scorecard  |  LightGBM  |  FastAPI  |  GitHub Pages", color=MUTED, fontsize=13)
    ax.text(1216, 590, "github.com/Omercan4/credit-risk-scorecard", color=MUTED, fontsize=12, ha="right")

    # the score slip, as on the page
    x0, y0, w = 740, 92, 476
    ax.add_patch(plt.Rectangle((x0, y0), w, 440, facecolor=CARD, edgecolor=LINE, lw=1.2))
    ax.text(x0 + 22, y0 + 38, "Score slip", color=FG, fontsize=15, fontweight="bold")
    reasons = {r["characteristic"] for r in result["reasons"]}
    short = {
        "fico": "FICO score", "annual_inc": "annual income", "dti": "debt-to-income", "loan_to_income": "loan / income",
        "mort_acc": "mortgage accounts", "home_ownership": "home", "inq_last_6mths": "inquiries, 6 months",
        "revol_util": "revolving utilisation", "purpose": "loan purpose",
    }
    for i, line in enumerate(result["points"]):
        y = y0 + 74 + i * 30
        color = RED if line["characteristic"] in reasons else FG
        ax.text(x0 + 22, y, short.get(line["characteristic"], line["characteristic"]), color=color, fontsize=11.5)
        ax.text(x0 + w - 86, y, f"{line['points']}", color=color, fontsize=11.5, fontweight="bold", ha="right")
        ax.text(x0 + w - 22, y, f"of {line['max_points']}", color=MUTED, fontsize=10, ha="right")
        ax.plot([x0 + 22, x0 + w - 22], [y + 10, y + 10], color=LINE, lw=0.6)
    ytot = y0 + 74 + len(result["points"]) * 30 + 22
    ax.plot([x0 + 22, x0 + w - 22], [ytot - 18, ytot - 18], color=FG, lw=1.6)
    ax.text(x0 + 22, ytot + 4, "Score", color=FG, fontsize=13)
    ax.text(x0 + w - 22, ytot + 10, f"{result['score']}", color=FG, fontsize=26, fontweight="bold", ha="right")
    ax.text(x0 + 22, ytot + 40, f"PD {result['pd']*100:.0f} %", color=MUTED, fontsize=11.5)
    ax.text(x0 + w - 22, ytot + 40, result["decision"].upper(), color=RED if result["decision"] == "decline" else GREEN,
            fontsize=14, fontweight="bold", ha="right", rotation=3)

    out = config.ROOT / "docs" / "img" / "social-preview.png"
    fig.savefig(out, dpi=100, facecolor=BG)
    print("wrote", out)


if __name__ == "__main__":
    main()
