"""Reject bias: the models only ever saw approved loans. How different were the people Lending Club turned down?

The rejected-applications file has only a few columns: amount, date, risk score, debt-to-income ratio,
employment length and region. It cannot be scored with the full scorecard, but it shows which
applicants are missing from the training data. Note: Lending Club's "Risk_Score" is a FICO score in the
early years and may be a different bureau score later; it is compared here as given (almost no values
are above 850, so the scale looks stable). Employment length is read but not compared: in the rejected file
the share of "< 1 year" jumps from 27 % to 85 % between 2009 and 2011, which looks like a form default,
not a change in applicants.
"""

import pandas as pd

from crs import config

REJECTED_GLOB = "rejected_2007_to_2018*.csv*"
REJECTED_PATH = config.PROCESSED_DIR / "rejected_2007_2015.parquet"
COLUMNS = {
    "Amount Requested": "amount",
    "Application Date": "date",
    "Risk_Score": "risk_score",
    "Debt-To-Income Ratio": "dti",
    "Employment Length": "emp_length",
}
LAST_YEAR = 2015


def build(raw_dir=config.RAW_DIR, out=REJECTED_PATH, chunksize=1_000_000):
    """Read the 27 M rejected applications in chunks and keep 2007-2015 with the columns we compare."""
    path = sorted(raw_dir.glob(REJECTED_GLOB))[0]
    parts = []
    for chunk in pd.read_csv(path, usecols=list(COLUMNS), dtype=str, chunksize=chunksize):
        chunk = chunk.rename(columns=COLUMNS)
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk[chunk["date"].dt.year <= LAST_YEAR]
        parts.append(clean(chunk))
    rejected = pd.concat(parts, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    rejected.to_parquet(out, index=False)
    return rejected


def clean(df):
    from crs.data.load import parse_emp_length

    out = pd.DataFrame({"date": df["date"]})
    out["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    out["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    out["dti"] = pd.to_numeric(df["dti"].str.rstrip("%"), errors="coerce")
    out["emp_length"] = parse_emp_length(df["emp_length"])
    return out


def load():
    return pd.read_parquet(REJECTED_PATH) if REJECTED_PATH.exists() else build()


def compare(accepted, rejected, fico_floor=660):
    """Accepted (all terms, 2007-2015) vs rejected applicants on the columns both files have."""
    acc = pd.DataFrame(
        {
            "risk_score": (accepted["fico_range_low"] + accepted["fico_range_high"]) / 2,
            "dti": accepted["dti"],
            "amount": accepted["loan_amnt"],
            "emp_length": accepted["emp_length"],
        }
    )

    def describe(d):
        score = d["risk_score"].dropna()
        return {
            "applications": len(d),
            "risk score known": round(float(d["risk_score"].notna().mean()), 3),
            "median risk score": float(score.median()),
            f"share below {fico_floor}": round(float((score < fico_floor).mean()), 3),
            "median DTI (%)": round(float(d["dti"].dropna().clip(upper=100).median()), 1),
            "share DTI above 40 %": round(float((d["dti"].dropna() > 40).mean()), 3),
            "median amount": float(d["amount"].median()),
        }

    return pd.DataFrame({"accepted": describe(acc), "rejected": describe(rejected)})
