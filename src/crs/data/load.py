"""Read the raw Lending Club CSV and turn its text columns into clean types.

The raw file stores numbers and dates as text ("36 months", "10+ years", "13.99%", "Dec-2015")
and has a few summary lines at the end that are not loans. clean() fixes both.
"""

import pandas as pd

from crs import config


def find_raw_file(raw_dir=config.RAW_DIR):
    files = sorted(raw_dir.glob(config.RAW_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No file matching {config.RAW_GLOB} in {raw_dir}. Run `make download`, or download "
            f"https://www.kaggle.com/datasets/{config.KAGGLE_DATASET} and unzip it there."
        )
    return files[0]


def read_raw(path):
    """Only the columns we use, all as text (clean() parses them)."""
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in config.ALL_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"The raw file has no column(s): {', '.join(missing)}")
    return pd.read_csv(path, usecols=config.ALL_COLUMNS, dtype=str)


def parse_months(text):
    return pd.to_datetime(text, format="%b-%Y", errors="coerce")


def parse_emp_length(text):
    """'< 1 year' -> 0, '1 year' -> 1, '10+ years' -> 10, 'n/a' or empty -> missing."""
    years = text.str.replace("< 1", "0", regex=False).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(years, errors="coerce")


def parse_status(text):
    """Old loans outside the credit policy are written 'Does not meet the credit policy. Status:Fully Paid'.

    Returns the plain status and a flag for these loans.
    """
    prefix = "Does not meet the credit policy. Status:"
    outside = text.str.startswith(prefix, na=False)
    return text.str.removeprefix(prefix), ~outside


def clean(raw):
    df = raw.copy()
    df = df[df["loan_amnt"].notna() & df["issue_d"].notna()]  # drops the summary lines at the end

    for col in config.NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in config.PERCENT:
        df[col] = pd.to_numeric(df[col].str.rstrip("%").str.strip(), errors="coerce")
    for col in config.MONTHS:
        df[col] = parse_months(df[col])

    df["term"] = pd.to_numeric(df["term"].str.extract(r"(\d+)", expand=False), errors="coerce").astype("Int64")
    df["emp_length"] = parse_emp_length(df["emp_length"])
    df["loan_status"], df["meets_policy"] = parse_status(df["loan_status"])

    for col in ["home_ownership", "verification_status", "purpose", "application_type", "grade", "sub_grade"]:
        df[col] = df[col].str.strip()

    df = df.drop_duplicates(subset="id")
    return df.sort_values(["issue_d", "id"]).reset_index(drop=True)


def build(raw_path=None, out_path=config.LOANS_PATH):
    raw_path = raw_path or find_raw_file()
    loans = clean(read_raw(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    loans.to_parquet(out_path, index=False)
    return loans


def load(path=config.LOANS_PATH):
    return pd.read_parquet(path)
