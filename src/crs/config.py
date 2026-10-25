"""Paths and the column groups.

Every column we load belongs to exactly one group. The groups decide what a model may see:
only APPLICATION columns are known when someone applies. The others are filled in later
(OUTCOME) or are Lending Club's own risk model (BENCHMARK), which we compare against but never use.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LOANS_PATH = PROCESSED_DIR / "loans.parquet"

KAGGLE_DATASET = "wordsforthewise/lending-club"
RAW_GLOB = "**/accepted_2007_to_2018*.csv*"

# Identify the loan and its contract.
KEYS = ["id", "issue_d", "term", "funded_amnt"]

# Known at application time: the only candidates for model features.
APPLICATION = [
    "loan_amnt",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "earliest_cr_line",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
]

# Lending Club's own pricing of the risk: the benchmark to beat, never a feature.
BENCHMARK = ["grade", "sub_grade", "int_rate", "installment"]

# Only known after the loan was paid out: used for the target and the profit, never a feature.
OUTCOME = ["loan_status", "total_pymnt", "total_rec_prncp", "total_rec_int", "recoveries", "last_pymnt_d"]

# Kept for monitoring and fairness checks only: a region can stand in for ethnicity ("redlining").
NOT_FOR_MODELS = ["addr_state"]

ALL_COLUMNS = KEYS + APPLICATION + BENCHMARK + OUTCOME + NOT_FOR_MODELS

NUMERIC = [
    "funded_amnt",
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "installment",
    "total_pymnt",
    "total_rec_prncp",
    "total_rec_int",
    "recoveries",
]
PERCENT = ["int_rate", "revol_util"]  # sometimes "13.99%", sometimes 13.99
MONTHS = ["issue_d", "earliest_cr_line", "last_pymnt_d"]  # "Dec-2015"

STATUSES = [
    "Fully Paid",
    "Charged Off",
    "Default",
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)",
]
