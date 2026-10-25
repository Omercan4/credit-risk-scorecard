"""Command line: python -m crs.cli <command>."""

import argparse

import pandas as pd

from crs import config
from crs.data import download, fixture, load, quality


def cmd_download(args):
    path = download.download()
    print(f"Downloaded to {path}")


def cmd_data(args):
    if args.synthetic:
        loans = load.clean(fixture.synthetic_raw(n=args.synthetic))
        out = config.PROCESSED_DIR / "loans_synthetic.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        loans.to_parquet(out, index=False)
    else:
        loans = load.build()
        out = config.LOANS_PATH
    print(f"{len(loans):,} loans -> {out}")

    problems = quality.check(loans)
    print("Data checks:", "all passed" if not problems else "; ".join(problems))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(quality.population(loans))
        print("\nMissing share per application column:")
        print(quality.missing_share(loans).head(10).to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="crs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download the Lending Club data from Kaggle").set_defaults(func=cmd_download)
    data = sub.add_parser("data", help="clean the raw CSV into data/processed/loans.parquet")
    data.add_argument("--synthetic", type=int, metavar="N", help="use N fake loans instead of the real data")
    data.set_defaults(func=cmd_data)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
