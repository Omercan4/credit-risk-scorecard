"""Command line: python -m crs.cli <command>."""

import argparse

import pandas as pd

from crs import applicant, config, decision, develop, monitor, policy, target, validation
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


def cmd_population(args):
    loans = load.load(config.PROCESSED_DIR / "loans_synthetic.parquet" if args.synthetic else config.LOANS_PATH)
    pop = target.population(loans)
    print("Who is in the modelling population:")
    print(target.exclusions(loans).to_string())
    print("\nDefault rate by issue year (36-month loans):")
    print(target.bad_rate_by_year(pop).to_string())
    dev, oot = target.split(pop)
    print(f"\nDevelopment 2007-2014: {len(dev):,} loans, bad rate {dev['bad'].mean():.3f}")
    print(f"Out-of-time test 2015: {len(oot):,} loans, bad rate {oot['bad'].mean():.3f}")


def cmd_scorecard(args):
    if args.synthetic:  # fake results go next to the fake data, never over the real scorecard
        out = config.PROCESSED_DIR
        develop.run(load.load(out / "loans_synthetic.parquet"), out / "scorecard_synthetic.json", out / "scorecard_synthetic.md")
    else:
        develop.run(load.load(config.LOANS_PATH))


def cmd_validate(args):
    validation.run(load.load(config.LOANS_PATH))


def cmd_strategy(args):
    decision.run(load.load(config.LOANS_PATH))


def cmd_monitor(args):
    monitor.run(load.load(config.LOANS_PATH))


def cmd_page(args):
    print(f"Wrote {applicant.export_page_card()}")


def cmd_policy(args):
    policy.run(load.load(config.LOANS_PATH))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="crs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download the Lending Club data from Kaggle").set_defaults(func=cmd_download)
    data = sub.add_parser("data", help="clean the raw CSV into data/processed/loans.parquet")
    data.add_argument("--synthetic", type=int, metavar="N", help="use N fake loans instead of the real data")
    data.set_defaults(func=cmd_data)
    popn = sub.add_parser("population", help="who is in the modelling population, and the default rate by year")
    popn.add_argument("--synthetic", action="store_true", help="use the fake loans from `data --synthetic`")
    popn.set_defaults(func=cmd_population)
    card = sub.add_parser("scorecard", help="time cross-validation on 2007-2014, then fit the scorecard")
    card.add_argument("--synthetic", action="store_true", help="use the fake loans from `data --synthetic`")
    card.set_defaults(func=cmd_scorecard)
    sub.add_parser("strategy", help="profit-based approve/decline rule on 2015, and reject bias").set_defaults(
        func=cmd_strategy
    )
    sub.add_parser("monitor", help="PSI/CSI and early defaults per quarter from 2015").set_defaults(func=cmd_monitor)
    sub.add_parser("policy", help="evidence for the lending rules that run before the score").set_defaults(func=cmd_policy)
    sub.add_parser("page", help="export the scorecard for the web page (docs/scorecard.json)").set_defaults(func=cmd_page)
    sub.add_parser("validate", help="out-of-time test on 2015 vs benchmark and challenger").set_defaults(func=cmd_validate)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
