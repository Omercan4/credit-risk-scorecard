# Credit-risk scorecard (work in progress)

A loan scorecard built the way a bank would build it: from raw loans to an approve/decline rule with reasons,
validated on later loans than it was trained on, and a monitor that warns when the applicants change.

## The task

> At the moment of application, estimate the probability that a 36-month loan will default (charged off) within its term,
> using only what is known at application time. Turn it into a score, an approve/decline rule chosen by expected profit,
> and the top reasons for each decline.

About half the columns in the Lending Club data are only filled in after the loan was paid out (payments, recoveries,
the last FICO score). A model that uses them looks almost perfect and is useless. Every column therefore belongs to one
group in [`src/crs/config.py`](src/crs/config.py), and only the application-time group can become a model feature.
Lending Club's own `grade` and `int_rate` are the benchmark to beat, not inputs.

## Validation design

- Development years 2007–2014, with expanding-window time cross-validation (train up to year *t*, test on *t+1*).
- One out-of-time test on 2015, which nothing was tuned on.
- 2016–2018 applications simulate the time after go-live for monitoring.

There is no random k-fold, because credit policy, prices and the economy changed over these years. Random folds would let later
loans leak into training and overstate how well the model works on next year's applicants.

## Run it

```bash
make install       # Python 3.12 virtual environment
make synthetic     # try everything on 20,000 fake loans, no download needed
make download      # real data, needs a Kaggle API token in ~/.kaggle/kaggle.json
make data          # clean the raw CSV into data/processed/loans.parquet and print the data checks
make test
```

## Data

Lending Club accepted loans 2007–2018, from Kaggle ([wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)).
The raw data is not part of this repository. Only aggregated numbers are published.
The tests and CI use a synthetic fixture ([`src/crs/data/fixture.py`](src/crs/data/fixture.py)) in the same raw format.

## Status

- [x] 3.0 scaffold, data cleaning, data checks
- [ ] 3.1 target, population and leakage rules
- [ ] 3.2 WoE binning and points scorecard with decline reasons
- [ ] 3.3 challenger model, calibration and out-of-time validation
- [ ] 3.4 profit-based cut-off and reject bias
- [ ] 3.5 drift monitoring (PSI/CSI, vintages)
- [ ] 3.6 API and public page
- [ ] 3.7 model card and packaging

Not for real lending decisions.
