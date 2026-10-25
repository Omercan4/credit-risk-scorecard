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

There is one caveat: the clock is the issue date. In practice a 2010 loan's outcome is only known around 2013, so a bank would
wait longer before training on it. The split is honest about the applicants, but slightly optimistic about how fast outcomes arrive.

## Population

2,260,668 loans, all data checks passed. The modelling population is the 36-month loans issued 2007–2015 that reached a final status:

| | loans |
|---|---:|
| in the population | 621,022 |
| 60-month term (left out) | 650,914 |
| issued 2016–2018, term not over (monitoring only) | 988,585 |
| 2007–2015, still open or late at the end of the data | 147 |

Bad = "Charged Off" or "Default". The 2,438 old loans marked "does not meet the credit policy" stay in, with their real outcome.

| issue year | loans | default rate |
|---|---:|---:|
| 2007 | 603 | 26.2 % |
| 2008 | 2,393 | 20.7 % |
| 2009 | 5,281 | 13.7 % |
| 2010 | 9,156 | 10.9 % |
| 2011 | 14,101 | 10.6 % |
| 2012 | 43,470 | 13.6 % |
| 2013 | 100,422 | 12.3 % |
| 2014 | 162,570 | 13.7 % |
| **2015 (out-of-time test)** | **283,026** | **14.9 %** |

Two things to note:
- Defaults were high in 2007–08 (the financial crisis, and a young platform), then rose again from 2013 as Lending Club grew fast.
- The test year 2015 is riskier than the development years (14.9 % vs 13.2 %). A model calibrated on the past will under-predict defaults there unless it is recalibrated, and the validation has to show this, not hide it.

## Run it

```bash
make install       # Python 3.12 virtual environment
make synthetic     # try everything on 20,000 fake loans, no download needed
make download      # real data (about 650 MB), needs a Kaggle API token in ~/.kaggle/
make data          # clean the raw CSV into data/processed/loans.parquet and print the data checks
make test
```

## Data

Lending Club accepted loans and rejected applications 2007–2018, from Kaggle ([wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)), license CC0: Public Domain.
The raw data is not part of this repository. Only aggregated numbers are published.
The tests and CI use a synthetic fixture ([`src/crs/data/fixture.py`](src/crs/data/fixture.py)) in the same raw format.

## Status

- [x] 3.0 scaffold, data cleaning, data checks
- [x] 3.1 target, population, time folds and the leakage gate
- [ ] 3.2 WoE binning and points scorecard with decline reasons
- [ ] 3.3 challenger model, calibration and out-of-time validation
- [ ] 3.4 profit-based cut-off and reject bias
- [ ] 3.5 drift monitoring (PSI/CSI, vintages)
- [ ] 3.6 API and public page
- [ ] 3.7 model card and packaging

Not for real lending decisions.
