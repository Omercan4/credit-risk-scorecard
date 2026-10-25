# Model card: credit-risk scorecard (Lending Club, 36-month loans)

Written like a short model validation report. Every number comes from the generated reports in [`reports/`](reports/);
`make data population scorecard validate strategy monitor` reproduces them from the raw data.

## Purpose and intended use
Estimate, at the moment of application, the probability that a 36-month unsecured personal loan defaults within its term.
Turn it into a points score, give the top 3 reasons for every decision, and price or decline by expected loss.
**A portfolio project for learning and discussion.** Not for real lending decisions: the data is from the US, 2007–2018,
and from one lender's approved loans only.

## Data
- Lending Club accepted loans and rejected applications 2007–2018 ([Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club), CC0). 2,260,668 loans.
- The raw data is not in this repository. Only aggregated numbers and the points table are published.

## Population and target
- **Population:** 36-month loans issued 2007–2015 that reached a final status: 621,022 loans. Left out: 60-month loans, loans issued 2016–2018 (whose term had not ended; used for monitoring) and 147 loans still open or late at the end of the data.
- **Target:** bad = "Charged Off" or "Default", good = "Fully Paid". Default rate 13.2 % in 2007–2014, 14.9 % in 2015.

## Leakage rules
- Every column belongs to exactly one group ([`src/crs/config.py`](src/crs/config.py)): application-time, benchmark, outcome, or not for models.
- Only the application-time group can reach a model. A test fails if a payment column, Lending Club's grade, interest rate or instalment, or the borrower's state gets in.
- **Why this matters:** one post-loan column alone (share of principal repaid) ranks the 2015 loans with Gini 1.0.

## Model
- Points scorecard on **9 characteristics**: FICO score, annual income, debt-to-income, loan amount / income, mortgage accounts, home ownership, credit inquiries in the last 6 months, revolving utilisation and loan purpose.
- **How it was built:** WoE binning (optbinning) with the default-rate trend forced where it is a business rule. Every bin holds at least 5 % of the loans. Missing values get their own bin, but a missing bin with fewer than 1 % of loans gets the riskiest bin's points.
- **Regression:** logistic regression on the WoE values. All coefficients are negative, as they must be, and the largest VIF is 1.6.
- **Scale:** 600 points = odds of 50:1, and 20 points double the odds. Scores run from 484 to 603.
- **Choices:** the number of characteristics was chosen by expanding-window time cross-validation on 2007–2014 (test years 2011–2014). Up to 12 characteristics gave +0.002 Gini, so 9 were kept.
- **Left out on purpose:**
  - Income verification: Lending Club verified mostly applicants it already found risky, so it carries Lending Club's own judgement and is not a fair decline reason.
  - Zip code and state: a region can stand in for ethnicity.

## Performance (out-of-time: 283,026 loans issued in 2015, never used for any choice)

| | Gini | KS | Brier | mean predicted PD |
|---|---:|---:|---:|---:|
| Scorecard | 0.303 | 0.218 | 0.122 | 13.5 % |
| LightGBM challenger (monotone constraints) | 0.343 | 0.249 | 0.121 | 13.3 % |
| Lending Club sub-grade (benchmark) | 0.357 | 0.262 | 0.121 | 12.8 % |
| observed | | | | 14.9 % |

- **Gini by year:** 2011 0.342, 2012 0.292, 2013 0.273, 2014 0.292, 2015 0.303. Stable by quarter of 2015 (0.296–0.309) and by loan purpose (0.26–0.33).
- **Benchmark:** the scorecard beat Lending Club's grade until 2012. The grade then improved every year, to 0.357 in 2015, using the full credit report.
- **Challenger:** +0.040 Gini over the scorecard (95 % bootstrap interval 0.037 to 0.042), ahead in every year. Two ratio characteristics added to the scorecard did not close the gap (development CV 0.302 → 0.300).

## Calibration
- All models under-predict 2015 defaults in every band (13.5 % predicted vs 14.9 % observed), because 2015 was riskier than any development year.
- Recalibrating on 2014 does not fix it: 2014 itself was safer than 2015. The ranking holds, but the level drifted.

## Decision rule
- **Lending rules first** ([`reports/policy.md`](reports/policy.md)):
  - **The limits:** FICO at least 660, income at least $10,000, loan at most $35,000 and at most 50 % of income, debt-to-income at most 40 % before and 50 % including the new payment, at most 7 inquiries in 6 months, utilisation at most 100 %. FICO, income, loan amount and debt-to-income must be given.
  - **Where they come from:** each limit sits at the edge of the approved 2007–2014 loans, where the scorecard's evidence ends. Without them, the open-ended last bins scored a loan of 8 times income like a loan of 0.6 times, and approved it.
  - **Effect on the approved loans:** together the rules would have declined 1.06 % of them, which defaulted at 24.7 % against 13.0 % for the rest.
- **Price:** the lowest annual rate at which the loan breaks even in expectation, (1 − PD) × gain = PD × 36.5 % loss rate, after a 1 % fee. Decline if even 30.99 % (Lending Club's highest rate) does not cover it.
- **Policy limit:** decline above a 40 % PD. The return on money lent in 2007–2014 fell with every PD band and reached zero there (161 loans, +0.2 %), and the model has almost no evidence beyond it.
- **Result on 2015:** at Lending Club's prices almost every approved loan was worth funding, so a profit rule declines under 1 % and changes profit by less than 1 M USD on 3.6 billion lent.
- **Where the score is worth money:** selection. An investor funding the safest 50 % earns 6.29 % on money lent with the scorecard, 6.09 % with Lending Club's grade and 6.60 % with the challenger. Both differences are significant in a bootstrap.

## Monitoring (go-live January 2015, 1.27 M applications to the end of 2018)
- **Inputs:** score PSI ≤ 0.006 in 2015–2016, rising to 0.094 in 2018. Revolving utilisation reached CSI 0.26 (red) in 2018, consistent with a tighter credit policy.
- **Outcomes:** early defaults (first 12 months) ran 1.13× expected for 2015 Q2 loans (known Sep 2016) and up to 1.49× for 2016 Q3 loans. Gini on early defaults stayed between 0.26 and 0.32.
- **Recommended action from mid-2017:** recalibrate the PD level. Then rebuild on recent vintages once the late-2017 policy change has settled.
- **One false alarm, explained:** mortgage accounts show CSI 0.86 only because the field was not recorded before 2012. Ignoring the missing bin, the CSI is 0.07.

## Fairness and regulation
- The data has no protected attributes, so fairness by group cannot be measured here. Region is excluded as a likely proxy.
- Income, home ownership and loan purpose can still correlate with protected groups. A real deployment would need a disparate-impact test with appropriate data.
- **Explained declines:** every decision returns its top 3 reasons from the points table. In the EU this matters: after the CJEU's SCHUFA ruling (C-634/21, 2023), automated credit scores fall under GDPR Art. 22. The EU AI Act also lists credit scoring of individuals as high-risk (Annex III).

## Known limits
- **Approved loans only (reject bias):** 63 % of the 6.3 M rejected applicants had a risk score below 660, against almost no approved loan. The PD for such applicants is an extrapolation and is likely too low. Reject inference is not done.
- **US data, 2007–2018, one platform:** the economy, competition and the lender's policy all changed over the period, as the monitoring shows.
- **Clock and outcome timing:** the issue date is the clock, while in practice outcomes arrive years later. Profit is undiscounted, with no funding or operating costs, so the break-even rate is a floor, not a price.
- **Challenger explanations:** the challenger is more accurate, but it has no reviewed reason codes yet, so it is not used for decisions.
