# A small neural-network market-risk case study

**Question:** Can an LSTM improve a one-day loss-quantile forecast for a fixed, daily rebalanced five-industry equity portfolio?

The public snapshot contains 6,791 sessions through 2025. The experiment uses 4,279 training targets through 2016, 754 validation targets in 2017–2019, and 1,508 untouched test targets in 2020–2025. All models forecast the same next-session loss. The default neural seed is 7, chosen in the protocol before test evaluation.

## 99% one-day VaR

| Model | Breaches | Breach rate | Expected | Pinball (bps) | Kupiec p | Independence p |
|---|---:|---:|---:|---:|---:|---:|
| EWMA | 37 | 2.45% | 15.08 | 4.561 | 1.705e-06 | 0.01258 |
| Gaussian | 42 | 2.79% | 15.08 | 6.402 | 1.081e-08 | 0.005795 |
| Historical | 26 | 1.72% | 15.08 | 5.860 | 0.0104 | 0.0007165 |
| LSTM | 20 | 1.33% | 15.08 | 4.160 | 0.2252 | 0.2656 |

## 95% one-day VaR

| Model | Breaches | Breach rate | Expected | Pinball (bps) | Kupiec p | Independence p |
|---|---:|---:|---:|---:|---:|---:|
| EWMA | 85 | 5.64% | 75.40 | 13.754 | 0.2659 | 0.1524 |
| Gaussian | 83 | 5.50% | 75.40 | 15.910 | 0.3766 | 0.0001127 |
| Historical | 76 | 5.04% | 75.40 | 15.709 | 0.9436 | 0.01595 |
| LSTM | 88 | 5.84% | 75.40 | 13.099 | 0.1465 | 0.09775 |

## What I learned

At 99%, the default LSTM has the lowest descriptive pinball score among the four models and fewer breaches, closer to the target than the benchmarks. Its 20 breaches still exceed the expected 15.08. Non-rejection of coverage or independence does not establish that it is safe or correctly specified.

At 95%, the default LSTM's 88 exceptions (5.84%) exceed the 5% target and the historical benchmark's 76. All three neural seeds show excess 95% breaches. The historical benchmark's overall frequency is close to target, but its exceptions cluster. This illustrates why frequency, clustering and quantile loss answer different questions.

The LSTM is frozen after validation while its input sequences update. Baselines update their histories daily. Results are limited to this portfolio and time period. No test-set tuning, production suitability or statistically significant model superiority is claimed.

## Initialization sensitivity

| Seed | 95% breaches | 95% pinball (bps) | 99% breaches | 99% pinball (bps) |
|---:|---:|---:|---:|---:|
| 7 | 88 | 13.099 | 20 | 4.160 |
| 19 | 95 | 13.054 | 19 | 4.276 |
| 42 | 90 | 12.903 | 21 | 4.193 |

These seeds measure sensitivity to optimization, not uncertainty over future market regimes. Comparisons and p-values are descriptive and unadjusted for multiple testing.

## Historical stress replay

| Window (after start close) | Portfolio return | Change on initial $1m | Worst one-day return |
|---|---:|---:|---:|
| Financial crisis: 2007-10-09 to 2009-03-09 | -51.19% | $-511,921 | -8.51% |
| COVID sell-off: 2020-02-19 to 2020-03-23 | -34.05% | $-340,530 | -11.55% |
| 2022: 2021-12-31 to 2022-12-30 | -14.69% | $-146,856 | -4.01% |
| Spring 2025: 2025-02-19 to 2025-04-08 | -17.47% | $-174,673 | -5.84% |

Stress windows are retrospective demonstrations. Financial-crisis observations are in training, so their replay is not held-out neural evidence. Compounded stress P&L differs from the constant-notional daily losses used in VaR backtesting.

## Reproduction

All three CPU fits and benchmark exports took **19.3 seconds** in the recorded environment, excluding download and app rendering. Best validation epochs were seed 7: 9, seed 19: 7, seed 42: 12.

`python -m risklab audit` verifies the raw-to-curated transform, source/code checksums, complete held-out forecast calendars, prior-session origins, actual portfolio losses, baseline replays, the train-only scaler, all three saved neural inference replays, noncrossing quantiles and metric recomputation.

Dataset: [Kenneth French five-industry returns](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_5_ind_port.html). See [methodology](METHODOLOGY.md) for equations and limits.
