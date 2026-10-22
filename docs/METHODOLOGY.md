# How the experiment works

## Data and book

Use the **first, value-weighted daily block** of Kenneth French's five-industry file, not the later equal-weighted block. Divide the supplied percentage returns by 100. Preserve the raw ZIP, retrieval time, source URLs, curated CSV and SHA-256 hashes. The frozen slice has 6,791 sessions from 4 January 1999 to 31 December 2025. Missing sentinels, duplicate dates, nonfinite values and incomplete end coverage cause an error; no interpolation or synthetic fallback is used.

The five broad portfolios are Consumer, Manufacturing, High tech, Healthcare and Other. These are SIC-based research industries, not five individual stocks, ETFs or a full modern sector classification. Each industry return is value weighted internally. Our hypothetical portfolio holds **20% in each industry**, resetting weights every day. Portfolio return is the weighted sum of simple returns. There are no transaction costs, liquidity constraints, borrow, cash flows or leverage. The data are the provider's returns including distributions, not the separate ex-dividend series.

Daily portfolio loss is `L[t] = -sum(w[i] * r[i,t])`. Positive loss means a fall in value. VaR and backtest dollar losses use an unchanged daily notional. Growth charts and stress-window P&L compound the daily returns; their notional evolves. Scaling the exposure leaves all percentage metrics unchanged. Start-of-day wealth times weighted industry return gives each industry's dollar contribution to a compounded stress-window P&L.

## Forecast target and timeline

For each target session t, predict the 95th and 99th percentiles of `L[t]` using data through the prior recorded session. These are one-day forecasts, with no square-root-of-time extrapolation. The date-only data cannot establish actual publication timestamps. This is a retrospective end-of-session research design, **not proof of an executable real-time forecast**. Industry histories can be revised and the snapshot is not a point-in-time vintage.

The first 250 sessions establish feature and baseline history. The first supervised target is 30 December 1999. There are 4,279 training targets through 30 December 2016, 754 validation targets in 2017–2019, and 1,508 test targets in 2020–2025. Adjacent input windows overlap, as real forecasts would. No training target uses a validation or test return. The first validation/test sequence may contain earlier-session returns from the preceding split; those observations were available at its origin.

## Four models

1. **Historical:** empirical inverse-CDF loss quantile from the previous 250 sessions. At 99%, with 250 equally weighted observations, this definition chooses the third-largest observed loss. It is a deliberately simple, somewhat coarse tail estimate.
2. **Gaussian:** previous-250-session loss mean plus `normal_quantile(q) * sample_standard_deviation`. It assumes a normal return distribution and uses `ddof=1`.
3. **EWMA:** `normal_quantile(q) * sqrt(v[t])`, with zero conditional mean and `v[t+1] = .94*v[t] + .06*r[t]^2`. Initialize variance on the first 250 observations, before the first forecast. Record the current forecast before incorporating its target return. EWMA remembers older observations with decreasing weights.
4. **LSTM:** 20 sessions × 11 standardized features, a single LSTM with 24 hidden units, and a two-output head. There are 3,602 trainable parameters. The first quantile is `softplus(a)`; the second is the first plus `softplus(b)`. This prevents crossing and restricts VaR to positive values, an explicit modeling constraint for this long-only example. Targets are percentage-point losses during training and decimal losses in saved forecasts.

Features: five industry returns, portfolio return, 5-session volatility, 20-session volatility, 20-session downside RMS, a span-10 EWMA of absolute returns, and drawdown relative to the past 60-session wealth maximum. Every feature is computed causally. The z-score scaler is fit only on the flattened **training sequences**; overlapping dates therefore receive repeated weight in that fit. The saved scaler and weights reproduce inference without refitting.

Train with Adam (learning rate .002), batches of 128, gradient norm capped at 1, at most 35 epochs and six epochs of validation patience. Select the lowest validation pinball epoch. Train seeds 7, 19 and 42; seed 7 is the display default declared in the protocol, not the test winner. Test outcomes did not determine epochs, features, seeds or confidence levels. The LSTM weights and scaler remain frozen throughout the test; baseline windows and LSTM inputs update each day. A walk-forward retrained LSTM is a separate possible experiment, not a feature of these results.

The small CPU training run is deterministic on this environment, with two PyTorch threads and deterministic algorithms. Hardware/library changes can affect exact floating point results. The experiment records versions, actual runtime and source identity; no claim is made about production latency.

## VaR, ES and model tests

For loss L and forecast v, a breach is strictly `L > v`, not `L >= v`. Pinball loss is `max(q*(L-v), (q-1)*(L-v))`. Multiply decimal pinball by 10,000 to report basis points. A low score balances quantile sharpness with asymmetric errors; it is not a profit or an accuracy percentage.

Historical ES integrates the top `(1-q)` mass of the empirical loss distribution, including fractional mass at a boundary observation. It is **separate from LSTM VaR**. Two predicted quantiles do not identify a whole tail distribution, and the LSTM does not estimate ES. For 250 observations at 99%, ES averages the two largest losses plus half of the third, divided by 2.5.

Report observed/expected breach counts, exact binomial intervals and a two-sided exact binomial test. Kupiec's unconditional coverage likelihood ratio uses a chi-square reference with one degree of freedom. Christoffersen independence compares a pooled exception probability against the two-state first-order Markov transition model. Conditional coverage adds their likelihood ratios with two degrees of freedom. Boundary likelihoods use `xlogy` so zero counts remain finite. If a transition state never occurs, independence and conditional coverage are undefined and left blank.

The exact binomial interval/test assumes independent Bernoulli exceptions. The asymptotic tests can be unreliable with sparse tail events. Calendar slices, three seeds, two quantiles and several models are descriptive comparisons with no multiple-testing correction. No p-value here establishes model equivalence or significant predictive superiority. Overlapping trailing-250 plots should not be treated as independent tests.

The 99% / 250-session traffic-light display borrows the familiar count boundaries (0–4, 5–9, 10+) only as an educational illustration. This portfolio return series is not a bank's actual/hypothetical trading P&L; the app does not implement a regulatory capital engine, P&L attribution test or internal-model approval process. Current market-risk capital methodology includes substantially more requirements.

## Stress and model limitations

The preset windows are reproducible illustrations: after the close on 9 October 2007 to 9 March 2009; 19 February to 23 March 2020; 31 December 2021 to 30 December 2022; and 19 February to 8 April 2025. They are selected retrospectively for demonstration. Their names are context, not evidence about the economic cause of each day's return. The financial crisis is in training, so no out-of-sample neural backtest is shown there. Test-period event windows show already-saved forecasts, with no event-specific refitting.

The experiment has one fixed US equity book and one retrospective test period. It lacks cross-asset exposures, options/nonlinear repricing, intraday data, current positions, historical publication vintages, liquidity/transaction costs, parameter uncertainty, GARCH and filtered historical benchmarks, formal predictive-ability tests and production model governance. These limits apply to interpretation of every result.

## Read the sources

- [French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and [five-industry construction](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_5_ind_port.html).
- [PyTorch LSTM reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).
- [SciPy binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
- [BIS MAR32 backtesting context](https://www.bis.org/basel_framework/chapter/MAR/32.htm?inforce=20230101&published=20200327).
- [Christoffersen (1998), Evaluating Interval Forecasts](https://doi.org/10.2307/2527341).
