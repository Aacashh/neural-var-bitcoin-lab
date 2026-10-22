"""Loss-positive VaR forecasts and coverage diagnostics."""
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binomtest, chi2, norm


def pinball(loss, forecast, q):
    error = np.asarray(loss) - np.asarray(forecast)
    return np.maximum(q * error, (q - 1) * error)


def empirical_es(losses, q):
    """Integral of the empirical upper quantile; fractional boundary mass included."""
    descending = np.sort(np.asarray(losses, dtype=float))[::-1]
    mass = len(descending) * (1 - q)
    if not len(descending) or not 0 < q < 1:
        raise ValueError("Nonempty sample and 0 < q < 1 required")
    whole = int(np.floor(mass))
    fraction = mass - whole
    numerator = descending[:whole].sum()
    if whole < len(descending):
        numerator += fraction * descending[whole]
    return float(numerator / mass)


def baselines(returns, quantiles=(.95, .99), window=250, decay=.94):
    losses = -returns
    records = []
    variance = float(returns.iloc[:window].var(ddof=1))
    for i in range(window, len(returns)):
        history = losses.iloc[i - window:i].to_numpy()
        for q in quantiles:
            es = empirical_es(history, q)
            predictions = {
                "Historical": float(np.quantile(history, q, method="inverted_cdf")),
                "Gaussian": float(history.mean() + norm.ppf(q) * history.std(ddof=1)),
                "EWMA": float(norm.ppf(q) * np.sqrt(variance)),
            }
            for name, value in predictions.items():
                records.append(dict(date=returns.index[i], origin=returns.index[i-1], model=name,
                                    seed=0, quantile=q, loss=float(losses.iloc[i]), var=value,
                                    historical_es=es))
        # This return is only consumed AFTER this session's forecast was recorded.
        variance = decay * variance + (1 - decay) * float(returns.iloc[i]) ** 2
    return pd.DataFrame(records)


def _binomial_log(k, n, p):
    return float(xlogy(k, p) + xlogy(n - k, 1 - p))


def coverage(loss, forecast, q):
    loss, forecast = np.asarray(loss), np.asarray(forecast)
    if loss.shape != forecast.shape or not len(loss) or not np.isfinite(loss).all() or not np.isfinite(forecast).all():
        raise ValueError("Matching, nonempty, finite arrays required")
    hits = loss > forecast
    n, k = len(hits), int(hits.sum())
    result = binomtest(k, n, p=1 - q)
    interval = result.proportion_ci(confidence_level=.95, method="exact")
    uc = max(0., 2 * (_binomial_log(k, n, k / n) - _binomial_log(k, n, 1 - q)))
    n00 = int((~hits[:-1] & ~hits[1:]).sum())
    n01 = int((~hits[:-1] & hits[1:]).sum())
    n10 = int((hits[:-1] & ~hits[1:]).sum())
    n11 = int((hits[:-1] & hits[1:]).sum())
    independence = independence_p = conditional_p = None
    # No invented p-value if a transition state was never observed.
    if n00 + n01 > 0 and n10 + n11 > 0:
        pooled = (n01 + n11) / (n - 1)
        null = _binomial_log(n01 + n11, n - 1, pooled)
        alt = _binomial_log(n01, n00 + n01, n01 / (n00 + n01)) + _binomial_log(n11, n10 + n11, n11 / (n10 + n11))
        independence = max(0., 2 * (alt - null))
        independence_p = float(chi2.sf(independence, 1))
        conditional_p = float(chi2.sf(uc + independence, 2))
    return dict(sessions=n, exceptions=k, expected=n*(1-q), exception_rate=k/n,
                ci_low=float(interval.low), ci_high=float(interval.high),
                pinball=float(pinball(loss, forecast, q).mean()), mean_var=float(forecast.mean()),
                binomial_p=float(result.pvalue), kupiec_lr=uc, kupiec_p=float(chi2.sf(uc, 1)),
                independence_lr=independence, independence_p=independence_p,
                conditional_coverage_p=conditional_p, consecutive_breaches=n11,
                n00=n00, n01=n01, n10=n10, n11=n11)


def score_table(predictions):
    rows = []
    for (model, seed, q), group in predictions.groupby(["model", "seed", "quantile"]):
        group = group.sort_values("date")
        rows.append(dict(model=model, seed=int(seed), quantile=float(q), **coverage(group.loss, group["var"], q)))
    return pd.DataFrame(rows)
