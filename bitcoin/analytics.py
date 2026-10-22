"""Descriptive daily alignment and a strictly prior-day historical risk threshold."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def daily_sentiment(posts,calendar,model="FinBERT",aggregation="Post mean",minimum=1):
    score="finbert_score" if model=="FinBERT" else "vader_score"
    label="finbert_label" if model=="FinBERT" else "vader_label"
    frame=posts.copy()
    frame["negative_flag"]=(frame[label]=="negative").astype(int)
    frame["positive_flag"]=(frame[label]=="positive").astype(int)
    grouped=frame.groupby("date").agg(posts=(score,"size"),sentiment=(score,"mean"),
                                      negative_share=("negative_flag","mean"),positive_share=("positive_flag","mean"),
                                      disagreement=("disagreement","mean"),dispersion=(score,"std"),sources=("source","nunique"))
    if aggregation=="Equal source mean":
        grouped["sentiment"]=frame.groupby(["date","source"])[score].mean().groupby("date").mean()
    result=grouped.reindex(pd.DatetimeIndex(calendar,name="date"))
    for column in ["posts","sources"]:
        result[column]=result[column].fillna(0).astype(int)
    # An absence of posts remains missing sentiment, never artificial neutral sentiment.
    result["eligible"]=(result.posts>=minimum)
    result.loc[~result.eligible,["sentiment","negative_share","positive_share","disagreement","dispersion"]]=np.nan
    return result


def market_metrics(prices):
    market=prices.set_index("date").sort_index().copy()
    market["return"]=market.close.pct_change(fill_method=None)
    market["realized_vol_30"]=market["return"].rolling(30).std()*np.sqrt(365)
    wealth=market.close/market.close.iloc[0]
    market["drawdown"]=wealth/wealth.cummax()-1
    market["loss"]=-market["return"]
    market["historical_var_95"]=market.loss.shift(1).rolling(60).quantile(.95,interpolation="higher")
    market["breach_95"]=market.loss>market.historical_var_95
    # Forward-close return for descriptive lag alignment; no missing-day drop before shifting.
    market["next_day_return"]=market.close.shift(-1)/market.close-1
    return market


def lag_associations(sentiment,market,lags=range(-3,4)):
    aligned=sentiment.reindex(market.index)
    rows=[]
    for lag in lags:
        # Positive lag: sentiment at t paired with the daily return at t+lag.
        target=market["return"].shift(-lag)
        paired=pd.DataFrame({"sentiment":aligned,"return":target}).dropna()
        rho=None
        if len(paired)>=15 and paired.sentiment.nunique()>1 and paired["return"].nunique()>1:
            rho=float(spearmanr(paired.sentiment,paired["return"]).statistic)
        rows.append(dict(lag_days=lag,paired_days=len(paired),spearman_rho=rho))
    return pd.DataFrame(rows)
