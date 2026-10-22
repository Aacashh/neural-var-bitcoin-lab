import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from bitcoin.analytics import daily_sentiment,market_metrics,lag_associations
from bitcoin.data import clean_text,BTC,load_data
from risklab.config import ROOT


def example_posts():
    return pd.DataFrame(dict(date=pd.to_datetime(["2021-01-01"]*3+["2021-01-03"]),
                             source=["Twitter","Twitter","Reddit","Reddit"],
                             finbert_score=[1.,1.,-1.,0.],vader_score=[1.,1.,-1.,0.],
                             finbert_label=["positive","positive","negative","neutral"],
                             vader_label=["positive","positive","negative","neutral"],disagreement=[False]*4))


def test_missing_social_days_stay_missing():
    day=pd.date_range("2021-01-01",periods=3)
    result=daily_sentiment(example_posts(),day)
    assert result.posts.tolist()==[3,0,1]
    assert np.isnan(result.sentiment.iloc[1])
    assert result.sentiment.iloc[2]==0  # real neutral is distinct from no posts


def test_source_balance_and_minimum_are_explicit():
    day=pd.date_range("2021-01-01",periods=3)
    post_mean=daily_sentiment(example_posts(),day)
    balanced=daily_sentiment(example_posts(),day,aggregation="Equal source mean",minimum=2)
    assert post_mean.sentiment.iloc[0]==pytest.approx(1/3)
    assert balanced.sentiment.iloc[0]==pytest.approx(0.)
    assert np.isnan(balanced.sentiment.iloc[2])
    assert balanced.posts.iloc[2]==1


def test_lag_shift_uses_calendar_before_dropping_missing():
    day=pd.date_range("2021-01-01",periods=40)
    ret=pd.Series(np.random.default_rng(3).normal(size=40),index=day)
    sentiment=ret.shift(-1)
    sentiment.iloc[1::2]=np.nan
    output=lag_associations(sentiment,pd.DataFrame({"return":ret}),[1])
    assert output.paired_days.iloc[0]==20
    assert output.spearman_rho.iloc[0]==pytest.approx(1.)


def test_bitcoin_var_excludes_target_return():
    day=pd.date_range("2021-01-01",periods=100)
    price=pd.Series(100*np.exp(np.linspace(0,1,100)),index=day)
    original=pd.DataFrame({"date":day,"close":price.values})
    shocked=original.copy()
    shocked.loc[70,"close"]*=.5
    a,b=market_metrics(original),market_metrics(shocked)
    assert a.historical_var_95.iloc[:61].isna().all()
    np.testing.assert_allclose(a.historical_var_95.iloc[:71],b.historical_var_95.iloc[:71],equal_nan=True)
    assert b.loss.iloc[70]>a.loss.iloc[70]


def test_cleaning_retains_negation_and_keyword_boundaries():
    result=clean_text("@person Bitcoin is NOT good!!! #BTC $BTC https://example.com/a")
    assert "NOT good!!!" in result
    assert "person" not in result and "https" not in result
    assert BTC.search(result)
    assert not BTC.search("mybitcointrader")


def test_archived_source_counts_and_daily_price_calendar():
    posts,prices,source=load_data()
    assert len(prices)==730
    assert not posts.duplicated(["source","text_hash"]).any()
    assert posts.source.value_counts().to_dict()==source["post_counts"]
    assert posts.text.map(lambda text:BTC.search(text) is not None).all()


@pytest.mark.parametrize("sources",[["Twitter","Reddit"],["Twitter"],["Reddit"],[]])
def test_bitcoin_dashboard_sources(sources):
    app=AppTest.from_file(str(ROOT/"bitcoin_app.py"),default_timeout=40).run()
    assert not app.exception
    app.sidebar.multiselect[0].set_value(sources).run()
    assert not app.exception
    if sources:
        assert len(app.tabs)==5
        assert len(app.metric)>=4
    else:
        assert app.warning


def test_bitcoin_dashboard_models_dates_and_candles():
    app=AppTest.from_file(str(ROOT/"bitcoin_app.py"),default_timeout=40).run()
    app.sidebar.selectbox[0].set_value("November 2022").run()
    app.sidebar.selectbox[1].set_value("VADER · baseline").run()
    app.sidebar.selectbox[2].set_value("Equal source mean").run()
    app.sidebar.slider[0].set_value(10).run()
    next(widget for widget in app.radio if widget.label=="Price display").set_value("OHLC candles").run()
    assert not app.exception


def test_bitcoin_no_posts_window_has_no_invented_sentiment():
    app=AppTest.from_file(str(ROOT/"bitcoin_app.py"),default_timeout=40).run()
    app.sidebar.date_input[0].set_value((pd.Timestamp("2021-01-01").date(),pd.Timestamp("2021-01-02").date())).run()
    assert not app.exception
    assert app.metric[2].value=="No eligible days"
