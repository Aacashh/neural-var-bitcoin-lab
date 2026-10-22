import numpy as np
import pandas as pd
import pytest
import torch

from risklab.backtest import empirical_es, coverage, baselines, pinball
from risklab.model import features_for, samples, QuantileLSTM
from risklab.data import load_data


def test_fractional_tail_mass():
    # 1.5 observations of tail mass: full 10 and half of 2.
    assert empirical_es([0., 1., 2., 10.], .625) == pytest.approx(11/1.5)
    assert empirical_es([0., 1., 2., 10.], .99) == pytest.approx(10.)


def test_loss_sign_and_strict_breach():
    result = coverage([-.02, .01, .03], [.01, .01, .02], .95)
    assert result["exceptions"] == 1
    assert result["expected"] == pytest.approx(.15)
    assert pinball([.03], [.02], .99)[0] == pytest.approx(.0099)


@pytest.mark.parametrize("hits", [np.zeros(20), np.ones(20), np.array([0, 1]*10)])
def test_boundary_coverage_is_finite(hits):
    result = coverage(hits, np.full(20,.5), .99)
    assert np.isfinite(result["kupiec_lr"])
    assert 0 <= result["binomial_p"] <= 1
    if hits.min() == hits.max():
        assert result["independence_p"] is None
    else:
        assert result["independence_p"] < .001  # alternating events are dependent too


def test_no_target_day_in_baseline():
    index = pd.bdate_range("2020-01-01", periods=40)
    returns = pd.Series(np.random.default_rng(7).normal(0,.01,40), index=index)
    changed = returns.copy()
    changed.iloc[30:] = -.5
    a = baselines(returns, window=10)
    b = baselines(changed, window=10)
    np.testing.assert_allclose(a.loc[a.date<=index[30],"var"], b.loc[b.date<=index[30],"var"])
    assert not np.allclose(a.loc[a.date==index[31],"var"], b.loc[b.date==index[31],"var"])


def test_feature_and_sample_causality():
    frame,_ = load_data()
    early,_ = features_for(frame.iloc[:500])
    full,_ = features_for(frame)
    np.testing.assert_allclose(early,full.iloc[:500],equal_nan=True)
    x,y,origin,dates,names = samples(frame.iloc[:600])
    assert (origin < dates).all()
    assert x.shape[1:] == (20,11)
    assert len(names) == 11
    _,portfolio = features_for(frame)
    np.testing.assert_allclose(y,-100*portfolio.reindex(dates),atol=1e-6)
    np.testing.assert_allclose(x[-1],full.loc[:origin[-1]].tail(20),atol=1e-7)


def test_stress_contribution_reconciles():
    frame,_ = load_data()
    observed = frame.loc[(frame.index>"2020-02-19")&(frame.index<="2020-03-23")]
    daily = observed.mean(axis=1)
    before = (1+daily).cumprod().shift(1,fill_value=1)
    contribution = observed.mul(.2).mul(before,axis=0).sum().sum()
    assert contribution == pytest.approx((1+daily).prod()-1,abs=1e-12)


def test_neural_quantiles_do_not_cross():
    model = QuantileLSTM(11)
    with torch.no_grad():
        pred = model(torch.randn(5,20,11)).numpy()
    assert (pred[:,0] > 0).all()
    assert (pred[:,1] > pred[:,0]).all()
