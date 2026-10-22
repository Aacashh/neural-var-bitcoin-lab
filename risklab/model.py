"""A direct quantile LSTM. No return-direction or trading-profit target."""
import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import PROTOCOL, WEIGHTS


def features_for(frame):
    portfolio = frame @ np.asarray(WEIGHTS)
    wealth = (1 + portfolio).cumprod()
    features = frame.copy()
    features["portfolio"] = portfolio
    features["vol_5"] = portfolio.rolling(5).std()
    features["vol_20"] = portfolio.rolling(20).std()
    features["downside_20"] = portfolio.clip(upper=0).pow(2).rolling(20).mean().pow(.5)
    features["absolute_ewma"] = portfolio.abs().ewm(span=10, adjust=False).mean()
    features["drawdown_60"] = wealth / wealth.rolling(60).max() - 1
    return features, portfolio


def samples(frame):
    features, portfolio = features_for(frame)
    x, y, origin, dates = [], [], [], []
    lookback = PROTOCOL["lookback"]
    for i in range(PROTOCOL["historical_window"] - 1, len(frame) - 1):
        sequence = features.iloc[i-lookback+1:i+1].to_numpy()
        if np.isfinite(sequence).all():
            x.append(sequence)
            y.append(-100 * portfolio.iloc[i+1])  # percentage points, for stable optimization
            origin.append(frame.index[i])
            dates.append(frame.index[i+1])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.DatetimeIndex(origin), pd.DatetimeIndex(dates), features.columns.tolist()


class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=24):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        encoded, _ = self.lstm(x)
        raw = self.head(encoded[:, -1, :])
        low = nn.functional.softplus(raw[:, 0])
        gap = nn.functional.softplus(raw[:, 1])
        return torch.stack((low, low + gap), dim=1)
