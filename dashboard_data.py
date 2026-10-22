"""Load checksummed display artifacts without importing the training runtime."""
import json
import pandas as pd
from risklab.config import ROOT
from risklab.data import sha256


def load_equity_results():
    folder=ROOT/"results"
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    for name,expected in manifest["checksums"].items():
        if sha256(folder/name)!=expected:
            raise ValueError(f"Result checksum mismatch: {name}")
    return pd.read_csv(folder/"forecasts.csv",parse_dates=["date","origin"]),manifest


def load_feature_history(source_sha256):
    folder=ROOT/"data"
    manifest=json.loads((folder/"dashboard_features.json").read_text(encoding="utf-8"))
    path=folder/"dashboard_features.csv"
    if manifest["source_sha256"]!=source_sha256 or sha256(path)!=manifest["features_sha256"]:
        raise ValueError("Feature-display cache does not match the frozen market data")
    return pd.read_csv(path,index_col="date",parse_dates=True)
