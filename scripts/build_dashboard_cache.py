"""Export the original causal features for the lightweight, inference-free viewer."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from risklab.data import load_data,sha256
from risklab.model import features_for

frame,source=load_data()
features,_=features_for(frame)
path=ROOT/"data"/"dashboard_features.csv"
features.to_csv(path,float_format="%.12g")
receipt=dict(source_sha256=source["csv_sha256"],features_sha256=sha256(path),
             columns=features.columns.tolist(),rows=len(features),
             computation="Unmodified risklab.model.features_for; causal full-history features for the saved LSTM input display.")
(ROOT/"data"/"dashboard_features.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
print(path)
