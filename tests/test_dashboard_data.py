import subprocess
import sys

import numpy as np

from dashboard_data import load_equity_results, load_feature_history
from risklab.config import ROOT
from risklab.data import load_data
from risklab.model import features_for


def test_display_cache_matches_causal_model_features():
    returns, source = load_data()
    expected, _ = features_for(returns)
    actual = load_feature_history(source["csv_sha256"])
    assert actual.index.equals(expected.index)
    assert list(actual.columns) == list(expected.columns)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
    predictions, _ = load_equity_results()
    assert len(predictions) == 18096


def test_saved_dashboard_browsing_does_not_load_training_or_inference_runtime():
    code = '''
import sys
from streamlit.testing.v1 import AppTest
app=AppTest.from_file("app.py", default_timeout=40).run()
assert not app.exception
app.sidebar.radio[0].set_value("Bitcoin").run()
assert not app.exception
assert "torch" not in sys.modules
assert "onnxruntime" not in sys.modules
'''
    result = subprocess.run([sys.executable,"-c",code],cwd=ROOT,capture_output=True,text=True,timeout=90)
    assert result.returncode==0, result.stderr
