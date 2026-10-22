import tempfile
from pathlib import Path

# Keep Streamlit's disposable test files inside this project on managed Windows.
scratch = Path(__file__).resolve().parents[1] / ".runtime" / "test-temp"
scratch.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(scratch)
