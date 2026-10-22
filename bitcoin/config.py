from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
START, END = "2021-01-01", "2022-12-31"
MODEL_ID = "Xenova/finbert"
MODEL_REVISION = "8f269abebfdd9009d7d9b5e96af7e5c6bfe50b20"
MAX_TOKENS = 128
BATCH_SIZE = 16
SOURCES = {
    "Twitter": "https://data.mendeley.com/datasets/8fbdhh72gs/5",
    "Reddit": "https://zenodo.org/records/12593440",
    "Prices": "https://zenodo.org/records/12593440",
}
