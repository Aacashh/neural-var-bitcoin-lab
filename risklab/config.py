from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/5_Industry_Portfolios_daily_CSV.zip"
DETAIL_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_5_ind_port.html"
SECTORS = ["Cnsmr", "Manuf", "HiTec", "Hlth", "Other"]
LABELS = {"Cnsmr": "Consumer", "Manuf": "Manufacturing", "HiTec": "High tech", "Hlth": "Healthcare", "Other": "Other industries"}
WEIGHTS = [.2] * 5
QUANTILES = [.95, .99]
SEEDS = [7, 19, 42]
PROTOCOL = {
    "data_start": "1999-01-01", "data_end": "2025-12-31",
    "train_end": "2016-12-31", "validation_end": "2019-12-31",
    "test_start": "2020-01-01", "test_end": "2025-12-31",
    "lookback": 20, "historical_window": 250, "ewma_decay": .94,
    "hidden_size": 24, "epochs": 35, "patience": 6, "batch_size": 128,
    "learning_rate": .002, "seeds": SEEDS, "quantiles": QUANTILES,
    "weights": WEIGHTS, "default_seed": 7,
}
