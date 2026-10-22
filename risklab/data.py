"""Explicit download; the app reads the frozen CSV without network calls."""
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from .config import ROOT, DATA_URL, DETAIL_URL, SECTORS, PROTOCOL


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_french(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Expected one source CSV")
        lines = archive.read(names[0]).decode("utf-8-sig").splitlines()
    # Select the value-weighted daily block explicitly; the second is equal-weighted.
    marker = next(i for i, line in enumerate(lines) if "Average Value Weighted Returns -- Daily" in line)
    header = next(i for i in range(marker + 1, len(lines)) if ",Cnsmr" in lines[i].replace(" ", ""))
    rows = []
    for line in lines[header + 1:]:
        if not re.match(r"^\s*\d{8},", line):
            if rows:
                break
            continue
        rows.append(line)
    frame = pd.read_csv(io.StringIO(lines[header] + "\n" + "\n".join(rows)), index_col=0)
    frame.columns = frame.columns.str.strip()
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    frame.index.name = "date"
    frame = frame[SECTORS].astype(float)
    if frame.isin([-99.99, -999]).any().any():
        raise ValueError("Missing-return sentinel in provider history")
    frame = frame.loc[PROTOCOL["data_start"]:PROTOCOL["data_end"]] / 100
    validate_returns(frame)
    return frame


def validate_returns(frame):
    if list(frame.columns) != SECTORS or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Unexpected sectors, duplicate dates, or unsorted history")
    if not np.isfinite(frame.to_numpy()).all() or (frame <= -1).any().any():
        raise ValueError("Nonfinite or impossible simple return")
    if len(frame) < 6000 or frame.index[-1] != pd.Timestamp(PROTOCOL["data_end"]):
        raise ValueError("The fixed experiment needs complete 1999–2025 coverage")


def fetch():
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    if (folder / "manifest.json").exists():
        raise FileExistsError("Frozen snapshot exists. Preserve it; use a separate project copy to refresh.")
    response = requests.get(DATA_URL, timeout=90)
    response.raise_for_status()
    frame = parse_french(response.content)
    raw = folder / "5_Industry_Portfolios_daily_CSV.zip"
    raw.write_bytes(response.content)
    output = folder / "sector_returns.csv"
    frame.to_csv(output, float_format="%.8f")
    manifest = dict(source="Kenneth R. French Data Library", url=DATA_URL, details=DETAIL_URL,
                    retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    description="First block: daily value-weighted five-industry returns, including distributions; provider percentages divided by 100.",
                    raw_sha256=sha256(raw), csv_sha256=sha256(output), rows=len(frame),
                    first_date=str(frame.index[0].date()), last_date=str(frame.index[-1].date()),
                    units="decimal daily simple returns", columns=SECTORS,
                    limitation="Retrospective research data; not a point-in-time vintage or a tradable ETF history.")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_data():
    folder = ROOT / "data"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if sha256(folder / "sector_returns.csv") != manifest["csv_sha256"]:
        raise ValueError("Snapshot checksum mismatch; restore the original file")
    frame = pd.read_csv(folder / "sector_returns.csv", index_col="date", parse_dates=True)
    validate_returns(frame)
    return frame, manifest
