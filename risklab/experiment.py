"""Train once, save evidence, then serve only frozen results."""
import copy
import hashlib
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import ROOT, PROTOCOL, QUANTILES, SEEDS, WEIGHTS
from .data import load_data, sha256, parse_french
from .backtest import baselines, score_table
from .model import QuantileLSTM, samples


def source_identity():
    h = hashlib.sha256()
    for path in sorted((ROOT / "risklab").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def train():
    out = ROOT / "results"
    if out.exists():
        raise FileExistsError("Saved experiment exists. Keep it intact; retrain in a separate copy.")
    stage = ROOT / ".runtime" / "training"
    if stage.exists():
        raise FileExistsError("Unfinished staging directory exists; inspect it before retrying")
    stage.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frame, source = load_data()
    x, y, origins, dates, names = samples(frame)
    training = dates <= PROTOCOL["train_end"]
    validation = (dates > PROTOCOL["train_end"]) & (dates <= PROTOCOL["validation_end"])
    testing = (dates >= PROTOCOL["test_start"]) & (dates <= PROTOCOL["test_end"])
    if min(training.sum(), validation.sum(), testing.sum()) < 700:
        raise ValueError("Insufficient fixed split coverage")
    scaler = StandardScaler().fit(x[training].reshape(-1, x.shape[-1]))
    scaled = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
    xt, yt = torch.from_numpy(scaled), torch.from_numpy(y).unsqueeze(1)
    q_tensor = torch.tensor(QUANTILES)
    reference = frame @ np.asarray(WEIGHTS)
    baseline = baselines(reference)
    baseline = baseline.loc[baseline.date.between(PROTOCOL["test_start"], PROTOCOL["test_end"])].copy()
    parts, logs, runtime = [baseline], [], []
    test_positions = np.flatnonzero(testing)
    for seed in SEEDS:
        begin = time.perf_counter()
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = QuantileLSTM(x.shape[-1], PROTOCOL["hidden_size"])
        optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL["learning_rate"])
        best, best_state, wait, best_epoch = float("inf"), None, 0, 0
        for epoch in range(PROTOCOL["epochs"]):
            model.train()
            order = rng.permutation(np.flatnonzero(training))
            batch_scores = []
            for offset in range(0, len(order), PROTOCOL["batch_size"]):
                batch = order[offset:offset+PROTOCOL["batch_size"]]
                residual = yt[batch] - model(xt[batch])
                loss = torch.maximum(q_tensor*residual, (q_tensor-1)*residual).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                batch_scores.append((float(loss.detach()), len(batch)))
            model.eval()
            with torch.no_grad():
                residual = yt[validation] - model(xt[validation])
                score = float(torch.maximum(q_tensor*residual, (q_tensor-1)*residual).mean())
            logs.append(dict(seed=seed, epoch=epoch+1,
                             train_pinball=sum(a*b for a,b in batch_scores)/sum(b for _,b in batch_scores)/100,
                             validation_pinball=score/100))
            if score < best - 1e-6:
                best, best_state, wait, best_epoch = score, copy.deepcopy(model.state_dict()), 0, epoch+1
            else:
                wait += 1
            if wait >= PROTOCOL["patience"]:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            predicted = model(xt[testing]).numpy()/100
        torch.save(best_state, stage / f"lstm_seed_{seed}.pt")
        records = []
        for j, index in enumerate(test_positions):
            for qi, q in enumerate(QUANTILES):
                records.append(dict(date=dates[index], origin=origins[index], model="LSTM", seed=seed,
                                    quantile=q, loss=float(y[index]/100), var=float(predicted[j,qi])))
        neural = pd.DataFrame(records)
        es_lookup = baseline.loc[baseline.model == "Historical", ["date", "quantile", "historical_es"]]
        neural = neural.merge(es_lookup, on=["date", "quantile"], validate="one_to_one")
        parts.append(neural)
        elapsed = time.perf_counter()-begin
        runtime.append(dict(seed=seed, seconds=elapsed, best_epoch=best_epoch, validation_pinball=best/100))
        print(f"LSTM seed {seed}: epoch {best_epoch}, {elapsed:.1f}s", flush=True)
    predictions = pd.concat(parts, ignore_index=True).sort_values(["model", "seed", "quantile", "date"])
    predictions["exception"] = predictions.loss > predictions["var"]
    predictions.to_csv(stage / "forecasts.csv", index=False, float_format="%.12g")
    score_table(predictions).to_csv(stage / "metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(logs).to_csv(stage / "learning_curves.csv", index=False)
    yearly = []
    for year, group in predictions.groupby(predictions.date.dt.year):
        scored = score_table(group)
        scored["year"] = int(year)
        yearly.append(scored)
    pd.concat(yearly).to_csv(stage / "yearly_metrics.csv", index=False)
    scaler_record = dict(features=names, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), target_units="percentage points")
    (stage / "scaler.json").write_text(json.dumps(scaler_record, indent=2), encoding="utf-8")
    splits = {}
    for name, mask in [("train", training), ("validation", validation), ("test", testing)]:
        splits[name] = dict(samples=int(mask.sum()), first=str(dates[mask][0].date()), last=str(dates[mask][-1].date()))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol=PROTOCOL,
                    source_sha256=source["csv_sha256"], code_sha256=source_identity(),
                    python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    splits=splits, features=names, parameters=sum(p.numel() for p in model.parameters()),
                    runtime=runtime, total_seconds=time.perf_counter()-started,
                    selection="Epoch selected by validation pinball; seed 7 is the predeclared display default. No selection on test scores.",
                    checksums={p.name:sha256(p) for p in stage.iterdir() if p.is_file()})
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(stage, out)
    return str(out)


def load_results():
    folder = ROOT / "results"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["checksums"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Result checksum mismatch: {name}")
    return pd.read_csv(folder / "forecasts.csv", parse_dates=["date", "origin"]), manifest


def audit():
    frame, source = load_data()
    predictions, manifest = load_results()
    if source["csv_sha256"] != manifest["source_sha256"] or source_identity() != manifest["code_sha256"]:
        raise ValueError("Source or experiment code no longer matches the saved experiment")
    raw = ROOT / "data" / "5_Industry_Portfolios_daily_CSV.zip"
    assert sha256(raw) == source["raw_sha256"]
    reconstructed = parse_french(raw.read_bytes())
    np.testing.assert_allclose(reconstructed, frame, atol=1e-12)
    assert (predictions.origin < predictions.date).all()
    assert not predictions.duplicated(["date", "model", "seed", "quantile"]).any()
    actual = -(frame @ np.asarray(WEIGHTS))
    np.testing.assert_allclose(predictions.loss, actual.reindex(predictions.date).to_numpy(), atol=2e-8)
    expected_days = frame.loc[PROTOCOL["test_start"]:PROTOCOL["test_end"]].index
    for _, group in predictions.groupby(["model", "seed", "quantile"]):
        assert group.date.tolist() == expected_days.tolist()
    saved = pd.read_csv(ROOT / "results" / "metrics.csv").sort_values(["model", "seed", "quantile"]).reset_index(drop=True)
    rescored = score_table(predictions).sort_values(["model", "seed", "quantile"]).reset_index(drop=True)
    for column in rescored.select_dtypes("number").columns:
        np.testing.assert_allclose(saved[column], rescored[column], atol=2e-7, rtol=2e-6, equal_nan=True)
    reference = baselines(frame @ np.asarray(WEIGHTS))
    reference = reference.loc[reference.date.between(PROTOCOL["test_start"], PROTOCOL["test_end"])]
    keys = ["model", "seed", "quantile", "date"]
    a = predictions.loc[predictions.model != "LSTM"].sort_values(keys)
    b = reference.sort_values(keys)
    np.testing.assert_allclose(a["var"], b["var"], atol=1e-10)
    x, _, _, dates, names = samples(frame)
    scaler = json.loads((ROOT / "results" / "scaler.json").read_text())
    training = dates <= PROTOCOL["train_end"]
    fitted = StandardScaler().fit(x[training].reshape(-1, x.shape[-1]))
    np.testing.assert_allclose(fitted.mean_, scaler["mean"], atol=1e-12)
    np.testing.assert_allclose(fitted.scale_, scaler["scale"], atol=1e-12)
    testing = dates >= PROTOCOL["test_start"]
    scaled = ((x[testing]-np.asarray(scaler["mean"]))/np.asarray(scaler["scale"])).astype(np.float32)
    torch.set_num_threads(2)
    for seed in SEEDS:
        model = QuantileLSTM(len(names), PROTOCOL["hidden_size"])
        model.load_state_dict(torch.load(ROOT / "results" / f"lstm_seed_{seed}.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            inferred = model(torch.from_numpy(scaled)).numpy()/100
        saved_neural = predictions.loc[(predictions.model == "LSTM") & (predictions.seed == seed)].pivot(index="date", columns="quantile", values="var")
        np.testing.assert_allclose(inferred, saved_neural[QUANTILES], atol=1e-7)
        assert (inferred[:, 0] <= inferred[:, 1]).all()
    return dict(status="passed", daily_observations=len(frame), held_out_sessions=len(expected_days),
                forecasts=len(predictions), model_quantile_series=len(saved),
                checks=["raw-to-curated returns", "checksums and code identity", "complete aligned test calendar",
                        "prior-session origins", "actual loss reconciliation", "baseline replay", "train-only scaler",
                        "all three saved LSTM inference replays", "noncrossing quantiles", "metric recomputation"])
