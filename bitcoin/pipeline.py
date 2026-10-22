import hashlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import ROOT, DATA, RAW, RESULTS, MODELS, MAX_TOKENS
from .data import load_data, digest
from .analytics import market_metrics, daily_sentiment


def code_identity():
    h=hashlib.sha256()
    for path in sorted(ROOT.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def build():
    from .sentiment import SentimentEngine
    if RESULTS.exists():
        raise FileExistsError("Frozen sentiment outputs exist; rebuild in a separate copy")
    stage=ROOT/".staging"
    stage.mkdir()
    posts,prices,source=load_data()
    begin=time.perf_counter()
    engine=SentimentEngine()
    init_seconds=time.perf_counter()-begin
    inference_begin=time.perf_counter()
    scores=pd.DataFrame(engine.predict(posts.text.tolist(),progress=True))
    inference_seconds=time.perf_counter()-inference_begin
    scored=pd.concat([posts.reset_index(drop=True),scores],axis=1)
    vectorizer=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=3,max_df=.75,max_features=4000,
                               token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b")
    matrix=vectorizer.fit_transform(scored.text)
    nmf=NMF(n_components=5,init="nndsvda",max_iter=400,random_state=7)
    representation=nmf.fit_transform(matrix)
    vocabulary=np.asarray(vectorizer.get_feature_names_out())
    topics=[]
    for i,weights in enumerate(nmf.components_):
        terms=vocabulary[np.argsort(weights)[-8:][::-1]].tolist()
        topics.append(dict(topic=i,label=" / ".join(terms[:3]),terms=terms))
    scored["topic_id"]=representation.argmax(axis=1)
    no_evidence=representation.sum(axis=1)<1e-10
    scored.loc[no_evidence,"topic_id"]=-1
    mapping={x["topic"]:x["label"] for x in topics}
    mapping[-1]="Unassigned"
    scored["topic"]=scored.topic_id.map(mapping)
    scored.to_csv(stage/"posts_scored.csv",index=False,float_format="%.10g")
    (stage/"topics.json").write_text(json.dumps(topics,indent=2),encoding="utf-8")
    market=market_metrics(prices)
    market.to_csv(stage/"daily_market.csv",float_format="%.10g")
    by_source=[]
    for name,group in scored.groupby("source"):
        daily=daily_sentiment(group,market.index)
        daily["source"]=name
        by_source.append(daily.reset_index())
    pd.concat(by_source).to_csv(stage/"daily_sources.csv",index=False,float_format="%.10g")
    model_manifest=json.loads((MODELS/"manifest.json").read_text())
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),code_sha256=code_identity(),
                  data_checksums=source["checksums"],model=model_manifest,max_tokens=MAX_TOKENS,
                  inference_seconds=inference_seconds,model_init_seconds=init_seconds,total_seconds=time.perf_counter()-begin,
                  posts=len(scored),truncated=int(scored.truncated.sum()),
                  low_confidence=int((scored.confidence<.65).sum()),disagreements=int(scored.disagreement.sum()),
                  topic_method="Retrospective TF-IDF + NMF; 5 components; labels are top terms, not causal or supervised topics.",
                  model_limits="FinBERT is trained on financial text, not validated on this crypto social sample. VADER is a rule-based comparator, not ground truth. Softmax confidence is not calibrated correctness.",
                  checksums={p.name:digest(p) for p in stage.iterdir() if p.is_file()})
    (stage/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    os.replace(stage,RESULTS)
    return manifest


def load_scored():
    manifest=json.loads((RESULTS/"manifest.json").read_text(encoding="utf-8"))
    for name,expected in manifest["checksums"].items():
        if digest(RESULTS/name)!=expected:
            raise ValueError(f"Sentiment checksum mismatch: {name}")
    scored=pd.read_csv(RESULTS/"posts_scored.csv",parse_dates=["date"])
    return scored,manifest


def audit(replay_model=False):
    posts,prices,source=load_data()
    scored,run=load_scored()
    assert code_identity()==run["code_sha256"]
    assert source["checksums"]==run["data_checksums"]
    for name,expected in source["raw_checksums"].items():
        assert digest(RAW/name)==expected
    assert posts.post_id.tolist()==scored.post_id.tolist()
    assert posts.text.tolist()==scored.text.tolist()
    assert not scored.duplicated(["source","text_hash"]).any()
    probabilities=scored[["positive","negative","neutral"]]
    assert ((probabilities>=0)&(probabilities<=1)).all().all()
    np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=2e-7)
    np.testing.assert_allclose(scored.finbert_score,scored.positive-scored.negative,atol=1e-8)
    assert np.array_equal(probabilities.idxmax(axis=1).to_numpy(),scored.finbert_label.to_numpy())
    assert np.array_equal((scored.finbert_label!=scored.vader_label).to_numpy(),scored.disagreement.to_numpy())
    saved=pd.read_csv(RESULTS/"daily_market.csv",parse_dates=["date"]).set_index("date")
    replay=market_metrics(prices)
    for col in ["return","realized_vol_30","drawdown","loss","historical_var_95","next_day_return"]:
        np.testing.assert_allclose(saved[col],replay[col],atol=1e-7,equal_nan=True)
    if replay_model:
        from .sentiment import SentimentEngine
        engine=SentimentEngine()
        # Replay all posts in the original batch order: dynamic padding can slightly
        # affect int8 inference in a different batch shape.
        output=pd.DataFrame(engine.predict(scored.text.tolist(),progress=True))
        np.testing.assert_allclose(output[["positive","negative","neutral"]],probabilities,atol=1e-7)
        np.testing.assert_allclose(output.vader_score,scored.vader_score,atol=1e-9)
    return dict(status="passed",posts=len(scored),price_days=len(prices),probability_simplex=True,
                source_and_result_checksums=True,market_recomputed=True,all_neural_scores_replayed=replay_model)
