import hashlib
import html
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from .config import DATA, RAW, START, END, SOURCES

BTC = re.compile(r"\b(?:bitcoin|btc|bitcoins)\b", re.I)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_text(text):
    text = html.unescape(str(text)).replace("\\n", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(?<!\w)@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"[#$](?=[A-Za-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def download_sources():
    RAW.mkdir(parents=True,exist_ok=True)
    # Public archive metadata pins file identity and the DOI version.
    r=requests.get("https://data.mendeley.com/public-api/datasets/8fbdhh72gs",timeout=60)
    r.raise_for_status()
    archive=r.json()
    if archive["version"] != 5:
        raise ValueError("Expected the reviewed version 5 tweet archive")
    item=next(x for x in archive["files"] if x["id"]=="159e4f05-0903-45c1-b12e-e4038805bd97")
    r=requests.get(item["content_details"]["download_url"],timeout=90)
    r.raise_for_status()
    if hashlib.sha256(r.content).hexdigest()!=item["content_details"]["sha256_hash"]:
        raise ValueError("Tweet archive hash mismatch")
    (RAW/"tweets.csv").write_bytes(r.content)
    (RAW/"tweet_archive_metadata.json").write_text(json.dumps(archive,indent=2),encoding="utf-8")
    r=requests.get("https://zenodo.org/api/records/12593440",timeout=60)
    r.raise_for_status()
    archive=r.json()
    (RAW/"reddit_archive_metadata.json").write_text(json.dumps(archive,indent=2),encoding="utf-8")
    for name in ["reddit_posts.csv","BTC-USD.csv"]:
        item=next(x for x in archive["files"] if x["key"]==name)
        r=requests.get(item["links"]["self"],timeout=90)
        r.raise_for_status()
        if "md5:"+hashlib.md5(r.content).hexdigest()!=item["checksum"]:
            raise ValueError(f"Archive MD5 mismatch: {name}")
        (RAW/name).write_bytes(r.content)


def curate():
    DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/"manifest.json").exists():
        raise FileExistsError("Frozen Bitcoin data exists; preserve it and rebuild in a separate copy")
    twitter=pd.read_csv(RAW/"tweets.csv")
    reddit=pd.read_csv(RAW/"reddit_posts.csv")
    prices=pd.read_csv(RAW/"BTC-USD.csv")
    twitter_day=pd.to_datetime(twitter.created_at,format="%m/%d/%Y",errors="raise")
    reddit_day=pd.to_datetime(reddit.date,format="%Y-%m-%d %H:%M:%S",errors="raise").dt.normalize()
    entries=[]
    for i,row in twitter.iterrows():
        text=clean_text(row.full_text)
        if BTC.search(text) and START<=str(twitter_day.iloc[i].date())<=END:
            entries.append(dict(post_id=f"tw-{int(row['Unnamed: 0'])}",date=twitter_day.iloc[i],source="Twitter",
                                text=text,title=text[:180],url=SOURCES["Twitter"],
                                timestamp_precision="day; source timezone unspecified",
                                engagement=int(row.favorite_count)+int(row.retweet_count)+int(row.reply_count)))
    for i,row in reddit.iterrows():
        title=clean_text(row.title)
        body=clean_text(row.body) if pd.notna(row.body) else ""
        text=title if body.startswith(title) else title+". "+body
        if body.startswith(title):
            text=body
        if BTC.search(text) and START<=str(reddit_day.iloc[i].date())<=END:
            url=str(row.url)
            if not url.startswith("https://"):
                url=SOURCES["Reddit"]
            entries.append(dict(post_id=f"rd-{int(row.X)}",date=reddit_day.iloc[i],source="Reddit",text=text.strip(),
                                title=title,url=url,timestamp_precision="timestamp in archive; timezone unspecified",
                                engagement=int(row.score)))
    posts=pd.DataFrame(entries)
    if posts.empty:
        raise ValueError("No Bitcoin posts in fixed window")
    posts["text_hash"]=posts.text.map(lambda x:hashlib.sha256(x.casefold().encode()).hexdigest())
    before=len(posts)
    posts=posts.sort_values(["date","source","post_id"]).drop_duplicates(["source","text_hash"],keep="first")
    if posts.post_id.duplicated().any():
        raise ValueError("Duplicate archive post identifiers")
    posts.to_csv(DATA/"posts.csv",index=False)
    prices=prices.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    prices["date"]=pd.to_datetime(prices.date)
    prices=prices.loc[prices.date.between(START,END),["date","open","high","low","close","volume"]].sort_values("date")
    if prices.date.duplicated().any() or not (prices[["open","high","low","close"]]>0).all().all():
        raise ValueError("Invalid daily price data")
    if not (prices.high>=prices[["open","close","low"]].max(axis=1)).all() or not (prices.low<=prices[["open","close"]].min(axis=1)).all():
        raise ValueError("OHLC bar bounds do not reconcile")
    expected=pd.date_range(START,END,freq="D")
    if prices.date.tolist()!=expected.tolist():
        raise ValueError("Missing calendar-day prices; do not fill")
    prices.to_csv(DATA/"prices.csv",index=False)
    raw_checksums={p.name:digest(p) for p in RAW.iterdir() if p.is_file()}
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=SOURCES,
                  period=dict(start=START,end=END),price_days=len(prices),
                  archive_rows=dict(tweets=len(twitter),reddit=len(reddit)),
                  retained_before_dedup=before,duplicate_texts_removed=before-len(posts),
                  post_counts={str(k):int(v) for k,v in posts.source.value_counts().items()},
                  source_coverage={name:dict(first=str(g.date.min().date()),last=str(g.date.max().date()),days=int(g.date.nunique())) for name,g in posts.groupby("source")},
                  raw_checksums=raw_checksums,checksums={name:digest(DATA/name) for name in ["posts.csv","prices.csv"]},
                  selection="Explicit whole-word bitcoin/btc/bitcoins in cleaned text; fixed 2021–2022 window; normalized exact text deduplication within source, keep earliest date. No sentiment-based selection.",
                  dates="Twitter archive has calendar dates only (month/day/year). Reddit timestamps and price dates lack timezone metadata. Daily joins are research approximations, not synchronized UTC or point-in-time availability.",
                  sample_bias="Tweets are selected influencers; Reddit archive favors popular posts. Engagement is measured retrospectively and is not used as a forecast weight.",
                  archive_count_note="Zenodo description mentions 770 posts; downloaded file contains 907. Counts above use the actual frozen file.",
                  licenses={"tweets":"CC BY 4.0; Jahanbin, Zare Chahooki and Rahmanian (2023), DOI 10.17632/8fbdhh72gs.5", "reddit_and_prices":"CC BY 4.0; George Vlahavas (2024), DOI 10.5281/zenodo.12593440"})
    (DATA/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def load_data():
    manifest=json.loads((DATA/"manifest.json").read_text(encoding="utf-8"))
    for name,expected in manifest["checksums"].items():
        if digest(DATA/name)!=expected:
            raise ValueError(f"Bitcoin data checksum mismatch: {name}")
    return pd.read_csv(DATA/"posts.csv",parse_dates=["date"]),pd.read_csv(DATA/"prices.csv",parse_dates=["date"]),manifest
