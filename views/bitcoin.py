"""Bitcoin workspace: cached social NLP and historical market risk."""
import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bitcoin.config import ROOT, DATA, RESULTS, MODELS
from bitcoin.data import load_data
from bitcoin.pipeline import load_scored
from bitcoin.analytics import daily_sentiment, market_metrics, lag_associations
from bitcoin_charts import style, market_pulse, sentiment_mix, source_timeline, risk_panels, lead_lag, UP, DOWN, BLUE, INK, ORANGE

def render():
    st.markdown("""<style>
     .block-container {padding-top:3.6rem;max-width:1500px;}
     h1 {font-size:2.45rem!important;font-weight:750;letter-spacing:-1px;}
     h2,h3 {letter-spacing:-.3px;}
     [data-testid="stMetric"] {background:#fff;border:1px solid #E0E7F0;border-radius:10px;padding:15px;}
     [data-testid="stMetricLabel"] {font-size:.8rem;color:#6A7D95;}
     [data-testid="stMetricValue"] {font-size:1.75rem;}
     [data-testid="stSidebar"] {border-right:1px solid #E0E7F0;}
     .kicker {font-size:.73rem;letter-spacing:1.8px;font-weight:700;color:#17867C;}
     .summary {color:#677D96;font-size:1rem;line-height:1.65;max-width:1000px;}
     .post-card {border:1px solid #E0E7F0;border-radius:10px;background:white;padding:16px;margin:10px 0;}
     .post-meta {font-size:.75rem;color:#6A7D95;margin-bottom:8px;}
     .post-text {font-size:.9rem;line-height:1.6;color:#19324D;white-space:pre-wrap;}
     </style>""",unsafe_allow_html=True)


    @st.cache_data
    def bundle(data_stamp,result_stamp):
        _,prices,source=load_data()
        scored,run=load_scored()
        return scored,market_metrics(prices),source,run


    try:
        posts,market,source,run=bundle((DATA/"manifest.json").stat().st_mtime_ns,(RESULTS/"manifest.json").stat().st_mtime_ns)
    except (FileNotFoundError,ValueError) as exc:
        st.error(f"The Bitcoin replay is unavailable: {exc}")
        st.code("python -m bitcoin fetch\npython -m bitcoin model\npython -m bitcoin score")
        st.stop()

    with st.sidebar:
        st.markdown("### ₿ Narrative Risk")
        st.caption("Bitcoin market & social research")
        presets={"Full archive":("2021-01-01","2022-12-31"),"2021":("2021-01-01","2021-12-31"),
                 "2022":("2022-01-01","2022-12-31"),"May–June 2022":("2022-05-01","2022-06-30"),
                 "November 2022":("2022-11-01","2022-11-30")}
        preset=st.selectbox("Research window",list(presets))
        bounds=presets[preset]
        # A range picker can temporarily contain one date while the user selects its end.
        dates=st.date_input("Date range",value=(pd.Timestamp(bounds[0]).date(),pd.Timestamp(bounds[1]).date()),
                             min_value=market.index.min().date(),max_value=market.index.max().date(),key=f"range-{preset}")
        if len(dates)!=2:
            st.info("Select both ends of the date range.")
            st.stop()
        start,end=map(pd.Timestamp,dates)
        sources=st.multiselect("Text sources",["Twitter","Reddit"],default=["Twitter","Reddit"])
        model=st.selectbox("Sentiment model",["FinBERT · int8","VADER · baseline"])
        aggregation=st.selectbox("Daily aggregation",["Post mean","Equal source mean"],help="Equal source mean gives each source present that day equal weight. Missing sources are not invented.")
        minimum=st.slider("Minimum sampled posts / day",1,10,1)
        st.divider()
        st.caption("HISTORICAL REPLAY\n\n2021–2022 · USD · daily\n\nProvider calendar dates; timezone metadata is incomplete.")
        st.caption("Coverage is a selected research sample. Missing posts do not mean neutral sentiment or zero platform activity.")

    if not sources:
        st.warning("Choose at least one text source to inspect the replay.")
        st.stop()
    if start>=end:
        st.warning("Choose a date range of at least two days.")
        st.stop()

    model_name="FinBERT" if model.startswith("FinBERT") else "VADER"
    score="finbert_score" if model_name=="FinBERT" else "vader_score"
    label="finbert_label" if model_name=="FinBERT" else "vader_label"
    selected=posts.loc[posts.source.isin(sources)&posts.date.between(start,end)].copy()
    view=market.loc[start:end].copy()
    daily=daily_sentiment(selected,view.index,model_name,aggregation,minimum)
    for name in ["Twitter","Reddit"]:
        daily[f"{name}_posts"]=selected.loc[selected.source==name].groupby("date").size().reindex(view.index,fill_value=0)

    st.markdown('<div class="kicker">MARKET INTELLIGENCE / HISTORICAL REPLAY</div>',unsafe_allow_html=True)
    st.title("Bitcoin | Narrative Risk")
    st.markdown('<div class="summary">Price action, sampled public discussion, and a compact financial-language model. Follow the market move into the underlying posts, then challenge what the sentiment signal actually tells you.</div>',unsafe_allow_html=True)
    st.caption(f"{start:%d %b %Y} — {end:%d %b %Y}  ·  {' + '.join(sources)}  ·  {model_name}  ·  {aggregation}  ·  At least {minimum} post(s) per scored day")

    cols=st.columns(4)
    cols[0].metric("BTC close · window end",f"${view.close.iloc[-1]:,.0f}")
    change=view.close.iloc[-1]/view.close.iloc[0]-1
    cols[1].metric("Window return",f"{change:+.1%}",help="Last close versus the first selected day's close. This is an observed return, not a trading backtest.")
    mean=daily.sentiment.mean()
    cols[2].metric("Daily sentiment · mean",f"{mean:+.2f}" if pd.notna(mean) else "No eligible days",help="Unweighted average of available daily language sentiment scores. Range −1 to +1; not a probability of a price rise.")
    cols[3].metric("Bitcoin-related posts",f"{len(selected):,}",help="Counts retained, deduplicated posts in the selected sources and dates. This is sampled attention, not platform-wide volume.")

    pulse_tab,narrative_tab,risk_tab,model_tab,sources_tab=st.tabs(["Market pulse","Narrative explorer","Sentiment vs risk","Model lab","Sources & notes"])

    with pulse_tab:
        chart_kind=st.radio("Price display",["Close","OHLC candles"],horizontal=True,label_visibility="collapsed")
        st.plotly_chart(market_pulse(view,daily,chart_kind=="OHLC candles"),use_container_width=True,config={"displaylogo":False})
        eligible=int(daily.eligible.sum())
        both=int(((daily.sources==2)&daily.eligible).sum())
        st.caption(f"Daily score coverage: {eligible} / {len(view)} selected days; {both} have both sources. Gaps stay missing. Volume bars count retained posts even when the minimum-per-day rule hides a sentiment score. Price is unfiltered by text availability.")
        left,right=st.columns([1.3,1])
        with left:
            source_rows=[]
            for name in sources:
                group=selected.loc[selected.source==name]
                part=daily_sentiment(group,view.index,model_name,minimum=minimum).reset_index()
                part["source"]=name
                source_rows.append(part)
            st.plotly_chart(source_timeline(pd.concat(source_rows)),use_container_width=True,config={"displaylogo":False})
        with right:
            st.plotly_chart(sentiment_mix(selected,label),use_container_width=True,config={"displaylogo":False})
            st.caption("Class proportions use all selected posts. The daily minimum rule applies to time-series scores, not this post-level distribution.")
        if selected.empty:
            st.info("No retained Bitcoin posts in this window. Price history is still available; the sentiment panels intentionally remain empty.")
        else:
            disagreement=float(selected.disagreement.mean())
            st.info(f"Model challenge: FinBERT and VADER disagree on **{disagreement:.0%}** of these posts. This is model disagreement, not an error rate. Read examples in Narrative explorer.")

    with narrative_tab:
        st.markdown("#### Go from a score to the language behind it")
        st.caption("FinBERT's classes describe language sentiment, not whether Bitcoin will rise. TF-IDF / NMF themes are discovered retrospectively across the full frozen corpus.")
        control1,control2=st.columns([1.4,1])
        with control1:
            query=st.text_input("Search the sampled posts",placeholder="e.g. mining, exchange, regulation")
        with control2:
            review=st.selectbox("Review queue",["All posts","Model disagreement","Low FinBERT confidence","Negative language","Truncated inputs"])
        topics=["All themes"]+sorted(selected.topic.dropna().unique().tolist())
        theme=st.selectbox("Discovered theme · top terms",topics)
        detail=selected.copy()
        if query:
            detail=detail.loc[detail.text.str.contains(query,case=False,regex=False)]
        if theme!="All themes":
            detail=detail.loc[detail.topic==theme]
        if review=="Model disagreement":
            detail=detail.loc[detail.disagreement]
        elif review=="Low FinBERT confidence":
            detail=detail.loc[detail.confidence<.65]
        elif review=="Negative language":
            detail=detail.loc[detail[label]=="negative"]
        elif review=="Truncated inputs":
            detail=detail.loc[detail.truncated]
        st.caption(f"{len(detail):,} matching posts · latest 8 shown below. Search and review filters affect this tab's detail and export; the market summary stays scoped to the sidebar.")
        for _,row in detail.sort_values(["date","post_id"],ascending=False).head(8).iterrows():
            text=html.escape(row.text[:1000])+("…" if len(row.text)>1000 else "")
            meta=f"{row.source} · {row.date:%d %b %Y} · {row[label].title()} · score {row[score]:+.2f} · FinBERT confidence {row.confidence:.0%}"
            st.markdown(f'<div class="post-card"><div class="post-meta">{html.escape(meta)}</div><div class="post-text">{text}</div></div>',unsafe_allow_html=True)
            with st.expander(f"Evidence & model comparison · {row.post_id}"):
                st.write(f"FinBERT: positive {row.positive:.3f}, neutral {row.neutral:.3f}, negative {row.negative:.3f}. VADER: {row.vader_label}, compound {row.vader_score:+.3f}.")
                st.write(f"Theme: {row.topic}. Tokens before the {run['max_tokens']}-token limit: {int(row.token_count)}. Truncated: {bool(row.truncated)}.")
                st.caption("Twitter entries link to their research archive; individual tweet IDs were not available in the selected source. Reddit URLs may point to an external article shared in the post.")
                st.link_button(f"Source record · {row.post_id} ↗",row.url)
        if detail.empty:
            st.info("No posts match these detail filters.")
        st.download_button("Download this review queue",detail.to_csv(index=False),"bitcoin_post_review.csv","text/csv")

    with risk_tab:
        st.markdown("#### Does the discussion add information about risk?")
        st.caption("Explore associations with returns and losses. Both models were pretrained; this is not an out-of-sample sentiment trading strategy or a test of causality.")
        col1,col2=st.columns([1.25,1])
        with col1:
            st.plotly_chart(risk_panels(view),use_container_width=True,config={"displaylogo":False})
        with col2:
            # Targets stay inside the selected daily price calendar.
            associations=lag_associations(daily.sentiment,view)
            st.plotly_chart(lead_lag(associations),use_container_width=True,config={"displaylogo":False})
            st.caption("Positive lag pairs sentiment at day t with returns on a later calendar day. Negative lag pairs it with earlier returns. Shifts occur on the full daily price calendar, before missing scores are removed.")
        st.markdown("#### A prior-history loss threshold")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=view.index,y=view.loss,name="Realized daily loss",line=dict(color="#9BAABD",width=1)))
        fig.add_trace(go.Scatter(x=view.index,y=view.historical_var_95,name="Prior 60-day · 95% historical VaR",line=dict(color=ORANGE,width=2)))
        breaches=view.loc[view.breach_95]
        fig.add_trace(go.Scatter(x=breaches.index,y=breaches.loss,mode="markers",name="VaR breach",marker=dict(color=DOWN,size=6)))
        fig.update_yaxes(tickformat=".0%",title="Loss positive · fraction of exposure")
        st.plotly_chart(style(fig,"Bitcoin downside risk · historical benchmark",340),use_container_width=True,config={"displaylogo":False})
        st.caption("This VaR is a 60-observation historical loss quantile that excludes the target day. The first 61 calendar days have no estimate. It uses the full archive before the selected window; changing the date filter does not refit the history. This Bitcoin benchmark is separate from the neural equity VaR experiment.")
        st.dataframe(associations,hide_index=True,use_container_width=True,column_config={"spearman_rho":st.column_config.NumberColumn(format="%.3f")})
        st.caption("Correlations are descriptive, with no multiple-testing correction, predictive-ability test or causality claim. The short, biased samples and uncertain publication times preclude a tradable timing claim.")
        export=view.join(daily.add_prefix("social_"))
        st.download_button("Download daily market & sentiment",export.to_csv(),"bitcoin_daily_research.csv","text/csv")

    with model_tab:
        st.markdown("#### A compact neural language model, with a simple challenger")
        st.markdown("**FinBERT** uses a BERT encoder pretrained for financial language and a three-class sentiment head. This project uses an **int8 ONNX** conversion on CPU. The displayed corpus scores are precomputed; browsing the dashboard needs no model calls, GPU, API key or paid service.")
        model_size=110.7
        cols=st.columns(3)
        cols[0].metric("Quantized model file",f"{model_size:.1f} MB")
        cols[1].metric("Cached posts scored",f"{run['posts']:,}")
        cols[2].metric("Token limit",str(run["max_tokens"]))
        st.code("Clean URLs / handles → tokenize → 128-token window → int8 FinBERT\n→ P(positive), P(neutral), P(negative)\n→ score = P(positive) − P(negative) → daily aggregation",language="text")
        st.markdown("VADER adds a lexicon-and-rule comparison with negation, punctuation and intensity handling. Its compound score also ranges from −1 to +1, but has a different meaning from a neural probability difference.")
        st.caption(f"Recorded CPU scoring time: {run['inference_seconds']:.1f} seconds for {run['posts']:,} posts, plus {run['model_init_seconds']:.1f} seconds to initialize. {run['truncated']} posts were truncated; {run['low_confidence']} had maximum softmax probability below 65%. These are runtime/diagnostic counts, not model-accuracy estimates.")
        st.info("FinBERT was not validated on this crypto social sample. Sarcasm, slang, mixed-topic posts, leading-token truncation and text about assets other than BTC can all mislead it. Model confidence is not calibrated correctness. NMF themes describe corpus language, not economic causes.")
        left,right=st.columns(2)
        with left:
            classes=["positive","neutral","negative"]
            cross=pd.crosstab(selected.finbert_label,selected.vader_label).reindex(index=classes,columns=classes,fill_value=0)
            fig=go.Figure(go.Heatmap(z=cross.values,x=classes,y=classes,colorscale="Blues",text=cross.values,texttemplate="%{text}",colorbar=dict(title="Posts")))
            fig.update_xaxes(title="VADER class")
            fig.update_yaxes(title="FinBERT class")
            st.plotly_chart(style(fig,"Model agreement · neither axis is ground truth",330,False),use_container_width=True,config={"displaylogo":False})
        with right:
            fig=go.Figure()
            for name,color in [("Twitter",BLUE),("Reddit",ORANGE)]:
                group=selected.loc[selected.source==name]
                fig.add_trace(go.Histogram(x=group.confidence,name=name,xbins=dict(start=0,end=1,size=.05),marker_color=color,opacity=.6))
            fig.update_layout(barmode="overlay")
            fig.update_xaxes(range=[0,1],title="Maximum softmax probability")
            fig.update_yaxes(title="Sampled posts")
            st.plotly_chart(style(fig,"FinBERT confidence · uncalibrated",330),use_container_width=True,config={"displaylogo":False})
        @st.fragment
        def local_analysis():
            st.markdown("#### Try one short sentence locally")
            st.caption("First use downloads the pinned 111 MB model. Text is analyzed on this app’s CPU and is not added to the archive.")
            sentence=st.text_area("Financial text",value="Bitcoin fell sharply as traders worried about exchange liquidity.",max_chars=2000)
            if st.button("Analyze locally",type="primary"):
                if not sentence.strip():
                    st.warning("Enter some text first.")
                else:
                    try:
                        @st.cache_resource
                        def engine():
                            from bitcoin.sentiment import SentimentEngine, download_model
                            download_model()
                            return SentimentEngine()
                        with st.spinner("Loading the CPU model and analyzing text…"):
                            result=engine().predict([sentence])[0]
                        st.dataframe(pd.DataFrame([result]),hide_index=True,use_container_width=True)
                        st.caption("One-text batches can differ slightly from saved corpus scores because quantized inference uses dynamic padding. This example is not added to historical data.")
                    except ImportError:
                        st.info("Text inference is unavailable in this environment. Historical results remain available.")
                    except Exception as exc:
                        st.error(f"Text analysis could not complete: {exc}")
        local_analysis()

    with sources_tab:
        st.markdown("#### Open data, explicit sampling, reproducible scores")
        st.markdown("**Tweets:** [Jahanbin, Zare Chahooki & Rahmanian · Mendeley Data v5](https://data.mendeley.com/datasets/8fbdhh72gs/5), CC BY 4.0. Selected public crypto influencers, collected by the researchers through the Apify Twitter API. We use raw text, not the archive's existing sentiment labels.")
        st.markdown("**Reddit + Bitcoin prices:** [George Vlahavas · Zenodo](https://zenodo.org/records/12593440), CC BY 4.0. Selected popular r/CryptoCurrency posts and archived daily BTC-USD OHLCV. Counts below come from the files, which differ from some counts in the archive description.")
        st.dataframe(pd.DataFrame([dict(source=k,posts=source["post_counts"][k],**v) for k,v in source["source_coverage"].items()]),hide_index=True,use_container_width=True)
        st.caption(f"From {source['archive_rows']['tweets']:,} archived tweets and {source['archive_rows']['reddit']:,} archived Reddit posts, the Bitcoin keyword/time filter retained {source['retained_before_dedup']:,}; {source['duplicate_texts_removed']} duplicate texts were then removed. Filtering is based on explicit bitcoin/btc/bitcoins mentions; implicit references can be missed. The source sample is not a population sample.")
        st.markdown("**Date caution:** Twitter has dates only, parsed as month/day/year. The Reddit and price files lack explicit timezone metadata. Alignment is by provider calendar date; no synchronized UTC, intraday timing or point-in-time availability claim is made. Engagement is stored for context and is never used as a predictive weight.")
        st.markdown("#### Technical blogs and GitHub references")
        st.markdown("""
    - [Plotly: Dashboard Design Mistakes That Mislead Decision Makers](https://plotly.com/blog/dashboard-design-mistakes/) — visible filter scope, dated KPIs, meaningful comparison and missing-data states.
    - [Plotly: Interactive Dashboards vs Static Reports](https://plotly.com/blog/interactive-dashboards-vs-static-reports/) — overview-to-detail investigation and downloadable evidence.
    - [Plotly Python financial charts](https://plotly.com/python/candlestick-charts/) — OHLC candlesticks and shared-axis market panels.
    - [ProsusAI / finBERT on GitHub](https://github.com/ProsusAI/finBERT) — the financial-language model; [Xenova ONNX weights](https://huggingface.co/Xenova/finbert) supply the pinned quantized conversion.
    - [cjhutto / vaderSentiment on GitHub](https://github.com/cjhutto/vaderSentiment) — rule-based social-language challenger.
    - [ONNX Runtime on GitHub](https://github.com/microsoft/onnxruntime) — local CPU inference.
    """)
        st.caption("App and pipeline code are original. Design patterns and library APIs were adapted; third-party weights/data retain their own rights. Full citations, reuse details and mathematical definitions are in bitcoin/METHODOLOGY.md and bitcoin/REFERENCES.md.")
        with st.expander("Show the recorded pipeline identity"):
            st.json({"source_created_utc":source["created_utc"],"code_sha256":run["code_sha256"],"model_revision":run["model"]["revision"],"inference_seconds":run["inference_seconds"],"data_checksums":run["data_checksums"]})

    st.divider()
    st.caption("BITCOIN / NARRATIVE RISK · Public research archives · Cached CPU sentiment · Descriptive market analysis · No live feed")
