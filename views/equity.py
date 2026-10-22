"""Offline Streamlit demo. Downloads and training are separate CLI commands."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from charts import COLORS, SECTOR_COLORS, style, line, forecasts
from risklab.config import ROOT, LABELS, SECTORS, WEIGHTS, PROTOCOL
from risklab.data import load_data
from risklab.backtest import coverage, score_table, empirical_es
from dashboard_data import load_equity_results, load_feature_history

def render():
    st.markdown("""<style>
     .block-container {padding-top:3.8rem; max-width:1440px;}
     h1 {font-weight:750; letter-spacing:-1.3px;}
     h2,h3 {letter-spacing:-.4px;}
     [data-testid="stMetric"] {background:white;border:1px solid #E2E7EF;padding:17px;border-radius:12px;}
     [data-testid="stMetricLabel"] {font-size:.83rem;color:#61718A;}
     [data-testid="stMetricValue"] {font-size:1.85rem;}
     .eyebrow {font-size:.75rem;letter-spacing:2px;font-weight:700;color:#147D73;}
     .intro {color:#61718A;max-width:850px;font-size:1.05rem;line-height:1.6;}
     [data-testid="stSidebar"] {border-right:1px solid #E2E7EF;}
     </style>""", unsafe_allow_html=True)


    @st.cache_data
    def bundle(data_stamp, result_stamp):
        frame, source = load_data()
        predictions, manifest = load_equity_results()
        return frame, source, predictions, manifest


    try:
        returns, source, predictions, manifest = bundle(
            (ROOT/"data"/"manifest.json").stat().st_mtime_ns,
            (ROOT/"results"/"manifest.json").stat().st_mtime_ns)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Saved case study unavailable: {exc}")
        st.code("python -m risklab fetch\npython -m risklab train")
        st.stop()

    portfolio = returns @ np.asarray(WEIGHTS)
    with st.sidebar:
        st.markdown("### Neural VaR Lab")
        st.caption("A weekend question, a market-risk experiment.")
        page = st.radio("Explore", ["Market overview", "Forecast & breaches", "Model backtesting", "Stress replay", "Inside the network", "Sources"])
        st.divider()
        q = st.selectbox("One-day confidence", [.95, .99], format_func=lambda x:f"{x:.0%}")
        notional = st.number_input("Illustrative exposure · USD", min_value=10000, max_value=100000000, value=1000000, step=100000)
        model = st.selectbox("Forecast model", ["LSTM", "Historical", "Gaussian", "EWMA"])
        seed = st.selectbox("LSTM seed", [7, 19, 42], help="Seed 7 was chosen before evaluating the test set. All three are reported.")
        st.divider()
        st.caption("BOOK  /  20% per industry\n\nConsumer · Manufacturing · High tech · Healthcare · Other")
        st.caption("FROZEN DATA  /  1999–2025\n\nHELD-OUT TEST  /  2020–2025\n\nUSD · daily · long only")

    selected = predictions.loc[(predictions.model == model) & (predictions.seed == (seed if model == "LSTM" else 0)) & (predictions["quantile"] == q)].sort_values("date")
    all_models = predictions.loc[(predictions["quantile"] == q) & ((predictions.model != "LSTM") | (predictions.seed == seed))]
    metrics = coverage(selected.loss, selected["var"], q)
    last = selected.iloc[-1]


    def title(kicker, headline, description):
        st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
        st.title(headline)
        st.markdown(f'<p class="intro">{description}</p>', unsafe_allow_html=True)


    def table_metrics(frame):
        display = frame[["model", "exceptions", "expected", "exception_rate", "pinball", "binomial_p", "kupiec_p", "independence_p"]].copy()
        display["pinball"] *= 10000
        display["exception_rate"] *= 100
        display.columns = ["Model", "Breaches", "Expected", "Breach rate · %", "Pinball · bps", "Binomial p", "Kupiec p", "Independence p"]
        st.dataframe(display, hide_index=True, use_container_width=True,
                     column_config={"Breach rate · %": st.column_config.NumberColumn(format="%.2f"),
                                    "Pinball · bps": st.column_config.NumberColumn(format="%.3f"),
                                    "Expected": st.column_config.NumberColumn(format="%.1f"),
                                    **{c:st.column_config.NumberColumn(format="%.3g") for c in ["Binomial p", "Kupiec p", "Independence p"]}})


    if page == "Market overview":
        title("01 / THE QUESTION", "Can a neural network see a bad day coming?",
              "I wanted to see whether a small LSTM could estimate tomorrow’s loss threshold better than simple risk models. This is a five-industry US equity case study, with every forecast checked against what actually happened.")
        cols = st.columns(4)
        cols[0].metric(f"{model} · {q:.0%} VaR", f"${last['var']*notional:,.0f}")
        cols[1].metric("Historical ES", f"${last.historical_es*notional:,.0f}", help="Same confidence as selected VaR; separate 250-session historical tail average. The LSTM predicts VaR, not ES.")
        cols[2].metric("Held-out breaches", f"{metrics['exceptions']} / {metrics['sessions']:,}")
        cols[3].metric("Expected breach rate", f"{1-q:.0%}")
        st.caption("Risk estimates above apply to the recorded 31 Dec 2025 session, using information through 30 Dec. They are not live forecasts. Breach counts cover the complete 2020–2025 test.")
        left, right = st.columns([1.65, 1])
        with left:
            market = returns.loc["2020-01-01":].rename(columns=LABELS)
            indices = 100*(1+market).cumprod()
            indices["Reference portfolio"] = 100*(1+portfolio.loc[market.index]).cumprod()
            st.plotly_chart(line(indices, "Industry returns · growth of 100", height=410,
                                 colors={"Reference portfolio":"#172B4D"}), use_container_width=True)
            st.caption("Industry indices use provider returns. The reference portfolio resets to 20% per industry daily, with no fees. These are research portfolios, not ETF prices.")
        with right:
            vol = portfolio.rolling(20).std()*np.sqrt(252)
            dd = (1+portfolio).cumprod()
            dd = dd/dd.cummax()-1
            panel = pd.DataFrame({"20-day realized volatility · annualized":vol, "Drawdown from prior peak":dd}).loc["2020":]
            panel.columns = ["Realized vol · annualized", "Drawdown"]
            st.plotly_chart(line(panel, "Volatility & drawdown", percent=True, height=410), use_container_width=True)
        st.info("The useful result is the comparison: a neural network can have a lower forecasting loss and still produce too many breaches in a crisis. Open Model backtesting to examine both.")

    elif page == "Forecast & breaches":
        title("02 / THE DAILY FORECAST", "The threshold, and the days it failed.",
              "A 99% one-day VaR is a loss threshold with a target exceedance probability of 1%. It is not a maximum possible loss. A red marker means the realized loss exceeded the forecast made the previous session.")
        period = st.selectbox("Zoom into a test window", ["Full test · 2020–2025", "COVID window · 2020", "2022", "2023", "2024", "2025"])
        visible = selected if period.startswith("Full") else selected.loc[selected.date.dt.year == (2020 if period.startswith("COVID") else int(period))]
        st.plotly_chart(forecasts(visible, notional), use_container_width=True)
        comparisons = all_models.pivot(index="date", columns="model", values="var").reindex(visible.date)
        st.plotly_chart(line(comparisons*notional, f"Four forecasts · {q:.0%} VaR in USD", colors=COLORS, height=300), use_container_width=True)
        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.markdown("#### Read one session")
            day = st.selectbox("Target session", visible.date.dt.strftime("%Y-%m-%d").tolist(), index=len(visible)-1)
            row = visible.loc[visible.date == day].iloc[0]
            st.write(f"Information cut-off: **{row.origin:%d %b %Y}**")
            st.write(f"Forecast threshold: **${row['var']*notional:,.0f}**")
            st.write(f"Realized loss: **${row.loss*notional:,.0f}**")
            st.write("**Breach**" if row.loss > row["var"] else "**Within threshold**")
            st.caption("A negative realized loss is a gain. USD figures use a fixed exposure each day; they are not a compounded account balance.")
        with col2:
            st.markdown("#### Largest breaches in this view")
            breaches = visible.loc[visible.loss > visible["var"]].copy()
            breaches["excess"] = (breaches.loss-breaches["var"])*notional
            breaches["loss"] *= notional
            breaches["var"] *= notional
            st.dataframe(breaches.nlargest(8,"excess")[["date","loss","var","excess"]], hide_index=True, use_container_width=True)
        export = visible.copy()
        export["loss_usd"] = export.loss*notional
        export["var_usd"] = export["var"]*notional
        st.download_button("Download this forecast view", export.to_csv(index=False), "forecast_view.csv", "text/csv")

    elif page == "Model backtesting":
        title("03 / MODEL CHALLENGE", "Good forecasts need more than a good chart.",
              "Compare quantile forecasting loss, breach frequency, and breach clustering over the same held-out dates. A smaller pinball score is better. A large p-value means a test did not reject its null; it does not prove the model is correct.")
        scored = score_table(all_models)
        table_metrics(scored)
        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            for _, row in scored.iterrows():
                fig.add_trace(go.Scatter(x=[row.model], y=[100*row.exception_rate], name=row.model, showlegend=False,
                                         mode="markers", marker=dict(size=13,color=COLORS[row.model]),
                                         error_y=dict(type="data",symmetric=False,array=[100*(row.ci_high-row.exception_rate)],arrayminus=[100*(row.exception_rate-row.ci_low)])))
            fig.add_hline(y=100*(1-q), line_dash="dash", line_color="#C94F5C", annotation_text="Target")
            fig.update_yaxes(title="Breach rate · %")
            st.plotly_chart(style(fig,"Coverage · exact 95% binomial intervals"),use_container_width=True)
            st.caption("Intervals assume independent Bernoulli exceptions. Market clustering can weaken that assumption; inspect the independence diagnostic alongside them.")
        with b:
            series = all_models.pivot(index="date",columns="model",values="var")
            actual = selected.set_index("date").loss
            rolling = series.lt(actual,axis=0).rolling(250).mean()
            fig = line(rolling,"Breach frequency · trailing 250 test sessions",percent=True,colors=COLORS)
            fig.add_hline(y=1-q,line_dash="dash",line_color="#C94F5C")
            st.plotly_chart(fig,use_container_width=True)
        st.markdown("#### Where did the performance change?")
        yearly = pd.read_csv(ROOT/"results"/"yearly_metrics.csv")
        yearly = yearly.loc[(yearly["quantile"]==q)&((yearly.model!="LSTM")|(yearly.seed==seed))]
        heat = yearly.pivot(index="model",columns="year",values="exception_rate")
        fig = go.Figure(go.Heatmap(z=heat.values*100,x=heat.columns.astype(str),y=heat.index,
                                   colorscale="YlOrRd",text=np.round(heat.values*100,1),texttemplate="%{text}%",
                                   colorbar=dict(title="Breach %")))
        st.plotly_chart(style(fig,"Calendar-year breach rates",280),use_container_width=True)
        with st.expander("What do these tests tell me?"):
            st.markdown("**Pinball loss** penalizes underestimating a loss quantile more heavily than overestimating it. **Kupiec** checks unconditional coverage. **Christoffersen independence** checks first-order clustering of exceptions. Conditional coverage combines those two likelihood ratios. The CSV also includes an exact binomial test and transition counts. Sparse 99% tails limit asymptotic tests; an undefined transition test is left blank. These are descriptive diagnostics across several comparisons, with no multiple-testing correction or statistical-superiority claim.")
            st.dataframe(scored[["model","conditional_coverage_p","consecutive_breaches","n00","n01","n10","n11"]],hide_index=True,use_container_width=True)
        if q == .99:
            count = int((selected.loss > selected["var"]).tail(250).sum())
            zone = "Green" if count <= 4 else "Amber" if count <= 9 else "Red"
            st.info(f"Educational 99% / 250-session traffic-light illustration: **{count} exceptions → {zone}**. This uses only a hypothetical portfolio return series; it is not a regulatory model approval or capital calculation.")
        st.download_button("Download all saved model scores",(ROOT/"results"/"metrics.csv").read_bytes(),"model_metrics.csv","text/csv")

    elif page == "Stress replay":
        title("04 / HISTORICAL STRESS", "What did the portfolio lose when markets broke?",
              "Replay observed industry returns over fixed historical windows. This shows the path and sector contribution of a hypothetical daily rebalanced portfolio. It does not predict the next crisis.")
        windows = {"Global financial crisis · peak-to-trough window":("2007-10-09","2009-03-09"),
                   "COVID sell-off":("2020-02-19","2020-03-23"),"2022 calendar year":("2021-12-31","2022-12-30"),
                   "Spring 2025":("2025-02-19","2025-04-08")}
        event = st.selectbox("Replay window",list(windows),index=1)
        start,end = windows[event]
        observed = returns.loc[(returns.index>pd.Timestamp(start))&(returns.index<=pd.Timestamp(end))]
        daily = observed @ np.asarray(WEIGHTS)
        wealth_before = (1+daily).cumprod().shift(1,fill_value=1)
        path = (1+daily).cumprod()-1
        contribution = observed.mul(WEIGHTS,axis=1).mul(wealth_before,axis=0).sum()*notional
        c = st.columns(3)
        c[0].metric("Window return",f"{path.iloc[-1]:.1%}")
        c[1].metric("Illustrative change in value",f"${path.iloc[-1]*notional:,.0f}")
        c[2].metric("Worst one-day return",f"{daily.min():.1%}")
        left,right = st.columns([1.4,1])
        with left:
            plot_path = pd.concat([pd.Series([0.],index=[pd.Timestamp(start)]),path]).to_frame("Reference portfolio")
            st.plotly_chart(line(plot_path,"Compounded return from the start close",percent=True),use_container_width=True)
        with right:
            fig=go.Figure(go.Bar(x=contribution.values,y=[LABELS[s] for s in contribution.index],orientation="h",marker_color=SECTOR_COLORS))
            fig.update_xaxes(title="USD contribution to the window P&L")
            st.plotly_chart(style(fig,"Sector contributions · sum equals window P&L"),use_container_width=True)
        st.caption(f"Window: after the {start} close through the {end} close. Daily 20% rebalancing, distributions included, no transaction costs. Contributions use start-of-day portfolio wealth and reconcile to the compounded window change.")
        if pd.Timestamp(start).year < 2020:
            st.info("The financial-crisis window is in the training history. It is shown as a historical replay only; no held-out neural performance is claimed for it.")
        else:
            stress = selected.loc[selected.date.between(observed.index[0],observed.index[-1])]
            m = coverage(stress.loss,stress["var"],q)
            st.write(f"For the selected {model} model at {q:.0%}: **{m['exceptions']} breaches across {m['sessions']} sessions** in this window. Full-test performance remains available under Model backtesting.")
        st.markdown("#### A simpler what-if: cut the exposure")
        retained = st.slider("Retain this share of the original exposure",0,100,70,5)
        st.write(f"At **{retained}%** exposure, the same historical window changes value by **${path.iloc[-1]*notional*retained/100:,.0f}**. A cash remainder has zero return here. Linear exposure scaling reduces both potential gains and losses.")

    elif page == "Inside the network":
        title("05 / NEURAL NETWORK", "Twenty sessions in. Two loss quantiles out.",
              "The LSTM reads a sequence of recent industry returns and risk features. It learns the 95th and 99th percentiles of next-day portfolio loss directly, using quantile loss rather than trying to predict a stock price.")
        cols=st.columns(4)
        for col, label, value in zip(cols,["Input window","Features per session","Trainable parameters","Random seeds"],["20 sessions",str(len(manifest["features"])),f"{manifest['parameters']:,}","3"]):
            col.metric(label,value)
        st.code("20 × 11 causal features → LSTM (24 hidden units) → last hidden state\n→ linear head → softplus q95; q99 = q95 + softplus(gap)",language="text")
        st.caption("The positive gap prevents quantile crossing. Outputs are positive loss thresholds. This architecture cannot represent a negative VaR, an explicit constraint for this long-only equity example.")
        split_rows=[dict(split=k,**v) for k,v in manifest["splits"].items()]
        st.dataframe(pd.DataFrame(split_rows),hide_index=True,use_container_width=True)
        curves=pd.read_csv(ROOT/"results"/"learning_curves.csv")
        curves=curves.loc[curves.seed==seed].set_index("epoch")[["train_pinball","validation_pinball"]]*10000
        curves.columns=["Training","Validation"]
        st.plotly_chart(line(curves,"Quantile loss by epoch · basis points",height=300),use_container_width=True)
        st.caption("Training curves average minibatches as weights change; validation uses the end-of-epoch model. Best epoch is chosen only on validation. The network is frozen throughout 2020–2025, while baseline histories update daily.")
        st.markdown("#### How much does initialization matter?")
        seed_scores=pd.read_csv(ROOT/"results"/"metrics.csv")
        seed_scores=seed_scores.loc[(seed_scores.model=="LSTM")&(seed_scores["quantile"]==q)]
        seed_scores=seed_scores[["seed","exceptions","exception_rate","pinball"]].copy()
        seed_scores["pinball"]*=10000
        seed_scores=seed_scores.rename(columns={"pinball":"pinball_bps"})
        st.dataframe(seed_scores,hide_index=True,use_container_width=True)
        st.markdown("#### Inspect the inputs for one forecast")
        day=st.selectbox("Forecast target",selected.date.dt.strftime("%Y-%m-%d").tolist(),index=len(selected)-1)
        cutoff=selected.loc[selected.date==day,"origin"].iloc[0]
        features=load_feature_history(source["csv_sha256"])
        window=features.loc[:cutoff].tail(20)
        scaler=json.loads((ROOT/"results"/"scaler.json").read_text())
        z=(window-np.asarray(scaler["mean"]))/np.asarray(scaler["scale"])
        fig=go.Figure(go.Heatmap(z=z.to_numpy().T,x=z.index.strftime("%Y-%m-%d"),y=z.columns,colorscale="RdBu_r",zmid=0,colorbar=dict(title="Train z-score")))
        st.plotly_chart(style(fig,"The actual 20-session input · scaled with training data only",380),use_container_width=True)
        st.caption(f"Latest input: {cutoff:%d %b %Y}. Target: {day}. Features are five industry returns, portfolio return, 5/20-day volatility, downside RMS, EWMA absolute return and 60-session drawdown. This is an input view, not a causal explanation of the model.")

    elif page == "Sources":
        title("06 / SOURCES", "The data behind the experiment.",
              "Frozen public industry returns, a recorded training protocol, and checksummed forecasts make this comparison reproducible.")
        st.markdown(f"[Kenneth French five-industry returns]({source['details']}) · [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) · [BIS backtesting context](https://www.bis.org/basel_framework/chapter/MAR/32.htm?inforce=20230101&published=20200327)")
        st.caption(f"Snapshot retrieved {source['retrieved_utc'][:10]}. Public research returns may be revised and are not point-in-time vintages. Data copyright belongs to its providers; this app’s code is separate.")
        st.markdown("Training: 1999–2016. Validation: 2017–2019. Held-out test: 2020–2025. All results refer to one fixed US equity research portfolio. No live prices or production risk calibration are implied.")
        with st.expander("Saved experiment identity"):
            st.json({"data_sha256":manifest["source_sha256"],"code_sha256":manifest["code_sha256"],"created_utc":manifest["created_utc"],"selection":manifest["selection"],"runtime_seconds":manifest["total_seconds"]})

    st.divider()
    st.caption("NEURAL VaR LAB  /  Personal research project  /  Frozen public data  /  One-day portfolio loss  /  Built to question the model")
