"""Shared Plotly design: aligned panels, restrained colors, explicit units."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INK="#19324D"
MUTED="#8293A7"
UP="#17867C"
DOWN="#C85965"
ORANGE="#D79B3A"
BLUE="#6685B2"
SOURCE_COLORS={"Twitter":BLUE,"Reddit":ORANGE}
LABEL_COLORS={"positive":UP,"neutral":"#A7B3C4","negative":DOWN}


def style(fig,title="",height=380,legend=True):
    fig.update_layout(title=dict(text=title,x=0,y=.99,font=dict(size=15,color=INK)),
                      height=height,template="plotly_white",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Arial, sans-serif",size=12,color=INK),
                      margin=dict(l=20,r=20,t=52,b=65 if legend else 40),
                      legend=dict(orientation="h",x=0,y=-.12,yanchor="top",font=dict(size=11)),
                      showlegend=legend,hovermode="x unified",bargap=.15)
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(gridcolor="#E6ECF3",zerolinecolor="#CDD7E4")
    return fig


def market_pulse(prices,daily,candles=False):
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.035,row_heights=[.55,.25,.20])
    if candles:
        fig.add_trace(go.Candlestick(x=prices.index,open=prices.open,high=prices.high,low=prices.low,close=prices.close,
                                     name="BTC / USD",increasing_line_color=UP,decreasing_line_color=DOWN),row=1,col=1)
    else:
        fig.add_trace(go.Scatter(x=prices.index,y=prices.close,name="BTC / USD",line=dict(color=INK,width=2.2)),row=1,col=1)
    colors=np.where(daily.sentiment.fillna(0)>=0,UP,DOWN)
    fig.add_trace(go.Bar(x=daily.index,y=daily.sentiment,marker_color=colors,name="Daily sentiment"),row=2,col=1)
    for source in ["Twitter","Reddit"]:
        col=f"{source}_posts"
        if col in daily:
            fig.add_trace(go.Bar(x=daily.index,y=daily[col],name=f"{source} posts",marker_color=SOURCE_COLORS[source]),row=3,col=1)
    fig.update_layout(barmode="stack",xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price · USD",tickprefix="$",tickformat=",.0f",row=1,col=1)
    fig.update_yaxes(title_text="Sentiment",range=[-1,1],tickvals=[-1,0,1],row=2,col=1)
    fig.update_yaxes(title_text="Posts",rangemode="tozero",row=3,col=1)
    fig=style(fig,"Bitcoin price, language sentiment and sampled attention",610)
    fig.update_layout(legend=dict(y=-.07))
    return fig


def source_timeline(daily_sources):
    fig=go.Figure()
    for source in ["Twitter","Reddit"]:
        group=daily_sources.loc[daily_sources.source==source]
        fig.add_trace(go.Scatter(x=group.date,y=group.sentiment,name=source,mode="lines+markers",connectgaps=False,
                                 marker=dict(size=3),line=dict(color=SOURCE_COLORS[source],width=1.5)))
    fig.add_hline(y=0,line_color="#B4C1D1",line_dash="dot")
    fig.update_yaxes(range=[-1,1],title="Mean score · −1 to +1")
    return style(fig,"Do the sampled communities sound different?",340)


def sentiment_mix(posts,label):
    frame=pd.crosstab(posts.source,posts[label],normalize="index").reindex(columns=["positive","neutral","negative"],fill_value=0)
    fig=go.Figure()
    for sentiment in frame.columns:
        fig.add_trace(go.Bar(x=frame[sentiment],y=frame.index,name=sentiment.title(),orientation="h",marker_color=LABEL_COLORS[sentiment]))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%",range=[0,1])
    return style(fig,"Language sentiment mix · within each source",280)


def risk_panels(market):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.1,row_heights=[.55,.45])
    fig.add_trace(go.Scatter(x=market.index,y=market.realized_vol_30,name="30-day vol · annualized",line=dict(color=BLUE,width=2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=market.index,y=market.drawdown,name="Drawdown",line=dict(color=DOWN,width=1.5),fill="tozeroy",fillcolor="rgba(200,89,101,.12)"),row=2,col=1)
    fig.update_yaxes(tickformat=".0%")
    return style(fig,"Volatility and drawdown · full-history context",430)


def lead_lag(table):
    fig=go.Figure(go.Bar(x=table.lag_days,y=table.spearman_rho,
                         marker_color=[BLUE if n<=0 else UP for n in table.lag_days],
                         customdata=table.paired_days,hovertemplate="Lag %{x} days<br>Spearman %{y:.3f}<br>%{customdata} paired dates<extra></extra>"))
    fig.update_yaxes(range=[-1,1],title="Spearman correlation")
    fig.update_xaxes(title="Return date minus sentiment date · days",dtick=1)
    return style(fig,"Lead / lag association · descriptive only",350,False)
