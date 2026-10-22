import plotly.graph_objects as go

COLORS = {"LSTM": "#147D73", "Historical": "#6376AA", "Gaussian": "#BD8B43", "EWMA": "#9467BD"}
SECTOR_COLORS = ["#147D73", "#6376AA", "#BD8B43", "#BD5E78", "#699DAD"]


def style(fig, title="", height=360, percent=False):
    fig.update_layout(title=dict(text=title, font=dict(size=15), x=0, y=.98), height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Arial, sans-serif", color="#172B4D", size=12),
                      margin=dict(l=20, r=20, t=55, b=85),
                      legend=dict(orientation="h", yanchor="top", y=-.18, x=0, font=dict(size=11)),
                      hovermode="x unified")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#E5E9EF", zerolinecolor="#C7CFDA")
    if percent:
        fig.update_yaxes(tickformat=".1%")
    return fig


def line(frame, title, percent=False, colors=None, height=360):
    fig = go.Figure()
    for i, col in enumerate(frame.columns):
        color = (colors or {}).get(col, SECTOR_COLORS[i % len(SECTOR_COLORS)])
        fig.add_trace(go.Scatter(x=frame.index, y=frame[col], name=str(col),
                                 mode="lines", line=dict(color=color, width=2)))
    return style(fig, title, height, percent)


def forecasts(group, notional):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=group.date, y=group.loss*notional, name="Realized loss",
                             line=dict(color="#9BA8BC", width=1)))
    fig.add_trace(go.Scatter(x=group.date, y=group["var"]*notional, name="Forecast VaR",
                             line=dict(color="#147D73", width=2.2)))
    hit = group.loss > group["var"]
    fig.add_trace(go.Scatter(x=group.loc[hit, "date"], y=group.loc[hit, "loss"]*notional,
                             name="VaR breach", mode="markers", marker=dict(color="#C94F5C", size=7)))
    fig = style(fig, "One-day loss versus the forecast made the previous session", 420)
    fig.update_yaxes(title="USD · loss positive", tickprefix="$")
    fig.update_xaxes(rangeslider_visible=False)
    return fig
