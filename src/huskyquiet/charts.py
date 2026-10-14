"""Plotly figures for the web app. Numbers come from the caller; nothing is hard-coded."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from huskyquiet.reporting import feature_label, model_name

_INK = "#1C2421"
_MUTED = "#5C6B73"
_TEAL = "#0E4D5C"
_GOLD = "#A6843D"
_GRID = "#E3E0D8"


def probability_histogram(series: dict[str, pd.Series], title: str) -> go.Figure:
    """Side-by-side counts of permits in 10-point probability bins."""
    figure = go.Figure()
    for (name, values), color, pattern in zip(series.items(), (_TEAL, _GOLD), ("", "/"), strict=False):
        figure.add_trace(go.Histogram(
            x=values,
            name=name,
            xbins=dict(start=0, end=1.0001, size=0.1),
            marker=dict(color=color, pattern_shape=pattern, line=dict(color="white", width=1)),
            hovertemplate="%{x}<br>%{y} permits<extra>" + name + "</extra>",
        ))
    figure.update_layout(
        title=title,
        barmode="group",
        xaxis=dict(title="Predicted chance of issuance within 90 days", tickformat=".0%", range=[0, 1], dtick=0.1),
        yaxis=dict(title="Pending permits", rangemode="tozero"),
        height=340,
        bargap=0.15,
    )
    return _polish(figure)


def reliability_figure(curves: dict[str, dict]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
        line=dict(color=_MUTED, dash="dash", width=1),
    ))
    styles = {"hgb_calibrated_platt": (_TEAL, "circle"), "historical_rate_baseline": (_GOLD, "square")}
    for key in ("hgb_calibrated_platt", "historical_rate_baseline"):
        if key not in curves:
            continue
        color, symbol = styles[key]
        figure.add_trace(go.Scatter(
            x=curves[key]["mean_predicted"],
            y=curves[key]["observed_rate"],
            mode="lines+markers",
            name=model_name(key),
            marker=dict(symbol=symbol, size=7),
            line=dict(color=color),
            hovertemplate="Predicted %{x:.0%}<br>Observed %{y:.0%}<extra></extra>",
        ))
    figure.update_layout(
        title="Predicted vs observed issuance, held-out test (Aug 2025 to Jul 2026)",
        xaxis=dict(title="Average predicted chance (10 equal-size groups)", tickformat=".0%", range=[0, 1]),
        yaxis=dict(title="Share actually issued within 90 days", tickformat=".0%", range=[0, 1]),
        height=420,
    )
    return _polish(figure)


def backtest_figure(rows: list[dict]) -> go.Figure:
    figure = go.Figure()
    x = [row["test_start"][:4] + "–" + row["test_end_exclusive"][2:4] for row in rows]
    for key, label, color, symbol in (
        ("hgb", "Gradient-boosted trees", _TEAL, "circle"),
        ("historical_rate", "Historical-rate baseline", _GOLD, "square"),
    ):
        figure.add_trace(go.Scatter(
            x=x,
            y=[row["metrics"][key]["roc_auc"] for row in rows],
            mode="lines+markers",
            name=label,
            marker=dict(symbol=symbol, size=8),
            line=dict(color=color),
            hovertemplate="%{x}: %{y:.2f}<extra>" + label + "</extra>",
        ))
    figure.update_layout(
        title="ROC-AUC in earlier one-year test windows (retrained before each)",
        xaxis=dict(title="Test year (August to July)", type="category"),
        yaxis=dict(title="ROC-AUC (0.5 = random ranking)", range=[0.5, 0.9], tickformat=".2f"),
        height=380,
    )
    return _polish(figure)


def importance_figure(rows: list[dict], limit: int = 10) -> go.Figure:
    chosen = list(reversed(rows[:limit]))
    figure = go.Figure(go.Bar(
        x=[row["roc_auc_drop"] for row in chosen],
        y=[feature_label(row["feature"]) for row in chosen],
        orientation="h",
        marker_color=_TEAL,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    figure.update_layout(
        title=f"Top {len(chosen)} inputs by permutation importance, test sample",
        xaxis=dict(title="Drop in ROC-AUC when the input is shuffled", tickformat=".2f", rangemode="tozero"),
        height=400,
        showlegend=False,
    )
    return _polish(figure)


def month_gap_figure(table: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=table["month"], y=table["observed"], mode="lines+markers", name="Share actually issued",
        marker=dict(symbol="square", size=7), line=dict(color=_GOLD),
        hovertemplate="%{x}: %{y:.0%}<extra>Observed</extra>",
    ))
    figure.add_trace(go.Scatter(
        x=table["month"], y=table["predicted"], mode="lines+markers", name="Average predicted chance",
        marker=dict(symbol="circle", size=7), line=dict(color=_TEAL),
        hovertemplate="%{x}: %{y:.0%}<extra>Predicted</extra>",
    ))
    top = max(0.5, float(table[["observed", "predicted"]].max().max()) + 0.05)
    figure.update_layout(
        title="Predicted vs observed by forecast month, held-out test",
        xaxis=dict(title="Forecast month", type="category"),
        yaxis=dict(title="Issued within 90 days", tickformat=".0%", range=[0, top]),
        height=380,
    )
    return _polish(figure)


def _polish(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        template="simple_white",
        font=dict(family="Inter, sans-serif", size=13, color=_INK),
        title=dict(font=dict(size=15), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=56, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title_text=""),
    )
    figure.update_yaxes(showgrid=True, gridcolor=_GRID)
    return figure
