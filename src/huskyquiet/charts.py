"""Plotly figures for the web app. Numbers come from the caller; nothing is hard-coded."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from huskyquiet.analysis import RAMP_HEX

_INK = "#1C2421"
_MUTED = "#5C6B73"
_TEAL = "#0E4D5C"
_GOLD = "#A6843D"
_CATEGORICAL = ("#0E4D5C", "#A6843D", "#3D6B58", "#8C4A3A", "#5C6B73", "#1F6F8B")


def legend_figure() -> go.Figure:
    labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    figure = go.Figure(go.Bar(
        x=labels,
        y=[1, 1, 1, 1, 1],
        marker_color=list(RAMP_HEX[:5]),
        hoverinfo="skip",
        showlegend=False,
    ))
    figure.update_layout(
        title="Color is P(issued within 90 days). Darker is higher. Not a noise scale.",
        yaxis_visible=False,
        height=120,
        margin=dict(l=8, r=8, t=36, b=8),
        bargap=0.08,
    )
    return _polish(figure)


def probability_histogram(series: dict[str, pd.Series]) -> go.Figure:
    figure = go.Figure()
    for i, (name, values) in enumerate(series.items()):
        figure.add_trace(go.Histogram(
            x=values,
            name=name,
            nbinsx=12,
            marker_color=_CATEGORICAL[i % len(_CATEGORICAL)],
            opacity=0.72,
        ))
    figure.update_layout(
        barmode="overlay",
        xaxis_title="Predicted P(issued within 90 days)",
        yaxis_title="Permits",
        xaxis_range=[0, 1],
        height=320,
        legend_title_text="",
    )
    return _polish(figure)


def category_bars(left_name: str, left: dict[str, int], right_name: str, right: dict[str, int]) -> go.Figure:
    keys = list(dict.fromkeys([*left.keys(), *right.keys()]))
    figure = go.Figure()
    figure.add_trace(go.Bar(name=left_name, x=keys, y=[left.get(k, 0) for k in keys], marker_color=_TEAL))
    figure.add_trace(go.Bar(name=right_name, x=keys, y=[right.get(k, 0) for k in keys], marker_color=_GOLD))
    figure.update_layout(barmode="group", yaxis_title="Pending permits", height=340, legend_title_text="")
    return _polish(figure)


def metric_bars(labels: list[str], values: list[float], title: str, y_title: str) -> go.Figure:
    figure = go.Figure(go.Bar(x=labels, y=values, marker_color=_TEAL, text=[f"{v:.3f}" for v in values],
                              textposition="outside"))
    figure.update_layout(title=title, yaxis_title=y_title, height=360, yaxis_range=[0, max(values + [1]) * 1.15])
    return _polish(figure)


def reliability_figure(curves: dict[str, dict]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                line=dict(color=_MUTED, dash="dash")))
    colors = {"historical_rate_baseline": _GOLD, "hgb_calibrated_platt": _TEAL}
    for i, (name, table) in enumerate(curves.items()):
        figure.add_trace(go.Scatter(
            x=table["mean_predicted"],
            y=table["observed_rate"],
            mode="lines+markers",
            name=name,
            line=dict(color=colors.get(name, _CATEGORICAL[i % len(_CATEGORICAL)])),
        ))
    figure.update_layout(
        title="Reliability on the test period",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed 90-day issuance rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        height=420,
    )
    return _polish(figure)


def backtest_figure(rows: list[dict]) -> go.Figure:
    figure = go.Figure()
    x = [row["test_start"] for row in rows]
    for key, label, color in (
        ("historical_rate", "Historical-rate baseline", _GOLD),
        ("hgb", "HistGradientBoosting", _TEAL),
    ):
        figure.add_trace(go.Scatter(
            x=x,
            y=[row["metrics"][key]["roc_auc"] for row in rows],
            mode="lines+markers",
            name=label,
            line=dict(color=color),
        ))
    figure.update_layout(
        title="Rolling-origin backtest, ROC-AUC",
        xaxis_title="Test year starting",
        yaxis_title="ROC-AUC",
        yaxis_range=[0.5, 0.9],
        height=380,
    )
    return _polish(figure)


def importance_figure(rows: list[dict], limit: int = 12) -> go.Figure:
    chosen = list(reversed(rows[:limit]))
    figure = go.Figure(go.Bar(
        x=[row["roc_auc_drop"] for row in chosen],
        y=[row["feature"] for row in chosen],
        orientation="h",
        marker_color=_TEAL,
    ))
    figure.update_layout(
        title="Permutation importance on a test sample (drop in ROC-AUC)",
        xaxis_title="ROC-AUC drop when shuffled",
        height=420,
    )
    return _polish(figure)


def month_gap_figure(table: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=table["month"], y=table["observed"], mode="lines+markers", name="Observed issuance rate",
        line=dict(color=_GOLD),
    ))
    figure.add_trace(go.Scatter(
        x=table["month"], y=table["predicted"], mode="lines+markers", name="Mean predicted probability",
        line=dict(color=_TEAL),
    ))
    figure.update_layout(
        title="Test-period calibration by forecast month",
        yaxis_title="Rate",
        yaxis_range=[0, max(0.6, float(table[["observed", "predicted"]].max().max()) + 0.05)],
        height=380,
    )
    return _polish(figure)


def _polish(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        template="simple_white",
        font=dict(family="Inter, sans-serif", size=13, color=_INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=16, r=16, t=48, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return figure
