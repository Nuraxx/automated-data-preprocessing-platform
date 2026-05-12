"""Plotly-based visualization helpers for the Streamlit app."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def quality_score_gauge(score: int) -> go.Figure:
    """Gauge-style indicator for a 0–100 quality score."""

    score = int(max(0, min(100, score)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.25},
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def histogram(df: pd.DataFrame, column: str, nbins: int = 30) -> go.Figure:
    s = pd.to_numeric(df[column], errors="coerce")
    fig = px.histogram(s, nbins=nbins, title=f"Histogram — {column}")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def box_plot(df: pd.DataFrame, column: str) -> go.Figure:
    s = pd.to_numeric(df[column], errors="coerce")
    fig = px.box(s, points="outliers", title=f"Box Plot — {column}")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def correlation_heatmap(df: pd.DataFrame, numeric_cols: Sequence[str]) -> go.Figure:
    if len(numeric_cols) < 2:
        return go.Figure()

    corr = df[list(numeric_cols)].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Heatmap",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def missing_value_heatmap(df: pd.DataFrame, max_rows: int = 300) -> go.Figure:
    if df.empty:
        return go.Figure()

    view = df
    if len(df) > max_rows:
        view = df.sample(n=max_rows, random_state=42)

    mat = view.isna().astype(int)
    fig = px.imshow(
        mat,
        aspect="auto",
        title=f"Missing Value Heatmap (sampled {len(view)} rows)",
        color_continuous_scale="Reds",
        zmin=0,
        zmax=1,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def null_percentage_bar(df: pd.DataFrame, top_n: int = 40) -> go.Figure:
    if df.empty:
        return go.Figure()

    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct.head(top_n)
    fig = px.bar(
        missing_pct,
        title=f"Null Percentage by Column (Top {min(top_n, len(missing_pct))})",
        labels={"value": "Missing %", "index": "Column"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-35)
    return fig


def pie_chart_categorical(df: pd.DataFrame, column: str, top_n: int = 10) -> go.Figure:
    s = df[column].astype(str).fillna("<missing>")
    counts = s.value_counts(dropna=False)

    if len(counts) > top_n:
        top = counts.head(top_n)
        other = counts.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({"Other": other})])

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title=f"Category Share — {column}",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def value_distribution(df: pd.DataFrame, column: str, top_n: int = 25) -> go.Figure:
    if pd.api.types.is_numeric_dtype(df[column]):
        return histogram(df, column)

    s = df[column].astype(str).fillna("<missing>")
    counts = s.value_counts().head(top_n)
    fig = px.bar(
        counts,
        title=f"Value Distribution — {column} (Top {min(top_n, len(counts))})",
        labels={"value": "Count", "index": "Value"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-35)
    return fig


def scatter_plot(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None) -> go.Figure:
    working = df[[x, y] + ([color] if color else [])].copy()
    fig = px.scatter(
        working,
        x=x,
        y=y,
        color=color,
        title=f"Scatter Plot — {x} vs {y}",
        opacity=0.8,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def empty_figure(title: str = "") -> go.Figure:
    fig = go.Figure()
    if title:
        fig.update_layout(title=title)
    fig.update_layout(height=200)
    return fig
