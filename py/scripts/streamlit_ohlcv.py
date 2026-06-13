#!/usr/bin/env -S uv run
"""
Streamlit dashboard for OHLCV data with EMAs, beta, and benchmark overlay.

Usage:
    uv run streamlit run scripts/streamlit_ohlcv.py

Requirements:
    streamlit, pandas, plotly, numpy (all added via uv add)
"""

from __future__ import annotations
import sqlite3

import sys
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

from typing import cast
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import get_local_candles
from src.db.models import SymbolSchema

# ── Page config ────────────────────────────────────────────────

st.set_page_config(
    page_title="OHLCV Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Available symbols ──────────────────────────────────────────


def get_tickers() -> list[str]:
    con = sqlite3.connect("../data/db.sqlite")
    cur = con.cursor()
    q = """
    SELECT ticker FROM symbol
    """
    res = cur.execute(q)
    data: list[tuple[str]] = res.fetchall()
    con.close()
    return list(map(lambda t: t[0], data))


AVAILABLE_SYMBOLS: list[str] = get_tickers()

AVAILABLE_BARS: list[str] = [
    "1h",
    "4h",
    "1d",
    "1w",
    "1M",
]

# ── Cache helpers ──────────────────────────────────────────────


@st.cache_resource
def fetch_candles(symbol: str, bar: str) -> pd.DataFrame:
    """Fetch OHLCV candles for a symbol at the given bar resolution."""
    # cache_resource avoids the broken pickle path in cache_data
    df = get_local_candles(symbol, bar=bar)
    # Convert index to a portable datetime column so pickling is reliable
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.set_index("Date")
    return df


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


# ── Sidebar ────────────────────────────────────────────────────

st.sidebar.title("OHLCV Dashboard")

symbol = st.sidebar.selectbox(
    "Symbol",
    options=AVAILABLE_SYMBOLS,
    index=AVAILABLE_SYMBOLS.index("SPY") if "SPY" in AVAILABLE_SYMBOLS else 0,
)

benchmark = st.sidebar.selectbox(
    "Benchmark (for beta / overlay)",
    options=AVAILABLE_SYMBOLS,
    index=AVAILABLE_SYMBOLS.index("SPY") if "SPY" in AVAILABLE_SYMBOLS else 0,
)

bar = st.sidebar.selectbox(
    "Bar",
    options=AVAILABLE_BARS,
    index=AVAILABLE_BARS.index("1d"),
)

n_bars = st.sidebar.slider(
    "Number of bars to display",
    min_value=50,
    max_value=2000,
    value=500,
    step=50,
)


# ── Fetch data ─────────────────────────────────────────────────

# Cache key based on selections — forces re-fetch on symbol/bar change
_cache_key = (symbol, bar, benchmark)

if (
    "cached_df" not in st.session_state
    or st.session_state.get("_cache_key") != _cache_key
):
    with st.spinner(f"Fetching {symbol} ({bar})..."):
        st.session_state["cached_df"] = fetch_candles(symbol, bar)

    with st.spinner(f"Fetching {benchmark} ({bar})..."):
        st.session_state["cached_bench"] = fetch_candles(benchmark, bar)

    st.session_state["_cache_key"] = _cache_key

df = st.session_state["cached_df"]
bench_df = st.session_state["cached_bench"]

if df.empty:
    st.error(f"No data available for {symbol}")
    st.stop()

# Apply bar limit
df = df.tail(n_bars).copy()
bench_df = bench_df.tail(n_bars).copy()

# ── Compute EMAs ───────────────────────────────────────────────

ema_9 = compute_ema(df["close"], 9)
ema_100 = compute_ema(df["close"], 100)
ema_200 = compute_ema(df["close"], 200)

# ── Compute beta ───────────────────────────────────────────────

# Align both series on index
combined = df[["close"]].merge(
    bench_df[["close"]],
    left_index=True,
    right_index=True,
    how="inner",
    suffixes=("", "_bench"),
)

beta_value: float | None = None
if len(combined) > 30:
    # Beta = Cov(R_stock, R_bench) / Var(R_bench)  using log returns
    stock_ret = np.log(combined["close"] / combined["close"].shift(1)).dropna()
    bench_ret = np.log(
        combined["close_bench"] / combined["close_bench"].shift(1)
    ).dropna()

    # Align again after returns
    ret_df = pd.DataFrame({"stock": stock_ret, "bench": bench_ret}).dropna()
    if len(ret_df) > 10:
        cov_matrix = np.cov(ret_df["stock"], ret_df["bench"])
        bench_var = cov_matrix[1, 1]
        if bench_var > 1e-15:
            beta_value = float(cov_matrix[0, 1] / bench_var)
        else:
            beta_value = None
    else:
        beta_value = None
else:
    beta_value = None

# ── Sidebar metrics ───────────────────────────────────────────

with st.sidebar:
    st.metric("Bars", len(df))
    st.metric(
        "Close",
        f"${df['close'].iloc[-1]:.2f}",
        delta=f"{(df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100:.2f}%"
        if len(df) >= 2
        else None,
    )
    st.metric(
        f"β vs {benchmark}", f"{beta_value:.3f}" if beta_value is not None else "N/A"
    )

# ── Chart ────────────────────────────────────────────────────-─

if (df["volume"] > 0).any():
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7, 0.3],
    )
    volume_row = 2
else:
    fig = go.Figure()
    volume_row = None

# Candlesticks
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=symbol,
    ),
    row=1,
    col=1,
)

# EMAs
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=ema_9,
        mode="lines",
        name="EMA 9",
        line=dict(color="orange", width=1.5),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=ema_100,
        mode="lines",
        name="EMA 100",
        line=dict(color="blue", width=1.5),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=ema_200,
        mode="lines",
        name="EMA 200",
        line=dict(color="purple", width=1.5),
    ),
    row=1,
    col=1,
)

# Benchmark overlay (normalised to same starting price)
if not bench_df.empty:
    bench_close = cast(pd.Series, bench_df["close"])
    # Normalise to the symbol's starting price for visual comparison
    norm_factor = float(df["close"].iloc[0]) / float(bench_close.iloc[0])
    bench_normalised = bench_close * norm_factor
    fig.add_trace(
        go.Scatter(
            x=bench_normalised.index,
            y=bench_normalised,
            mode="lines",
            name=f"{benchmark} (norm.)",
            line=dict(color="gray", width=1.5, dash="dot"),
        ),
        row=1,
        col=1,
    )

# Volume bars
if volume_row is not None:
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker_color="rgba(100, 140, 240, 0.5)",
            showlegend=False,
        ),
        row=volume_row,
        col=1,
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1)

# Layout
fig.update_layout(
    title=f"{symbol} — {bar.upper()} OHLCV with EMA(9, 100, 200) & {benchmark} overlay",
    xaxis_title="Date",
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", y=(-0.15 if volume_row else -0.08)),
    height=700,
)
fig.update_yaxes(title_text="Price", row=1, col=1)

st.plotly_chart(fig, width="stretch")

# ── Raw data table (collapsible) ───────────────────────────────

with st.expander("View raw data"):
    display_df = df.copy()
    display_df["EMA 9"] = ema_9
    display_df["EMA 100"] = ema_100
    display_df["EMA 200"] = ema_200
    st.dataframe(
        display_df.style.format(
            {
                "open": "{:.2f}",
                "high": "{:.2f}",
                "low": "{:.2f}",
                "close": "{:.2f}",
                "volume": "{:.0f}",
                "EMA 9": "{:.2f}",
                "EMA 100": "{:.2f}",
                "EMA 200": "{:.2f}",
            }
        ),
        width="stretch",
    )
