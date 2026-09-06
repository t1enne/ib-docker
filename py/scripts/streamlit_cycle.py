#!/usr/bin/env -S uv run
"""
Cycle Screener Dashboard — Visual regime ratio explorer.

Replaces the arbitrary-threshold terminal screener with an interactive
Streamlit dashboard. Plots the full history of each cross-asset ratio
with rolling statistics so you can visually assess regime state.

Usage:
    uv run streamlit run scripts/streamlit_cycle.py
"""

from __future__ import annotations
from src.data import resample_ohlcv

import sqlite3
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

from src.utils import get_local_candles

# ── Page config ────────────────────────────────────────────────

st.set_page_config(
    page_title="Cycle Ratios Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Available tickers ──────────────────────────────────────────


@st.cache_resource
def get_tickers() -> list[str]:
    con = sqlite3.connect("../data/db.sqlite")
    cur = con.cursor()
    rows = cur.execute("SELECT ticker FROM symbol ORDER BY ticker").fetchall()
    con.close()
    return [r[0] for r in rows]


AVAILABLE_SYMBOLS: list[str] = get_tickers()

# ── Ratio definitions ordered by signal hierarchy ──
# 1. Credit (leads equities), 2. Yield curve (recession clock),
# 3. Equity sector, 4. Breadth, 5-6. Commodities (real-economy),
# 7-9. Secondary confirmation
_RATIO_DEFS: list[tuple[str, str, str]] = [
    ("HYG / TLT (credit/treasury)", "HYG", "TLT"),
    ("HYG / LQD (credit quality)", "HYG", "LQD"),
    ("CPER / GLD (copper/gold)", "CPER", "GLD"),
    ("QQQ / SPY (Tech concentration)", "QQQ", "SPY"),
    ("IWM / SPY (small/large cap)", "IWM", "SPY"),
    ("XLY / XLP (discretionary/staples)", "XLY", "XLP"),
    ("SHY / IEF (2Y/10Y proxy)", "SHY", "IEF"),
    ("USO / GLD (oil/gold)", "USO", "GLD"),
    ("XLF / XLU (financial/utility)", "XLF", "XLU"),
    ("SPY / TLT", "SPY", "TLT"),
]


# ── Data fetch ─────────────────────────────────────────────────


@st.cache_resource
def fetch_daily_close(symbol: str) -> pd.Series:
    """Fetch daily close prices for a symbol."""
    df = get_local_candles(symbol, bar="1h")
    if df.empty:
        return pd.Series(dtype=float)
    daily = resample_ohlcv(df, "1d")
    if daily.empty:
        return pd.Series(dtype=float)
    return daily["close"]


def compute_ratio_series(
    num_series: pd.Series,
    den_series: pd.Series,
) -> pd.Series:
    """Compute ratio of two aligned price series."""
    combined = pd.DataFrame({"num": num_series, "den": den_series}).dropna()
    if combined.empty:
        return pd.Series(dtype=float)
    return combined["num"] / combined["den"]


def compute_rolling_stats(
    ratio: pd.Series,
    window: int = 63,
) -> dict[str, pd.Series]:
    """Compute rolling mean and ±2σ bands for a ratio series."""
    roll_mean = ratio.rolling(window, min_periods=max(5, window // 4)).mean()
    roll_std = ratio.rolling(window, min_periods=max(5, window // 4)).std()
    return {
        "sma": roll_mean,
        "upper": roll_mean + 2 * roll_std,
        "lower": roll_mean - 2 * roll_std,
    }


# ── Sidebar ────────────────────────────────────────────────────

st.sidebar.title("Cycle Ratios")

bar = st.sidebar.selectbox(
    "Bar resolution (fetch)", options=["1h", "4h", "1d"], index=0
)

lookback = st.sidebar.slider(
    "Lookback (trading days)",
    min_value=60,
    max_value=1260,
    value=504,
    step=21,
)

rolling_window = st.sidebar.slider(
    "Rolling window (days)",
    min_value=21,
    max_value=252,
    value=63,
    step=21,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Each plot shows the ratio history (blue), a rolling mean (orange), "
    "and ±2σ bands (shaded). Use these to visually assess where the "
    "current ratio sits relative to its own history — no arbitrary thresholds."
)

# ── Fetch all needed data ──────────────────────────────────────

_NEEDED_TICKERS: list[str] = sorted(
    {den for _, _, den in _RATIO_DEFS} | {num for _, num, _ in _RATIO_DEFS}
)

with st.spinner("Fetching data for all cross-asset tickers..."):
    closes: dict[str, pd.Series] = {}
    missing: list[str] = []
    for ticker in _NEEDED_TICKERS:
        s = fetch_daily_close(ticker)
        if s.empty:
            missing.append(ticker)
        else:
            closes[ticker] = s

if missing:
    st.warning(f"No data for: {', '.join(missing)}")

if not closes:
    st.error("No ticker data available. Check database.")
    st.stop()

# ── Apply lookback ─────────────────────────────────────────────

cutoff_date: pd.Timestamp | None = None
if lookback > 0:
    all_dates = sorted({d for s in closes.values() for d in s.index})
    if len(all_dates) >= lookback:
        cutoff_date = all_dates[-lookback]
        closes = {
            k: v[v.index >= cutoff_date]  # type: ignore[misc]
            for k, v in closes.items()
        }

# ── Compute ratios & stats ─────────────────────────────────────

st.title("Macro Cycle Ratio Explorer")

_RATIO_PLOTS: list[tuple[str, str, str]] = _RATIO_DEFS


def _plot_ratio(
    label: str,
    num_ticker: str,
    den_ticker: str,
    col: DeltaGenerator,
) -> None:
    """Plot a single ratio with rolling statistics into a Streamlit column."""
    num_s = closes.get(num_ticker)
    den_s = closes.get(den_ticker)

    if num_s is None or den_s is None:
        with col:
            st.caption(f"{label} — data missing")
        return

    ratio = compute_ratio_series(num_s, den_s)
    if ratio.empty or len(ratio) < 5:
        with col:
            st.caption(f"{label} — insufficient data")
        return

    stats = compute_rolling_stats(ratio, rolling_window)
    current = ratio.iloc[-1]
    sma_val = stats["sma"].iloc[-1]
    upper_val = stats["upper"].iloc[-1]
    lower_val = stats["lower"].iloc[-1]

    # Z-score style positioning
    if pd.notna(upper_val) and pd.notna(lower_val) and (upper_val - lower_val) > 0:
        z_score = (current - sma_val) / (
            (upper_val - lower_val) / 4.0
        )  # 2σ → 4σ total band
    else:
        z_score = 0.0

    # Build plot
    fig = go.Figure()

    # ±2σ band (shaded)
    if stats["upper"].notna().any() and stats["lower"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=ratio.index.tolist() + ratio.index.tolist()[::-1],
                y=stats["upper"].tolist() + stats["lower"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(128, 128, 128, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Ratio line
    fig.add_trace(
        go.Scatter(
            x=ratio.index,
            y=ratio,
            mode="lines",
            name=label,
            line=dict(color="royalblue", width=1.5),
        )
    )

    # Rolling mean
    fig.add_trace(
        go.Scatter(
            x=stats["sma"].index,
            y=stats["sma"],
            mode="lines",
            name=f"SMA({rolling_window})",
            line=dict(color="darkorange", width=1.2, dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{label}<br>"
                f"<sup>current: {current:.4f} | z≈{z_score:+.1f} "
                f"| sma: {sma_val:.4f}</sup>"
            ),
            x=0.01,
        ),
        xaxis_title=None,
        template="plotly_white",
        hovermode="x unified",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="top", y=-0.12),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    with col:
        st.plotly_chart(fig, width="stretch")


# ── Layout: 2 columns per row ──────────────────────────────────

for i in range(0, len(_RATIO_PLOTS), 2):
    cols = st.columns(2)
    for j, (label, num, den) in enumerate(_RATIO_PLOTS[i : i + 2]):
        _plot_ratio(label, num, den, cols[j])  # type: ignore[arg-type]

# ── Summary table ──────────────────────────────────────────────

st.markdown("---")
st.subheader("Current Summary")

rows: list[dict[str, object]] = []
for label, num, den in _RATIO_PLOTS:
    num_s = closes.get(num)
    den_s = closes.get(den)
    if num_s is None or den_s is None:
        rows.append(
            {
                "Ratio": label,
                "Current": "N/A",
                f"SMA({rolling_window})": "N/A",
                "Z≈": "N/A",
                "Trend (21d)": "N/A",
            }
        )
        continue
    ratio = compute_ratio_series(num_s, den_s)
    if ratio.empty or len(ratio) < 5:
        rows.append(
            {
                "Ratio": label,
                "Current": "N/A",
                f"SMA({rolling_window})": "N/A",
                "Z≈": "N/A",
                "Trend (21d)": "N/A",
            }
        )
        continue
    stats = compute_rolling_stats(ratio, rolling_window)
    current_v = ratio.iloc[-1]
    sma_v = stats["sma"].iloc[-1]
    upper_v = stats["upper"].iloc[-1]
    lower_v = stats["lower"].iloc[-1]
    if pd.notna(upper_v) and pd.notna(lower_v) and (upper_v - lower_v) > 0:
        z_v = (current_v - sma_v) / ((upper_v - lower_v) / 4.0)
    else:
        z_v = np.nan
    # 21-day trend (simple return)
    if len(ratio) >= 22:
        trend_21 = (ratio.iloc[-1] / ratio.iloc[-22] - 1.0) * 100.0
    else:
        trend_21 = np.nan

    def _fmt(v: float | None) -> str:
        if v is None or pd.isna(v):
            return "N/A"
        return f"{v:.4f}"

    rows.append(
        {
            "Ratio": label,
            "Current": _fmt(current_v),
            f"SMA({rolling_window})": _fmt(sma_v),
            "Z≈": f"{z_v:+.1f}" if not pd.isna(z_v) else "N/A",
            "Trend (21d)": f"{trend_21:+.2f}%" if not pd.isna(trend_21) else "N/A",
        }
    )

summary_df = pd.DataFrame(rows)

st.dataframe(
    summary_df,
    width="stretch",
    hide_index=True,
    column_config={
        "Ratio": st.column_config.TextColumn("Ratio"),
        "Current": st.column_config.TextColumn("Current"),
        f"SMA({rolling_window})": st.column_config.TextColumn(f"SMA({rolling_window})"),
        "Z≈": st.column_config.TextColumn("Z≈"),
        "Trend (21d)": st.column_config.TextColumn("Trend (21d)"),
    },
)

st.caption(
    f"Z≈ = (current − sma) / (σ)  where σ = band_width / 4. "
    f"Data window: last {lookback} trading days. Charts use {rolling_window}-day rolling stats."
)
