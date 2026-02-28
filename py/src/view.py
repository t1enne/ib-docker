import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import numpy as np

from src.db import db
from src.db.models import SymbolSchema, CandleSchema
from src.utils import get_log_returns, symmetric_cointegration_p, list_to_axes


def get_symbols():
    symbols = SymbolSchema.select(SymbolSchema.ticker).order_by(SymbolSchema.ticker)
    return [s.ticker for s in symbols]


def get_candle_data(tickers: list[str], days: int | None = None):
    query = CandleSchema.select().where(CandleSchema.ticker.in_(tickers))
    if days:
        cutoff_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        query = query.where(CandleSchema.timestamp >= cutoff_ts)

    candles = list(query.order_by(CandleSchema.timestamp))

    if not candles:
        return pd.DataFrame()

    data = []
    for c in candles:
        data.append(
            {
                "ticker": c.ticker,
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
        )

    df = pd.DataFrame(data)
    df["datetime"] = (
        pd.to_datetime(df["timestamp"], unit="ms")
        .dt.tz_localize("UTC")
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    return df


def main():
    st.set_page_config(page_title="Historical Data Viewer", layout="wide")
    st.title("Historical Data Viewer")

    symbols = get_symbols()
    if not symbols:
        st.warning("No symbols found in database. Run sync command first.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_symbols = col1.multiselect("Select symbols", symbols, default="QQQ")

        if not selected_symbols:
            col1.info("Select at least one symbol to view data.")
            return
        window = col1.select_slider(
            "Time window",
            options=[30, 90, 180, 365, None],
            value=30,
            format_func=lambda x: f"{x} days" if x else "All",
        )

    df = get_candle_data(selected_symbols, window)

    if df.empty:
        st.warning("No data found for selected symbols and time window.")
        return

    normalized_dfs = []
    for ticker in selected_symbols:
        ticker_df = df[df["ticker"] == ticker].copy()
        ticker_df = ticker_df.sort_values("datetime")
        ticker_df.loc[:, "pct_change"] = (
            ticker_df["close"] / ticker_df["close"].iloc[0] - 1
        ) * 100
        normalized_dfs.append(ticker_df[["datetime", "pct_change", "ticker"]])

    all_tickers_df = pd.concat(normalized_dfs)

    fig = px.line(
        all_tickers_df,
        x="datetime",
        y="pct_change",
        color="ticker",
        title="Normalized Price (% Change)",
    )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        xaxis=dict(
            rangebreaks=[
                dict(bounds=[16, 9.5], pattern="hour"),
                dict(bounds=["sat", "mon"], pattern="day of week"),
            ],
        ),
    )
    with col2:
        col2.plotly_chart(fig, width="stretch")

        # Correlation and Cointegration Matrices
        if len(selected_symbols) >= 2:
            col2.subheader("Correlation & Cointegration")

            price_pivot = df.pivot(index="datetime", columns="ticker", values="close")
            returns_df = price_pivot.pct_change().dropna()

            tickers = list(returns_df.columns)
            corr_matrix = pd.DataFrame(
                index=list_to_axes(tickers), columns=list_to_axes(tickers), dtype=float
            )
            cointegration_matrix = pd.DataFrame(
                index=list_to_axes(tickers), columns=list_to_axes(tickers), dtype=float
            )

            for i, sym1 in enumerate(tickers):
                for j, sym2 in enumerate(tickers):
                    if i == j:
                        corr_matrix.loc[sym1, sym2] = 1.0
                        cointegration_matrix.loc[sym1, sym2] = 0.0
                        continue

                    corr = np.corrcoef(returns_df[sym1], returns_df[sym2])[0, 1]
                    p_val = symmetric_cointegration_p(
                        price_pivot[sym1], price_pivot[sym2]
                    )
                    corr_matrix.loc[sym1, sym2] = corr
                    cointegration_matrix.loc[sym1, sym2] = p_val

            corr_fig = px.imshow(
                corr_matrix.astype(float),
                text_auto=".2f",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Correlation Matrix",
            )
            coin_fig = px.imshow(
                cointegration_matrix.astype(float),
                text_auto=".3f",
                color_continuous_scale="RdYlGn_r",
                title="Cointegration Matrix (p-values)",
            )

            col2.plotly_chart(corr_fig, width="stretch")
            col2.plotly_chart(coin_fig, width="stretch")

        # Data table
        col2.subheader("Data Table")
        display_df = df[
            ["datetime", "ticker", "open", "high", "low", "close", "volume"]
        ].copy()
        display_df.loc[:, "datetime"] = display_df["datetime"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        col2.dataframe(display_df, width="stretch")


if __name__ == "__main__":
    main()
