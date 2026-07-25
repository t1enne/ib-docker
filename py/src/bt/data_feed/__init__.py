"""Functional data feed module.

Provides pure functions for loading and transforming market data into Candles.
Can be injected into Backtest engine via EngineDependencies.

Usage:
    from src.bt.data_feed import load_candles, candles_from_dataframe

    # Load data
    df = load_candles(["AAPL", "MSFT"], start, end, "1h")

    # Generate candles
    async for candle in candles_from_dataframe(df, ["AAPL", "MSFT"]):
        process(candle)
"""

import pandas as pd
from pandas import Timestamp
from typing import List, cast

from src.bt.state import Candle
from src.utils import get_local_candles
from src.syncm import sync_data


def load_candles(
    symbols: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar: str,
) -> pd.DataFrame:
    """Pure function - loads candles for given symbols and date range.

    Args:
        symbols: List of ticker symbols
        start: Start timestamp
        end: End timestamp
        bar: Bar size (e.g., "1h", "1d")

    Returns:
        DataFrame with MultiIndex (symbol, timestamp) containing OHLCV data
    """
    time_series = [get_local_candles(s, start, end, bar) for s in symbols]
    return pd.concat(time_series, axis=1, keys=symbols)


def candles_from_dataframe(
    df: pd.DataFrame,
    symbols: List[str],
):
    """Pure async generator - yields Candles (OHLCV bars) from a DataFrame.

    This function is stateless - given the same DataFrame and symbols,
    it will always yield the same sequence of Candles.

    Args:
        df: DataFrame with MultiIndex (symbol, timestamp) containing OHLCV
        symbols: List of symbols to generate candles for

    Yields:
        Candle objects for each symbol at each timestamp
    """
    if df.empty or len(df.index) == 0:
        return

    all_timestamps = sorted(
        set().union(*[set(df.xs(s, axis=1).index) for s in symbols])
    )

    for ts in all_timestamps:
        for s in symbols:
            try:
                row = df.xs(s, axis=1).loc[ts]
                pdt = cast(pd.Timestamp, Timestamp(ts))
                assert not pd.isna(pdt)
                yield Candle(
                    timestamp=pdt,
                    symbol=s,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            except KeyError:
                continue


def sync_and_load(
    symbols: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar: str,
) -> pd.DataFrame:
    """Sync data from IBKR and load into DataFrame.

    This is the main entry point for production use.
    """
    import asyncio

    async def _sync():
        await sync_data(symbols, from_date=start.date())

    asyncio.run(_sync())
    return load_candles(symbols, start, end, bar)
