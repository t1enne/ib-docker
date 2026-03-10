from dataclasses import asdict
from src.bt.types import StrategyConfig
from src.bt.state import Tick, BacktestState
from typing import List, Generator, TypedDict
from src.market_data.resample import resample_multiindex
import pandas as pd


def _df_to_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MultiIndex columns df to MultiIndex (symbol, timestamp) format."""
    if df.empty:
        return df

    # Check if columns are MultiIndex (symbol, field)
    if isinstance(df.columns, pd.MultiIndex):
        # Convert to (symbol, timestamp) format
        frames = []
        for symbol in df.columns.get_level_values(0).unique():
            sym_df = df.xs(symbol, axis=1, level=0).copy()
            # Reset the DatetimeIndex to get timestamp column
            sym_df = sym_df.reset_index()
            # First column is the timestamp (original index)
            ts_col = sym_df.columns[0]
            sym_df = sym_df.rename(columns={ts_col: "timestamp"})
            # Add symbol column
            sym_df["symbol"] = symbol
            sym_df = sym_df.set_index(["symbol", "timestamp"])
            frames.append(sym_df)

        if frames:
            return pd.concat(frames)

    return df

    # Check if columns are MultiIndex (symbol, field)
    if isinstance(df.columns, pd.MultiIndex):
        # Convert to (symbol, timestamp) format
        frames = []
        for symbol in df.columns.get_level_values(0).unique():
            sym_df = df.xs(symbol, axis=1, level=0).copy()
            # Reset the DatetimeIndex to get timestamp column
            sym_df = sym_df.reset_index()
            # First column is the timestamp (original index)
            ts_col = sym_df.columns[0]
            sym_df = sym_df.rename(columns={ts_col: "timestamp"})
            sym_df = sym_df.set_index(["symbol", "timestamp"])
            frames.append(sym_df)

        if frames:
            return pd.concat(frames)

    return df

    # Check if columns are MultiIndex (symbol, field)
    if isinstance(df.columns, pd.MultiIndex):
        # Convert to (symbol, timestamp) format
        frames = []
        for symbol in df.columns.get_level_values(0).unique():
            sym_df = df.xs(symbol, axis=1, level=0).copy()
            # Reset the DatetimeIndex to get 'timestamp' column
            sym_df = sym_df.reset_index()
            sym_df = sym_df.rename(columns={sym_df.columns[0]: "timestamp"})
            sym_df = sym_df.set_index(["symbol", "timestamp"])
            frames.append(sym_df)

        if frames:
            return pd.concat(frames)

    return df

    # Check if columns are MultiIndex (symbol, field)
    if isinstance(df.columns, pd.MultiIndex):
        # Convert to (symbol, timestamp) format
        frames = []
        for symbol in df.columns.get_level_values(0).unique():
            sym_cols = df.xs(symbol, axis=1, level=0)
            sym_cols["symbol"] = symbol
            sym_cols = sym_cols.set_index("symbol", append=True)
            sym_cols = sym_cols.swaplevel(0, 1)
            frames.append(sym_cols)

        if frames:
            return pd.concat(frames)

    return df


def ticks_generator(
    df: pd.DataFrame,
    config,
) -> Generator[Tick, None, None]:
    """Create a generator that yields ticks from a DataFrame.

    This is a pure function - given the same inputs, it always yields
    the same sequence of ticks. Yields base timeframe ticks and HTF ticks
    (if configured) interleaved by timestamp.

    Args:
        df: DataFrame with MultiIndex (symbol, timestamp) containing OHLCV
        config: StrategyConfig with symbols, bar, and htf settings, or list of symbols (legacy)

    Yields:
        Tick objects for each symbol at each timestamp
    """
    # Handle both StrategyConfig and legacy list of symbols
    if hasattr(config, "symbols"):
        symbols = config.symbols
        base_interval = getattr(config, "bar", "1h")
        htf_intervals = getattr(config, "htf", []) or []
    else:
        symbols = config
        base_interval = "1h"
        htf_intervals = []

    if df.empty or len(df.index) == 0:
        return

    # Convert to MultiIndex format for resampling
    df_mi = _df_to_multiindex(df)

    # Pre-compute HTF DataFrames
    htf_dfs = {}
    for freq in htf_intervals:
        htf_dfs[freq] = resample_multiindex(df_mi, freq, completed_only=False)

    # Get all unique timestamps across base and HTF
    all_timestamps = set()
    for s in symbols:
        all_timestamps.update(df.xs(s, axis=1).index)
    for freq, htf_df in htf_dfs.items():
        if not htf_df.empty:
            for s in symbols:
                try:
                    all_timestamps.update(htf_df.xs(s, level="symbol").index)
                except KeyError:
                    pass

    sorted_timestamps = sorted(all_timestamps)

    for ts in sorted_timestamps:
        # Yield base interval ticks first
        for s in symbols:
            try:
                row = df.xs(s, axis=1).loc[ts]
                timestamp = pd.Timestamp(ts)
                if pd.isna(timestamp):
                    continue
                yield Tick(
                    timestamp=timestamp,
                    symbol=s,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    interval=base_interval,
                )
            except KeyError:
                continue

        # Yield HTF ticks at their boundaries
        for freq in htf_intervals:
            htf_df = htf_dfs.get(freq)
            if htf_df is None or htf_df.empty:
                continue
            for s in symbols:
                try:
                    row = htf_df.xs(s, level="symbol").loc[ts]
                    timestamp = pd.Timestamp(ts)
                    if pd.isna(timestamp):
                        continue
                    # Check if this is a valid HTF candle (not partial)
                    # by checking if the timestamp aligns with the frequency
                    yield Tick(
                        timestamp=timestamp,
                        symbol=s,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        interval=freq,
                    )
                except KeyError:
                    continue


def merge_bt_state(a: BacktestState, b: dict):
    return BacktestState(**{**asdict(a), **b})
