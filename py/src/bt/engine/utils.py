from dataclasses import asdict
from src.bt.types import StrategyConfig
from src.bt.state import Tick, BacktestState
from typing import List, Generator, TypedDict, cast
from src.market_data.resample import resample_multiindex
import pandas as pd
import numpy as np


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


def ticks_generator(
    df: pd.DataFrame,
    config,
) -> Generator[Tick, None, None]:
    """Create a generator that yields ticks from a DataFrame.

    This is a pure function - given the same inputs, it always yields
    the same sequence of ticks. Yields base timeframe ticks and HTF ticks
    (if configured) interleaved by timestamp.

    Uses pre-extracted numpy arrays to avoid per-tick DataFrame lookups.

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

    # Pre-extract base OHLCV into numpy arrays for fast iteration.
    # df has MultiIndex columns: (symbol, field). Extract each (sym, field)
    # into a plain numpy array indexed by row position.
    fields = ["open", "high", "low", "close", "volume"]
    base_arrays: dict[tuple[str, str], np.ndarray] = {}
    for s in symbols:
        for f in fields:
            col = (s, f)
            if col in df.columns:
                base_arrays[(s, f)] = df[col].to_numpy(dtype=float)

    timestamps = df.index

    # Pre-compute HTF DataFrames
    htf_dfs: dict[str, pd.DataFrame] = {}
    for freq in htf_intervals:
        htf_dfs[freq] = resample_multiindex(df_mi, freq, completed_only=False)

    # Get all unique timestamps across base and HTF
    all_timestamps: set = set()
    for s in symbols:
        all_timestamps.update(timestamps)
    for freq, htf_df in htf_dfs.items():
        if not htf_df.empty:
            for s in symbols:
                try:
                    all_timestamps.update(htf_df.xs(s, level="symbol").index)
                except KeyError:
                    pass

    sorted_timestamps = sorted(all_timestamps)
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}  # fast lookup

    # Pre-compute HTF arrays for fast iteration
    htf_arrays: dict[tuple[str, str, str], np.ndarray] = {}  # (freq, sym, field)
    htf_timestamps_map: dict[str, dict[str, np.ndarray]] = {}  # freq -> sym -> timestamps
    for freq, htf_df in htf_dfs.items():
        if htf_df.empty:
            continue
        htf_timestamps_map[freq] = {}
        for s in symbols:
            try:
                sym_htf = htf_df.xs(s, level="symbol")
                htf_timestamps_map[freq][s] = sym_htf.index.to_numpy()
                for f in fields:
                    htf_arrays[(freq, s, f)] = sym_htf[f].to_numpy(dtype=float)
            except KeyError:
                continue

    for ts in sorted_timestamps:
        pdt = cast(pd.Timestamp, pd.Timestamp(ts))
        if pd.isna(pdt):
            continue

        idx = ts_to_idx.get(ts)
        if idx is not None:
            for s in symbols:
                yield Tick(
                    timestamp=pdt,
                    symbol=s,
                    open=float(base_arrays[(s, "open")][idx]),
                    high=float(base_arrays[(s, "high")][idx]),
                    low=float(base_arrays[(s, "low")][idx]),
                    close=float(base_arrays[(s, "close")][idx]),
                    volume=float(base_arrays[(s, "volume")][idx]),
                    interval=base_interval,
                )

        # Yield HTF ticks at their boundaries
        for freq in htf_intervals:
            ts_map = htf_timestamps_map.get(freq, {})
            sym_ts = ts_map.get(s, None)
            if sym_ts is None:
                continue
            # Find if ts matches any HTF timestamp for this symbol
            htf_idx = np.searchsorted(sym_ts, ts)
            if htf_idx < len(sym_ts) and sym_ts[htf_idx] == ts:
                yield Tick(
                    timestamp=pdt,
                    symbol=s,
                    open=float(htf_arrays[(freq, s, "open")][htf_idx]),
                    high=float(htf_arrays[(freq, s, "high")][htf_idx]),
                    low=float(htf_arrays[(freq, s, "low")][htf_idx]),
                    close=float(htf_arrays[(freq, s, "close")][htf_idx]),
                    volume=float(htf_arrays[(freq, s, "volume")][htf_idx]),
                    interval=freq,
                )


def merge_bt_state(a: BacktestState, b: dict):
    return BacktestState(
        portfolio=b.get("portfolio", a.portfolio),
        model_state=b.get("model_state", a.model_state),
        timestamp=b.get("timestamp", a.timestamp),
        pending_signals=b.get("pending_signals", a.pending_signals),
        risk_events=b.get("risk_events", a.risk_events),
        candles=b.get("candles", a.candles),
        htf_data=b.get("htf_data", a.htf_data),
    )
