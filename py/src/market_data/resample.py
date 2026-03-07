"""Pure functional OHLCV resampling utilities.

Provides bt-agnostic resampling with lookahead protection.
"""

from typing import Optional, cast
import pandas as pd


OHLCV_COLS: list[str] = ["open", "high", "low", "close", "volume"]


def resample_ohlcv(
    df: pd.DataFrame,
    freq: str,
    *,
    completed_only: bool = True,
    current_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe.

    Args:
        df: DataFrame with timestamp index and OHLCV columns
        freq: Resample frequency (e.g., "1h", "4h", "1D")
        completed_only: If True, exclude the current incomplete bucket (no lookahead)
        current_ts: Current timestamp for completed_only filtering

    Returns:
        Resampled DataFrame with OHLCV columns
    """
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLS)  # type: ignore[arg-type]

    resampled = df.resample(freq).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    resampled = resampled.dropna()

    if completed_only and current_ts is not None:
        bucket_end = pd.Timestamp(current_ts.floor(freq))
        resampled = resampled[resampled.index < bucket_end]

    return resampled


def resample_multiindex(
    df: pd.DataFrame,
    freq: str,
    *,
    completed_only: bool = True,
    current_ts: Optional[pd.Timestamp] = None,
    symbol_col: str = "symbol",
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Resample MultiIndex (symbol, timestamp) OHLCV data.

    Args:
        df: MultiIndex DataFrame with (symbol, timestamp) index and OHLCV columns
        freq: Resample frequency (e.g., "1h", "4h", "1D")
        completed_only: If True, exclude the current incomplete bucket (no lookahead)
        current_ts: Current timestamp for completed_only filtering
        symbol_col: Column name for symbol (if df is not MultiIndex)
        ts_col: Column name for timestamp (if df is not MultiIndex)

    Returns:
        Resampled DataFrame with MultiIndex (symbol, timestamp)
    """
    if df.empty:
        return pd.DataFrame(  # noqa: ARG-TYPE
            columns=OHLCV_COLS,  # type: ignore[arg-type]
            index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
        )

    if isinstance(df.index, pd.MultiIndex):
        symbols = df.index.get_level_values("symbol").unique()
        resampled_frames = []

        for symbol in symbols:
            sym_df = df.xs(symbol, level="symbol")
            sym_resampled = resample_ohlcv(
                sym_df, freq, completed_only=False, current_ts=None
            )
            if not sym_resampled.empty:
                sym_resampled = sym_resampled.reset_index()
                sym_resampled["symbol"] = symbol
                sym_resampled = sym_resampled.set_index(["symbol", "timestamp"])
                resampled_frames.append(sym_resampled)

        if not resampled_frames:
            return pd.DataFrame(  # noqa: ARG-TYPE
                columns=OHLCV_COLS,  # type: ignore[arg-type]
                index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
            )

        result = pd.concat(resampled_frames)

        if completed_only and current_ts is not None:
            bucket_end = pd.Timestamp(current_ts.floor(freq))
            result = result[result.index.get_level_values("timestamp") < bucket_end]

        return cast(pd.DataFrame, result)
    else:
        return resample_ohlcv(
            df, freq, completed_only=completed_only, current_ts=current_ts
        )
