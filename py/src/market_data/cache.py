"""Generic resample cache for higher-timeframe data.

Bt-agnostic cache that tracks completed buckets to avoid recomputation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, cast, Tuple
import pandas as pd

from src.market_data.resample import resample_multiindex, resample_ohlcv


OHLCV_COLS: list[str] = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class ResampleCache:
    """Immutable cache for resampled higher-timeframe data.

    Attributes:
        cache: Dict mapping frequency -> resampled DataFrame
        anchor: Dict mapping frequency -> last completed bucket timestamp
    """

    cache: Dict[str, pd.DataFrame] = field(default_factory=dict)
    anchor: Dict[str, pd.Timestamp] = field(default_factory=dict)


def update_resample_cache(
    cache: ResampleCache,
    candles: pd.DataFrame,
    frequencies: List[str],
    current_ts: pd.Timestamp,
) -> ResampleCache:
    """Update resample cache when new data arrives.

    Only recomputes for frequencies where the bucket has changed (anchor changed).
    This ensures no lookahead - cache always contains completed buckets only.

    Args:
        cache: Current cache state
        candles: Raw candle DataFrame (MultiIndex or timestamp index)
        frequencies: List of frequencies to maintain (e.g., ["1h", "4h"])
        current_ts: Current timestamp (used to detect bucket transitions)

    Returns:
        Updated ResampleCache with new anchors
    """
    new_cache = dict(cache.cache)
    new_anchor = dict(cache.anchor)

    for freq in frequencies:
        if current_ts is pd.NaT:
            continue
        current_bucket = pd.Timestamp(current_ts.floor(freq))
        if current_bucket is pd.NaT:
            continue
        prev_anchor = new_anchor.get(freq)

        if prev_anchor is None or current_bucket != prev_anchor:
            if isinstance(candles.index, pd.MultiIndex):
                resampled = resample_multiindex(
                    candles,
                    freq,
                    completed_only=False,
                    current_ts=None,
                )
            else:
                resampled = resample_ohlcv(
                    candles,
                    freq,
                    completed_only=False,
                    current_ts=None,
                )

            new_cache[freq] = resampled
            new_anchor[freq] = cast(pd.Timestamp, current_bucket)

    return ResampleCache(cache=new_cache, anchor=new_anchor)


def _append_bucket(
    cache_df: pd.DataFrame,
    symbol: str,
    bucket_ts: pd.Timestamp,
    ohlcv: Dict[str, float],
) -> pd.DataFrame:
    new_row = pd.DataFrame(
        {
            "open": [ohlcv["open"]],
            "high": [ohlcv["high"]],
            "low": [ohlcv["low"]],
            "close": [ohlcv["close"]],
            "volume": [ohlcv["volume"]],
        },
        index=pd.MultiIndex.from_tuples(
            [(symbol, bucket_ts)], names=["symbol", "timestamp"]
        ),
    )

    if cache_df.empty:
        return new_row

    updated = cache_df.copy()
    updated.loc[(symbol, bucket_ts), ["open", "high", "low", "close", "volume"]] = [
        ohlcv["open"],
        ohlcv["high"],
        ohlcv["low"],
        ohlcv["close"],
        ohlcv["volume"],
    ]
    return updated


def update_resample_cache_incremental(
    cache: ResampleCache,
    partial: Dict[str, Dict[str, dict]],
    *,
    symbol: str,
    timestamp: pd.Timestamp,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    frequencies: List[str],
) -> Tuple[ResampleCache, Dict[str, Dict[str, dict]]]:
    """Update resample cache using a single tick.

    Maintains a partial bucket per (freq, symbol), finalizing buckets when
    the timestamp crosses into a new bucket.
    """
    new_cache = dict(cache.cache)
    new_anchor = dict(cache.anchor)
    new_partial: Dict[str, Dict[str, dict]] = {
        freq: dict(partial.get(freq, {})) for freq in frequencies
    }

    for freq in frequencies:
        bucket_ts = pd.Timestamp(timestamp.floor(freq))
        sym_partial = new_partial.setdefault(freq, {})
        current = sym_partial.get(symbol)

        if current is None or current["timestamp"] != bucket_ts:
            if current is not None:
                cache_df = new_cache.get(
                    freq,
                    pd.DataFrame(
                        columns=pd.Index(OHLCV_COLS),
                        index=pd.MultiIndex.from_tuples(
                            [], names=["symbol", "timestamp"]
                        ),
                    ),
                )
                new_cache[freq] = _append_bucket(
                    cache_df,
                    symbol,
                    current["timestamp"],
                    {
                        "open": current["open"],
                        "high": current["high"],
                        "low": current["low"],
                        "close": current["close"],
                        "volume": current["volume"],
                    },
                )
                new_anchor[freq] = current["timestamp"]

            sym_partial[symbol] = {
                "timestamp": bucket_ts,
                "open": open,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        else:
            current["high"] = max(current["high"], high)
            current["low"] = min(current["low"], low)
            current["close"] = close
            current["volume"] += volume

    return ResampleCache(cache=new_cache, anchor=new_anchor), new_partial


def get_from_cache(
    cache: ResampleCache,
    freq: str,
    *,
    completed_only: bool = True,
    current_ts: Optional[pd.Timestamp] = None,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Get resampled data from cache with optional filtering.

    Args:
        cache: ResampleCache to read from
        freq: Frequency to retrieve
        completed_only: If True, exclude incomplete bucket (no lookahead)
        current_ts: Current timestamp for completed_only filtering
        symbol: Optional symbol to filter to single-symbol DataFrame

    Returns:
        Resampled DataFrame
    """
    if freq not in cache.cache:
        return pd.DataFrame(columns=OHLCV_COLS)  # type: ignore[arg-type]

    cached = cache.cache[freq]

    if cached.empty:
        return cached

    if completed_only and current_ts is not None:
        bucket_end = pd.Timestamp(current_ts.floor(freq))
        if isinstance(cached.index, pd.MultiIndex):
            cached = cached[cached.index.get_level_values("timestamp") < bucket_end]
        else:
            cached = cached[cached.index < bucket_end]

    if symbol:
        if isinstance(cached.index, pd.MultiIndex):
            if symbol not in cached.index.get_level_values("symbol"):
                return pd.DataFrame(columns=OHLCV_COLS)  # type: ignore[arg-type]
            return cast(pd.DataFrame, cached.xs(symbol, level="symbol"))
        else:
            return cast(pd.DataFrame, cached)

    return cast(pd.DataFrame, cached)
