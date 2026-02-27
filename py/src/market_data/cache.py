"""Generic resample cache for higher-timeframe data.

Bt-agnostic cache that tracks completed buckets to avoid recomputation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, cast
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
        current_bucket = pd.Timestamp(current_ts.floor(freq))
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
            new_anchor[freq] = current_bucket

    return ResampleCache(cache=new_cache, anchor=new_anchor)


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
