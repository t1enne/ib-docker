from typing import List, Optional, TYPE_CHECKING, Any
import pandas as pd

if TYPE_CHECKING:
    from src.bt.state import BacktestState

from src.bt.state import (
    BacktestState,
    TradeSignal,
    ActionType,
    Tick,
    Position,
)
from src.market_data.cache import get_from_cache
from src.market_data.resample import resample_multiindex


def close(
    tick: Tick,
    position: Position,
    reason: Any,
    z: Optional[float] = None,
) -> List[TradeSignal]:
    return [
        TradeSignal(
            action=ActionType.close,
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            price=tick.close,
            z_score=z,
            qty=abs(position.qty),
            reason=reason,
        )
    ]


def open(
    tick: Tick,
    dir: ActionType,
    reason: Optional[str] = "",
    hedge: Optional[float] = None,
) -> List[TradeSignal]:
    return [
        TradeSignal(
            action=dir,
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            price=tick.close,
            reason=reason,
            hedge_beta=hedge,
        ),
    ]


def htf_candles(
    state: "BacktestState",
    freq: str,
    tick: Tick,
    completed_only: bool = True,
) -> pd.DataFrame:
    """Get higher-timeframe candles for the tick's symbol.

    Convenience wrapper around get_resampled_candles with a simpler signature.
    Returns only completed buckets by default (no lookahead).

    Args:
        state: Current backtest state
        freq: Resample frequency (e.g., "4h", "1D")
        tick: Current tick (used to infer symbol)
        completed_only: If True, exclude incomplete bucket (default True)

    Returns:
        DataFrame with timestamp index and OHLCV columns for the tick's symbol
    """
    return get_resampled_candles(
        state, freq, symbol=tick.symbol, completed_only=completed_only
    )


def get_resampled_candles(
    state: "BacktestState",
    freq: str,
    symbol: Optional[str] = None,
    completed_only: bool = True,
) -> pd.DataFrame:
    """Get resampled candles from cache.

    Args:
        state: Current backtest state
        freq: Resample frequency (e.g., "1h", "4h", "1D")
        symbol: Optional symbol to filter to single-symbol DataFrame
        completed_only: If True, exclude incomplete bucket (no lookahead)

    Returns:
        Resampled DataFrame with OHLCV columns
    """
    cache = state.model_state.resample_cache
    anchor = state.model_state.resample_anchor
    partial = getattr(state.model_state, "resample_partial", {})
    candles = state.candles
    current_ts = state.timestamp

    if not cache or freq not in cache:
        if freq in partial:
            if completed_only:
                return pd.DataFrame(
                    columns=pd.Index(["open", "high", "low", "close", "volume"])
                )
            cached = pd.DataFrame(
                columns=pd.Index(["open", "high", "low", "close", "volume"])
            )
        else:
            return resample_multiindex(
                candles,
                freq,
                completed_only=completed_only,
                current_ts=current_ts,
            )
    else:
        from src.market_data.cache import ResampleCache

        cache_obj = ResampleCache(cache=cache, anchor=anchor)

        cached = get_from_cache(
            cache_obj,
            freq,
            completed_only=completed_only,
            current_ts=current_ts,
            symbol=symbol,
        )

    if completed_only:
        return cached

    freq_partial = partial.get(freq, {})
    if not freq_partial:
        return cached

    partial_rows = []
    if symbol:
        bucket = freq_partial.get(symbol)
        if bucket:
            partial_rows.append((symbol, bucket))
    else:
        partial_rows = list(freq_partial.items())

    if not partial_rows:
        return cached

    if symbol:
        partial_df = pd.DataFrame(
            [
                {
                    "open": b["open"],
                    "high": b["high"],
                    "low": b["low"],
                    "close": b["close"],
                    "volume": b["volume"],
                    "timestamp": b["timestamp"],
                }
                for _s, b in partial_rows
            ]
        ).set_index("timestamp")
    else:
        partial_df = pd.DataFrame(
            [
                {
                    "open": b["open"],
                    "high": b["high"],
                    "low": b["low"],
                    "close": b["close"],
                    "volume": b["volume"],
                    "symbol": s,
                    "timestamp": b["timestamp"],
                }
                for s, b in partial_rows
            ]
        ).set_index(["symbol", "timestamp"])

    if cached.empty:
        return partial_df

    combined = pd.concat([cached, partial_df])
    return combined
