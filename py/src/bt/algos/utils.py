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
    tick: Tick, dir: ActionType, z: Optional[float], hedge: Optional[float] = None
) -> List[TradeSignal]:
    return [
        TradeSignal(
            action=dir,
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            price=tick.close,
            z_score=z,
            hedge_beta=hedge,
        ),
    ]


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
    candles = state.candles
    current_ts = state.timestamp

    if not cache or freq not in cache:
        return resample_multiindex(
            candles,
            freq,
            completed_only=completed_only,
            current_ts=current_ts,
        )

    from src.market_data.cache import ResampleCache

    cache_obj = ResampleCache(cache=cache, anchor=anchor)

    return get_from_cache(
        cache_obj,
        freq,
        completed_only=completed_only,
        current_ts=current_ts,
        symbol=symbol,
    )
