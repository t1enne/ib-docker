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
    state: BacktestState,
    freq: str,
    tick: Tick,
) -> pd.DataFrame:
    """Get completed higher-timeframe candles for the tick's symbol.

    Returns only completed buckets (timestamp <= tick.timestamp) to prevent
    lookahead bias.

    Args:
        state: Current backtest state
        freq: Resample frequency (e.g., "4h", "1D")
        tick: Current tick (used to infer symbol and timestamp for filtering)

    Returns:
        DataFrame with MultiIndex (symbol, timestamp) for completed HTF candles
    """
    df = state.htf_data.get(freq)
    if df is None or df.empty:
        return pd.DataFrame(
            columns=pd.Index(["open", "high", "low", "close", "volume"]),
            index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
        )

    # Filter to completed buckets only (no lookahead)
    completed = df[df.index.get_level_values("timestamp") <= tick.timestamp]

    # Filter to tick's symbol
    try:
        return completed.xs(tick.symbol, level="symbol")
    except KeyError:
        return pd.DataFrame(
            columns=pd.Index(["open", "high", "low", "close", "volume"]),
            index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
        )


def get_resampled_candles(
    state: BacktestState,
    freq: str,
    symbol: Optional[str] = None,
    completed_only: bool = True,
) -> pd.DataFrame:
    """Legacy function for backward compatibility.

    Prefer using htf_candles(state, freq, tick) instead.
    """
    if not symbol:
        raise ValueError("symbol is required")

    if state.htf_data.get(freq) is None:
        return pd.DataFrame(
            columns=pd.Index(["open", "high", "low", "close", "volume"]),
            index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
        )

    df = state.htf_data[freq]

    if completed_only:
        df = df[df.index.get_level_values("timestamp") <= state.timestamp]

    try:
        return df.xs(symbol, level="symbol")
    except KeyError:
        return pd.DataFrame(
            columns=pd.Index(["open", "high", "low", "close", "volume"]),
            index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
        )
