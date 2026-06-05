"""Functional pairs trading strategy.

Pure functions that generate signals based on state.
"""

from src.bt.algos.utils import close, open

from typing import Optional, List, TYPE_CHECKING, cast
from src.bt.state import (
    BacktestState,
    TradeSignal,
    ActionType,
    TradeExitReason,
    Tick,
    Position,
)
from src.bt.types import PlotConfig
import pandas as pd
import numpy as np
from src.bt.zscore import calculate_rolling_z
from src.utils import calculate_zscore_spread

if TYPE_CHECKING:
    from src.bt.types import StrategyConfig


def on_tick(
    state: BacktestState, tick: Tick, strategy_params: dict
) -> List[TradeSignal]:
    """Generate trading signals based on z-score.

    Args:
        state: Current backtest state
        tick: Current tick event
        entry_z: Z-score threshold for entry
        exit_z: Z-score threshold for exit

    Returns:
        Tuple of TradeSignal objects
    """

    signals = []
    entry_z = strategy_params.get("entry_z", 2.5)
    exit_z = strategy_params.get("exit_z", 0.5)
    window = strategy_params.get("rolling_window_size")
    symbols = strategy_params.get("symbols")

    if not symbols:
        if not state.candles:
            return signals
        symbols = list(state.candles.keys())

    if len(symbols) != 2:
        return signals

    sym1, sym2 = symbols
    if sym1 not in state.candles or sym2 not in state.candles:
        return signals

    try:
        candles1 = state.candles[sym1]
        candles2 = state.candles[sym2]
    except KeyError:
        return signals

    if candles1.empty or candles2.empty:
        return signals

    closes1, closes2 = candles1["close"], candles2["close"]
    aligned1, aligned2 = closes1.align(closes2, join="inner")
    aligned1 = cast(pd.Series, aligned1)
    aligned2 = cast(pd.Series, aligned2)
    aligned = pd.concat([aligned1, aligned2], axis=1).dropna()
    if aligned.empty:
        return signals

    aligned1 = aligned.iloc[:, 0]
    aligned2 = aligned.iloc[:, 1]

    if window is None:
        window = min(len(aligned1), len(aligned2))

    if len(aligned1) < window or len(aligned2) < window:
        return signals

    z_score, _alpha, hedge = calculate_rolling_z(aligned1, aligned2, window)
    if np.isnan(z_score):
        return signals

    last_ts = cast(pd.Timestamp, aligned1.index[-1])
    row1 = candles1.loc[last_ts]
    row2 = candles2.loc[last_ts]
    tick1 = Tick(
        timestamp=last_ts,
        symbol=sym1,
        open=float(row1["open"]),
        high=float(row1["high"]),
        low=float(row1["low"]),
        close=float(row1["close"]),
        volume=float(row1["volume"]),
    )
    tick2 = Tick(
        timestamp=last_ts,
        symbol=sym2,
        open=float(row2["open"]),
        high=float(row2["high"]),
        low=float(row2["low"]),
        close=float(row2["close"]),
        volume=float(row2["volume"]),
    )

    # Check for existing positions
    position1 = state.portfolio.positions.get(sym1)
    position2 = state.portfolio.positions.get(sym2)

    have_positions = True if position1 or position2 else False
    is_regressed = abs(z_score) < exit_z

    # If we have positions, check for exit
    if have_positions and is_regressed:
        # Exit signal
        signal1 = (
            close(tick1, position1, TradeExitReason.regression, z_score)
            if position1
            else []
        )
        signal2 = (
            close(tick2, position2, TradeExitReason.regression, z_score)
            if position2
            else []
        )
        return signals + signal1 + signal2

    # Only check for entry if we don't have positions
    if have_positions:
        return signals
        # Check for entry
    if z_score < -entry_z:
        # Long sym1, short sym2
        buy_signals = open(tick1, ActionType.long, f"z: {z_score}") + open(
            tick2, ActionType.short, f"z: {z_score}", hedge
        )
        return signals + buy_signals

    if z_score > entry_z:
        # Short sym1, long sym2
        buy_signals = open(tick1, ActionType.short, f"z: {z_score}") + open(
            tick2, ActionType.long, f"z: {z_score}", hedge
        )
        return signals + buy_signals

    return signals


def plot(state: BacktestState, config: "StrategyConfig") -> PlotConfig:
    """Return z-score as a separate subplot."""
    if not state.candles:
        return PlotConfig()
    symbols = list(config.symbols)
    if len(symbols) != 2:
        return PlotConfig()
    sym1, sym2 = symbols

    try:
        candles1 = state.candles[sym1]
        candles2 = state.candles[sym2]
    except KeyError:
        return PlotConfig()

    closes1, closes2 = candles1["close"], candles2["close"]
    aligned1, aligned2 = closes1.align(closes2, join="inner")
    aligned1 = cast(pd.Series, aligned1)
    aligned2 = cast(pd.Series, aligned2)
    aligned = pd.concat([aligned1, aligned2], axis=1).dropna()
    if aligned.empty:
        return PlotConfig()

    aligned1 = aligned.iloc[:, 0]
    aligned2 = aligned.iloc[:, 1]

    window = config.rolling_window_size
    z_series = calculate_zscore_spread(aligned1, aligned2, window)

    return PlotConfig(subplots=[("Z-Score", z_series)])
