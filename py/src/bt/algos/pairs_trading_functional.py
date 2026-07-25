"""Functional pairs trading strategy.

Pure functions that generate signals based on state.
Uses model_state.price_buffers to cache aligned closes for performance.
"""

from src.bt.algos.utils import close, open

from typing import Optional, List, TYPE_CHECKING, cast
from src.bt.state import (
    BacktestState,
    TradeSignal,
    ActionType,
    TradeExitReason,
    Candle,
    Position,
)
from src.bt.types import PlotConfig
import pandas as pd
import numpy as np
from src.bt.zscore import _ols_log_params
from src.utils import calculate_zscore_spread

if TYPE_CHECKING:
    from src.bt.types import StrategyConfig


def _compute_zscore(
    arr1: np.ndarray, arr2: np.ndarray, window: int
) -> tuple[float, float, float]:
    """Compute z-score from aligned price arrays using direct numpy OLS."""
    if len(arr1) < window or len(arr2) < window:
        return (float("nan"), 0.0, 1.0)

    w1 = arr1[-window:]
    w2 = arr2[-window:]

    alpha, beta = _ols_log_params(w2, w1)

    log_w1 = np.log(w1)
    log_w2 = np.log(w2)
    spread = log_w1 - (alpha + beta * log_w2)

    mean = spread.mean()
    std = spread.std(ddof=1)

    current_spread = log_w1[-1] - (alpha + beta * log_w2[-1])
    z = (current_spread - mean) / std if std != 0 else 0.0

    return (round(float(z), 3), alpha, beta)


def on_candle(
    state: BacktestState, candle: Candle, strategy_params: dict
) -> List[TradeSignal]:
    """Generate trading signals based on z-score.

    Uses model_state.price_buffers as an aligned-closes cache to avoid
    DataFrame align/concat/dropna on every candle.
    Each candle for the last symbol appends {sym: close} pairs to the buffer.

    Args:
        state: Current backtest state
        candle: Current candle (OHLCV bar)
        strategy_params: entry_z, exit_z, rolling_window_size, symbols

    Returns:
        List of TradeSignal objects
    """
    signals: list[TradeSignal] = []
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

    # Fast path: use pre-aligned close prices from model_state.price_buffers.
    # The engine appends a {sym1: close1, sym2: close2} dict on each tick
    # of the last symbol, so buffers contains only fully-aligned pairs.
    buffers = state.model_state.price_buffers
    if len(buffers) < (window or 2):
        return signals

    # Extract aligned closes as numpy arrays
    n = len(buffers)
    arr1 = np.empty(n, dtype=float)
    arr2 = np.empty(n, dtype=float)
    for i, pair in enumerate(buffers):
        arr1[i] = float(pair[sym1])
        arr2[i] = float(pair[sym2])

    z_window = window if window is not None else 50
    z_score, _alpha, hedge = _compute_zscore(arr1, arr2, z_window)
    if np.isnan(z_score):
        return signals

    # Build Candle objects from state.candles DataFrames.
    # DataFrames are refreshed every _CANDLE_BATCH_SIZE rows, so .iloc[-1] is up-to-date.
    df1 = state.candles.get(sym1)
    df2 = state.candles.get(sym2)
    if df1 is None or df2 is None or len(df1) == 0 or len(df2) == 0:
        return signals

    row1 = df1.iloc[-1]
    row2 = df2.iloc[-1]

    tick1 = Candle(
        timestamp=cast(pd.Timestamp, row1.name),
        symbol=sym1,
        open=cast(float, row1["open"]),
        high=cast(float, row1["high"]),
        low=cast(float, row1["low"]),
        close=cast(float, row1["close"]),
        volume=cast(float, row1["volume"]),
    )
    tick2 = Candle(
        timestamp=cast(pd.Timestamp, row2.name),
        symbol=sym2,
        open=cast(float, row2["open"]),
        high=cast(float, row2["high"]),
        low=cast(float, row2["low"]),
        close=cast(float, row2["close"]),
        volume=cast(float, row2["volume"]),
    )

    # Check for existing positions
    position1 = state.portfolio.positions.get(sym1)
    position2 = state.portfolio.positions.get(sym2)

    have_positions = position1 is not None or position2 is not None
    is_regressed = abs(z_score) < exit_z

    # If we have positions, check for exit
    if have_positions and is_regressed:
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
