"""Functional pairs trading strategy.

Pure functions that generate signals based on state.
"""

from src.bt.algos.utils import close, open

from typing import Tuple, Optional, List, TYPE_CHECKING
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

if TYPE_CHECKING:
    from src.bt.types import StrategyConfig


def on_tick(
    state: BacktestState, tick_group: dict, strategy_params: dict
) -> List[TradeSignal]:
    """Generate trading signals based on z-score.

    Args:
        state: Current backtest state
        tick_group: Dictionary of ticks by symbol for current timestamp
        entry_z: Z-score threshold for entry
        exit_z: Z-score threshold for exit

    Returns:
        Tuple of TradeSignal objects
    """

    signals = []
    z_score = state.model_state.z_score
    hedge = state.model_state.hedge_beta

    # Check if z_score is available (pair trading strategies only)
    if z_score is None:
        return signals

    # Check if we have enough data
    if len(state.model_state.price_buffers) < 2:
        return signals

    # Get symbols
    symbols = list(tick_group.keys())
    if len(symbols) != 2:
        return signals

    sym1, sym2 = symbols

    # Check for existing positions
    position1 = state.portfolio.positions.get(sym1)
    position2 = state.portfolio.positions.get(sym2)
    tick1 = tick_group[sym1]
    tick2 = tick_group[sym2]

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
        buy_signals = open(tick1, ActionType.long, z_score) + open(
            tick2, ActionType.short, z_score, hedge
        )
        return signals + buy_signals

    if z_score > entry_z:
        # Short sym1, long sym2
        buy_signals = open(tick1, ActionType.short, z_score) + open(
            tick2, ActionType.long, z_score, hedge
        )
        return signals + buy_signals

    return signals


def plot(state: BacktestState, config: "StrategyConfig") -> PlotConfig:
    """Return z-score as a separate subplot."""
    z_score = state.model_state.z_score
    if z_score is None:
        return PlotConfig()

    timestamps = [buf.get("timestamp") for buf in state.model_state.price_buffers]
    z_series = pd.Series([z_score], index=[timestamps[-1]] if timestamps else None)

    return PlotConfig(subplots=[("Z-Score", z_series)])
