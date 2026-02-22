"""Functional pairs trading strategy.

Pure functions that generate signals based on state.
"""

from typing import Tuple, Optional, List
from src.bt.state import (
    BacktestState,
    TradeSignal,
    ActionType,
    TradeExitReason,
    Tick,
    Position,
)


def _close(tick: Tick, position: Position, z: float) -> List[TradeSignal]:
    return [
        TradeSignal(
            action=ActionType.close,
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            price=tick.close,
            z_score=z,
            qty=abs(position.qty),
            reason=TradeExitReason.regression,
        )
    ]


def _open(
    tick: Tick, dir: ActionType, z: float, hedge: Optional[float] = None
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


def pairs_trading_on_tick(
    state: BacktestState,
    tick_group: dict,
    entry_z: float,
    exit_z: float,
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
        signal1 = _close(tick1, position1, z_score) if position1 else []
        signal2 = _close(tick2, position2, z_score) if position2 else []
        return signals + signal1 + signal2

    # Only check for entry if we don't have positions
    if have_positions:
        return signals
        # Check for entry
    if z_score < -entry_z:
        # Long sym1, short sym2
        buy_signals = _open(tick1, ActionType.long, z_score) + _open(
            tick2, ActionType.short, z_score, hedge
        )
        return signals + buy_signals

    if z_score > entry_z:
        # Short sym1, long sym2
        buy_signals = _open(tick1, ActionType.short, z_score) + _open(
            tick2, ActionType.long, z_score, hedge
        )
        return signals + buy_signals

    return signals
