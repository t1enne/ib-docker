"""Bar aggregation — convert raw ticks into completed OHLCV bars.

Public API:
    create_bar_aggregator(symbol, interval) → BarAggregator
    AggregatedBar                    — frozen dataclass for a completed bar
    BarAggregator                    — Protocol for tick → bar aggregation
    AggregatorState                  — immutable aggregation state
    baragg_tick(state, tick)         — pure function: process one tick
    baragg_current(state)            — pure function: in-progress bar
"""

from src.live.baragg.types import (
    AggregatedBar,
    BarAggregator,
    AggregatorState,
    PartialBar,
)
from src.live.baragg.pure import baragg_tick, baragg_current, create_initial_state
from src.live.baragg.registry import create_bar_aggregator

__all__ = [
    "AggregatedBar",
    "BarAggregator",
    "AggregatorState",
    "PartialBar",
    "baragg_tick",
    "baragg_current",
    "create_initial_state",
    "create_bar_aggregator",
]
