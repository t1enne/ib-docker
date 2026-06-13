"""Immutable types for bar aggregation.

All types are frozen dataclasses for determinism, testability, and replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.bt.state.types import Candle


@dataclass(frozen=True)
class PartialBar:
    """In-progress bar — OHLCV accumulator for a single time bucket.

    Updated on each tick within the same bucket. Converted to AggregatedBar
    when the bucket rolls over.
    """

    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int


@dataclass(frozen=True)
class AggregatedBar:
    """A completed OHLCV bar aggregated from raw ticks.

    Bar boundaries are aligned to UTC clock intervals. The timestamp
    is the right edge of the interval (bar close time).
    """

    symbol: str
    timestamp: pd.Timestamp  # bar close (right edge of interval)
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str  # e.g. "5min", "1h", "1d"
    tick_count: int = 0

    def to_tick(self) -> Candle:
        """Convert to a Tick for compatibility with the existing StrategyFn pipeline.

        The Tick.interval field carries the aggregation interval so strategies
        can distinguish bar resolutions.
        """
        from src.bt.state.types import Candle

        return Candle(
            timestamp=self.timestamp,
            symbol=self.symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            interval=self.interval,
        )


@dataclass(frozen=True)
class AggregatorState:
    """Immutable state of a single-symbol bar aggregator.

    Holds the in-progress partial bar and emits completed bars when
    a tick falls into a new interval bucket.

    The `completed` field contains only bars emitted by the most recent
    update — callers read and discard them; no accumulation.
    """

    symbol: str
    interval: str
    partial: PartialBar | None = None
    bucket: pd.Timestamp | None = None
    completed: tuple[AggregatedBar, ...] = ()


class BarAggregator(Protocol):
    """Protocol for tick → bar aggregation.

    Implementations accept raw ticks and emit completed bars.
    Single-symbol: one aggregator instance per symbol.

    The strategy pipeline receives completed bars as Tick objects
    (via AggregatedBar.to_tick()), making the aggregation transparent.
    """

    def update(self, tick: Candle) -> tuple[AggregatedBar, ...]:
        """Process one tick. Returns completed bars emitted by this update.

        Returns an empty tuple if no bar was completed. Bars are not
        accumulated across calls — the caller is responsible for consuming
        and forwarding each batch.
        """
        ...

    @property
    def current_bar(self) -> AggregatedBar | None:
        """The in-progress bar (None before the first tick)."""
        ...

    def reset(self) -> None:
        """Clear all state (e.g., on market session reset)."""
        ...
