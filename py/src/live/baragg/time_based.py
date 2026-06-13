"""TimeBarAggregator — mutable wrapper around pure bar aggregation functions.

This is the concrete implementation used in the live engine loop.
It holds a mutable reference to an AggregatorState and delegates to
pure functions. The mutable wrapper avoids the allocate-on-every-tick
overhead of threading immutable state through the hot path, while the
core logic remains pure and independently testable.
"""

from __future__ import annotations

from src.live.baragg.types import AggregatedBar, AggregatorState
from src.live.baragg.pure import baragg_tick, baragg_current, create_initial_state
from src.bt.state.types import Tick


class TimeBarAggregator:
    """Fixed-interval bar aggregation from raw ticks.

    Aggregates ticks into bars aligned to UTC clock boundaries.
    A bar is completed and emitted when a tick falls into the next interval.

    Configurable interval: "1min", "5min", "15min", "1h", "4h", "1d"

    Single-symbol: each instance handles one symbol.
    """

    def __init__(self, interval: str, symbol: str) -> None:
        """Create a time-based bar aggregator.

        Args:
            interval: Bar interval string (e.g. "5min", "1h", "1d")
            symbol: Ticker symbol to aggregate (e.g. "AAPL")
        """
        self.interval = interval
        self.symbol = symbol
        self._state: AggregatorState = create_initial_state(symbol, interval)

    def update(self, tick: Tick) -> tuple[AggregatedBar, ...]:
        """Process one tick. Returns completed bars emitted by this update.

        Returns an empty tuple if no bar was completed.
        """
        self._state = baragg_tick(self._state, tick)
        return self._state.completed

    @property
    def current_bar(self) -> AggregatedBar | None:
        """The in-progress bar (None before the first tick)."""
        return baragg_current(self._state)

    def reset(self) -> None:
        """Clear all state (e.g., on market session reset)."""
        self._state = create_initial_state(self.symbol, self.interval)
