"""Registry for creating bar aggregators.

Factory that selects the appropriate aggregator implementation for a
given symbol and interval. Currently only supports time-based aggregation;
extensible for tick-count and volume-based aggregators.
"""

from __future__ import annotations

from src.live.baragg.types import BarAggregator


def create_bar_aggregator(symbol: str, interval: str) -> BarAggregator:
    """Create a bar aggregator for a single symbol.

    Args:
        symbol: Ticker symbol (e.g. "AAPL")
        interval: Bar interval (e.g. "5min", "1h", "1d")

    Returns:
        A BarAggregator implementation for the given symbol and interval.

    Raises:
        ValueError: If the interval is not supported.
    """
    from src.live.baragg.time_based import TimeBarAggregator

    return TimeBarAggregator(interval=interval, symbol=symbol)
