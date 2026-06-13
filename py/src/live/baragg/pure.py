"""Pure functions for bar aggregation.

All functions are pure: they take immutable state and inputs, return
new immutable state. No side effects, no mutations.

These are the testable core — the TimeBarAggregator class is a thin
mutable wrapper around them for convenience in the live engine loop.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.live.baragg.types import AggregatedBar, AggregatorState, PartialBar
from src.bt.state.types import Candle


def create_initial_state(symbol: str, interval: str) -> AggregatorState:
    """Create the initial (empty) aggregator state for a symbol.

    Args:
        symbol: Ticker symbol (e.g. "AAPL")
        interval: Bar interval (e.g. "5min", "1h", "1d")
    """
    return AggregatorState(symbol=symbol, interval=interval)


def baragg_tick(state: AggregatorState, tick: Candle) -> AggregatorState:
    """Process one tick. Returns new state with any completed bars.

    Bars are emitted when a tick crosses into a new interval bucket.
    The completed bars appear in `state.completed` — callers consume
    them from the returned state and discard on the next call.

    If the tick's symbol doesn't match the aggregator's symbol, the
    state is returned unchanged.

    Args:
        state: Current aggregator state
        tick: Incoming tick

    Returns:
        New AggregatorState (may be the same object if symbol mismatch)
    """
    if tick.symbol != state.symbol:
        return state

    bucket = tick.timestamp.floor(state.interval)

    if state.partial is None:
        # First tick — start a new partial bar
        return _start_bar(state, tick, bucket)

    assert state.bucket is not None

    if bucket > state.bucket:
        # Bucket rolled over: emit completed bar, start new
        completed_bar = _partial_to_bar(
            symbol=state.symbol,
            partial=state.partial,
            bucket=state.bucket,
            interval=state.interval,
        )
        new_state = _start_bar(state, tick, bucket)
        return replace(new_state, completed=(completed_bar,))

    # Same bucket — update the in-progress bar
    return _update_bar(state, tick)


def baragg_current(state: AggregatorState) -> AggregatedBar | None:
    """Return the in-progress bar, or None if no ticks have been received.

    Args:
        state: Current aggregator state

    Returns:
        The current partial bar as an AggregatedBar, or None
    """
    if state.partial is None or state.bucket is None:
        return None
    return _partial_to_bar(
        symbol=state.symbol,
        partial=state.partial,
        bucket=state.bucket,
        interval=state.interval,
    )


# ── Internal helpers ────────────────────────────────────────────────


def _start_bar(
    state: AggregatorState,
    tick: Candle,
    bucket: pd.Timestamp,
) -> AggregatorState:
    """Start a new partial bar from a tick."""
    partial = PartialBar(
        open=tick.close,
        high=tick.close,
        low=tick.close,
        close=tick.close,
        volume=tick.volume,
        tick_count=1,
    )
    return replace(state, partial=partial, bucket=bucket, completed=())


def _update_bar(state: AggregatorState, tick: Candle) -> AggregatorState:
    """Update the in-progress bar with a new tick in the same bucket."""
    assert state.partial is not None
    p = state.partial
    updated = PartialBar(
        open=p.open,
        high=max(p.high, tick.close),
        low=min(p.low, tick.close),
        close=tick.close,
        volume=p.volume + tick.volume,
        tick_count=p.tick_count + 1,
    )
    return replace(state, partial=updated, completed=())


def _partial_to_bar(
    symbol: str,
    partial: PartialBar,
    bucket: pd.Timestamp,
    interval: str,
) -> AggregatedBar:
    """Convert a PartialBar to a completed AggregatedBar."""
    return AggregatedBar(
        symbol=symbol,
        timestamp=bucket,
        open=partial.open,
        high=partial.high,
        low=partial.low,
        close=partial.close,
        volume=partial.volume,
        interval=interval,
        tick_count=partial.tick_count,
    )
